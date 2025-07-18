"""
Tool manager for agent integration.

This module provides the ToolManager class which serves as the high-level interface
between agents and the tool system. It handles tool selection logic, query generation,
result processing, and integration with conversation context.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

from .base import BaseTool, ToolResult, ToolContext, ToolCapability
from .registry import ToolRegistry
from .executor import ToolExecutor
from .exceptions import ToolError, ToolExecutionError
from ..models import Message


logger = logging.getLogger(__name__)


@dataclass
class ToolSelectionCriteria:
    """
    Criteria for tool selection based on conversation context.
    
    Attributes:
        required_capabilities: Capabilities that tools must have
        preferred_capabilities: Capabilities that are preferred but not required
        context_keywords: Keywords that indicate tool relevance
        temporal_indicators: Words that suggest need for current information
        factual_indicators: Words that suggest need for factual data
        max_tools: Maximum number of tools to select
        relevance_threshold: Minimum relevance score for tool selection
    """
    required_capabilities: Set[ToolCapability] = field(default_factory=set)
    preferred_capabilities: Set[ToolCapability] = field(default_factory=set)
    context_keywords: Set[str] = field(default_factory=set)
    temporal_indicators: Set[str] = field(default_factory=lambda: {
        "current", "latest", "recent", "today", "now", "2024", "2025",
        "this year", "this month", "this week", "up to date", "real-time"
    })
    factual_indicators: Set[str] = field(default_factory=lambda: {
        "what is", "how many", "statistics", "data", "research", "facts",
        "information", "details", "evidence", "proof", "study", "report"
    })
    max_tools: int = 2
    relevance_threshold: float = 0.6

@dataclass
class QueryGenerationContext:
    """
    Context for generating tool queries from conversation.
    
    Attributes:
        conversation_summary: Brief summary of recent conversation
        key_topics: Main topics being discussed
        specific_questions: Specific questions that need answers
        context_entities: Named entities mentioned in context
        user_intent: Inferred user intent or goal
    """
    conversation_summary: str = ""
    key_topics: List[str] = field(default_factory=list)
    specific_questions: List[str] = field(default_factory=list)
    context_entities: List[str] = field(default_factory=list)
    user_intent: str = ""


@dataclass
class ToolExecutionPlan:
    """
    Plan for executing selected tools.
    
    Attributes:
        selected_tools: Tools selected for execution
        queries: Generated queries for each tool
        execution_order: Order in which tools should be executed
        expected_token_usage: Estimated token usage for results
        timeout: Maximum execution time for the plan
    """
    selected_tools: List[BaseTool] = field(default_factory=list)
    queries: Dict[str, str] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    expected_token_usage: int = 0
    timeout: float = 60.0


@dataclass
class ToolManagerConfig:
    """
    Configuration for the tool manager.
    
    Attributes:
        enabled: Whether tool usage is enabled
        max_concurrent_tools: Maximum number of tools to run concurrently
        default_timeout: Default timeout for tool execution
        result_integration_strategy: How to integrate tool results ("append", "summarize", "selective")
        token_budget_percentage: Percentage of available tokens to allocate for tool results
        relevance_analysis_enabled: Whether to perform relevance analysis
        query_optimization_enabled: Whether to optimize generated queries
        fallback_on_failure: Whether to continue without tools if they fail
    """
    enabled: bool = True
    max_concurrent_tools: int = 2
    default_timeout: float = 45.0
    result_integration_strategy: str = "summarize"
    token_budget_percentage: float = 0.3  # 30% of available tokens
    relevance_analysis_enabled: bool = True
    query_optimization_enabled: bool = True
    fallback_on_failure: bool = True

class ToolManager:
    """
    High-level interface for agents to interact with tools.
    
    The ToolManager serves as the orchestration layer between agents and tools,
    providing intelligent tool selection, query generation, execution management,
    and result integration capabilities.
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        executor: Optional[ToolExecutor] = None,
        config: Optional[ToolManagerConfig] = None
    ):
        """
        Initialize the tool manager.
        
        Args:
            registry: Tool registry for discovering available tools
            executor: Tool executor for running tools (creates default if None)
            config: Configuration for the manager
        """
        self.registry = registry
        self.executor = executor or ToolExecutor()
        self.config = config or ToolManagerConfig()
        self.logger = logging.getLogger(f"{__name__}.ToolManager")
        
        # Metrics tracking
        self.execution_stats = {
            'total_evaluations': 0,
            'tools_selected': 0,
            'tools_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time': 0.0,
            'token_usage': 0
        }
        
        self.logger.info("ToolManager initialized")
    
    async def evaluate_and_execute_tools(
        self,
        context: ToolContext,
        agent_capabilities: Optional[Set[str]] = None
    ) -> List[ToolResult]:
        """
        Evaluate which tools to use and execute them based on conversation context.
        
        This is the main entry point for agents to request tool assistance.
        
        Args:
            context: The conversation context for tool evaluation
            agent_capabilities: Optional set of capabilities the agent supports
            
        Returns:
            List of tool results from executed tools
        """
        if not self.config.enabled:
            self.logger.debug("Tool usage is disabled")
            return []
        
        self.execution_stats['total_evaluations'] += 1
        start_time = time.time()
        
        try:
            # Step 1: Analyze conversation context for tool relevance
            if not self._should_use_tools(context):
                self.logger.debug("No tools needed for current context")
                return []
            
            # Step 2: Select appropriate tools
            execution_plan = await self._create_execution_plan(context, agent_capabilities)
            
            if not execution_plan.selected_tools:
                self.logger.debug("No relevant tools found for context")
                return []
            
            # Step 3: Execute selected tools
            results = await self._execute_tools(execution_plan, context)
            
            # Step 4: Process and integrate results
            processed_results = self._process_results(results, context)
            
            # Update statistics
            execution_time = time.time() - start_time
            self.execution_stats['total_execution_time'] += execution_time
            self.execution_stats['tools_selected'] += len(execution_plan.selected_tools)
            self.execution_stats['tools_executed'] += len(results)
            self.execution_stats['successful_executions'] += sum(1 for r in results if r.success)
            self.execution_stats['failed_executions'] += sum(1 for r in results if not r.success)
            self.execution_stats['token_usage'] += sum(r.token_count for r in results)
            
            self.logger.info(
                f"Tool execution completed: {len(results)} tools executed in {execution_time:.2f}s"
            )
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Tool evaluation and execution failed: {str(e)}")
            if self.config.fallback_on_failure:
                return []
            else:
                raise ToolError(f"Tool manager execution failed: {str(e)}")  
  
    def _should_use_tools(self, context: ToolContext) -> bool:
        """
        Determine if tools should be used based on conversation context.
        
        Args:
            context: The conversation context to analyze
            
        Returns:
            bool: True if tools should be used, False otherwise
        """
        if not self.config.relevance_analysis_enabled:
            return True
        
        # Analyze recent messages for tool relevance indicators
        recent_messages = context.get_recent_messages(3)
        
        if not recent_messages:
            return False
        
        # Combine recent message content for analysis
        combined_content = " ".join(msg.content.lower() for msg in recent_messages)
        
        # Check for temporal indicators (need for current information)
        criteria = ToolSelectionCriteria()
        temporal_score = sum(
            1 for indicator in criteria.temporal_indicators
            if indicator in combined_content
        ) / len(criteria.temporal_indicators)
        
        # Check for factual indicators (need for factual data)
        factual_score = sum(
            1 for indicator in criteria.factual_indicators
            if indicator in combined_content
        ) / len(criteria.factual_indicators)
        
        # Check for question patterns
        question_patterns = [
            r'\bwhat\s+is\b', r'\bhow\s+many\b', r'\bwhen\s+did\b',
            r'\bwhere\s+is\b', r'\bwho\s+is\b', r'\bwhy\s+did\b'
        ]
        question_score = sum(
            1 for pattern in question_patterns
            if re.search(pattern, combined_content, re.IGNORECASE)
        ) / len(question_patterns)
        
        # Calculate overall relevance score
        relevance_score = (temporal_score + factual_score + question_score) / 3
        
        self.logger.debug(
            f"Tool relevance analysis: temporal={temporal_score:.2f}, "
            f"factual={factual_score:.2f}, questions={question_score:.2f}, "
            f"overall={relevance_score:.2f}"
        )
        
        return relevance_score >= 0.1  # Lower threshold for broader tool usage
    
    async def _create_execution_plan(
        self,
        context: ToolContext,
        agent_capabilities: Optional[Set[str]] = None
    ) -> ToolExecutionPlan:
        """
        Create an execution plan by selecting tools and generating queries.
        
        Args:
            context: The conversation context
            agent_capabilities: Optional agent capabilities
            
        Returns:
            ToolExecutionPlan: The execution plan
        """
        # Step 1: Get available tools
        available_tools = self.registry.get_all_tools(enabled_only=True)
        
        if not available_tools:
            return ToolExecutionPlan()
        
        # Step 2: Filter tools by relevance
        relevant_tools = []
        for tool in available_tools:
            try:
                if tool.is_relevant(context):
                    relevant_tools.append(tool)
            except Exception as e:
                self.logger.warning(f"Error checking tool relevance for {tool}: {e}")
        
        if not relevant_tools:
            return ToolExecutionPlan()
        
        # Step 3: Select best tools based on criteria
        selected_tools = self._select_tools(relevant_tools, context)
        
        if not selected_tools:
            return ToolExecutionPlan()
        
        # Step 4: Generate queries for selected tools
        query_context = self._extract_query_context(context)
        queries = {}
        
        for tool in selected_tools:
            try:
                query = self._generate_query_for_tool(tool, query_context, context)
                if query:
                    tool_info = tool.get_tool_info()
                    queries[tool_info.name] = query
            except Exception as e:
                self.logger.warning(f"Error generating query for tool {tool}: {e}")
        
        # Step 5: Determine execution order and estimate token usage
        execution_order = [tool.get_tool_info().name for tool in selected_tools if tool.get_tool_info().name in queries]
        expected_tokens = sum(tool.get_tool_info().token_cost_estimate for tool in selected_tools)
        
        return ToolExecutionPlan(
            selected_tools=selected_tools,
            queries=queries,
            execution_order=execution_order,
            expected_token_usage=expected_tokens,
            timeout=self.config.default_timeout
        )    

    def _select_tools(self, available_tools: List[BaseTool], context: ToolContext) -> List[BaseTool]:
        """
        Select the best tools from available options.
        
        Args:
            available_tools: List of available tools
            context: The conversation context
            
        Returns:
            List of selected tools
        """
        # For now, implement a simple selection strategy
        # In the future, this could be enhanced with more sophisticated scoring
        
        # Prioritize tools by capability and recent usage
        tool_scores = []
        
        for tool in available_tools:
            score = 0.0
            tool_info = tool.get_tool_info()
            
            # Base score from tool capabilities
            if ToolCapability.SEARCH in tool_info.capabilities:
                score += 1.0  # Search is highly valuable
            
            # Adjust score based on estimated execution time
            if tool_info.max_execution_time < 10.0:
                score += 0.5  # Prefer faster tools
            
            # Adjust score based on token cost
            if tool_info.token_cost_estimate < 500:
                score += 0.3  # Prefer lower token cost
            
            tool_scores.append((tool, score))
        
        # Sort by score and select top tools
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        max_tools = min(self.config.max_concurrent_tools, len(tool_scores))
        
        selected_tools = [tool for tool, score in tool_scores[:max_tools]]
        
        self.logger.debug(f"Selected {len(selected_tools)} tools from {len(available_tools)} available")
        
        return selected_tools
    
    def _extract_query_context(self, context: ToolContext) -> QueryGenerationContext:
        """
        Extract relevant information from conversation context for query generation.
        
        Args:
            context: The conversation context
            
        Returns:
            QueryGenerationContext: Extracted context for query generation
        """
        recent_messages = context.get_recent_messages(5)
        
        if not recent_messages:
            return QueryGenerationContext()
        
        # Extract key information from recent messages
        combined_content = " ".join(msg.content for msg in recent_messages)
        
        # Simple keyword extraction (could be enhanced with NLP)
        key_topics = self._extract_key_topics(combined_content)
        specific_questions = self._extract_questions(combined_content)
        context_entities = self._extract_entities(combined_content)
        
        # Generate conversation summary
        conversation_summary = context.get_conversation_summary()
        
        # Infer user intent (simplified)
        user_intent = self._infer_user_intent(combined_content)
        
        return QueryGenerationContext(
            conversation_summary=conversation_summary,
            key_topics=key_topics,
            specific_questions=specific_questions,
            context_entities=context_entities,
            user_intent=user_intent
        )  
  
    def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content."""
        # Simple implementation - could be enhanced with NLP
        words = content.lower().split()
        
        # Filter out common words and extract potential topics
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those'
        }
        
        topics = []
        for word in words:
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 3 and clean_word not in stop_words:
                topics.append(clean_word)
        
        # Return most frequent topics (simplified)
        from collections import Counter
        topic_counts = Counter(topics)
        return [topic for topic, count in topic_counts.most_common(5)]
    
    def _extract_questions(self, content: str) -> List[str]:
        """Extract questions from content."""
        sentences = re.split(r'[.!?]+', content)
        questions = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and (sentence.endswith('?') or 
                           any(sentence.lower().startswith(q) for q in ['what', 'how', 'when', 'where', 'who', 'why'])):
                questions.append(sentence)
        
        return questions[:3]  # Return up to 3 questions
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content."""
        # Simple implementation - could be enhanced with NER
        # Look for capitalized words that might be entities
        words = content.split()
        entities = []
        
        for word in words:
            # Simple heuristic: capitalized words that aren't at sentence start
            clean_word = re.sub(r'[^\w]', '', word)
            if (clean_word and clean_word[0].isupper() and 
                len(clean_word) > 2 and clean_word not in ['The', 'This', 'That']):
                entities.append(clean_word)
        
        return list(set(entities))[:5]  # Return unique entities, max 5
    
    def _infer_user_intent(self, content: str) -> str:
        """Infer user intent from content."""
        content_lower = content.lower()
        
        # Simple intent classification
        if any(word in content_lower for word in ['search', 'find', 'look for', 'information about']):
            return "information_seeking"
        elif any(word in content_lower for word in ['current', 'latest', 'recent', 'now']):
            return "current_information"
        elif any(word in content_lower for word in ['how many', 'statistics', 'data', 'numbers']):
            return "data_inquiry"
        elif any(word in content_lower for word in ['what is', 'define', 'explain']):
            return "definition_request"
        else:
            return "general_inquiry"    
    
    def _generate_query_for_tool(
        self,
        tool: BaseTool,
        query_context: QueryGenerationContext,
        context: ToolContext
    ) -> str:
        """
        Generate an optimized query for a specific tool.
        
        Args:
            tool: The tool to generate a query for
            query_context: Context for query generation
            context: The conversation context
            
        Returns:
            str: Generated query for the tool
        """
        tool_info = tool.get_tool_info()
        
        # Start with key topics and questions
        query_parts = []
        
        # Add specific questions if available
        if query_context.specific_questions:
            query_parts.extend(query_context.specific_questions[:2])
        
        # Add key topics
        if query_context.key_topics:
            topic_query = " ".join(query_context.key_topics[:3])
            query_parts.append(topic_query)
        
        # Add entities for context
        if query_context.context_entities:
            entity_query = " ".join(query_context.context_entities[:2])
            query_parts.append(entity_query)
        
        # Combine and optimize based on tool capabilities
        if ToolCapability.SEARCH in tool_info.capabilities:
            # For search tools, create a focused search query
            if query_context.user_intent == "current_information":
                query_parts.insert(0, "latest")
            elif query_context.user_intent == "data_inquiry":
                query_parts.insert(0, "statistics data")
        
        # Combine parts into final query
        query = " ".join(query_parts).strip()
        
        # Apply query optimization if enabled
        if self.config.query_optimization_enabled:
            query = self._optimize_query(query, tool_info)
        
        self.logger.debug(f"Generated query for {tool_info.name}: '{query}'")
        
        return query
    
    def _optimize_query(self, query: str, tool_info) -> str:
        """
        Optimize a query for better tool performance.
        
        Args:
            query: The original query
            tool_info: Information about the tool
            
        Returns:
            str: Optimized query
        """
        if not query:
            return query
        
        # Remove redundant words
        words = query.split()
        unique_words = []
        seen = set()
        
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                unique_words.append(word)
                seen.add(word_lower)
        
        optimized_query = " ".join(unique_words)
        
        # Limit query length for efficiency
        if len(optimized_query) > 200:
            optimized_query = optimized_query[:197] + "..."
        
        return optimized_query    

    async def _execute_tools(
        self,
        execution_plan: ToolExecutionPlan,
        context: ToolContext
    ) -> List[ToolResult]:
        """
        Execute the tools according to the execution plan.
        
        Args:
            execution_plan: The execution plan
            context: The conversation context
            
        Returns:
            List of tool results
        """
        if not execution_plan.selected_tools:
            return []
        
        # Execute tools concurrently
        tasks = []
        
        for tool in execution_plan.selected_tools:
            tool_info = tool.get_tool_info()
            query = execution_plan.queries.get(tool_info.name, "")
            
            if query:
                task = self.executor.execute_tool(tool, query, context)
                tasks.append(task)
        
        if not tasks:
            return []
        
        # Wait for all tools to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    tool_name = execution_plan.selected_tools[i].get_tool_info().name
                    self.logger.error(f"Tool {tool_name} execution failed: {result}")
                    
                    # Create error result
                    error_result = ToolResult(
                        success=False,
                        content="",
                        tool_name=tool_name,
                        errors=[str(result)]
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Tool execution batch failed: {e}")
            return [] 
   
    def _process_results(
        self,
        results: List[ToolResult],
        context: ToolContext
    ) -> List[ToolResult]:
        """
        Process and integrate tool results.
        
        Args:
            results: Raw tool results
            context: The conversation context
            
        Returns:
            List of processed tool results
        """
        if not results:
            return results
        
        processed_results = []
        token_budget = int(context.available_tokens * self.config.token_budget_percentage)
        used_tokens = 0
        
        for result in results:
            if not result.success:
                processed_results.append(result)
                continue
            
            # Apply result integration strategy
            if self.config.result_integration_strategy == "summarize":
                # Summarize result if it's too long
                if result.token_count > 500:
                    summarized_content = self._summarize_result(result.content, 300)
                    result.content = summarized_content
                    result.token_count = len(summarized_content.split()) * 1.3  # Rough estimate
            
            elif self.config.result_integration_strategy == "selective":
                # Select most relevant parts of the result
                result.content = self._select_relevant_content(result.content, context)
                result.token_count = len(result.content.split()) * 1.3  # Rough estimate
            
            # Check token budget
            if used_tokens + result.token_count <= token_budget:
                processed_results.append(result)
                used_tokens += result.token_count
            else:
                # Truncate result to fit budget
                remaining_budget = token_budget - used_tokens
                if remaining_budget > 50:  # Only include if meaningful space left
                    truncated_content = self._truncate_content(result.content, remaining_budget)
                    result.content = truncated_content
                    result.token_count = remaining_budget
                    processed_results.append(result)
                break
        
        self.logger.debug(f"Processed {len(processed_results)} tool results using {used_tokens} tokens")
        
        return processed_results
    
    def _summarize_result(self, content: str, max_length: int) -> str:
        """
        Summarize tool result content to fit within token limits.
        
        Args:
            content: The content to summarize
            max_length: Maximum length in characters
            
        Returns:
            str: Summarized content
        """
        if len(content) <= max_length:
            return content
        
        # Simple summarization: take first part and last part
        first_part = content[:max_length // 2]
        last_part = content[-(max_length // 2):]
        
        return f"{first_part}...\n\n...{last_part}"
    
    def _select_relevant_content(self, content: str, context: ToolContext) -> str:
        """
        Select the most relevant parts of tool result content.
        
        Args:
            content: The content to filter
            context: The conversation context
            
        Returns:
            str: Filtered content
        """
        # Simple implementation: return first 1000 characters
        # Could be enhanced with more sophisticated relevance scoring
        if len(content) <= 1000:
            return content
        
        return content[:997] + "..."
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """
        Truncate content to fit within token limit.
        
        Args:
            content: The content to truncate
            max_tokens: Maximum number of tokens
            
        Returns:
            str: Truncated content
        """
        # Rough estimation: 1 token ≈ 0.75 characters
        max_chars = int(max_tokens * 0.75)
        
        if len(content) <= max_chars:
            return content
        
        return content[:max_chars - 3] + "..."    

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tool manager execution statistics.
        
        Returns:
            Dict containing execution statistics
        """
        stats = self.execution_stats.copy()
        
        # Add derived metrics
        if stats['total_evaluations'] > 0:
            stats['tool_usage_rate'] = stats['tools_executed'] / stats['total_evaluations']
            stats['success_rate'] = (
                stats['successful_executions'] / stats['tools_executed']
                if stats['tools_executed'] > 0 else 0.0
            )
            stats['average_execution_time'] = (
                stats['total_execution_time'] / stats['total_evaluations']
            )
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the tool manager.
        
        Returns:
            Dict containing health status information
        """
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'enabled': self.config.enabled,
                'max_concurrent_tools': self.config.max_concurrent_tools,
                'default_timeout': self.config.default_timeout
            },
            'registry_status': {
                'total_tools': len(self.registry),
                'enabled_tools': len(self.registry.list_tools(enabled_only=True))
            },
            'statistics': self.get_statistics()
        }
        
        # Check executor health
        try:
            executor_health = await self.executor.health_check()
            health['executor_status'] = executor_health['status']
        except Exception as e:
            health['executor_status'] = 'error'
            health['executor_error'] = str(e)
        
        return health
    
    async def close(self):
        """Clean up resources."""
        await self.executor.close()
        self.logger.info("Tool manager closed")