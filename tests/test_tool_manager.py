"""
Unit tests for the ToolManager class.

This module tests the tool manager's functionality including tool selection logic,
execution orchestration, query generation, result processing, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Set
from datetime import datetime

from src.agentic_conversation.tools.manager import (
    ToolManager,
    ToolManagerConfig,
    ToolSelectionCriteria,
    QueryGenerationContext,
    ToolExecutionPlan
)
from src.agentic_conversation.tools.base import (
    BaseTool,
    ToolResult,
    ToolContext,
    ToolInfo,
    ToolCapability
)
from src.agentic_conversation.tools.registry import ToolRegistry
from src.agentic_conversation.tools.executor import ToolExecutor
from src.agentic_conversation.tools.exceptions import ToolError, ToolExecutionError
from src.agentic_conversation.models import Message


class MockTool(BaseTool):
    """Mock tool for testing purposes."""
    
    def __init__(self, name: str = "mock_tool", capabilities: List[ToolCapability] = None, 
                 relevant: bool = True, config: Dict[str, Any] = None):
        self.name = name
        self.capabilities = capabilities or [ToolCapability.SEARCH]
        self.relevant = relevant
        super().__init__(config or {})
    
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        """Mock execute method."""
        return ToolResult(
            success=True,
            content=f"Mock result for query: {query}",
            tool_name=self.name,
            execution_time=0.1,
            token_count=50
        )
    
    def is_relevant(self, context: ToolContext) -> bool:
        """Mock relevance check."""
        return self.relevant
    
    def get_tool_info(self) -> ToolInfo:
        """Mock tool info."""
        return ToolInfo(
            name=self.name,
            description=f"Mock tool: {self.name}",
            capabilities=self.capabilities,
            max_execution_time=10.0,
            token_cost_estimate=100
        )


class TestToolManagerConfig:
    """Test ToolManagerConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ToolManagerConfig()
        
        assert config.enabled is True
        assert config.max_concurrent_tools == 2
        assert config.default_timeout == 45.0
        assert config.result_integration_strategy == "summarize"
        assert config.token_budget_percentage == 0.3
        assert config.relevance_analysis_enabled is True
        assert config.query_optimization_enabled is True
        assert config.fallback_on_failure is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ToolManagerConfig(
            enabled=False,
            max_concurrent_tools=5,
            default_timeout=60.0,
            result_integration_strategy="selective",
            token_budget_percentage=0.5,
            relevance_analysis_enabled=False,
            query_optimization_enabled=False,
            fallback_on_failure=False
        )
        
        assert config.enabled is False
        assert config.max_concurrent_tools == 5
        assert config.default_timeout == 60.0
        assert config.result_integration_strategy == "selective"
        assert config.token_budget_percentage == 0.5
        assert config.relevance_analysis_enabled is False
        assert config.query_optimization_enabled is False
        assert config.fallback_on_failure is False


class TestToolSelectionCriteria:
    """Test ToolSelectionCriteria functionality."""
    
    def test_default_criteria(self):
        """Test default selection criteria."""
        criteria = ToolSelectionCriteria()
        
        assert len(criteria.required_capabilities) == 0
        assert len(criteria.preferred_capabilities) == 0
        assert len(criteria.context_keywords) == 0
        assert len(criteria.temporal_indicators) > 0
        assert len(criteria.factual_indicators) > 0
        assert criteria.max_tools == 2
        assert criteria.relevance_threshold == 0.6
    
    def test_temporal_indicators(self):
        """Test temporal indicators are properly set."""
        criteria = ToolSelectionCriteria()
        
        expected_temporal = {
            "current", "latest", "recent", "today", "now", "2024", "2025",
            "this year", "this month", "this week", "up to date", "real-time"
        }
        
        assert criteria.temporal_indicators == expected_temporal
    
    def test_factual_indicators(self):
        """Test factual indicators are properly set."""
        criteria = ToolSelectionCriteria()
        
        expected_factual = {
            "what is", "how many", "statistics", "data", "research", "facts",
            "information", "details", "evidence", "proof", "study", "report"
        }
        
        assert criteria.factual_indicators == expected_factual


class TestQueryGenerationContext:
    """Test QueryGenerationContext functionality."""
    
    def test_default_context(self):
        """Test default query generation context."""
        context = QueryGenerationContext()
        
        assert context.conversation_summary == ""
        assert len(context.key_topics) == 0
        assert len(context.specific_questions) == 0
        assert len(context.context_entities) == 0
        assert context.user_intent == ""
    
    def test_custom_context(self):
        """Test custom query generation context."""
        context = QueryGenerationContext(
            conversation_summary="Test summary",
            key_topics=["AI", "machine learning"],
            specific_questions=["What is AI?"],
            context_entities=["OpenAI", "GPT"],
            user_intent="information_seeking"
        )
        
        assert context.conversation_summary == "Test summary"
        assert context.key_topics == ["AI", "machine learning"]
        assert context.specific_questions == ["What is AI?"]
        assert context.context_entities == ["OpenAI", "GPT"]
        assert context.user_intent == "information_seeking"


class TestToolExecutionPlan:
    """Test ToolExecutionPlan functionality."""
    
    def test_default_plan(self):
        """Test default execution plan."""
        plan = ToolExecutionPlan()
        
        assert len(plan.selected_tools) == 0
        assert len(plan.queries) == 0
        assert len(plan.execution_order) == 0
        assert plan.expected_token_usage == 0
        assert plan.timeout == 60.0
    
    def test_custom_plan(self):
        """Test custom execution plan."""
        mock_tool = MockTool("test_tool")
        plan = ToolExecutionPlan(
            selected_tools=[mock_tool],
            queries={"test_tool": "test query"},
            execution_order=["test_tool"],
            expected_token_usage=100,
            timeout=30.0
        )
        
        assert len(plan.selected_tools) == 1
        assert plan.queries == {"test_tool": "test query"}
        assert plan.execution_order == ["test_tool"]
        assert plan.expected_token_usage == 100
        assert plan.timeout == 30.0


class TestToolManager:
    """Test ToolManager functionality."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        return registry
    
    @pytest.fixture
    def mock_executor(self):
        """Create a mock tool executor."""
        executor = Mock(spec=ToolExecutor)
        return executor
    
    @pytest.fixture
    def tool_manager(self, mock_registry, mock_executor):
        """Create a ToolManager instance for testing."""
        config = ToolManagerConfig()
        return ToolManager(mock_registry, mock_executor, config)
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample tool context for testing."""
        messages = [
            Message(
                content="What is the latest news about AI?",
                agent_id="user",
                timestamp=datetime.now(),
                token_count=10
            ),
            Message(
                content="I need current information about machine learning trends.",
                agent_id="user", 
                timestamp=datetime.now(),
                token_count=12
            )
        ]
        
        return ToolContext(
            conversation_history=messages,
            current_turn=2,
            agent_id="test_agent",
            available_tokens=4000,
            max_result_tokens=1000,
            timeout=30.0
        )
    
    def test_initialization(self, mock_registry, mock_executor):
        """Test ToolManager initialization."""
        config = ToolManagerConfig(enabled=False)
        manager = ToolManager(mock_registry, mock_executor, config)
        
        assert manager.registry == mock_registry
        assert manager.executor == mock_executor
        assert manager.config == config
        assert manager.execution_stats['total_evaluations'] == 0
    
    def test_initialization_with_defaults(self, mock_registry):
        """Test ToolManager initialization with default executor and config."""
        manager = ToolManager(mock_registry)
        
        assert manager.registry == mock_registry
        assert manager.executor is not None
        assert isinstance(manager.config, ToolManagerConfig)
    
    @pytest.mark.asyncio
    async def test_evaluate_and_execute_tools_disabled(self, tool_manager, sample_context):
        """Test tool evaluation when tools are disabled."""
        tool_manager.config.enabled = False
        
        results = await tool_manager.evaluate_and_execute_tools(sample_context)
        
        assert results == []
        assert tool_manager.execution_stats['total_evaluations'] == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_and_execute_tools_no_relevant_tools(self, tool_manager, sample_context):
        """Test tool evaluation when no tools are relevant."""
        # Mock _should_use_tools to return False
        with patch.object(tool_manager, '_should_use_tools', return_value=False):
            results = await tool_manager.evaluate_and_execute_tools(sample_context)
        
        assert results == []
        assert tool_manager.execution_stats['total_evaluations'] == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_and_execute_tools_success(self, tool_manager, sample_context):
        """Test successful tool evaluation and execution."""
        # Create mock tools
        mock_tool1 = MockTool("search_tool", [ToolCapability.SEARCH])
        
        # Mock the execution plan creation
        execution_plan = ToolExecutionPlan(
            selected_tools=[mock_tool1],
            queries={"search_tool": "latest AI news"},
            execution_order=["search_tool"],
            expected_token_usage=100,
            timeout=30.0
        )
        
        # Mock tool results
        tool_result = ToolResult(
            success=True,
            content="Latest AI news content",
            tool_name="search_tool",
            execution_time=1.5,
            token_count=200
        )
        
        with patch.object(tool_manager, '_should_use_tools', return_value=True), \
             patch.object(tool_manager, '_create_execution_plan', return_value=execution_plan), \
             patch.object(tool_manager, '_execute_tools', return_value=[tool_result]), \
             patch.object(tool_manager, '_process_results', return_value=[tool_result]):
            
            results = await tool_manager.evaluate_and_execute_tools(sample_context)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].tool_name == "search_tool"
        assert tool_manager.execution_stats['total_evaluations'] == 1
        assert tool_manager.execution_stats['tools_selected'] == 1
        assert tool_manager.execution_stats['tools_executed'] == 1
        assert tool_manager.execution_stats['successful_executions'] == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_and_execute_tools_with_failure(self, tool_manager, sample_context):
        """Test tool evaluation with execution failure."""
        tool_manager.config.fallback_on_failure = True
        
        with patch.object(tool_manager, '_should_use_tools', return_value=True), \
             patch.object(tool_manager, '_create_execution_plan', side_effect=Exception("Test error")):
            
            results = await tool_manager.evaluate_and_execute_tools(sample_context)
        
        assert results == []
        assert tool_manager.execution_stats['total_evaluations'] == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_and_execute_tools_no_fallback(self, tool_manager, sample_context):
        """Test tool evaluation with failure and no fallback."""
        tool_manager.config.fallback_on_failure = False
        
        with patch.object(tool_manager, '_should_use_tools', return_value=True), \
             patch.object(tool_manager, '_create_execution_plan', side_effect=Exception("Test error")):
            
            with pytest.raises(ToolError):
                await tool_manager.evaluate_and_execute_tools(sample_context)
    
    def test_should_use_tools_disabled_analysis(self, tool_manager, sample_context):
        """Test _should_use_tools when relevance analysis is disabled."""
        tool_manager.config.relevance_analysis_enabled = False
        
        result = tool_manager._should_use_tools(sample_context)
        
        assert result is True
    
    def test_should_use_tools_no_messages(self, tool_manager):
        """Test _should_use_tools with no conversation history."""
        empty_context = ToolContext(conversation_history=[])
        
        result = tool_manager._should_use_tools(empty_context)
        
        assert result is False
    
    def test_should_use_tools_with_temporal_indicators(self, tool_manager, sample_context):
        """Test _should_use_tools with temporal indicators in messages."""
        # The sample_context already contains temporal indicators like "latest" and "current"
        result = tool_manager._should_use_tools(sample_context)
        
        assert result is True
    
    def test_should_use_tools_with_factual_indicators(self, tool_manager):
        """Test _should_use_tools with factual indicators."""
        messages = [
            Message(
                content="What is machine learning? How many types are there?",
                agent_id="user",
                timestamp=datetime.now(),
                token_count=10
            )
        ]
        
        context = ToolContext(conversation_history=messages)
        result = tool_manager._should_use_tools(context)
        
        assert result is True
    
    def test_should_use_tools_with_question_patterns(self, tool_manager):
        """Test _should_use_tools with question patterns."""
        messages = [
            Message(
                content="When did artificial intelligence start? Where is it used?",
                agent_id="user",
                timestamp=datetime.now(),
                token_count=10
            )
        ]
        
        context = ToolContext(conversation_history=messages)
        result = tool_manager._should_use_tools(context)
        
        assert result is True
    
    def test_should_use_tools_low_relevance(self, tool_manager):
        """Test _should_use_tools with low relevance content."""
        messages = [
            Message(
                content="Hello, how are you doing today?",
                agent_id="user",
                timestamp=datetime.now(),
                token_count=8
            )
        ]
        
        context = ToolContext(conversation_history=messages)
        result = tool_manager._should_use_tools(context)
        
        # Should still return True due to low threshold (0.1)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_create_execution_plan_no_tools(self, tool_manager, sample_context):
        """Test execution plan creation with no available tools."""
        tool_manager.registry.get_all_tools.return_value = []
        
        plan = await tool_manager._create_execution_plan(sample_context)
        
        assert len(plan.selected_tools) == 0
        assert len(plan.queries) == 0
    
    @pytest.mark.asyncio
    async def test_create_execution_plan_no_relevant_tools(self, tool_manager, sample_context):
        """Test execution plan creation with no relevant tools."""
        mock_tool = MockTool("irrelevant_tool", relevant=False)
        tool_manager.registry.get_all_tools.return_value = [mock_tool]
        
        plan = await tool_manager._create_execution_plan(sample_context)
        
        assert len(plan.selected_tools) == 0
    
    @pytest.mark.asyncio
    async def test_create_execution_plan_success(self, tool_manager, sample_context):
        """Test successful execution plan creation."""
        mock_tool = MockTool("search_tool", [ToolCapability.SEARCH])
        tool_manager.registry.get_all_tools.return_value = [mock_tool]
        
        with patch.object(tool_manager, '_select_tools', return_value=[mock_tool]), \
             patch.object(tool_manager, '_extract_query_context') as mock_extract, \
             patch.object(tool_manager, '_generate_query_for_tool', return_value="test query"):
            
            mock_extract.return_value = QueryGenerationContext()
            plan = await tool_manager._create_execution_plan(sample_context)
        
        assert len(plan.selected_tools) == 1
        assert "search_tool" in plan.queries
        assert plan.queries["search_tool"] == "test query"
        assert plan.execution_order == ["search_tool"]
    
    def test_select_tools(self, tool_manager, sample_context):
        """Test tool selection logic."""
        # Create tools with different characteristics
        fast_tool = MockTool("fast_tool", [ToolCapability.SEARCH])
        fast_tool.get_tool_info().max_execution_time = 5.0
        fast_tool.get_tool_info().token_cost_estimate = 100
        
        slow_tool = MockTool("slow_tool", [ToolCapability.ANALYSIS])
        slow_tool.get_tool_info().max_execution_time = 30.0
        slow_tool.get_tool_info().token_cost_estimate = 500
        
        available_tools = [fast_tool, slow_tool]
        
        selected = tool_manager._select_tools(available_tools, sample_context)
        
        # Should select both tools, but fast_tool should be preferred
        assert len(selected) <= tool_manager.config.max_concurrent_tools
        assert fast_tool in selected
    
    def test_extract_query_context(self, tool_manager, sample_context):
        """Test query context extraction."""
        with patch.object(tool_manager, '_extract_key_topics', return_value=["AI", "news"]), \
             patch.object(tool_manager, '_extract_questions', return_value=["What is AI?"]), \
             patch.object(tool_manager, '_extract_entities', return_value=["OpenAI"]), \
             patch.object(tool_manager, '_infer_user_intent', return_value="information_seeking"):
            
            context = tool_manager._extract_query_context(sample_context)
        
        assert context.key_topics == ["AI", "news"]
        assert context.specific_questions == ["What is AI?"]
        assert context.context_entities == ["OpenAI"]
        assert context.user_intent == "information_seeking"
    
    def test_extract_key_topics(self, tool_manager):
        """Test key topic extraction."""
        content = "I want to learn about machine learning and artificial intelligence algorithms"
        
        topics = tool_manager._extract_key_topics(content)
        
        # Check that some key topics are extracted (implementation filters by length > 3)
        assert "machine" in topics
        assert "learning" in topics
        assert "artificial" in topics
        assert "intelligence" in topics
        assert "algorithms" in topics
        # Check that topics list is not empty
        assert len(topics) > 0
    
    def test_extract_questions(self, tool_manager):
        """Test question extraction."""
        content = "What is AI? How does machine learning work? This is not a question."
        
        questions = tool_manager._extract_questions(content)
        
        assert "What is AI?" in questions
        assert "How does machine learning work?" in questions
        assert len(questions) <= 3  # Should limit to 3 questions
    
    def test_extract_entities(self, tool_manager):
        """Test entity extraction."""
        content = "OpenAI created GPT-4 and ChatGPT for natural language processing"
        
        entities = tool_manager._extract_entities(content)
        
        assert "OpenAI" in entities
        assert "GPT" in entities or "GPT-4" in entities
        assert "ChatGPT" in entities
        assert len(entities) <= 5  # Should limit to 5 entities
    
    def test_infer_user_intent(self, tool_manager):
        """Test user intent inference."""
        # Test different intent patterns
        search_content = "I want to search for information about AI"
        assert tool_manager._infer_user_intent(search_content) == "information_seeking"
        
        current_content = "What is the latest news about technology?"
        assert tool_manager._infer_user_intent(current_content) == "current_information"
        
        data_content = "How many people use AI? Show me statistics"
        assert tool_manager._infer_user_intent(data_content) == "data_inquiry"
        
        definition_content = "What is machine learning? Please explain"
        assert tool_manager._infer_user_intent(definition_content) == "definition_request"
        
        general_content = "Tell me something interesting"
        assert tool_manager._infer_user_intent(general_content) == "general_inquiry"
    
    def test_generate_query_for_tool(self, tool_manager):
        """Test query generation for specific tools."""
        mock_tool = MockTool("search_tool", [ToolCapability.SEARCH])
        
        query_context = QueryGenerationContext(
            key_topics=["AI", "machine learning"],
            specific_questions=["What is AI?"],
            context_entities=["OpenAI"],
            user_intent="current_information"
        )
        
        tool_context = ToolContext()
        
        query = tool_manager._generate_query_for_tool(mock_tool, query_context, tool_context)
        
        assert "What is AI?" in query
        assert "AI" in query
        assert "machine learning" in query
        assert "OpenAI" in query
        # Should add "latest" for current_information intent
        assert "latest" in query
    
    def test_optimize_query(self, tool_manager):
        """Test query optimization."""
        tool_info = ToolInfo(name="test_tool", description="Test tool")
        
        # Test deduplication
        query = "AI machine learning AI artificial intelligence"
        optimized = tool_manager._optimize_query(query, tool_info)
        
        # Should remove duplicate "AI"
        assert optimized.count("AI") == 1
        assert "machine learning" in optimized
        assert "artificial intelligence" in optimized
        
        # Test length limiting
        long_query = "word " * 100  # 500 characters
        optimized_long = tool_manager._optimize_query(long_query, tool_info)
        
        assert len(optimized_long) <= 200
        assert optimized_long.endswith("...")
    
    @pytest.mark.asyncio
    async def test_execute_tools(self, tool_manager, sample_context):
        """Test tool execution."""
        mock_tool = MockTool("test_tool")
        execution_plan = ToolExecutionPlan(
            selected_tools=[mock_tool],
            queries={"test_tool": "test query"}
        )
        
        # Mock executor
        expected_result = ToolResult(
            success=True,
            content="Test result",
            tool_name="test_tool"
        )
        tool_manager.executor.execute_tool = AsyncMock(return_value=expected_result)
        
        results = await tool_manager._execute_tools(execution_plan, sample_context)
        
        assert len(results) == 1
        assert results[0] == expected_result
        tool_manager.executor.execute_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_tools_with_exception(self, tool_manager, sample_context):
        """Test tool execution with exceptions."""
        mock_tool = MockTool("test_tool")
        execution_plan = ToolExecutionPlan(
            selected_tools=[mock_tool],
            queries={"test_tool": "test query"}
        )
        
        # Mock executor to raise exception
        tool_manager.executor.execute_tool = AsyncMock(side_effect=Exception("Test error"))
        
        results = await tool_manager._execute_tools(execution_plan, sample_context)
        
        assert len(results) == 1
        assert results[0].success is False
        assert "Test error" in results[0].errors
    
    def test_process_results_summarize_strategy(self, tool_manager, sample_context):
        """Test result processing with summarize strategy."""
        tool_manager.config.result_integration_strategy = "summarize"
        
        # Create a result that exceeds the token limit
        large_result = ToolResult(
            success=True,
            content="x" * 1000,  # Large content
            tool_name="test_tool",
            token_count=600  # Exceeds 500 token limit
        )
        
        with patch.object(tool_manager, '_summarize_result', return_value="summarized content"):
            processed = tool_manager._process_results([large_result], sample_context)
        
        assert len(processed) == 1
        assert processed[0].content == "summarized content"
    
    def test_process_results_selective_strategy(self, tool_manager, sample_context):
        """Test result processing with selective strategy."""
        tool_manager.config.result_integration_strategy = "selective"
        
        result = ToolResult(
            success=True,
            content="test content",
            tool_name="test_tool",
            token_count=100
        )
        
        with patch.object(tool_manager, '_select_relevant_content', return_value="selected content"):
            processed = tool_manager._process_results([result], sample_context)
        
        assert len(processed) == 1
        assert processed[0].content == "selected content"
    
    def test_process_results_token_budget(self, tool_manager, sample_context):
        """Test result processing respects token budget."""
        # Set a small token budget
        sample_context.available_tokens = 1000
        tool_manager.config.token_budget_percentage = 0.3  # 300 tokens budget
        
        # Create results that exceed budget
        result1 = ToolResult(success=True, content="content1", tool_name="tool1", token_count=200)
        result2 = ToolResult(success=True, content="content2", tool_name="tool2", token_count=200)
        
        processed = tool_manager._process_results([result1, result2], sample_context)
        
        # Should only include first result due to budget constraints
        assert len(processed) == 1
        assert processed[0].tool_name == "tool1"
    
    def test_summarize_result(self, tool_manager):
        """Test result summarization."""
        content = "A" * 1000  # 1000 character content
        
        summarized = tool_manager._summarize_result(content, 200)
        
        assert len(summarized) <= 200
        assert "..." in summarized
        assert summarized.startswith("A")
        assert summarized.endswith("A")
    
    def test_select_relevant_content(self, tool_manager, sample_context):
        """Test relevant content selection."""
        content = "A" * 2000  # 2000 character content
        
        selected = tool_manager._select_relevant_content(content, sample_context)
        
        assert len(selected) <= 1000
        if len(content) > 1000:
            assert selected.endswith("...")
    
    def test_truncate_content(self, tool_manager):
        """Test content truncation."""
        content = "A" * 1000  # 1000 character content
        
        truncated = tool_manager._truncate_content(content, 100)  # 100 tokens ≈ 75 chars
        
        assert len(truncated) <= 75
        assert truncated.endswith("...")
    
    def test_get_statistics(self, tool_manager):
        """Test statistics retrieval."""
        # Update some statistics
        tool_manager.execution_stats['total_evaluations'] = 10
        tool_manager.execution_stats['tools_executed'] = 8
        tool_manager.execution_stats['successful_executions'] = 6
        
        stats = tool_manager.get_statistics()
        
        assert stats['total_evaluations'] == 10
        assert stats['tools_executed'] == 8
        assert stats['successful_executions'] == 6
        assert 'tool_usage_rate' in stats
        assert 'success_rate' in stats
        assert stats['tool_usage_rate'] == 0.8  # 8/10
        assert stats['success_rate'] == 0.75  # 6/8
    
    def test_get_statistics_no_executions(self, tool_manager):
        """Test statistics with no executions."""
        stats = tool_manager.get_statistics()
        
        assert stats['total_evaluations'] == 0
        assert stats['tools_executed'] == 0
        assert stats['tool_usage_rate'] == 0.0
        assert stats['success_rate'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])