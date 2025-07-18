"""
Web search tool implementation with support for multiple search providers.

This module implements a web search tool that can use different search providers
(Tavily, SerpAPI, DuckDuckGo) to retrieve current information from the web.
The tool includes query optimization, result processing, and configurable relevance detection.
"""

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import ssl

from .base import BaseTool, ToolResult, ToolContext, ToolInfo, ToolCapability
from .exceptions import ToolExecutionError, ToolTimeoutError, ToolConfigurationError


class SearchProvider:
    """Base class for search providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30.0)
        self.max_results = config.get("max_results", 5)
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Execute search query and return results."""
        raise NotImplementedError("Subclasses must implement search method")
    
    def _make_request(self, url: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        if headers is None:
            headers = {}
        
        # Add default headers
        headers.setdefault("User-Agent", "Mozilla/5.0 (compatible; AgenticConversation/1.0)")
        
        try:
            request = Request(url, headers=headers)
            
            # Create SSL context that doesn't verify certificates for development
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            with urlopen(request, timeout=self.timeout, context=ssl_context) as response:
                content = response.read().decode('utf-8')
                return json.loads(content)
                
        except HTTPError as e:
            raise ToolExecutionError(f"HTTP error {e.code}: {e.reason}")
        except URLError as e:
            raise ToolExecutionError(f"URL error: {e.reason}")
        except json.JSONDecodeError as e:
            raise ToolExecutionError(f"JSON decode error: {e}")
        except Exception as e:
            raise ToolExecutionError(f"Request failed: {str(e)}")


class TavilySearchProvider(SearchProvider):
    """Tavily search provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.api_key:
            raise ToolConfigurationError("Tavily API key is required")
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_images": False,
            "include_raw_content": False,
            "max_results": self.max_results
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            # Convert to POST request with JSON payload
            request = Request(
                url, 
                data=json.dumps(payload).encode('utf-8'),
                headers=headers,
                method='POST'
            )
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            with urlopen(request, timeout=self.timeout, context=ssl_context) as response:
                content = response.read().decode('utf-8')
                data = json.loads(content)
            
            results = []
            if "results" in data:
                for item in data["results"]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                        "published_date": item.get("published_date", ""),
                        "score": item.get("score", 0.0)
                    })
            
            return results
            
        except Exception as e:
            raise ToolExecutionError(f"Tavily search failed: {str(e)}")


class SerpAPIProvider(SearchProvider):
    """SerpAPI search provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not self.api_key:
            raise ToolConfigurationError("SerpAPI key is required")
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search using SerpAPI."""
        base_url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": str(self.max_results),
            "format": "json"
        }
        
        # Build URL with parameters
        param_string = "&".join([f"{k}={quote_plus(str(v))}" for k, v in params.items()])
        url = f"{base_url}?{param_string}"
        
        try:
            data = self._make_request(url)
            
            results = []
            if "organic_results" in data:
                for item in data["organic_results"]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "published_date": item.get("date", ""),
                        "score": 1.0  # SerpAPI doesn't provide relevance scores
                    })
            
            return results
            
        except Exception as e:
            raise ToolExecutionError(f"SerpAPI search failed: {str(e)}")


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider implementation (no API key required)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # DuckDuckGo doesn't require API key
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo instant answer API."""
        # Use DuckDuckGo's instant answer API
        base_url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        param_string = "&".join([f"{k}={quote_plus(str(v))}" for k, v in params.items()])
        url = f"{base_url}?{param_string}"
        
        try:
            data = self._make_request(url)
            
            results = []
            
            # Process abstract (main answer)
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "DuckDuckGo Answer"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", ""),
                    "published_date": "",
                    "score": 1.0
                })
            
            # Process related topics
            for topic in data.get("RelatedTopics", [])[:self.max_results-1]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else "Related Topic",
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "published_date": "",
                        "score": 0.8
                    })
            
            # If no results, try to extract from definition
            if not results and data.get("Definition"):
                results.append({
                    "title": f"Definition: {query}",
                    "url": data.get("DefinitionURL", ""),
                    "snippet": data.get("Definition", ""),
                    "published_date": "",
                    "score": 0.9
                })
            
            return results[:self.max_results]
            
        except Exception as e:
            raise ToolExecutionError(f"DuckDuckGo search failed: {str(e)}")


class WebSearchTool(BaseTool):
    """
    Web search tool with support for multiple search providers.
    
    This tool can search the web using different providers (Tavily, SerpAPI, DuckDuckGo)
    and return processed, summarized results that are relevant to the conversation context.
    Features configurable relevance detection with customizable weights and thresholds.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the web search tool."""
        # Set attributes before calling super().__init__ to avoid validation issues
        self.config = config or {}
        
        # Get provider configuration
        self.provider_name = self.get_config_value("provider", "duckduckgo").lower()
        self.max_results = self.get_config_value("max_results", 5)
        self.result_summary_length = self.get_config_value("result_summary_length", 200)
        self.relevance_threshold = self.get_config_value("relevance_threshold", 0.7)
        
        # Initialize search provider
        self.provider = self._create_provider()
        
        # Now call parent initialization
        super().__init__(config)
        
        # Configurable relevance detection settings
        self.relevance_score_threshold = self.get_config_value("relevance_score_threshold", 2.0)
        self.temporal_weight = self.get_config_value("temporal_weight", 2.0)
        self.factual_weight = self.get_config_value("factual_weight", 1.5)
        self.question_weight = self.get_config_value("question_weight", 1.0)
        self.conversational_penalty = self.get_config_value("conversational_penalty", -1.0)
        
        # Configurable relevance detection keywords (general indicators)
        default_relevance_indicators = [
            "current", "latest", "recent", "today", "now", "2024", "2025",
            "what is", "how many", "news", "update", "happening", "status", 
            "price", "cost", "when did", "where is", "who is", "define", "explain"
        ]
        self.relevance_indicators = self.get_config_value("relevance_indicators", default_relevance_indicators)
        
        # Configurable temporal indicators that suggest need for current information
        default_temporal_indicators = [
            "today", "yesterday", "this week", "this month", "this year",
            "currently", "now", "recently", "lately", "ongoing",
            "latest", "newest", "current", "up-to-date", "real-time"
        ]
        self.temporal_indicators = self.get_config_value("temporal_indicators", default_temporal_indicators)
        
        # Configurable factual indicators
        default_factual_indicators = [
            "statistics", "data", "research", "study", "report", "analysis",
            "facts", "information", "details", "evidence", "findings",
            "survey", "poll", "census", "measurement", "calculation"
        ]
        self.factual_indicators = self.get_config_value("factual_indicators", default_factual_indicators)
        
        # Configurable question patterns (compiled regex patterns)
        default_question_patterns = [
            r'\bwhat\s+is\s+\w+', r'\bhow\s+many\s+\w+', r'\bwhen\s+did\s+\w+',
            r'\bwhere\s+is\s+\w+', r'\bwho\s+is\s+\w+', r'\bwhy\s+is\s+\w+',
            r'\bhow\s+to\s+\w+', r'\bcan\s+you\s+find\s+\w+', r'\btell\s+me\s+about\s+\w+',
            r'\bshow\s+me\s+\w+', r'\bsearch\s+for\s+\w+', r'\blook\s+up\s+\w+'
        ]
        question_patterns = self.get_config_value("question_patterns", default_question_patterns)
        self.question_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in question_patterns]
        
        # Configurable conversational patterns to exclude
        default_conversational_patterns = [
            r'\bhow\s+are\s+you\b', r'\bhello\b', r'\bhi\b', r'\bthanks?\b',
            r'\bthank\s+you\b', r'\bgoodbye\b', r'\bbye\b', r'\bplease\b',
            r'\bsorry\b', r'\bexcuse\s+me\b', r'\bpardon\b'
        ]
        conversational_patterns = self.get_config_value("conversational_patterns", default_conversational_patterns)
        self.conversational_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in conversational_patterns] 
   
    def _create_provider(self) -> SearchProvider:
        """Create the appropriate search provider based on configuration."""
        provider_config = {
            "api_key": self.get_config_value("api_key"),
            "timeout": self.get_config_value("timeout", 30.0),
            "max_results": self.max_results
        }
        
        if self.provider_name == "tavily":
            return TavilySearchProvider(provider_config)
        elif self.provider_name == "serpapi":
            return SerpAPIProvider(provider_config)
        elif self.provider_name == "duckduckgo":
            return DuckDuckGoProvider(provider_config)
        else:
            raise ToolConfigurationError(f"Unknown search provider: {self.provider_name}")
    
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        """Execute web search and return processed results."""
        start_time = time.time()
        
        try:
            # Validate input
            validation_errors = self.validate_input(query, context)
            if validation_errors:
                return ToolResult(
                    success=False,
                    content="",
                    errors=validation_errors,
                    tool_name="WebSearchTool",
                    execution_time=time.time() - start_time
                )
            
            # Optimize query based on context
            optimized_query = self._optimize_query(query, context)
            
            # Execute search with timeout
            try:
                search_task = asyncio.create_task(self.provider.search(optimized_query))
                raw_results = await asyncio.wait_for(search_task, timeout=context.timeout)
            except asyncio.TimeoutError:
                raise ToolTimeoutError(f"Search timed out after {context.timeout} seconds")
            
            # Process and summarize results
            processed_content = self._process_results(raw_results, context)
            
            # Calculate token count (rough estimate)
            token_count = len(processed_content.split()) * 1.3  # Rough token estimation
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                content=processed_content,
                metadata={
                    "original_query": query,
                    "optimized_query": optimized_query,
                    "provider": self.provider_name,
                    "result_count": len(raw_results),
                    "processing_time": execution_time
                },
                execution_time=execution_time,
                token_count=int(token_count),
                tool_name="WebSearchTool"
            )
            
        except (ToolExecutionError, ToolTimeoutError) as e:
            return ToolResult(
                success=False,
                content="",
                errors=[str(e)],
                tool_name="WebSearchTool",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                errors=[f"Unexpected error: {str(e)}"],
                tool_name="WebSearchTool",
                execution_time=time.time() - start_time
            )
    
    def is_relevant(self, context: ToolContext) -> bool:
        """
        Determine if web search is relevant based on conversation context.
        
        Uses configurable relevance indicators, weights, and thresholds to analyze
        the conversation for temporal indicators, factual questions, and other
        patterns that suggest web search would be beneficial.
        
        Args:
            context: The conversation context to analyze
            
        Returns:
            bool: True if web search is relevant, False otherwise
        """
        # Get recent messages for analysis
        recent_messages = context.get_recent_messages(3)
        if not recent_messages:
            return False
        
        # Combine recent message content for analysis
        combined_content = " ".join([msg.content for msg in recent_messages]).lower()
        
        # Initialize relevance score
        relevance_score = 0.0
        
        # Analyze direct relevance indicators (configurable keywords)
        for indicator in self.relevance_indicators:
            if indicator.lower() in combined_content:
                relevance_score += self.temporal_weight
        
        # Analyze temporal indicators (time-sensitive information needs)
        temporal_matches = 0
        for indicator in self.temporal_indicators:
            if indicator.lower() in combined_content:
                temporal_matches += 1
        
        if temporal_matches > 0:
            # Apply diminishing returns for multiple temporal indicators
            relevance_score += self.temporal_weight * min(temporal_matches, 3) * 0.7
        
        # Analyze factual indicators (data/research requests)
        factual_matches = 0
        for indicator in self.factual_indicators:
            if indicator.lower() in combined_content:
                factual_matches += 1
        
        if factual_matches > 0:
            relevance_score += self.factual_weight * min(factual_matches, 2)
        
        # Analyze question patterns using compiled regex
        question_matches = 0
        for pattern in self.question_patterns:
            if pattern.search(combined_content):
                question_matches += 1
        
        if question_matches > 0:
            relevance_score += self.question_weight * min(question_matches, 3)
        
        # Apply penalty for conversational patterns that shouldn't trigger search
        conversational_matches = 0
        for pattern in self.conversational_patterns:
            if pattern.search(combined_content):
                conversational_matches += 1
        
        if conversational_matches > 0:
            relevance_score += self.conversational_penalty * conversational_matches
        
        # Additional context-aware analysis
        relevance_score += self._analyze_conversation_context(recent_messages)
        
        # Return True if relevance score meets the configurable threshold
        return relevance_score >= self.relevance_score_threshold
    
    def _analyze_conversation_context(self, messages: List) -> float:
        """
        Perform additional context-aware analysis of the conversation.
        
        Args:
            messages: List of recent messages to analyze
            
        Returns:
            float: Additional relevance score based on context analysis
        """
        context_score = 0.0
        
        # Analyze message recency and urgency
        if len(messages) >= 2:
            # Check if there's a follow-up question or clarification
            last_two = [msg.content.lower() for msg in messages[-2:]]
            
            # Look for follow-up patterns
            followup_patterns = [
                r'\bcan\s+you\s+(find|search|look\s+up)',
                r'\bwhat\s+about\b',
                r'\bmore\s+(info|information|details)',
                r'\bspecifically\b',
                r'\bexactly\b'
            ]
            
            for content in last_two:
                for pattern in followup_patterns:
                    if re.search(pattern, content):
                        context_score += 0.5
        
        # Check for uncertainty indicators that might benefit from search
        uncertainty_patterns = [
            r'\bi\s+(think|believe|assume)',
            r'\bmaybe\b',
            r'\bprobably\b',
            r'\bnot\s+sure\b',
            r'\buncertain\b'
        ]
        
        combined_content = " ".join([msg.content.lower() for msg in messages])
        for pattern in uncertainty_patterns:
            if re.search(pattern, combined_content):
                context_score += 0.3
        
        # Check for specific domains that often benefit from current information
        domain_patterns = [
            r'\b(stock|crypto|bitcoin|market|price)\b',
            r'\b(weather|temperature|forecast)\b',
            r'\b(news|breaking|headline)\b',
            r'\b(covid|pandemic|virus)\b',
            r'\b(election|politics|government)\b',
            r'\b(technology|tech|ai|software)\b'
        ]
        
        for pattern in domain_patterns:
            if re.search(pattern, combined_content):
                context_score += 0.8
                break  # Only count once to avoid over-weighting
        
        return context_score  
  
    def _optimize_query(self, query: str, context: ToolContext) -> str:
        """
        Optimize search query based on conversation context.
        
        Performs context-aware query generation by analyzing the conversation
        for temporal needs, domain-specific context, and other indicators
        that can improve search relevance.
        
        Args:
            query: The original search query
            context: The conversation context
            
        Returns:
            str: The optimized search query
        """
        # Start with the original query
        optimized_query = query.strip()
        
        # Get recent conversation context
        recent_messages = context.get_recent_messages(5)  # Look at more messages for better context
        conversation_text = " ".join([msg.content for msg in recent_messages]).lower()
        
        # Extract domain context from conversation
        domain_context = self._extract_domain_context(conversation_text)
        if domain_context and domain_context not in optimized_query.lower():
            optimized_query = f"{optimized_query} {domain_context}"
        
        # Add temporal context if conversation suggests current information is needed
        temporal_context = self._extract_temporal_context(conversation_text, optimized_query)
        if temporal_context:
            optimized_query = f"{optimized_query} {temporal_context}"
        
        # Add location context if relevant
        location_context = self._extract_location_context(conversation_text)
        if location_context and location_context not in optimized_query.lower():
            optimized_query = f"{optimized_query} {location_context}"
        
        # Enhance query based on question type
        question_enhancement = self._enhance_for_question_type(optimized_query, conversation_text)
        if question_enhancement:
            optimized_query = question_enhancement
        
        # Remove potentially sensitive information
        optimized_query = self._sanitize_query(optimized_query)
        
        # Clean up and limit query length
        optimized_query = self._clean_and_limit_query(optimized_query)
        
        return optimized_query
    
    def _extract_domain_context(self, conversation_text: str) -> str:
        """Extract domain-specific context from conversation."""
        domain_mappings = {
            r'\b(stock|stocks|trading|market|nasdaq|dow|s&p)\b': 'stock market',
            r'\b(crypto|bitcoin|ethereum|blockchain|cryptocurrency)\b': 'cryptocurrency',
            r'\b(weather|temperature|rain|snow|forecast|climate)\b': 'weather',
            r'\b(covid|pandemic|virus|vaccine|health)\b': 'health news',
            r'\b(election|politics|government|congress|senate)\b': 'politics',
            r'\b(technology|tech|ai|software|programming)\b': 'technology',
            r'\b(sports|football|basketball|baseball|soccer)\b': 'sports',
            r'\b(movie|film|entertainment|celebrity|actor)\b': 'entertainment'
        }
        
        for pattern, context in domain_mappings.items():
            if re.search(pattern, conversation_text):
                return context
        
        return ""
    
    def _extract_temporal_context(self, conversation_text: str, current_query: str) -> str:
        """Extract temporal context and add appropriate time qualifiers."""
        # Check if query already has temporal context
        if any(year in current_query for year in ["2024", "2025", "2023"]):
            return ""
        
        # Determine appropriate temporal context
        temporal_mappings = {
            r'\b(today|now|current|currently|right now)\b': "2025",
            r'\b(this week|recent|recently|latest)\b': "2025 recent",
            r'\b(this year|2025)\b': "2025",
            r'\b(last year|2024)\b': "2024",
            r'\b(breaking|urgent|just happened)\b': "breaking news 2025"
        }
        
        for pattern, context in temporal_mappings.items():
            if re.search(pattern, conversation_text):
                return context
        
        # Default temporal context if any temporal indicators are present
        if any(indicator in conversation_text for indicator in self.temporal_indicators):
            return "2025"
        
        return ""
    
    def _extract_location_context(self, conversation_text: str) -> str:
        """Extract location context from conversation."""
        # Look for common location patterns
        location_patterns = [
            r'\bin\s+([\w\s]+(?:city|state|country|usa|america|uk|canada))\b',
            r'\b(new york|california|texas|florida|london|paris|tokyo|beijing)\b',
            r'\b(usa|america|united states|uk|united kingdom|canada|australia)\b'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, conversation_text, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return ""
    
    def _enhance_for_question_type(self, query: str, conversation_text: str) -> str:
        """Enhance query based on the type of question being asked."""
        # Statistical/numerical questions
        if re.search(r'\b(how many|how much|statistics|data|numbers|percentage)\b', conversation_text):
            if not re.search(r'\b(statistics|data|numbers)\b', query.lower()):
                return f"{query} statistics data"
        
        # Definition questions
        if re.search(r'\b(what is|define|definition|meaning|explain)\b', conversation_text):
            if not re.search(r'\b(definition|meaning|explanation)\b', query.lower()):
                return f"{query} definition explanation"
        
        # Comparison questions
        if re.search(r'\b(vs|versus|compared to|difference between|better than)\b', conversation_text):
            if not re.search(r'\b(comparison|vs|versus)\b', query.lower()):
                return f"{query} comparison"
        
        # How-to questions
        if re.search(r'\b(how to|tutorial|guide|instructions|steps)\b', conversation_text):
            if not re.search(r'\b(tutorial|guide|how to)\b', query.lower()):
                return f"{query} tutorial guide"
        
        return query
    
    def _clean_and_limit_query(self, query: str) -> str:
        """Clean up the query and ensure it meets length requirements."""
        # Remove duplicate words
        words = query.split()
        seen = set()
        cleaned_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                cleaned_words.append(word)
        
        cleaned_query = " ".join(cleaned_words)
        
        # Limit query length
        if len(cleaned_query) > 200:
            # Try to keep the most important parts
            words = cleaned_query.split()
            if len(words) > 20:
                # Keep first 15 words and last 5 words
                cleaned_query = " ".join(words[:15] + ["..."] + words[-5:])
            else:
                cleaned_query = cleaned_query[:197] + "..."
        
        return cleaned_query
    
    def _sanitize_query(self, query: str) -> str:
        """Remove potentially sensitive information from search queries."""
        # Remove common sensitive patterns
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number pattern
        ]
        
        sanitized = query
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized
    
    def _process_results(self, raw_results: List[Dict[str, Any]], context: ToolContext) -> str:
        """Process and summarize search results."""
        if not raw_results:
            return "No search results found."
        
        # Filter results by relevance if scores are available
        filtered_results = []
        for result in raw_results:
            score = result.get("score", 1.0)
            if score >= self.relevance_threshold:
                filtered_results.append(result)
        
        # If filtering removed too many results, use original results
        if len(filtered_results) < 2 and len(raw_results) >= 2:
            filtered_results = raw_results
        
        # Format results
        formatted_results = []
        for i, result in enumerate(filtered_results[:self.max_results], 1):
            title = result.get("title", "").strip()
            url = result.get("url", "").strip()
            snippet = result.get("snippet", "").strip()
            
            # Summarize snippet if too long
            if len(snippet) > self.result_summary_length:
                snippet = snippet[:self.result_summary_length-3] + "..."
            
            # Format result entry
            result_entry = f"{i}. **{title}**"
            if snippet:
                result_entry += f"\n   {snippet}"
            if url:
                result_entry += f"\n   Source: {url}"
            
            formatted_results.append(result_entry)
        
        # Combine results with summary
        result_summary = f"Found {len(raw_results)} search results:\n\n"
        result_summary += "\n\n".join(formatted_results)
        
        # Ensure result fits within token limits
        max_tokens = context.max_result_tokens
        estimated_tokens = len(result_summary.split()) * 1.3
        
        if estimated_tokens > max_tokens:
            # Truncate results to fit token limit
            target_length = int(len(result_summary) * (max_tokens / estimated_tokens))
            result_summary = result_summary[:target_length-50] + "\n\n[Results truncated to fit context window]"
        
        return result_summary
    
    def get_tool_info(self) -> ToolInfo:
        """Get metadata about the web search tool."""
        return ToolInfo(
            name="WebSearchTool",
            description="Search the web for current information using multiple search providers with configurable relevance detection",
            capabilities=[ToolCapability.SEARCH],
            required_config=[] if self.provider_name == "duckduckgo" else ["api_key"],
            optional_config=[
                "provider", "max_results", "result_summary_length", 
                "relevance_threshold", "timeout", "relevance_score_threshold",
                "temporal_weight", "factual_weight", "question_weight", 
                "conversational_penalty", "relevance_indicators", 
                "temporal_indicators", "factual_indicators", 
                "question_patterns", "conversational_patterns"
            ],
            version="1.1.0",
            author="Agentic Conversation System",
            max_execution_time=30.0,
            token_cost_estimate=300
        )
    
    def validate_input(self, query: str, context: ToolContext) -> List[str]:
        """Validate input parameters for web search."""
        errors = super().validate_input(query, context)
        
        # Additional validation for web search
        if len(query.strip()) < 3:
            errors.append("Search query must be at least 3 characters long")
        
        if len(query) > 500:
            errors.append("Search query is too long (max 500 characters)")
        
        # Check if provider is properly configured
        if not self.provider:
            errors.append("Search provider is not properly configured")
        
        return errors