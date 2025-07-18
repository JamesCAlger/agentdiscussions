"""
Unit tests for the WebSearchTool implementation.

This module tests the web search tool functionality including multiple providers,
query optimization, result processing, and relevance detection.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from urllib.error import HTTPError, URLError

from src.agentic_conversation.tools.web_search import (
    WebSearchTool, TavilySearchProvider, SerpAPIProvider, DuckDuckGoProvider
)
from src.agentic_conversation.tools.base import ToolContext, ToolCapability
from src.agentic_conversation.tools.exceptions import (
    ToolExecutionError, ToolTimeoutError, ToolConfigurationError
)
from src.agentic_conversation.models import Message
from datetime import datetime


class TestSearchProviders:
    """Test cases for individual search providers."""
    
    @pytest.fixture
    def mock_response_data(self):
        """Mock response data for different providers."""
        return {
            "tavily": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example.com/1",
                        "content": "This is test content for result 1",
                        "published_date": "2024-01-01",
                        "score": 0.9
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example.com/2",
                        "content": "This is test content for result 2",
                        "published_date": "2024-01-02",
                        "score": 0.8
                    }
                ]
            },
            "serpapi": {
                "organic_results": [
                    {
                        "title": "SerpAPI Result 1",
                        "link": "https://example.com/serp1",
                        "snippet": "SerpAPI test content 1",
                        "date": "2024-01-01"
                    },
                    {
                        "title": "SerpAPI Result 2",
                        "link": "https://example.com/serp2",
                        "snippet": "SerpAPI test content 2",
                        "date": "2024-01-02"
                    }
                ]
            },
            "duckduckgo": {
                "Abstract": "DuckDuckGo abstract content",
                "AbstractURL": "https://example.com/ddg",
                "Heading": "DuckDuckGo Heading",
                "RelatedTopics": [
                    {
                        "Text": "Related topic 1 - Description",
                        "FirstURL": "https://example.com/related1"
                    },
                    {
                        "Text": "Related topic 2 - Description",
                        "FirstURL": "https://example.com/related2"
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_tavily_provider_success(self, mock_response_data):
        """Test successful Tavily search."""
        config = {"api_key": "test_key", "max_results": 5}
        provider = TavilySearchProvider(config)
        
        with patch('src.agentic_conversation.tools.web_search.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_response_data["tavily"]).encode('utf-8')
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_urlopen.return_value = mock_response
            
            results = await provider.search("test query")
            
            assert len(results) == 2
            assert results[0]["title"] == "Test Result 1"
            assert results[0]["url"] == "https://example.com/1"
            assert results[0]["snippet"] == "This is test content for result 1"
            assert results[0]["score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_serpapi_provider_success(self, mock_response_data):
        """Test successful SerpAPI search."""
        config = {"api_key": "test_key", "max_results": 5}
        provider = SerpAPIProvider(config)
        
        with patch('src.agentic_conversation.tools.web_search.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_response_data["serpapi"]).encode('utf-8')
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_urlopen.return_value = mock_response
            
            results = await provider.search("test query")
            
            assert len(results) == 2
            assert results[0]["title"] == "SerpAPI Result 1"
            assert results[0]["url"] == "https://example.com/serp1"
            assert results[0]["snippet"] == "SerpAPI test content 1"
            assert results[0]["score"] == 1.0  # Default score for SerpAPI
    
    @pytest.mark.asyncio
    async def test_duckduckgo_provider_success(self, mock_response_data):
        """Test successful DuckDuckGo search."""
        config = {"max_results": 5}
        provider = DuckDuckGoProvider(config)
        
        with patch('src.agentic_conversation.tools.web_search.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = json.dumps(mock_response_data["duckduckgo"]).encode('utf-8')
            mock_response.__enter__.return_value = mock_response
            mock_response.__exit__.return_value = None
            mock_urlopen.return_value = mock_response
            
            results = await provider.search("test query")
            
            assert len(results) >= 1
            assert results[0]["title"] == "DuckDuckGo Heading"
            assert results[0]["snippet"] == "DuckDuckGo abstract content"
    
    def test_tavily_provider_missing_api_key(self):
        """Test Tavily provider with missing API key."""
        config = {"max_results": 5}
        
        with pytest.raises(ToolConfigurationError, match="Tavily API key is required"):
            TavilySearchProvider(config)
    
    def test_serpapi_provider_missing_api_key(self):
        """Test SerpAPI provider with missing API key."""
        config = {"max_results": 5}
        
        with pytest.raises(ToolConfigurationError, match="SerpAPI key is required"):
            SerpAPIProvider(config)
    
    @pytest.mark.asyncio
    async def test_provider_http_error(self):
        """Test provider handling of HTTP errors."""
        config = {"api_key": "test_key", "max_results": 5}
        provider = TavilySearchProvider(config)
        
        with patch('src.agentic_conversation.tools.web_search.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                url="test", code=404, msg="Not Found", hdrs=None, fp=None
            )
            
            with pytest.raises(ToolExecutionError, match="Tavily search failed"):
                await provider.search("test query")
    
    @pytest.mark.asyncio
    async def test_provider_url_error(self):
        """Test provider handling of URL errors."""
        config = {"api_key": "test_key", "max_results": 5}
        provider = TavilySearchProvider(config)
        
        with patch('src.agentic_conversation.tools.web_search.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection failed")
            
            with pytest.raises(ToolExecutionError, match="Tavily search failed"):
                await provider.search("test query")


class TestWebSearchTool:
    """Test cases for the WebSearchTool class."""
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample tool context for testing."""
        messages = [
            Message(
                agent_id="agent_a",
                content="What is the current weather like?",
                timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
                token_count=10
            ),
            Message(
                agent_id="agent_b", 
                content="I need current information about the weather today.",
                timestamp=datetime.fromisoformat("2024-01-01T10:01:00"),
                token_count=12
            )
        ]
        
        return ToolContext(
            conversation_history=messages,
            current_turn=2,
            agent_id="agent_a",
            available_tokens=8000,
            max_result_tokens=1000,
            timeout=30.0
        )
    
    @pytest.fixture
    def mock_search_results(self):
        """Mock search results for testing."""
        return [
            {
                "title": "Weather Today",
                "url": "https://weather.com/today",
                "snippet": "Current weather conditions show sunny skies with temperature of 75°F",
                "published_date": "2024-01-01",
                "score": 0.9
            },
            {
                "title": "Weather Forecast",
                "url": "https://weather.com/forecast",
                "snippet": "Extended weather forecast for the next 7 days",
                "published_date": "2024-01-01",
                "score": 0.8
            }
        ]
    
    def test_tool_initialization_duckduckgo(self):
        """Test tool initialization with DuckDuckGo provider."""
        config = {"provider": "duckduckgo", "max_results": 3}
        tool = WebSearchTool(config)
        
        assert tool.provider_name == "duckduckgo"
        assert tool.max_results == 3
        assert isinstance(tool.provider, DuckDuckGoProvider)
    
    def test_tool_initialization_tavily(self):
        """Test tool initialization with Tavily provider."""
        config = {"provider": "tavily", "api_key": "test_key"}
        tool = WebSearchTool(config)
        
        assert tool.provider_name == "tavily"
        assert isinstance(tool.provider, TavilySearchProvider)
    
    def test_tool_initialization_invalid_provider(self):
        """Test tool initialization with invalid provider."""
        config = {"provider": "invalid_provider"}
        
        with pytest.raises(ToolConfigurationError, match="Unknown search provider"):
            WebSearchTool(config)
    
    def test_get_tool_info(self):
        """Test tool info retrieval."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        info = tool.get_tool_info()
        
        assert info.name == "WebSearchTool"
        assert ToolCapability.SEARCH in info.capabilities
        assert info.version == "1.1.0"
        assert "provider" in info.optional_config
    
    def test_relevance_detection_positive(self, sample_context):
        """Test relevance detection with relevant context."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Context contains "current" and "today" - should be relevant
        assert tool.is_relevant(sample_context) is True
    
    def test_relevance_detection_negative(self):
        """Test relevance detection with irrelevant context."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        messages = [
            Message(
                agent_id="agent_a",
                content="Hello, how are you doing?",
                timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
                token_count=8
            )
        ]
        
        context = ToolContext(
            conversation_history=messages,
            agent_id="agent_a"
        )
        
        assert tool.is_relevant(context) is False
    
    def test_query_optimization(self, sample_context):
        """Test query optimization based on context."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        original_query = "weather"
        optimized = tool._optimize_query(original_query, sample_context)
        
        # Should add temporal context
        assert "2024" in optimized or "2025" in optimized
        assert len(optimized) <= 200
    
    def test_query_sanitization(self):
        """Test query sanitization to remove sensitive information."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        sensitive_query = "My SSN is 123-45-6789 and email test@example.com"
        sanitized = tool._sanitize_query(sensitive_query)
        
        assert "123-45-6789" not in sanitized
        assert "test@example.com" not in sanitized
        assert "[REDACTED]" in sanitized
    
    def test_result_processing(self, sample_context, mock_search_results):
        """Test search result processing and formatting."""
        tool = WebSearchTool({"provider": "duckduckgo", "max_results": 2})
        
        processed = tool._process_results(mock_search_results, sample_context)
        
        assert "Found 2 search results" in processed
        assert "Weather Today" in processed
        assert "https://weather.com/today" in processed
        assert "Current weather conditions" in processed
    
    def test_result_processing_empty(self, sample_context):
        """Test processing of empty search results."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        processed = tool._process_results([], sample_context)
        
        assert processed == "No search results found."
    
    def test_result_processing_long_snippets(self, sample_context):
        """Test processing of results with long snippets."""
        tool = WebSearchTool({"provider": "duckduckgo", "result_summary_length": 50})
        
        long_results = [
            {
                "title": "Long Result",
                "url": "https://example.com",
                "snippet": "This is a very long snippet that should be truncated because it exceeds the maximum length configured for result summaries in the tool configuration",
                "score": 0.9
            }
        ]
        
        processed = tool._process_results(long_results, sample_context)
        
        # Should be truncated with ellipsis
        assert "..." in processed
        assert len(processed.split("This is a very long snippet")[1].split("...")[0]) <= 50
    
    @pytest.mark.asyncio
    async def test_execute_success(self, sample_context, mock_search_results):
        """Test successful tool execution."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        with patch.object(tool.provider, 'search', return_value=mock_search_results):
            result = await tool.execute("current weather", sample_context)
            
            assert result.success is True
            assert "Weather Today" in result.content
            assert result.tool_name == "WebSearchTool"
            assert result.metadata["provider"] == "duckduckgo"
            assert result.metadata["result_count"] == 2
            assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self, sample_context):
        """Test tool execution timeout."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Mock provider to simulate slow response
        async def slow_search(query):
            await asyncio.sleep(2)
            return []
        
        with patch.object(tool.provider, 'search', side_effect=slow_search):
            # Set very short timeout
            sample_context.timeout = 0.1
            
            result = await tool.execute("test query", sample_context)
            
            assert result.success is False
            assert any("timed out" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_execute_provider_error(self, sample_context):
        """Test tool execution with provider error."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        with patch.object(tool.provider, 'search', side_effect=ToolExecutionError("Provider failed")):
            result = await tool.execute("test query", sample_context)
            
            assert result.success is False
            assert any("Provider failed" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_execute_validation_error(self, sample_context):
        """Test tool execution with validation error."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Empty query should fail validation
        result = await tool.execute("", sample_context)
        
        assert result.success is False
        assert any("cannot be empty" in error for error in result.errors)
    
    def test_input_validation(self, sample_context):
        """Test input validation."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Test empty query
        errors = tool.validate_input("", sample_context)
        assert any("cannot be empty" in error for error in errors)
        
        # Test short query
        errors = tool.validate_input("ab", sample_context)
        assert any("at least 3 characters" in error for error in errors)
        
        # Test long query
        long_query = "a" * 501
        errors = tool.validate_input(long_query, sample_context)
        assert any("too long" in error for error in errors)
        
        # Test valid query
        errors = tool.validate_input("valid query", sample_context)
        assert len(errors) == 0
    
    def test_relevance_indicators(self):
        """Test various relevance indicators."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        test_cases = [
            ("What is the current price of Bitcoin?", True),
            ("How many people live in Tokyo today?", True),
            ("Latest news about AI research", True),
            ("Define machine learning", True),
            ("Hello, how are you?", False),
            ("I like pizza", False),
            ("Tell me a joke", False)
        ]
        
        for content, expected in test_cases:
            messages = [Message(
                agent_id="test", 
                content=content, 
                timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
                token_count=len(content.split())
            )]
            context = ToolContext(conversation_history=messages, agent_id="test")
            
            result = tool.is_relevant(context)
            assert result == expected, f"Failed for content: '{content}'"
    
    def test_configurable_relevance_thresholds(self):
        """Test configurable relevance score thresholds."""
        # Test with low threshold
        low_threshold_tool = WebSearchTool({
            "provider": "duckduckgo",
            "relevance_score_threshold": 1.0
        })
        
        # Test with high threshold
        high_threshold_tool = WebSearchTool({
            "provider": "duckduckgo", 
            "relevance_score_threshold": 5.0
        })
        
        messages = [Message(
            agent_id="test",
            content="What is machine learning?",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=4
        )]
        context = ToolContext(conversation_history=messages, agent_id="test")
        
        # Low threshold should be relevant
        assert low_threshold_tool.is_relevant(context) is True
        
        # High threshold should not be relevant
        assert high_threshold_tool.is_relevant(context) is False
    
    def test_configurable_weights(self):
        """Test configurable weights for different indicator types."""
        # Test with high temporal weight
        high_temporal_tool = WebSearchTool({
            "provider": "duckduckgo",
            "temporal_weight": 5.0,
            "factual_weight": 0.1,
            "relevance_score_threshold": 3.0
        })
        
        # Test with high factual weight
        high_factual_tool = WebSearchTool({
            "provider": "duckduckgo",
            "temporal_weight": 0.1,
            "factual_weight": 5.0,
            "relevance_score_threshold": 3.0
        })
        
        temporal_messages = [Message(
            agent_id="test",
            content="What's happening today?",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=3
        )]
        
        factual_messages = [Message(
            agent_id="test",
            content="I need research data on climate change.",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=7
        )]
        
        temporal_context = ToolContext(conversation_history=temporal_messages, agent_id="test")
        factual_context = ToolContext(conversation_history=factual_messages, agent_id="test")
        
        # High temporal weight tool should be relevant for temporal content
        assert high_temporal_tool.is_relevant(temporal_context) is True
        assert high_temporal_tool.is_relevant(factual_context) is False
        
        # High factual weight tool should be relevant for factual content
        assert high_factual_tool.is_relevant(factual_context) is True
        assert high_factual_tool.is_relevant(temporal_context) is False
    
    def test_configurable_keywords(self):
        """Test configurable relevance indicators and keywords."""
        custom_tool = WebSearchTool({
            "provider": "duckduckgo",
            "relevance_indicators": ["custom_keyword", "special_term"],
            "temporal_indicators": ["right_now", "immediately"],
            "factual_indicators": ["custom_data", "special_research"],
            "relevance_score_threshold": 1.0
        })
        
        # Test custom relevance indicators
        messages1 = [Message(
            agent_id="test",
            content="I need custom_keyword information",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=4
        )]
        context1 = ToolContext(conversation_history=messages1, agent_id="test")
        assert custom_tool.is_relevant(context1) is True
        
        # Test custom temporal indicators
        messages2 = [Message(
            agent_id="test",
            content="I need this right_now",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=4
        )]
        context2 = ToolContext(conversation_history=messages2, agent_id="test")
        assert custom_tool.is_relevant(context2) is True
        
        # Test custom factual indicators
        messages3 = [Message(
            agent_id="test",
            content="Show me custom_data analysis",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=4
        )]
        context3 = ToolContext(conversation_history=messages3, agent_id="test")
        assert custom_tool.is_relevant(context3) is True
        
        # Test that default keywords don't work (but question patterns still do)
        messages4 = [Message(
            agent_id="test",
            content="Tell me about some random topic",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=6
        )]
        context4 = ToolContext(conversation_history=messages4, agent_id="test")
        # This should be relevant due to question pattern, but let's test with higher threshold
        custom_tool_high_threshold = WebSearchTool({
            "provider": "duckduckgo",
            "relevance_indicators": ["custom_keyword", "special_term"],
            "temporal_indicators": ["right_now", "immediately"],
            "factual_indicators": ["custom_data", "special_research"],
            "relevance_score_threshold": 2.0,  # Higher threshold
            "question_weight": 0.5  # Lower question weight
        })
        assert custom_tool_high_threshold.is_relevant(context4) is False
    
    def test_configurable_question_patterns(self):
        """Test configurable question patterns."""
        custom_tool = WebSearchTool({
            "provider": "duckduckgo",
            "question_patterns": [r'\bcustom\s+question\s+\w+', r'\bspecial\s+query\s+\w+'],
            "question_weight": 3.0,
            "relevance_score_threshold": 2.0
        })
        
        # Test custom question pattern
        messages = [Message(
            agent_id="test",
            content="This is a custom question about AI",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=7
        )]
        context = ToolContext(conversation_history=messages, agent_id="test")
        assert custom_tool.is_relevant(context) is True
        
        # Test that default patterns don't work (use content without relevance indicators)
        messages2 = [Message(
            agent_id="test",
            content="Tell me about machine learning concepts",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=6
        )]
        context2 = ToolContext(conversation_history=messages2, agent_id="test")
        assert custom_tool.is_relevant(context2) is False
    
    def test_conversational_penalty(self):
        """Test conversational penalty configuration."""
        high_penalty_tool = WebSearchTool({
            "provider": "duckduckgo",
            "conversational_penalty": -10.0,  # Much higher penalty
            "temporal_weight": 1.0,  # Lower temporal weight
            "relevance_score_threshold": 1.0
        })
        
        low_penalty_tool = WebSearchTool({
            "provider": "duckduckgo",
            "conversational_penalty": -0.1,
            "relevance_score_threshold": 1.0
        })
        
        # Message with both relevance indicators and conversational patterns
        messages = [Message(
            agent_id="test",
            content="Hello, what is the current weather?",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=6
        )]
        context = ToolContext(conversation_history=messages, agent_id="test")
        
        # High penalty should make it not relevant
        assert high_penalty_tool.is_relevant(context) is False
        
        # Low penalty should still be relevant
        assert low_penalty_tool.is_relevant(context) is True
    
    def test_context_aware_analysis(self):
        """Test additional context-aware analysis features."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Test follow-up patterns
        followup_messages = [
            Message(
                agent_id="agent_a",
                content="Tell me about AI",
                timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
                token_count=4
            ),
            Message(
                agent_id="agent_b",
                content="Can you find more information about that?",
                timestamp=datetime.fromisoformat("2024-01-01T10:01:00"),
                token_count=7
            )
        ]
        followup_context = ToolContext(conversation_history=followup_messages, agent_id="agent_b")
        assert tool.is_relevant(followup_context) is True
        
        # Test uncertainty indicators
        uncertainty_messages = [Message(
            agent_id="test",
            content="I think Bitcoin price is around $50k but I'm not sure",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=11
        )]
        uncertainty_context = ToolContext(conversation_history=uncertainty_messages, agent_id="test")
        assert tool.is_relevant(uncertainty_context) is True
        
        # Test domain-specific patterns
        domain_messages = [Message(
            agent_id="test",
            content="What's the latest crypto market news?",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=6
        )]
        domain_context = ToolContext(conversation_history=domain_messages, agent_id="test")
        assert tool.is_relevant(domain_context) is True
    
    @pytest.mark.asyncio
    async def test_token_limit_handling(self, sample_context):
        """Test handling of token limits in results."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Create very long mock results
        long_results = [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "snippet": "Very long content " * 100,  # Very long snippet
                "score": 0.9
            }
            for i in range(10)
        ]
        
        # Set very low token limit
        sample_context.max_result_tokens = 50
        
        processed = tool._process_results(long_results, sample_context)
        
        # Should be truncated
        assert "[Results truncated to fit context window]" in processed
    
    def test_relevance_threshold_filtering(self, sample_context):
        """Test filtering of results based on relevance threshold."""
        tool = WebSearchTool({"provider": "duckduckgo", "relevance_threshold": 0.8})
        
        mixed_results = [
            {"title": "High Relevance", "url": "https://example.com/1", "snippet": "Good content", "score": 0.9},
            {"title": "Low Relevance", "url": "https://example.com/2", "snippet": "Poor content", "score": 0.5},
            {"title": "Medium Relevance", "url": "https://example.com/3", "snippet": "OK content", "score": 0.8}
        ]
        
        processed = tool._process_results(mixed_results, sample_context)
        
        # Should only include high and medium relevance results
        assert "High Relevance" in processed
        assert "Medium Relevance" in processed
        assert "Low Relevance" not in processed
    
    def test_enhanced_query_optimization(self):
        """Test enhanced context-aware query optimization."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Test domain context extraction
        crypto_messages = [Message(
            agent_id="test",
            content="I want to know about Bitcoin and cryptocurrency trends",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=8
        )]
        crypto_context = ToolContext(conversation_history=crypto_messages, agent_id="test")
        
        optimized = tool._optimize_query("price", crypto_context)
        assert "cryptocurrency" in optimized.lower()
        
        # Test temporal context extraction
        temporal_messages = [Message(
            agent_id="test",
            content="What's happening right now with the stock market?",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=8
        )]
        temporal_context = ToolContext(conversation_history=temporal_messages, agent_id="test")
        
        optimized = tool._optimize_query("market", temporal_context)
        assert "2025" in optimized
        
        # Test location context extraction
        location_messages = [Message(
            agent_id="test",
            content="What's the weather like in New York today?",
            timestamp=datetime.fromisoformat("2024-01-01T10:00:00"),
            token_count=8
        )]
        location_context = ToolContext(conversation_history=location_messages, agent_id="test")
        
        optimized = tool._optimize_query("weather", location_context)
        assert "new york" in optimized.lower()
    
    def test_domain_context_extraction(self):
        """Test domain context extraction from conversation."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        test_cases = [
            ("I want to check stock prices", "stock market"),
            ("Tell me about Bitcoin and crypto", "cryptocurrency"),
            ("What's the weather forecast?", "weather"),
            ("Latest COVID news and updates", "health news"),
            ("Election results and politics", "politics"),
            ("New technology and AI developments", "technology"),
            ("Football scores and sports news", "sports"),
            ("Movie reviews and entertainment", "entertainment")
        ]
        
        for conversation_text, expected_domain in test_cases:
            domain = tool._extract_domain_context(conversation_text.lower())
            assert domain == expected_domain, f"Failed for: '{conversation_text}'"
    
    def test_temporal_context_extraction(self):
        """Test temporal context extraction from conversation."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        test_cases = [
            ("What's happening today?", "2025"),
            ("I need current information", "2025"),
            ("Show me recent developments", "2025 recent"),
            ("This week's updates", "2025 recent"),
            ("Breaking news just happened", "breaking news 2025"),
            ("Last year's data", "2024")
        ]
        
        for conversation_text, expected_temporal in test_cases:
            temporal = tool._extract_temporal_context(conversation_text.lower(), "test query")
            assert temporal == expected_temporal, f"Failed for: '{conversation_text}'"
    
    def test_location_context_extraction(self):
        """Test location context extraction from conversation."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        test_cases = [
            ("Weather in New York", "new york"),
            ("Events in California", "california"),
            ("News from London", "london"),
            ("Updates from USA", "usa"),
            ("Information about Canada", "canada")
        ]
        
        for conversation_text, expected_location in test_cases:
            location = tool._extract_location_context(conversation_text.lower())
            assert expected_location in location.lower(), f"Failed for: '{conversation_text}'"
    
    def test_question_type_enhancement(self):
        """Test query enhancement based on question type."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        test_cases = [
            ("How many people live in Tokyo?", "population", "statistics data"),
            ("What is machine learning?", "ML", "definition explanation"),
            ("Python vs Java comparison", "languages", "comparison"),
            ("How to learn programming?", "coding", "tutorial guide")
        ]
        
        for conversation_text, query, expected_enhancement in test_cases:
            enhanced = tool._enhance_for_question_type(query, conversation_text.lower())
            assert expected_enhancement in enhanced.lower(), f"Failed for: '{conversation_text}'"
    
    def test_query_cleaning_and_limiting(self):
        """Test query cleaning and length limiting."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        
        # Test duplicate word removal
        duplicate_query = "machine learning machine learning AI artificial intelligence AI"
        cleaned = tool._clean_and_limit_query(duplicate_query)
        words = cleaned.split()
        assert len(words) == len(set(word.lower() for word in words)), "Duplicates not removed"
        
        # Test length limiting with diverse words
        very_long_query = " ".join([f"word{i}" for i in range(50)])  # 50 unique words
        limited = tool._clean_and_limit_query(very_long_query)
        assert len(limited) <= 200, "Query not properly limited"
        if len(very_long_query) > 200:  # Only check for truncation if original was long
            assert "..." in limited, "Truncation indicator missing"
    
    def test_enhanced_tool_info(self):
        """Test that tool info includes new configuration options."""
        tool = WebSearchTool({"provider": "duckduckgo"})
        info = tool.get_tool_info()
        
        # Check that new configuration options are included
        expected_configs = [
            "relevance_score_threshold", "temporal_weight", "factual_weight",
            "question_weight", "conversational_penalty", "relevance_indicators",
            "temporal_indicators", "factual_indicators", "question_patterns",
            "conversational_patterns"
        ]
        
        for config in expected_configs:
            assert config in info.optional_config, f"Missing config option: {config}"
        
        # Check version is updated
        assert info.version == "1.1.0"
        assert "configurable relevance detection" in info.description.lower()


if __name__ == "__main__":
    pytest.main([__file__])