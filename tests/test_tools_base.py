"""
Unit tests for the tool system base classes and data models.

This module tests the core interfaces, data models, and exception classes
that form the foundation of the tool system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

from src.agentic_conversation.tools.base import (
    BaseTool, ToolResult, ToolContext, ToolInfo, ToolCapability
)
from src.agentic_conversation.tools.exceptions import (
    ToolError, ToolExecutionError, ToolTimeoutError, 
    ToolConfigurationError, ToolRateLimitError
)
from src.agentic_conversation.models import Message


class TestToolResult:
    """Test cases for ToolResult data model."""
    
    def test_tool_result_creation(self):
        """Test basic ToolResult creation."""
        result = ToolResult(
            success=True,
            content="Test result content",
            tool_name="test_tool"
        )
        
        assert result.success is True
        assert result.content == "Test result content"
        assert result.tool_name == "test_tool"
        assert result.metadata == {}
        assert result.execution_time == 0.0
        assert result.token_count == 0
        assert result.errors == []
        assert isinstance(result.timestamp, datetime)
    
    def test_tool_result_with_metadata(self):
        """Test ToolResult creation with metadata."""
        metadata = {"query": "test query", "result_count": 5}
        result = ToolResult(
            success=True,
            content="Test content",
            metadata=metadata,
            execution_time=1.5,
            token_count=100
        )
        
        assert result.metadata == metadata
        assert result.execution_time == 1.5
        assert result.token_count == 100
    
    def test_add_error(self):
        """Test adding errors to ToolResult."""
        result = ToolResult(success=True, content="Test")
        
        result.add_error("First error")
        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0] == "First error"
        
        result.add_error("Second error")
        assert len(result.errors) == 2
        assert result.errors[1] == "Second error"
    
    def test_has_errors(self):
        """Test error detection in ToolResult."""
        result = ToolResult(success=True, content="Test")
        assert result.has_errors() is False
        
        result.add_error("Test error")
        assert result.has_errors() is True
    
    def test_get_summary(self):
        """Test ToolResult summary generation."""
        result = ToolResult(
            success=True,
            content="Test content with some length",
            tool_name="test_tool",
            execution_time=2.5
        )
        
        summary = result.get_summary()
        assert "test_tool" in summary
        assert "SUCCESS" in summary
        assert "2.50s" in summary
        assert str(len(result.content)) in summary
    
    def test_get_summary_failed(self):
        """Test ToolResult summary for failed execution."""
        result = ToolResult(
            success=False,
            content="Error content",
            tool_name="test_tool",
            execution_time=1.0
        )
        
        summary = result.get_summary()
        assert "FAILED" in summary
        assert "test_tool" in summary
    
    def test_to_dict(self):
        """Test ToolResult serialization to dictionary."""
        timestamp = datetime.now()
        result = ToolResult(
            success=True,
            content="Test content",
            metadata={"key": "value"},
            execution_time=1.5,
            token_count=100,
            errors=["error1"],
            tool_name="test_tool",
            timestamp=timestamp
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["content"] == "Test content"
        assert result_dict["metadata"] == {"key": "value"}
        assert result_dict["execution_time"] == 1.5
        assert result_dict["token_count"] == 100
        assert result_dict["errors"] == ["error1"]
        assert result_dict["tool_name"] == "test_tool"
        assert result_dict["timestamp"] == timestamp.isoformat()
    
    def test_from_dict(self):
        """Test ToolResult deserialization from dictionary."""
        timestamp = datetime.now()
        data = {
            "success": True,
            "content": "Test content",
            "metadata": {"key": "value"},
            "execution_time": 1.5,
            "token_count": 100,
            "errors": ["error1"],
            "tool_name": "test_tool",
            "timestamp": timestamp.isoformat()
        }
        
        result = ToolResult.from_dict(data)
        
        assert result.success is True
        assert result.content == "Test content"
        assert result.metadata == {"key": "value"}
        assert result.execution_time == 1.5
        assert result.token_count == 100
        assert result.errors == ["error1"]
        assert result.tool_name == "test_tool"
        assert result.timestamp == timestamp


class TestToolContext:
    """Test cases for ToolContext data model."""
    
    def test_tool_context_creation(self):
        """Test basic ToolContext creation."""
        context = ToolContext(
            agent_id="test_agent",
            current_turn=5,
            available_tokens=4000
        )
        
        assert context.agent_id == "test_agent"
        assert context.current_turn == 5
        assert context.available_tokens == 4000
        assert context.conversation_history == []
        assert context.metadata == {}
        assert context.max_result_tokens == 1000
        assert context.timeout == 30.0
    
    def test_tool_context_with_messages(self):
        """Test ToolContext with conversation history."""
        messages = [
            Message("agent1", "Hello", datetime.now(), 10),
            Message("agent2", "Hi there", datetime.now(), 15),
            Message("agent1", "How are you?", datetime.now(), 20)
        ]
        
        context = ToolContext(
            conversation_history=messages,
            agent_id="agent1",
            current_turn=3
        )
        
        assert len(context.conversation_history) == 3
        assert context.conversation_history[0].agent_id == "agent1"
        assert context.conversation_history[1].agent_id == "agent2"
    
    def test_get_recent_messages(self):
        """Test getting recent messages from context."""
        messages = [
            Message(f"agent{i}", f"Message {i}", datetime.now(), 10)
            for i in range(10)
        ]
        
        context = ToolContext(conversation_history=messages)
        
        # Test default count (5)
        recent = context.get_recent_messages()
        assert len(recent) == 5
        assert recent[0].content == "Message 5"
        assert recent[-1].content == "Message 9"
        
        # Test custom count
        recent_3 = context.get_recent_messages(3)
        assert len(recent_3) == 3
        assert recent_3[0].content == "Message 7"
        
        # Test with fewer messages than requested
        short_context = ToolContext(conversation_history=messages[:2])
        recent_short = short_context.get_recent_messages(5)
        assert len(recent_short) == 2
    
    def test_get_messages_by_agent(self):
        """Test filtering messages by agent."""
        messages = [
            Message("agent1", "Message 1", datetime.now(), 10),
            Message("agent2", "Message 2", datetime.now(), 15),
            Message("agent1", "Message 3", datetime.now(), 20),
            Message("agent3", "Message 4", datetime.now(), 25)
        ]
        
        context = ToolContext(conversation_history=messages)
        
        agent1_messages = context.get_messages_by_agent("agent1")
        assert len(agent1_messages) == 2
        assert agent1_messages[0].content == "Message 1"
        assert agent1_messages[1].content == "Message 3"
        
        agent2_messages = context.get_messages_by_agent("agent2")
        assert len(agent2_messages) == 1
        assert agent2_messages[0].content == "Message 2"
        
        nonexistent_messages = context.get_messages_by_agent("nonexistent")
        assert len(nonexistent_messages) == 0
    
    def test_get_conversation_summary(self):
        """Test conversation summary generation."""
        # Test empty conversation
        empty_context = ToolContext()
        summary = empty_context.get_conversation_summary()
        assert summary == "No conversation history available"
        
        # Test with messages
        messages = [
            Message("agent1", "This is a short message", datetime.now(), 10),
            Message("agent2", "This is a much longer message that should be truncated because it exceeds the preview length limit and goes beyond one hundred characters", datetime.now(), 15),
            Message("agent1", "Final message", datetime.now(), 20)
        ]
        
        context = ToolContext(conversation_history=messages)
        summary = context.get_conversation_summary()
        
        assert "agent1: This is a short message" in summary
        # The message should be truncated at 97 characters + "..."
        assert "agent2: This is a much longer message that should be truncated because it exceeds the preview length limi..." in summary
        assert "agent1: Final message" in summary
        assert " | " in summary  # Check separator
    
    def test_has_sufficient_tokens(self):
        """Test token availability checking."""
        context = ToolContext(available_tokens=1000)
        
        assert context.has_sufficient_tokens(500) is True
        assert context.has_sufficient_tokens(1000) is True
        assert context.has_sufficient_tokens(1001) is False
        assert context.has_sufficient_tokens(0) is True
    
    def test_to_dict(self):
        """Test ToolContext serialization to dictionary."""
        messages = [
            Message("agent1", "Test message", datetime.now(), 10)
        ]
        
        context = ToolContext(
            conversation_history=messages,
            current_turn=5,
            agent_id="test_agent",
            available_tokens=4000,
            metadata={"key": "value"},
            max_result_tokens=500,
            timeout=45.0
        )
        
        context_dict = context.to_dict()
        
        assert len(context_dict["conversation_history"]) == 1
        assert context_dict["current_turn"] == 5
        assert context_dict["agent_id"] == "test_agent"
        assert context_dict["available_tokens"] == 4000
        assert context_dict["metadata"] == {"key": "value"}
        assert context_dict["max_result_tokens"] == 500
        assert context_dict["timeout"] == 45.0
    
    def test_from_dict(self):
        """Test ToolContext deserialization from dictionary."""
        message_data = {
            "agent_id": "agent1",
            "content": "Test message",
            "timestamp": datetime.now().isoformat(),
            "token_count": 10,
            "metadata": {}
        }
        
        data = {
            "conversation_history": [message_data],
            "current_turn": 5,
            "agent_id": "test_agent",
            "available_tokens": 4000,
            "metadata": {"key": "value"},
            "max_result_tokens": 500,
            "timeout": 45.0
        }
        
        context = ToolContext.from_dict(data)
        
        assert len(context.conversation_history) == 1
        assert context.conversation_history[0].agent_id == "agent1"
        assert context.current_turn == 5
        assert context.agent_id == "test_agent"
        assert context.available_tokens == 4000
        assert context.metadata == {"key": "value"}
        assert context.max_result_tokens == 500
        assert context.timeout == 45.0


class TestToolInfo:
    """Test cases for ToolInfo data model."""
    
    def test_tool_info_creation(self):
        """Test basic ToolInfo creation."""
        info = ToolInfo(
            name="test_tool",
            description="A test tool for testing"
        )
        
        assert info.name == "test_tool"
        assert info.description == "A test tool for testing"
        assert info.capabilities == []
        assert info.required_config == []
        assert info.optional_config == []
        assert info.version == "1.0.0"
        assert info.author == ""
        assert info.max_execution_time == 30.0
        assert info.token_cost_estimate == 100
    
    def test_tool_info_with_capabilities(self):
        """Test ToolInfo with capabilities."""
        capabilities = [ToolCapability.SEARCH, ToolCapability.ANALYSIS]
        info = ToolInfo(
            name="search_tool",
            description="A search tool",
            capabilities=capabilities,
            required_config=["api_key"],
            optional_config=["timeout", "max_results"],
            version="2.1.0",
            author="Test Author",
            max_execution_time=60.0,
            token_cost_estimate=200
        )
        
        assert info.capabilities == capabilities
        assert info.required_config == ["api_key"]
        assert info.optional_config == ["timeout", "max_results"]
        assert info.version == "2.1.0"
        assert info.author == "Test Author"
        assert info.max_execution_time == 60.0
        assert info.token_cost_estimate == 200
    
    def test_has_capability(self):
        """Test capability checking."""
        info = ToolInfo(
            name="test_tool",
            description="Test",
            capabilities=[ToolCapability.SEARCH, ToolCapability.ANALYSIS]
        )
        
        assert info.has_capability(ToolCapability.SEARCH) is True
        assert info.has_capability(ToolCapability.ANALYSIS) is True
        assert info.has_capability(ToolCapability.CALCULATION) is False
        assert info.has_capability(ToolCapability.FILE_OPERATIONS) is False
    
    def test_get_config_requirements(self):
        """Test configuration requirements mapping."""
        info = ToolInfo(
            name="test_tool",
            description="Test",
            required_config=["api_key", "endpoint"],
            optional_config=["timeout", "retries"]
        )
        
        config_map = info.get_config_requirements()
        
        assert config_map["api_key"] is True
        assert config_map["endpoint"] is True
        assert config_map["timeout"] is False
        assert config_map["retries"] is False
        assert len(config_map) == 4
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        info = ToolInfo(
            name="test_tool",
            description="Test",
            required_config=["api_key", "endpoint"],
            optional_config=["timeout"]
        )
        
        config = {
            "api_key": "test_key",
            "endpoint": "https://api.example.com",
            "timeout": 30
        }
        
        errors = info.validate_config(config)
        assert len(errors) == 0
    
    def test_validate_config_missing_required(self):
        """Test configuration validation with missing required parameters."""
        info = ToolInfo(
            name="test_tool",
            description="Test",
            required_config=["api_key", "endpoint"]
        )
        
        config = {
            "api_key": "test_key"
            # Missing "endpoint"
        }
        
        errors = info.validate_config(config)
        assert len(errors) == 1
        assert "endpoint" in errors[0]
        assert "Missing required configuration parameter" in errors[0]
    
    def test_validate_config_none_values(self):
        """Test configuration validation with None values."""
        info = ToolInfo(
            name="test_tool",
            description="Test",
            required_config=["api_key"]
        )
        
        config = {
            "api_key": None
        }
        
        errors = info.validate_config(config)
        assert len(errors) == 1
        assert "api_key" in errors[0]
        assert "cannot be None" in errors[0]
    
    def test_to_dict(self):
        """Test ToolInfo serialization to dictionary."""
        capabilities = [ToolCapability.SEARCH, ToolCapability.ANALYSIS]
        info = ToolInfo(
            name="test_tool",
            description="A test tool",
            capabilities=capabilities,
            required_config=["api_key"],
            optional_config=["timeout"],
            version="1.5.0",
            author="Test Author",
            max_execution_time=45.0,
            token_cost_estimate=150
        )
        
        info_dict = info.to_dict()
        
        assert info_dict["name"] == "test_tool"
        assert info_dict["description"] == "A test tool"
        assert info_dict["capabilities"] == ["search", "analysis"]
        assert info_dict["required_config"] == ["api_key"]
        assert info_dict["optional_config"] == ["timeout"]
        assert info_dict["version"] == "1.5.0"
        assert info_dict["author"] == "Test Author"
        assert info_dict["max_execution_time"] == 45.0
        assert info_dict["token_cost_estimate"] == 150
    
    def test_from_dict(self):
        """Test ToolInfo deserialization from dictionary."""
        data = {
            "name": "test_tool",
            "description": "A test tool",
            "capabilities": ["search", "analysis"],
            "required_config": ["api_key"],
            "optional_config": ["timeout"],
            "version": "1.5.0",
            "author": "Test Author",
            "max_execution_time": 45.0,
            "token_cost_estimate": 150
        }
        
        info = ToolInfo.from_dict(data)
        
        assert info.name == "test_tool"
        assert info.description == "A test tool"
        assert len(info.capabilities) == 2
        assert ToolCapability.SEARCH in info.capabilities
        assert ToolCapability.ANALYSIS in info.capabilities
        assert info.required_config == ["api_key"]
        assert info.optional_config == ["timeout"]
        assert info.version == "1.5.0"
        assert info.author == "Test Author"
        assert info.max_execution_time == 45.0
        assert info.token_cost_estimate == 150


class MockTool(BaseTool):
    """Mock tool implementation for testing BaseTool."""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Set tool info before calling super().__init__() so validation works
        super().__init__(config)
    
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        """Mock execute implementation."""
        return ToolResult(
            success=True,
            content=f"Mock result for query: {query}",
            tool_name="mock_tool",
            execution_time=0.1
        )
    
    def is_relevant(self, context: ToolContext) -> bool:
        """Mock relevance check."""
        return "mock" in context.get_conversation_summary().lower()
    
    def get_tool_info(self) -> ToolInfo:
        """Return mock tool info."""
        return ToolInfo(
            name="mock_tool",
            description="A mock tool for testing",
            required_config=["required_param"],
            optional_config=["optional_param"]
        )


class TestBaseTool:
    """Test cases for BaseTool abstract base class."""
    
    def test_tool_creation_valid_config(self):
        """Test tool creation with valid configuration."""
        config = {
            "required_param": "test_value",
            "optional_param": "optional_value"
        }
        
        tool = MockTool(config)
        assert tool.config == config
        assert tool.is_configured() is True
    
    def test_tool_creation_invalid_config(self):
        """Test tool creation with invalid configuration."""
        config = {
            # Missing required_param
            "optional_param": "optional_value"
        }
        
        with pytest.raises(ToolConfigurationError) as exc_info:
            MockTool(config)
        
        assert "required_param" in str(exc_info.value)
        assert "mock_tool" in str(exc_info.value)
    
    def test_tool_creation_no_config(self):
        """Test tool creation without configuration."""
        with pytest.raises(ToolConfigurationError):
            MockTool()
    
    @pytest.mark.asyncio
    async def test_execute_method(self):
        """Test tool execution."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        context = ToolContext(agent_id="test_agent")
        result = await tool.execute("test query", context)
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "test query" in result.content
        assert result.tool_name == "mock_tool"
    
    def test_is_relevant_method(self):
        """Test tool relevance checking."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        # Test relevant context
        messages = [Message("agent1", "This is about mock testing", datetime.now(), 10)]
        relevant_context = ToolContext(conversation_history=messages)
        assert tool.is_relevant(relevant_context) is True
        
        # Test irrelevant context
        messages = [Message("agent1", "This is about something else", datetime.now(), 10)]
        irrelevant_context = ToolContext(conversation_history=messages)
        assert tool.is_relevant(irrelevant_context) is False
    
    def test_get_tool_info_method(self):
        """Test tool info retrieval."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        info = tool.get_tool_info()
        assert isinstance(info, ToolInfo)
        assert info.name == "mock_tool"
        assert info.description == "A mock tool for testing"
    
    def test_validate_input_valid(self):
        """Test input validation with valid parameters."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        context = ToolContext(timeout=30.0)
        errors = tool.validate_input("valid query", context)
        
        assert len(errors) == 0
    
    def test_validate_input_empty_query(self):
        """Test input validation with empty query."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        context = ToolContext(timeout=30.0)
        errors = tool.validate_input("", context)
        
        assert len(errors) == 1
        assert "Query cannot be empty" in errors[0]
    
    def test_validate_input_none_context(self):
        """Test input validation with None context."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        errors = tool.validate_input("valid query", None)
        
        assert len(errors) == 1
        assert "Context cannot be None" in errors[0]
    
    def test_validate_input_invalid_timeout(self):
        """Test input validation with invalid timeout."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        context = ToolContext(timeout=0)
        errors = tool.validate_input("valid query", context)
        
        assert len(errors) == 1
        assert "Timeout must be greater than 0" in errors[0]
    
    def test_get_config_value(self):
        """Test configuration value retrieval."""
        config = {
            "required_param": "test_value",
            "optional_param": "optional_value"
        }
        tool = MockTool(config)
        
        assert tool.get_config_value("required_param") == "test_value"
        assert tool.get_config_value("optional_param") == "optional_value"
        assert tool.get_config_value("nonexistent") is None
        assert tool.get_config_value("nonexistent", "default") == "default"
    
    def test_string_representations(self):
        """Test string representations of tool."""
        config = {"required_param": "test_value"}
        tool = MockTool(config)
        
        str_repr = str(tool)
        assert "mock_tool" in str_repr
        assert "1.0.0" in str_repr
        
        repr_str = repr(tool)
        assert "MockTool" in repr_str
        assert "mock_tool" in repr_str
        assert "1.0.0" in repr_str


class TestToolExceptions:
    """Test cases for tool exception classes."""
    
    def test_tool_error_basic(self):
        """Test basic ToolError creation."""
        error = ToolError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.tool_name is None
        assert error.context == {}
    
    def test_tool_error_with_tool_name(self):
        """Test ToolError with tool name."""
        error = ToolError("Test error", tool_name="test_tool")
        
        assert str(error) == "[test_tool] Test error"
        assert error.tool_name == "test_tool"
    
    def test_tool_error_with_context(self):
        """Test ToolError with context."""
        context = {"key": "value", "number": 42}
        error = ToolError("Test error", context=context)
        
        assert error.context == context
    
    def test_tool_error_to_dict(self):
        """Test ToolError serialization."""
        context = {"key": "value"}
        error = ToolError("Test error", tool_name="test_tool", context=context)
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "ToolError"
        assert error_dict["message"] == "Test error"
        assert error_dict["tool_name"] == "test_tool"
        assert error_dict["context"] == context
    
    def test_tool_execution_error(self):
        """Test ToolExecutionError specific attributes."""
        original_error = ValueError("Original error")
        error = ToolExecutionError(
            "Execution failed",
            tool_name="test_tool",
            original_error=original_error,
            retry_count=3
        )
        
        assert error.original_error == original_error
        assert error.retry_count == 3
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ToolExecutionError"
        assert error_dict["original_error"] == "Original error"
        assert error_dict["retry_count"] == 3
    
    def test_tool_timeout_error(self):
        """Test ToolTimeoutError specific attributes."""
        error = ToolTimeoutError(
            "Tool timed out",
            tool_name="test_tool",
            timeout_duration=30.0,
            elapsed_time=35.5
        )
        
        assert error.timeout_duration == 30.0
        assert error.elapsed_time == 35.5
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ToolTimeoutError"
        assert error_dict["timeout_duration"] == 30.0
        assert error_dict["elapsed_time"] == 35.5
    
    def test_tool_configuration_error(self):
        """Test ToolConfigurationError specific attributes."""
        error = ToolConfigurationError(
            "Invalid config",
            tool_name="test_tool",
            config_field="api_key",
            expected_value="string",
            actual_value="None"
        )
        
        assert error.config_field == "api_key"
        assert error.expected_value == "string"
        assert error.actual_value == "None"
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ToolConfigurationError"
        assert error_dict["config_field"] == "api_key"
        assert error_dict["expected_value"] == "string"
        assert error_dict["actual_value"] == "None"
    
    def test_tool_rate_limit_error(self):
        """Test ToolRateLimitError specific attributes."""
        error = ToolRateLimitError(
            "Rate limit exceeded",
            tool_name="test_tool",
            rate_limit=100,
            current_usage=105,
            reset_time=1234567890.0
        )
        
        assert error.rate_limit == 100
        assert error.current_usage == 105
        assert error.reset_time == 1234567890.0
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "ToolRateLimitError"
        assert error_dict["rate_limit"] == 100
        assert error_dict["current_usage"] == 105
        assert error_dict["reset_time"] == 1234567890.0


if __name__ == "__main__":
    pytest.main([__file__])