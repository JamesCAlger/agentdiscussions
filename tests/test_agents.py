"""
Unit tests for the agent interface and related classes.

This module tests the BaseAgent abstract class, AgentInfo, AgentResponse,
ConversationContext, and related functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from typing import List

from src.agentic_conversation.agents import (
    BaseAgent, AgentInfo, AgentResponse, ConversationContext,
    AgentError, AgentTimeoutError, AgentValidationError,
    AgentResponseError, AgentConfigurationError
)
from src.agentic_conversation.models import (
    Message, ConversationState, ConversationStatus, AgentMetrics
)


class TestAgentInfo:
    """Test cases for AgentInfo dataclass."""
    
    def test_agent_info_creation(self):
        """Test creating an AgentInfo instance."""
        info = AgentInfo(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent for unit testing",
            model_name="gpt-4",
            capabilities=["text_generation", "conversation"],
            metadata={"version": "1.0"}
        )
        
        assert info.agent_id == "test_agent"
        assert info.name == "Test Agent"
        assert info.description == "A test agent for unit testing"
        assert info.model_name == "gpt-4"
        assert info.capabilities == ["text_generation", "conversation"]
        assert info.metadata == {"version": "1.0"}
    
    def test_agent_info_defaults(self):
        """Test AgentInfo with default values."""
        info = AgentInfo(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            model_name="gpt-4"
        )
        
        assert info.capabilities == []
        assert info.metadata == {}
    
    def test_agent_info_to_dict(self):
        """Test converting AgentInfo to dictionary."""
        info = AgentInfo(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            model_name="gpt-4",
            capabilities=["text_generation"],
            metadata={"version": "1.0"}
        )
        
        expected = {
            "agent_id": "test_agent",
            "name": "Test Agent",
            "description": "A test agent",
            "model_name": "gpt-4",
            "capabilities": ["text_generation"],
            "metadata": {"version": "1.0"}
        }
        
        assert info.to_dict() == expected
    
    def test_agent_info_from_dict(self):
        """Test creating AgentInfo from dictionary."""
        data = {
            "agent_id": "test_agent",
            "name": "Test Agent",
            "description": "A test agent",
            "model_name": "gpt-4",
            "capabilities": ["text_generation"],
            "metadata": {"version": "1.0"}
        }
        
        info = AgentInfo.from_dict(data)
        
        assert info.agent_id == "test_agent"
        assert info.name == "Test Agent"
        assert info.description == "A test agent"
        assert info.model_name == "gpt-4"
        assert info.capabilities == ["text_generation"]
        assert info.metadata == {"version": "1.0"}


class TestAgentResponse:
    """Test cases for AgentResponse dataclass."""
    
    def test_agent_response_creation(self):
        """Test creating an AgentResponse instance."""
        timestamp = datetime.now()
        response = AgentResponse(
            content="Hello, world!",
            agent_id="test_agent",
            timestamp=timestamp,
            token_count=3,
            response_time=1.5,
            model_calls=1,
            confidence=0.95,
            reasoning="Simple greeting response",
            metadata={"temperature": 0.7},
            errors=[]
        )
        
        assert response.content == "Hello, world!"
        assert response.agent_id == "test_agent"
        assert response.timestamp == timestamp
        assert response.token_count == 3
        assert response.response_time == 1.5
        assert response.model_calls == 1
        assert response.confidence == 0.95
        assert response.reasoning == "Simple greeting response"
        assert response.metadata == {"temperature": 0.7}
        assert response.errors == []
    
    def test_agent_response_defaults(self):
        """Test AgentResponse with default values."""
        response = AgentResponse(
            content="Hello",
            agent_id="test_agent"
        )
        
        assert response.token_count == 0
        assert response.response_time == 0.0
        assert response.model_calls == 1
        assert response.confidence is None
        assert response.reasoning is None
        assert response.metadata == {}
        assert response.errors == []
        assert isinstance(response.timestamp, datetime)
    
    def test_agent_response_to_message(self):
        """Test converting AgentResponse to Message."""
        timestamp = datetime.now()
        response = AgentResponse(
            content="Hello, world!",
            agent_id="test_agent",
            timestamp=timestamp,
            token_count=3,
            response_time=1.5,
            model_calls=1,
            confidence=0.95,
            reasoning="Simple greeting",
            errors=["minor_warning"]
        )
        
        message = response.to_message()
        
        assert isinstance(message, Message)
        assert message.agent_id == "test_agent"
        assert message.content == "Hello, world!"
        assert message.timestamp == timestamp
        assert message.token_count == 3
        assert message.metadata["response_time"] == 1.5
        assert message.metadata["model_calls"] == 1
        assert message.metadata["confidence"] == 0.95
        assert message.metadata["reasoning"] == "Simple greeting"
        assert message.metadata["errors"] == ["minor_warning"]
    
    def test_agent_response_to_agent_metrics(self):
        """Test converting AgentResponse to AgentMetrics."""
        response = AgentResponse(
            content="Hello",
            agent_id="test_agent",
            response_time=2.0,
            token_count=5,
            model_calls=2,
            errors=["error1", "error2"]
        )
        
        metrics = response.to_agent_metrics()
        
        assert isinstance(metrics, AgentMetrics)
        assert metrics.response_time == 2.0
        assert metrics.token_count == 5
        assert metrics.model_calls == 2
        assert metrics.errors == ["error1", "error2"]
    
    def test_agent_response_error_handling(self):
        """Test error handling methods in AgentResponse."""
        response = AgentResponse(
            content="Hello",
            agent_id="test_agent"
        )
        
        assert not response.has_errors()
        
        response.add_error("Test error")
        assert response.has_errors()
        assert "Test error" in response.errors
        
        response.add_error("Another error")
        assert len(response.errors) == 2
    
    def test_agent_response_serialization(self):
        """Test AgentResponse serialization methods."""
        timestamp = datetime.now()
        response = AgentResponse(
            content="Hello",
            agent_id="test_agent",
            timestamp=timestamp,
            token_count=3,
            response_time=1.0,
            confidence=0.8
        )
        
        # Test to_dict
        data = response.to_dict()
        assert data["content"] == "Hello"
        assert data["agent_id"] == "test_agent"
        assert data["timestamp"] == timestamp.isoformat()
        assert data["token_count"] == 3
        assert data["response_time"] == 1.0
        assert data["confidence"] == 0.8
        
        # Test from_dict
        restored = AgentResponse.from_dict(data)
        assert restored.content == response.content
        assert restored.agent_id == response.agent_id
        assert restored.timestamp == response.timestamp
        assert restored.token_count == response.token_count
        assert restored.response_time == response.response_time
        assert restored.confidence == response.confidence


class TestConversationContext:
    """Test cases for ConversationContext dataclass."""
    
    def create_sample_conversation_state(self) -> ConversationState:
        """Create a sample conversation state for testing."""
        state = ConversationState(
            max_context_tokens=1000,
            current_context_tokens=300
        )
        
        # Add some sample messages
        msg1 = Message(
            agent_id="agent_a",
            content="Hello",
            timestamp=datetime.now(),
            token_count=50
        )
        msg2 = Message(
            agent_id="agent_b",
            content="Hi there!",
            timestamp=datetime.now(),
            token_count=100
        )
        msg3 = Message(
            agent_id="agent_a",
            content="How are you?",
            timestamp=datetime.now(),
            token_count=150
        )
        
        state.add_message(msg1)
        state.add_message(msg2)
        state.add_message(msg3)
        
        return state
    
    def test_conversation_context_creation(self):
        """Test creating a ConversationContext instance."""
        state = self.create_sample_conversation_state()
        
        context = ConversationContext(
            conversation_state=state,
            system_prompt="You are a helpful assistant",
            available_tokens=500,
            turn_number=3,
            other_agent_id="agent_b",
            metadata={"temperature": 0.7}
        )
        
        assert context.conversation_state == state
        assert context.system_prompt == "You are a helpful assistant"
        assert context.available_tokens == 500
        assert context.turn_number == 3
        assert context.other_agent_id == "agent_b"
        assert context.metadata == {"temperature": 0.7}
    
    def test_conversation_context_defaults(self):
        """Test ConversationContext with default values."""
        state = ConversationState()
        
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test prompt",
            available_tokens=1000,
            turn_number=1
        )
        
        assert context.other_agent_id is None
        assert context.metadata == {}
    
    def test_get_conversation_history(self):
        """Test getting conversation history."""
        state = self.create_sample_conversation_state()
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test",
            available_tokens=1000,
            turn_number=1
        )
        
        history = context.get_conversation_history()
        assert len(history) == 3
        assert all(isinstance(msg, Message) for msg in history)
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        state = self.create_sample_conversation_state()
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test",
            available_tokens=1000,
            turn_number=1
        )
        
        # Get last 2 messages
        recent = context.get_recent_messages(2)
        assert len(recent) == 2
        assert recent[0].content == "Hi there!"
        assert recent[1].content == "How are you?"
        
        # Get more messages than available
        all_recent = context.get_recent_messages(10)
        assert len(all_recent) == 3
        
        # Get zero messages
        none_recent = context.get_recent_messages(0)
        assert len(none_recent) == 0
    
    def test_get_messages_by_agent(self):
        """Test getting messages by specific agent."""
        state = self.create_sample_conversation_state()
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test",
            available_tokens=1000,
            turn_number=1
        )
        
        agent_a_messages = context.get_messages_by_agent("agent_a")
        assert len(agent_a_messages) == 2
        assert all(msg.agent_id == "agent_a" for msg in agent_a_messages)
        
        agent_b_messages = context.get_messages_by_agent("agent_b")
        assert len(agent_b_messages) == 1
        assert agent_b_messages[0].agent_id == "agent_b"
        
        nonexistent_messages = context.get_messages_by_agent("nonexistent")
        assert len(nonexistent_messages) == 0
    
    def test_get_last_message(self):
        """Test getting the last message."""
        state = self.create_sample_conversation_state()
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test",
            available_tokens=1000,
            turn_number=1
        )
        
        last_message = context.get_last_message()
        assert last_message is not None
        assert last_message.content == "How are you?"
        assert last_message.agent_id == "agent_a"
        
        # Test with empty conversation
        empty_state = ConversationState()
        empty_context = ConversationContext(
            conversation_state=empty_state,
            system_prompt="Test",
            available_tokens=1000,
            turn_number=1
        )
        
        assert empty_context.get_last_message() is None
    
    def test_context_utilization_methods(self):
        """Test context utilization methods."""
        state = ConversationState(
            max_context_tokens=1000,
            current_context_tokens=900  # 90% utilization
        )
        
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test",
            available_tokens=100,
            turn_number=1
        )
        
        assert context.get_context_utilization() == 90.0
        assert context.is_context_near_limit()  # Default threshold is 90%
        assert context.is_context_near_limit(85.0)  # Above 85%
        assert not context.is_context_near_limit(95.0)  # Below 95%
    
    def test_conversation_context_serialization(self):
        """Test ConversationContext serialization methods."""
        state = self.create_sample_conversation_state()
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test prompt",
            available_tokens=500,
            turn_number=2,
            other_agent_id="agent_b",
            metadata={"key": "value"}
        )
        
        # Test to_dict
        data = context.to_dict()
        assert "conversation_state" in data
        assert data["system_prompt"] == "Test prompt"
        assert data["available_tokens"] == 500
        assert data["turn_number"] == 2
        assert data["other_agent_id"] == "agent_b"
        assert data["metadata"] == {"key": "value"}
        
        # Test from_dict
        restored = ConversationContext.from_dict(data)
        assert restored.system_prompt == context.system_prompt
        assert restored.available_tokens == context.available_tokens
        assert restored.turn_number == context.turn_number
        assert restored.other_agent_id == context.other_agent_id
        assert restored.metadata == context.metadata


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def __init__(self, agent_id: str, name: str, system_prompt: str, 
                 response_content: str = "Test response"):
        super().__init__(agent_id, name, system_prompt)
        self.response_content = response_content
        self.generate_response_called = False
        self.get_agent_info_called = False
    
    async def generate_response(self, context: ConversationContext) -> AgentResponse:
        """Generate a test response."""
        self.generate_response_called = True
        return AgentResponse(
            content=self.response_content,
            agent_id=self.agent_id,
            token_count=len(self.response_content.split()),
            response_time=0.1
        )
    
    def get_agent_info(self) -> AgentInfo:
        """Get test agent info."""
        self.get_agent_info_called = True
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description="Test agent for unit testing",
            model_name="test-model",
            capabilities=["testing"]
        )


class TestBaseAgent:
    """Test cases for BaseAgent abstract class."""
    
    def test_base_agent_initialization(self):
        """Test BaseAgent initialization."""
        agent = ConcreteAgent("test_agent", "Test Agent", "You are a test agent")
        
        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.system_prompt == "You are a test agent"
        assert agent.logger is not None
    
    def test_base_agent_validation_errors(self):
        """Test BaseAgent initialization validation."""
        # Empty agent_id
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            ConcreteAgent("", "Test Agent", "Test prompt")
        
        # Whitespace-only agent_id
        with pytest.raises(ValueError, match="Agent ID cannot be empty"):
            ConcreteAgent("   ", "Test Agent", "Test prompt")
        
        # Empty name
        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            ConcreteAgent("test_agent", "", "Test prompt")
        
        # Empty system_prompt
        with pytest.raises(ValueError, match="System prompt cannot be empty"):
            ConcreteAgent("test_agent", "Test Agent", "")
    
    @pytest.mark.asyncio
    async def test_generate_response_interface(self):
        """Test the generate_response interface."""
        agent = ConcreteAgent("test_agent", "Test Agent", "Test prompt")
        
        state = ConversationState()
        context = ConversationContext(
            conversation_state=state,
            system_prompt="Test prompt",
            available_tokens=1000,
            turn_number=1
        )
        
        response = await agent.generate_response(context)
        
        assert agent.generate_response_called
        assert isinstance(response, AgentResponse)
        assert response.content == "Test response"
        assert response.agent_id == "test_agent"
    
    def test_get_agent_info_interface(self):
        """Test the get_agent_info interface."""
        agent = ConcreteAgent("test_agent", "Test Agent", "Test prompt")
        
        info = agent.get_agent_info()
        
        assert agent.get_agent_info_called
        assert isinstance(info, AgentInfo)
        assert info.agent_id == "test_agent"
        assert info.name == "Test Agent"
    
    def test_validate_context(self):
        """Test context validation."""
        agent = ConcreteAgent("test_agent", "Test Agent", "Test prompt")
        
        # Valid context
        valid_context = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Valid prompt",
            available_tokens=1000,
            turn_number=1
        )
        
        errors = agent.validate_context(valid_context)
        assert len(errors) == 0
        assert agent.is_context_valid(valid_context)
        
        # Invalid context - zero available tokens
        invalid_context1 = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Valid prompt",
            available_tokens=0,
            turn_number=1
        )
        
        errors = agent.validate_context(invalid_context1)
        assert len(errors) == 1
        assert "Available tokens must be greater than 0" in errors
        assert not agent.is_context_valid(invalid_context1)
        
        # Invalid context - negative turn number
        invalid_context2 = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Valid prompt",
            available_tokens=1000,
            turn_number=-1
        )
        
        errors = agent.validate_context(invalid_context2)
        assert len(errors) == 1
        assert "Turn number cannot be negative" in errors
        
        # Invalid context - empty system prompt
        invalid_context3 = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="",
            available_tokens=1000,
            turn_number=1
        )
        
        errors = agent.validate_context(invalid_context3)
        assert len(errors) == 1
        assert "System prompt cannot be empty" in errors
    
    @pytest.mark.asyncio
    async def test_prepare_context(self):
        """Test context preparation (default implementation)."""
        agent = ConcreteAgent("test_agent", "Test Agent", "Test prompt")
        
        original_context = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Test prompt",
            available_tokens=1000,
            turn_number=1
        )
        
        prepared_context = await agent.prepare_context(original_context)
        
        # Default implementation should return the same context
        assert prepared_context is original_context
    
    @pytest.mark.asyncio
    async def test_post_process_response(self):
        """Test response post-processing (default implementation)."""
        agent = ConcreteAgent("test_agent", "Test Agent", "Test prompt")
        
        original_response = AgentResponse(
            content="Test response",
            agent_id="test_agent"
        )
        
        context = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Test prompt",
            available_tokens=1000,
            turn_number=1
        )
        
        processed_response = await agent.post_process_response(original_response, context)
        
        # Default implementation should return the same response
        assert processed_response is original_response
    
    def test_string_representations(self):
        """Test string representations of BaseAgent."""
        agent = ConcreteAgent("test_agent", "Test Agent", "You are a helpful test agent")
        
        str_repr = str(agent)
        assert "ConcreteAgent" in str_repr
        assert "test_agent" in str_repr
        assert "Test Agent" in str_repr
        
        repr_str = repr(agent)
        assert "ConcreteAgent" in repr_str
        assert "test_agent" in repr_str
        assert "Test Agent" in repr_str
        assert "You are a helpful test agent" in repr_str


class TestAgentErrors:
    """Test cases for agent error classes."""
    
    def test_agent_error_basic(self):
        """Test basic AgentError functionality."""
        error = AgentError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.agent_id is None
        assert error.context is None
        assert error.original_error is None
    
    def test_agent_error_with_details(self):
        """Test AgentError with additional details."""
        original_error = ValueError("Original error")
        context = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Test",
            available_tokens=1000,
            turn_number=1
        )
        
        error = AgentError(
            "Test error message",
            agent_id="test_agent",
            context=context,
            original_error=original_error
        )
        
        assert error.message == "Test error message"
        assert error.agent_id == "test_agent"
        assert error.context is context
        assert error.original_error is original_error
    
    def test_agent_error_to_dict(self):
        """Test AgentError serialization to dictionary."""
        original_error = ValueError("Original error")
        error = AgentError(
            "Test error message",
            agent_id="test_agent",
            original_error=original_error
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "AgentError"
        assert error_dict["message"] == "Test error message"
        assert error_dict["agent_id"] == "test_agent"
        assert "timestamp" in error_dict
        assert error_dict["original_error"] == "Original error"
    
    def test_specific_error_types(self):
        """Test specific agent error types."""
        timeout_error = AgentTimeoutError("Operation timed out")
        assert isinstance(timeout_error, AgentError)
        assert str(timeout_error) == "Operation timed out"
        
        validation_error = AgentValidationError("Validation failed")
        assert isinstance(validation_error, AgentError)
        assert str(validation_error) == "Validation failed"
        
        response_error = AgentResponseError("Response generation failed")
        assert isinstance(response_error, AgentError)
        assert str(response_error) == "Response generation failed"
        
        config_error = AgentConfigurationError("Configuration invalid")
        assert isinstance(config_error, AgentError)
        assert str(config_error) == "Configuration invalid"


if __name__ == "__main__":
    pytest.main([__file__])

class TestLangChainAgent:
    """Test cases for LangChainAgent implementation."""
    
    @pytest.fixture
    def model_config(self):
        """Create a test model configuration."""
        from src.agentic_conversation.models import ModelConfig
        return ModelConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    @pytest.fixture
    def token_counter(self):
        """Create a mock token counter."""
        from unittest.mock import Mock
        from src.agentic_conversation.token_counter import TokenCountResult, ModelType
        
        mock_counter = Mock()
        mock_counter.count_tokens.return_value = TokenCountResult(
            token_count=10,
            model_type=ModelType.GPT_4,
            encoding_name="cl100k_base",
            content_length=50
        )
        return mock_counter
    
    @pytest.fixture
    def conversation_context(self):
        """Create a test conversation context."""
        state = ConversationState(max_context_tokens=2000)
        
        # Add some sample messages
        msg1 = Message(
            agent_id="agent_a",
            content="Hello there!",
            timestamp=datetime.now(),
            token_count=20
        )
        msg2 = Message(
            agent_id="agent_b", 
            content="Hi! How can I help you?",
            timestamp=datetime.now(),
            token_count=30
        )
        
        state.add_message(msg1)
        state.add_message(msg2)
        
        return ConversationContext(
            conversation_state=state,
            system_prompt="You are a helpful assistant",
            available_tokens=1500,
            turn_number=2,
            other_agent_id="agent_a"
        )
    
    def test_langchain_agent_initialization(self, model_config, token_counter):
        """Test LangChainAgent initialization."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        
        with pytest.raises(Exception):
            # This will fail because we don't have actual API keys, but we can test the initialization logic
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="You are a test agent",
                model_config=model_config,
                token_counter=token_counter,
                max_retries=2,
                retry_delay=0.5,
                timeout=10.0
            )
    
    def test_langchain_agent_validation_errors(self, token_counter):
        """Test LangChainAgent initialization validation."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from src.agentic_conversation.models import ModelConfig
        from src.agentic_conversation.agents import AgentConfigurationError
        
        # Invalid model config
        invalid_config = ModelConfig(
            model_name="",  # Empty model name
            temperature=0.7,
            max_tokens=1000
        )
        
        with pytest.raises(AgentConfigurationError):
            LangChainAgent(
                agent_id="test_agent",
                name="Test Agent", 
                system_prompt="Test prompt",
                model_config=invalid_config,
                token_counter=token_counter
            )
    
    def test_convert_messages_to_langchain(self, model_config, token_counter):
        """Test message conversion to LangChain format."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from unittest.mock import patch, Mock
        import os
        
        # Mock the LLM creation to avoid API calls
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
                mock_llm = Mock()
                mock_create_llm.return_value = mock_llm
                
                agent = LangChainAgent(
                    agent_id="test_agent",
                    name="Test Agent",
                    system_prompt="Test prompt",
                    model_config=model_config,
                    token_counter=token_counter
                )
            
            # Create test messages
            messages = [
                Message(
                    agent_id="test_agent",
                    content="Hello from me",
                    timestamp=datetime.now(),
                    token_count=10
                ),
                Message(
                    agent_id="other_agent",
                    content="Hello from other",
                    timestamp=datetime.now(),
                    token_count=15
                )
            ]
            
            langchain_messages = agent._convert_messages_to_langchain(messages)
            
            assert len(langchain_messages) == 2
            # First message should be AIMessage (from this agent)
            assert langchain_messages[0].__class__.__name__ == "AIMessage"
            assert langchain_messages[0].content == "Hello from me"
            # Second message should be HumanMessage (from other agent)
            assert langchain_messages[1].__class__.__name__ == "HumanMessage"
            assert langchain_messages[1].content == "Hello from other"
    
    def test_prepare_messages(self, model_config, token_counter, conversation_context):
        """Test message preparation for LLM."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from unittest.mock import patch, Mock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="agent_b",  # This agent is agent_b
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter
            )
            
            messages = agent._prepare_messages(conversation_context)
            
            # Should have system message + conversation messages
            assert len(messages) >= 3
            # First message should be SystemMessage
            assert messages[0].__class__.__name__ == "SystemMessage"
            assert messages[0].content == "You are a helpful assistant"
    
    @pytest.mark.asyncio
    async def test_generate_response_with_mock(self, model_config, token_counter, conversation_context):
        """Test response generation with mocked LLM."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from unittest.mock import patch, Mock, AsyncMock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            # Create mock LLM
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "This is a test response from the mocked LLM."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter,
                max_retries=1,
                timeout=5.0
            )
            
            response = await agent.generate_response(conversation_context)
            
            assert isinstance(response, AgentResponse)
            assert response.content == "This is a test response from the mocked LLM."
            assert response.agent_id == "test_agent"
            assert response.token_count > 0
            assert response.response_time > 0
            assert response.model_calls == 1
            assert not response.has_errors()
    
    @pytest.mark.asyncio
    async def test_generate_response_with_retry(self, model_config, token_counter, conversation_context):
        """Test response generation with retry logic."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from src.agentic_conversation.agents import AgentResponseError
        from unittest.mock import patch, Mock, AsyncMock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            # Create mock LLM that fails first, then succeeds
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "Success on retry!"
            
            # First call fails, second succeeds
            mock_llm.ainvoke = AsyncMock(side_effect=[
                Exception("API Error"),
                mock_response
            ])
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter,
                max_retries=2,
                retry_delay=0.1,  # Short delay for testing
                timeout=5.0
            )
            
            response = await agent.generate_response(conversation_context)
            
            assert isinstance(response, AgentResponse)
            assert response.content == "Success on retry!"
            assert response.model_calls == 2  # One failure + one success
            assert response.response_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_timeout(self, model_config, token_counter, conversation_context):
        """Test response generation timeout handling."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from src.agentic_conversation.agents import AgentTimeoutError
        from unittest.mock import patch, Mock, AsyncMock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            # Create mock LLM that times out
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError("Request timed out"))
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter,
                max_retries=1,
                retry_delay=0.1,
                timeout=0.1  # Very short timeout
            )
            
            with pytest.raises(AgentTimeoutError):
                await agent.generate_response(conversation_context)
    
    @pytest.mark.asyncio
    async def test_generate_response_all_retries_fail(self, model_config, token_counter, conversation_context):
        """Test response generation when all retries fail."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from src.agentic_conversation.agents import AgentResponseError
        from unittest.mock import patch, Mock, AsyncMock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            # Create mock LLM that always fails
            mock_llm = Mock()
            mock_llm.ainvoke = AsyncMock(side_effect=Exception("Persistent API Error"))
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter,
                max_retries=2,
                retry_delay=0.1,
                timeout=5.0
            )
            
            with pytest.raises(AgentResponseError):
                await agent.generate_response(conversation_context)
    
    @pytest.mark.asyncio
    async def test_generate_response_empty_content(self, model_config, token_counter, conversation_context):
        """Test handling of empty response content."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from src.agentic_conversation.agents import AgentResponseError
        from unittest.mock import patch, Mock, AsyncMock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            # Create mock LLM that returns empty content
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = ""  # Empty response
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter,
                max_retries=1,
                timeout=5.0
            )
            
            with pytest.raises(AgentResponseError, match="empty response"):
                await agent.generate_response(conversation_context)
    
    @pytest.mark.asyncio
    async def test_generate_response_validation_error(self, model_config, token_counter):
        """Test response generation with invalid context."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from src.agentic_conversation.agents import AgentValidationError
        from unittest.mock import patch, Mock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter
            )
            
            # Create invalid context
            invalid_context = ConversationContext(
                conversation_state=ConversationState(),
                system_prompt="",  # Empty system prompt
                available_tokens=0,  # Zero available tokens
                turn_number=-1  # Negative turn number
            )
            
            with pytest.raises(AgentValidationError):
                await agent.generate_response(invalid_context)
    
    def test_get_agent_info(self, model_config, token_counter):
        """Test getting agent information."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from unittest.mock import patch, Mock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter
            )
            
            info = agent.get_agent_info()
            
            assert isinstance(info, AgentInfo)
            assert info.agent_id == "test_agent"
            assert info.name == "Test Agent"
            assert info.model_name == "gpt-4"
            assert "text_generation" in info.capabilities
            assert "conversation" in info.capabilities
            assert "openai_models" in info.capabilities  # Because model is gpt-4
            assert info.metadata["temperature"] == 0.7
            assert info.metadata["max_tokens"] == 1000
    
    def test_update_model_config(self, model_config, token_counter):
        """Test updating model configuration."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from src.agentic_conversation.models import ModelConfig
        from src.agentic_conversation.agents import AgentConfigurationError
        from unittest.mock import patch, Mock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter
            )
            
            # Test valid update
            new_config = ModelConfig(
                model_name="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=500
            )
            
            agent.update_model_config(new_config)
            assert agent.model_config.model_name == "gpt-3.5-turbo"
            assert agent.model_config.temperature == 0.5
            assert agent.model_config.max_tokens == 500
            
            # Test invalid update
            invalid_config = ModelConfig(
                model_name="",  # Empty model name
                temperature=0.5,
                max_tokens=500
            )
            
            with pytest.raises(AgentConfigurationError):
                agent.update_model_config(invalid_config)
    
    def test_get_model_info(self, model_config, token_counter):
        """Test getting model information."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from unittest.mock import patch, Mock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter,
                max_retries=3,
                retry_delay=1.0,
                timeout=30.0
            )
            
            model_info = agent.get_model_info()
            
            assert model_info["model_name"] == "gpt-4"
            assert model_info["temperature"] == 0.7
            assert model_info["max_tokens"] == 1000
            assert model_info["top_p"] == 0.9
            assert model_info["frequency_penalty"] == 0.0
            assert model_info["presence_penalty"] == 0.0
            assert model_info["timeout"] == 30.0
            assert model_info["max_retries"] == 3
            assert model_info["retry_delay"] == 1.0
    
    def test_string_representations(self, model_config, token_counter):
        """Test string representations of LangChainAgent."""
        from src.agentic_conversation.langchain_agent import LangChainAgent
        from unittest.mock import patch, Mock
        
        with patch.object(LangChainAgent, '_create_llm') as mock_create_llm:
            mock_llm = Mock()
            mock_create_llm.return_value = mock_llm
            
            agent = LangChainAgent(
                agent_id="test_agent",
                name="Test Agent",
                system_prompt="Test prompt",
                model_config=model_config,
                token_counter=token_counter
            )
            
            str_repr = str(agent)
            assert "LangChainAgent" in str_repr
            assert "test_agent" in str_repr
            assert "Test Agent" in str_repr
            assert "gpt-4" in str_repr
            
            repr_str = repr(agent)
            assert "LangChainAgent" in repr_str
            assert "test_agent" in repr_str
            assert "Test Agent" in repr_str
            assert "gpt-4" in repr_str
            assert "temperature=0.7" in repr_str
            assert "max_tokens=1000" in repr_str