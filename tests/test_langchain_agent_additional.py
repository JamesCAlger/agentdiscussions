"""
Additional unit tests for LangChain agent implementation to improve coverage.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agentic_conversation.langchain_agent import LangChainAgent
from src.agentic_conversation.models import (
    AgentConfig, ModelConfig, ConversationState, Message
)
from src.agentic_conversation.agents import ConversationContext


class TestLangChainAgentAdditional:
    """Additional tests for LangChainAgent to improve coverage."""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration."""
        return AgentConfig(
            name="TestAgent",
            system_prompt="You are a test agent."
        )
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.1
        )
    
    @pytest.fixture
    def conversation_context(self):
        """Create test conversation context."""
        state = ConversationState()
        state.add_message(Message(
            agent_id="other_agent",
            content="Hello, how are you?",
            timestamp=datetime.now(),
            token_count=15
        ))
        
        return ConversationContext(
            conversation_state=state,
            system_prompt="You are a helpful assistant.",
            available_tokens=1500,
            turn_number=1,
            other_agent_id="other_agent"
        )
    
    def test_initialization_with_api_key(self, agent_config, model_config):
        """Test agent initialization with API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            assert agent.agent_id == "test_agent"
            assert agent.agent_config == agent_config
            assert agent.model_config == model_config
    
    def test_initialization_without_api_key(self, agent_config, model_config):
        """Test agent initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                LangChainAgent(
                    agent_id="test_agent",
                    agent_config=agent_config,
                    model_config=model_config
                )
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, agent_config, model_config, conversation_context):
        """Test successful response generation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            # Mock the LLM response
            mock_response = "This is a test response."
            with patch.object(agent, '_call_llm', return_value=mock_response):
                response = await agent.generate_response(conversation_context)
                
                assert response.content == mock_response
                assert response.agent_id == "test_agent"
                assert response.token_count > 0
                assert response.response_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_with_retry(self, agent_config, model_config, conversation_context):
        """Test response generation with retry on failure."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            # Mock first call to fail, second to succeed
            mock_response = "Success after retry."
            with patch.object(agent, '_call_llm', side_effect=[Exception("API Error"), mock_response]):
                response = await agent.generate_response(conversation_context)
                
                assert response.content == mock_response
                assert len(response.errors) == 1
                assert "API Error" in response.errors[0]
    
    @pytest.mark.asyncio
    async def test_generate_response_max_retries_exceeded(self, agent_config, model_config, conversation_context):
        """Test response generation when max retries are exceeded."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config,
                max_retries=2
            )
            
            # Mock all calls to fail
            with patch.object(agent, '_call_llm', side_effect=Exception("Persistent API Error")):
                response = await agent.generate_response(conversation_context)
                
                assert "I apologize, but I'm experiencing technical difficulties" in response.content
                assert len(response.errors) == 3  # Initial + 2 retries
    
    def test_build_prompt_with_history(self, agent_config, model_config, conversation_context):
        """Test prompt building with conversation history."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            prompt = agent._build_prompt(conversation_context)
            
            assert agent_config.system_prompt in prompt
            assert "Hello, how are you?" in prompt
            assert "other_agent" in prompt
    
    def test_build_prompt_empty_history(self, agent_config, model_config):
        """Test prompt building with empty conversation history."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            empty_context = ConversationContext(
                conversation_state=ConversationState(),
                system_prompt="You are a helpful assistant.",
                available_tokens=1500,
                turn_number=0
            )
            
            prompt = agent._build_prompt(empty_context)
            
            assert agent_config.system_prompt in prompt
            assert "No previous conversation" in prompt
    
    def test_call_llm_with_different_models(self, agent_config, conversation_context):
        """Test LLM calls with different model configurations."""
        models_to_test = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
        
        for model_name in models_to_test:
            model_config = ModelConfig(
                model_name=model_name,
                temperature=0.7,
                max_tokens=2000
            )
            
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                agent = LangChainAgent(
                    agent_id="test_agent",
                    agent_config=agent_config,
                    model_config=model_config
                )
                
                # Mock the LangChain LLM
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    mock_llm = Mock()
                    mock_llm.invoke.return_value = Mock(content="Test response")
                    mock_llm_class.return_value = mock_llm
                    
                    prompt = agent._build_prompt(conversation_context)
                    response = agent._call_llm(prompt)
                    
                    assert response == "Test response"
                    mock_llm_class.assert_called_with(
                        model=model_name,
                        temperature=0.7,
                        max_tokens=2000
                    )
    
    def test_call_llm_with_optional_parameters(self, agent_config, conversation_context):
        """Test LLM calls with optional model parameters."""
        model_config = ModelConfig(
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.95,
            frequency_penalty=0.2,
            presence_penalty=0.1
        )
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                mock_llm = Mock()
                mock_llm.invoke.return_value = Mock(content="Test response")
                mock_llm_class.return_value = mock_llm
                
                prompt = agent._build_prompt(conversation_context)
                response = agent._call_llm(prompt)
                
                # Verify all parameters were passed
                call_kwargs = mock_llm_class.call_args[1]
                assert call_kwargs['temperature'] == 0.8
                assert call_kwargs['max_tokens'] == 1500
                assert call_kwargs['top_p'] == 0.95
                assert call_kwargs['frequency_penalty'] == 0.2
                assert call_kwargs['presence_penalty'] == 0.1
    
    def test_get_agent_info(self, agent_config, model_config):
        """Test getting agent information."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            info = agent.get_agent_info()
            
            assert info.agent_id == "test_agent"
            assert info.name == agent_config.name
            assert info.model_name == model_config.model_name
            assert info.capabilities == ["text_generation", "conversation"]
    
    def test_calculate_exponential_backoff(self, agent_config, model_config):
        """Test exponential backoff calculation."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            # Test different retry attempts
            delay_0 = agent._calculate_exponential_backoff(0)
            delay_1 = agent._calculate_exponential_backoff(1)
            delay_2 = agent._calculate_exponential_backoff(2)
            
            assert delay_0 == 1.0  # Base delay
            assert delay_1 == 2.0  # 2^1
            assert delay_2 == 4.0  # 2^2
    
    @pytest.mark.asyncio
    async def test_response_validation(self, agent_config, model_config, conversation_context):
        """Test response validation and sanitization."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            # Test with empty response
            with patch.object(agent, '_call_llm', return_value=""):
                response = await agent.generate_response(conversation_context)
                assert "I apologize, but I couldn't generate a proper response" in response.content
            
            # Test with whitespace-only response
            with patch.object(agent, '_call_llm', return_value="   \n\t   "):
                response = await agent.generate_response(conversation_context)
                assert "I apologize, but I couldn't generate a proper response" in response.content
    
    def test_error_handling_in_llm_call(self, agent_config, model_config, conversation_context):
        """Test error handling in LLM calls."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            # Test different types of errors
            error_types = [
                Exception("Generic error"),
                ConnectionError("Network error"),
                TimeoutError("Request timeout"),
                ValueError("Invalid parameter")
            ]
            
            for error in error_types:
                with patch('langchain_openai.ChatOpenAI') as mock_llm_class:
                    mock_llm = Mock()
                    mock_llm.invoke.side_effect = error
                    mock_llm_class.return_value = mock_llm
                    
                    with pytest.raises(type(error)):
                        agent._call_llm("test prompt")
    
    def test_token_counting_integration(self, agent_config, model_config, conversation_context):
        """Test integration with token counting."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            agent = LangChainAgent(
                agent_id="test_agent",
                agent_config=agent_config,
                model_config=model_config
            )
            
            # Mock token counter
            with patch('src.agentic_conversation.token_counter.TokenCounter') as mock_counter_class:
                mock_counter = Mock()
                mock_counter.count_tokens.return_value.total_tokens = 25
                mock_counter_class.return_value = mock_counter
                
                with patch.object(agent, '_call_llm', return_value="Test response"):
                    response = agent.generate_response(conversation_context)
                    
                    # Verify token counting was called
                    mock_counter.count_tokens.assert_called()