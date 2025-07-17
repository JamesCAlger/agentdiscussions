"""
Unit tests for orchestrator error handling and recovery mechanisms.

These tests verify the error handling, circuit breaker integration,
and recovery strategies in the ConversationOrchestrator.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agentic_conversation.orchestrator import ConversationOrchestrator, OrchestrationError
from src.agentic_conversation.models import (
    SystemConfig, AgentConfig, ModelConfig, ConversationConfig, LoggingConfig,
    ConversationState, ConversationStatus, Message
)
from src.agentic_conversation.circuit_breaker import CircuitBreakerError, CircuitState
from src.agentic_conversation.agents import AgentError, AgentTimeoutError


class TestOrchestratorErrorHandling:
    """Test suite for orchestrator error handling mechanisms."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample system configuration for testing."""
        return SystemConfig(
            agent_a=AgentConfig(
                name="TestAgent1",
                system_prompt="You are a test agent."
            ),
            agent_b=AgentConfig(
                name="TestAgent2",
                system_prompt="You are another test agent."
            ),
            model=ModelConfig(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            ),
            conversation=ConversationConfig(
                max_turns=4,
                context_window_strategy="sliding",
                context_window_size=2000,
                turn_timeout=5.0
            ),
            logging=LoggingConfig(
                log_level="INFO",
                output_directory="./test_logs",
                real_time_display=False
            )
        )
    
    def test_error_handling_initialization(self, sample_config):
        """Test that error handling components are properly initialized."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Check circuit breakers are initialized
            assert "agent_a" in orchestrator.circuit_breakers
            assert "agent_b" in orchestrator.circuit_breakers
            assert "conversation" in orchestrator.circuit_breakers
            
            # Check error recovery strategies are initialized
            assert "context_overflow" in orchestrator.error_recovery_strategies
            assert "agent_failure" in orchestrator.error_recovery_strategies
            assert "timeout" in orchestrator.error_recovery_strategies
            assert "api_failure" in orchestrator.error_recovery_strategies
            
            # Check max recovery attempts is set
            assert orchestrator.max_recovery_attempts == 3
    
    def test_circuit_breaker_stats(self, sample_config):
        """Test getting circuit breaker statistics."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            stats = orchestrator.get_circuit_breaker_stats()
            
            assert "agent_a" in stats
            assert "agent_b" in stats
            assert "conversation" in stats
            
            # Check stats structure
            agent_a_stats = stats["agent_a"]
            assert "name" in agent_a_stats
            assert "state" in agent_a_stats
            assert "total_calls" in agent_a_stats
            assert "total_failures" in agent_a_stats
            assert "success_rate" in agent_a_stats
    
    def test_reset_circuit_breakers(self, sample_config):
        """Test resetting all circuit breakers."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Force a circuit breaker to open
            orchestrator.circuit_breakers["agent_a"].force_open()
            assert orchestrator.circuit_breakers["agent_a"].state == CircuitState.OPEN
            
            # Reset all circuit breakers
            orchestrator.reset_circuit_breakers()
            assert orchestrator.circuit_breakers["agent_a"].state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_context_overflow_recovery(self, sample_config):
        """Test recovery from context overflow errors."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Create a mock conversation state with overflow
            conversation_state = ConversationState(max_context_tokens=1000)
            conversation_state.current_context_tokens = 1200  # Over limit
            
            # Add some messages
            for i in range(5):
                message = Message(
                    agent_id=f"agent_{i % 2}",
                    content=f"Test message {i}" * 20,  # Long content
                    timestamp=datetime.now(),
                    token_count=100
                )
                conversation_state.messages.append(message)
            
            context = {"conversation_state": conversation_state}
            error = Exception("Context overflow")
            
            # Test recovery
            recovery_result = await orchestrator._recover_from_context_overflow(error, context)
            
            assert recovery_result["recovery_successful"] is True
            assert recovery_result["strategy"] == "aggressive_context_truncation"
            assert "messages_removed" in recovery_result
            assert "tokens_saved" in recovery_result
    
    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, sample_config):
        """Test recovery from agent failure errors."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Create mock conversation context
            mock_context_obj = Mock()
            mock_context_obj.system_prompt = "You are a complex agent with detailed instructions."
            
            context = {
                "agent_id": "agent_a",
                "conversation_context": mock_context_obj
            }
            error = AgentError("Agent failed to generate response")
            
            # Test recovery
            recovery_result = await orchestrator._recover_from_agent_failure(error, context)
            
            assert recovery_result["recovery_successful"] is True
            assert recovery_result["strategy"] == "simplified_prompt"
            assert "original_prompt_length" in recovery_result
            assert "simplified_prompt_length" in recovery_result
            
            # Check that prompt was simplified
            assert mock_context_obj.system_prompt == "You are a helpful assistant. Please provide a brief response."
    
    @pytest.mark.asyncio
    async def test_timeout_recovery(self, sample_config):
        """Test recovery from timeout errors."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Store original timeout
            original_timeout = orchestrator.agent_a.timeout
            
            context = {"agent_id": "agent_a"}
            error = AgentTimeoutError("Request timed out")
            
            # Test recovery
            recovery_result = await orchestrator._recover_from_timeout(error, context)
            
            assert recovery_result["recovery_successful"] is True
            assert recovery_result["strategy"] == "increased_timeout"
            assert recovery_result["original_timeout"] == original_timeout
            assert recovery_result["new_timeout"] > original_timeout
            
            # Check that timeout was actually increased
            assert orchestrator.agent_a.timeout > original_timeout
    
    @pytest.mark.asyncio
    async def test_api_failure_recovery(self, sample_config):
        """Test recovery from API failure errors."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            context = {"attempt": 2}
            error = Exception("API rate limit exceeded")
            
            # Mock asyncio.sleep to avoid actual delay in tests
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                recovery_result = await orchestrator._recover_from_api_failure(error, context)
                
                assert recovery_result["recovery_successful"] is True
                assert recovery_result["strategy"] == "exponential_backoff"
                assert recovery_result["backoff_delay"] == 4  # 2^2
                assert recovery_result["attempt"] == 2
                
                # Check that sleep was called with correct delay
                mock_sleep.assert_called_once_with(4)
    
    @pytest.mark.asyncio
    async def test_attempt_error_recovery_success(self, sample_config):
        """Test successful error recovery attempt."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            error = Exception("Test error")
            context = {"test": "data"}
            
            # Mock a recovery strategy
            mock_recovery = AsyncMock(return_value={
                "recovery_successful": True,
                "strategy": "test_strategy"
            })
            orchestrator.error_recovery_strategies["test_error"] = mock_recovery
            
            result = await orchestrator._attempt_error_recovery(error, "test_error", context, 1)
            
            assert result["recovery_successful"] is True
            assert result["strategy"] == "test_strategy"
            mock_recovery.assert_called_once_with(error, context)
    
    @pytest.mark.asyncio
    async def test_attempt_error_recovery_max_attempts(self, sample_config):
        """Test error recovery with max attempts exceeded."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            error = Exception("Test error")
            context = {"test": "data"}
            
            # Attempt recovery with attempt count exceeding max
            result = await orchestrator._attempt_error_recovery(
                error, "test_error", context, orchestrator.max_recovery_attempts + 1
            )
            
            assert result["recovery_successful"] is False
            assert result["strategy"] == "max_attempts_exceeded"
            assert result["attempts"] == orchestrator.max_recovery_attempts + 1
    
    @pytest.mark.asyncio
    async def test_attempt_error_recovery_no_strategy(self, sample_config):
        """Test error recovery with no available strategy."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            error = Exception("Test error")
            context = {"test": "data"}
            
            result = await orchestrator._attempt_error_recovery(error, "unknown_error", context, 1)
            
            assert result["recovery_successful"] is False
            assert result["strategy"] == "no_strategy_available"
            assert result["error_type"] == "unknown_error"
    
    @pytest.mark.asyncio
    async def test_attempt_error_recovery_strategy_failure(self, sample_config):
        """Test error recovery when strategy itself fails."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            error = Exception("Test error")
            context = {"test": "data"}
            
            # Mock a recovery strategy that fails
            mock_recovery = AsyncMock(side_effect=Exception("Recovery failed"))
            orchestrator.error_recovery_strategies["test_error"] = mock_recovery
            
            result = await orchestrator._attempt_error_recovery(error, "test_error", context, 1)
            
            assert result["recovery_successful"] is False
            assert result["strategy"] == "test_error"
            assert "recovery_error" in result
            assert "original_error" in result
    
    @pytest.mark.asyncio
    async def test_context_overflow_recovery_failure(self, sample_config):
        """Test context overflow recovery when it fails."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Create context without conversation_state to trigger failure
            context = {"invalid": "context"}
            error = Exception("Context overflow")
            
            recovery_result = await orchestrator._recover_from_context_overflow(error, context)
            
            assert recovery_result["recovery_successful"] is False
            assert recovery_result["strategy"] == "aggressive_context_truncation"
            assert "error" in recovery_result
    
    @pytest.mark.asyncio
    async def test_agent_failure_recovery_failure(self, sample_config):
        """Test agent failure recovery when it fails."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Create context without conversation_context to trigger failure
            context = {"agent_id": "agent_a"}
            error = AgentError("Agent failed")
            
            recovery_result = await orchestrator._recover_from_agent_failure(error, context)
            
            assert recovery_result["recovery_successful"] is False
            assert recovery_result["strategy"] == "simplified_prompt"
            assert "error" in recovery_result
    
    @pytest.mark.asyncio
    async def test_timeout_recovery_failure(self, sample_config):
        """Test timeout recovery when it fails."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Create context without agent_id to trigger failure
            context = {"invalid": "context"}
            error = AgentTimeoutError("Timeout")
            
            recovery_result = await orchestrator._recover_from_timeout(error, context)
            
            assert recovery_result["recovery_successful"] is False
            assert recovery_result["strategy"] == "increased_timeout"
            assert "error" in recovery_result
    
    @pytest.mark.asyncio
    async def test_api_failure_recovery_failure(self, sample_config):
        """Test API failure recovery when it fails."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            context = {"attempt": 1}
            error = Exception("API failure")
            
            # Mock asyncio.sleep to raise an exception
            with patch('asyncio.sleep', new_callable=AsyncMock, side_effect=Exception("Sleep failed")):
                recovery_result = await orchestrator._recover_from_api_failure(error, context)
                
                assert recovery_result["recovery_successful"] is False
                assert recovery_result["strategy"] == "exponential_backoff"
                assert "error" in recovery_result


class TestCircuitBreakerIntegration:
    """Test suite for circuit breaker integration in orchestrator."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample system configuration for testing."""
        return SystemConfig(
            agent_a=AgentConfig(name="TestAgent1", system_prompt="Test prompt"),
            agent_b=AgentConfig(name="TestAgent2", system_prompt="Test prompt"),
            model=ModelConfig(model_name="gpt-3.5-turbo"),
            conversation=ConversationConfig(max_turns=2, turn_timeout=1.0),
            logging=LoggingConfig()
        )
    
    def test_circuit_breaker_configuration(self, sample_config):
        """Test that circuit breakers are configured correctly."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            # Use a unique orchestrator ID to avoid circuit breaker reuse
            orchestrator = ConversationOrchestrator(
                orchestrator_id="test_circuit_config_unique",
                config=sample_config
            )
            
            # Check circuit breaker configuration
            agent_a_breaker = orchestrator.circuit_breakers["agent_a"]
            assert agent_a_breaker.config.failure_threshold == 3
            assert agent_a_breaker.config.recovery_timeout == 30.0
            assert agent_a_breaker.config.success_threshold == 2
            # The timeout should match the conversation turn_timeout
            assert agent_a_breaker.config.timeout == sample_config.conversation.turn_timeout
    
    def test_circuit_breaker_names(self, sample_config):
        """Test that circuit breakers have correct names."""
        with patch('src.agentic_conversation.langchain_agent.ChatOpenAI') as mock_openai:
            mock_llm = Mock()
            mock_openai.return_value = mock_llm
            
            orchestrator = ConversationOrchestrator(config=sample_config)
            
            # Check circuit breaker names include orchestrator ID
            agent_a_breaker = orchestrator.circuit_breakers["agent_a"]
            assert orchestrator.orchestrator_id in agent_a_breaker.name
            assert "agent_a" in agent_a_breaker.name


if __name__ == "__main__":
    pytest.main([__file__])