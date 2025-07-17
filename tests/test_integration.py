"""
Integration tests for the agentic conversation system.

These tests verify that all components work together correctly
and test complete conversation flows.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import pytest

from src.agentic_conversation.orchestrator import ConversationOrchestrator
from src.agentic_conversation.config import ConfigurationLoader
from src.agentic_conversation.models import SystemConfig, ConversationStatus


class TestIntegrationScenarios:
    """Integration tests for complete conversation scenarios."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for test configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_config_dict(self):
        """Create a sample configuration dictionary."""
        return {
            "agents": {
                "agent_a": {
                    "name": "Researcher",
                    "system_prompt": "You are a thorough researcher who asks probing questions and seeks detailed information."
                },
                "agent_b": {
                    "name": "Analyst", 
                    "system_prompt": "You are an analytical thinker who synthesizes information and provides insights."
                }
            },
            "model": {
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "conversation": {
                "max_turns": 6,
                "context_window_strategy": "sliding",
                "initial_prompt": "Discuss the benefits and challenges of remote work.",
                "turn_timeout": 30.0
            },
            "logging": {
                "log_level": "INFO",
                "output_directory": "./test_logs",
                "real_time_display": False,
                "export_formats": ["json"]
            }
        }
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Create mock LLM responses for testing."""
        return [
            "Remote work offers flexibility and work-life balance. What specific challenges have organizations faced?",
            "Organizations struggle with communication, collaboration, and maintaining company culture. How do teams stay connected?",
            "Teams use video calls, chat tools, and project management software. What about productivity concerns?",
            "Productivity can vary - some thrive while others struggle with distractions. What metrics help measure success?",
            "Key metrics include task completion rates, quality of work, and employee satisfaction surveys.",
            "These metrics provide valuable insights for optimizing remote work policies and support systems."
        ]
    
    def test_configuration_loading_and_validation(self, sample_config_dict):
        """Test that configuration loading and validation works correctly."""
        loader = ConfigurationLoader()
        
        # Test loading from dictionary
        config = loader.load_from_dict(sample_config_dict)
        
        assert isinstance(config, SystemConfig)
        assert config.agent_a.name == "Researcher"
        assert config.agent_b.name == "Analyst"
        assert config.model.model_name == "gpt-3.5-turbo"
        assert config.conversation.max_turns == 6
        assert config.logging.log_level == "INFO"
    
    def test_configuration_file_loading(self, temp_config_dir, sample_config_dict):
        """Test loading configuration from a YAML file."""
        import yaml
        
        # Create a test config file
        config_file = temp_config_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config_dict, f)
        
        loader = ConfigurationLoader()
        config = loader.load_from_file(config_file)
        
        assert isinstance(config, SystemConfig)
        assert config.agent_a.name == "Researcher"
        assert config.conversation.max_turns == 6
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, sample_config_dict):
        """Test that the orchestrator initializes correctly with configuration."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(sample_config_dict)
        
        # Mock the API key for testing
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config=config)
            
            assert orchestrator.config == config
            assert orchestrator.agent_a is not None
            assert orchestrator.agent_b is not None
            assert orchestrator.conversation_graph is not None
            assert orchestrator.telemetry_logger is not None
    
    @pytest.mark.asyncio
    async def test_mock_conversation_flow(self, sample_config_dict, mock_llm_responses):
        """Test a complete conversation flow with mocked LLM responses."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(sample_config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config=config)
            
            # Mock the LLM calls to return predefined responses
            response_iter = iter(mock_llm_responses)
            
            def mock_llm_call(*args, **kwargs):
                try:
                    return next(response_iter)
                except StopIteration:
                    return "I think we've covered this topic thoroughly."
            
            # Patch the LLM calls in both agents
            with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_llm_call):
                with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_llm_call):
                    
                    # Run the conversation
                    result = await orchestrator.run_conversation("test-conversation")
                    
                    # Verify the conversation completed successfully
                    assert result is not None
                    assert result.status in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
                    assert len(result.messages) > 0
                    assert result.current_turn > 0
    
    @pytest.mark.asyncio
    async def test_conversation_with_different_configurations(self, mock_llm_responses):
        """Test conversations with different configuration scenarios."""
        base_config = {
            "agents": {
                "agent_a": {"name": "Agent A", "system_prompt": "You are Agent A."},
                "agent_b": {"name": "Agent B", "system_prompt": "You are Agent B."}
            },
            "model": {"model_name": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 1000},
            "logging": {"log_level": "INFO", "output_directory": "./test_logs"}
        }
        
        # Test different conversation configurations
        test_configs = [
            {"max_turns": 2, "context_window_strategy": "truncate"},
            {"max_turns": 4, "context_window_strategy": "sliding"},
            {"max_turns": 6, "context_window_strategy": "sliding", "turn_timeout": 60.0}
        ]
        
        for conv_config in test_configs:
            config_dict = base_config.copy()
            config_dict["conversation"] = conv_config
            
            loader = ConfigurationLoader()
            config = loader.load_from_dict(config_dict)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                orchestrator = ConversationOrchestrator(config=config)
                
                # Mock LLM responses
                response_iter = iter(mock_llm_responses)
                def mock_llm_call(*args, **kwargs):
                    try:
                        return next(response_iter)
                    except StopIteration:
                        return "Conversation complete."
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_llm_call):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_llm_call):
                        
                        result = await orchestrator.run_conversation(f"test-{conv_config['max_turns']}")
                        
                        # Verify conversation respects max_turns
                        assert result.current_turn <= conv_config['max_turns']
                        assert result.status in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, sample_config_dict):
        """Test error recovery and graceful degradation."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(sample_config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config=config)
            
            # Test scenario 1: Agent A fails, Agent B continues
            def mock_agent_a_fail(*args, **kwargs):
                raise Exception("Agent A API Error")
            
            def mock_agent_b_success(*args, **kwargs):
                return "I'll continue the conversation despite the error."
            
            with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_agent_a_fail):
                with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_agent_b_success):
                    
                    result = await orchestrator.run_conversation("error-recovery-test")
                    
                    # Conversation should handle the error gracefully
                    assert result is not None
                    assert result.status in [ConversationStatus.ERROR, ConversationStatus.COMPLETED]
    
    @pytest.mark.asyncio
    async def test_context_window_management_integration(self, sample_config_dict):
        """Test context window management under different scenarios."""
        # Create config with small context window to trigger management
        config_dict = sample_config_dict.copy()
        config_dict["conversation"]["context_window_strategy"] = "sliding"
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config=config)
            
            # Mock context manager to simulate context window pressure
            original_manage_context = orchestrator.context_manager.manage_conversation_context
            
            def mock_context_management(state):
                # Simulate context management by truncating messages
                if len(state.messages) > 3:
                    state.messages = state.messages[-3:]  # Keep only last 3 messages
                return original_manage_context(state)
            
            with patch.object(orchestrator.context_manager, 'manage_conversation_context', 
                            side_effect=mock_context_management):
                
                # Generate long responses to fill context
                long_responses = [
                    "This is a very long response that contains a lot of detailed information about the topic. " * 10,
                    "Another lengthy response with extensive details and comprehensive analysis of the subject matter. " * 10,
                    "Yet another detailed response that provides thorough coverage of all aspects of the discussion. " * 10,
                    "A final comprehensive response that summarizes all the key points and insights. " * 10
                ]
                
                response_iter = iter(long_responses)
                def mock_llm_call(*args, **kwargs):
                    try:
                        return next(response_iter)
                    except StopIteration:
                        return "Summary complete."
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_llm_call):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_llm_call):
                        
                        result = await orchestrator.run_conversation("context-management-test")
                        
                        # Verify conversation completed despite context management
                        assert result is not None
                        assert result.status in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
    
    def test_telemetry_integration(self, sample_config_dict, temp_config_dir):
        """Test that telemetry logging works correctly in integration."""
        # Set up logging directory
        log_dir = temp_config_dir / "logs"
        log_dir.mkdir()
        
        config_dict = sample_config_dict.copy()
        config_dict["logging"]["output_directory"] = str(log_dir)
        config_dict["logging"]["export_formats"] = ["json", "csv"]
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config=config)
            
            # Verify telemetry logger is properly configured
            assert orchestrator.telemetry_logger is not None
            assert orchestrator.telemetry_logger.config.output_directory == str(log_dir)
            assert "json" in orchestrator.telemetry_logger.config.export_formats
            assert "csv" in orchestrator.telemetry_logger.config.export_formats
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, sample_config_dict):
        """Test system performance under simulated load."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(sample_config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            # Run multiple conversations concurrently
            async def run_single_conversation(conv_id):
                orchestrator = ConversationOrchestrator(config=config)
                
                # Mock fast responses
                def mock_fast_response(*args, **kwargs):
                    return f"Quick response for conversation {conv_id}"
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_fast_response):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_fast_response):
                        
                        start_time = time.time()
                        result = await orchestrator.run_conversation(f"load-test-{conv_id}")
                        end_time = time.time()
                        
                        return {
                            'conversation_id': conv_id,
                            'duration': end_time - start_time,
                            'status': result.status,
                            'turns': result.current_turn
                        }
            
            # Run 5 conversations concurrently
            tasks = [run_single_conversation(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all conversations completed successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == 5
            
            # Check performance metrics
            durations = [r['duration'] for r in successful_results]
            avg_duration = sum(durations) / len(durations)
            
            # Conversations should complete in reasonable time (adjust threshold as needed)
            assert avg_duration < 10.0  # seconds
            
            # All conversations should have completed successfully
            for result in successful_results:
                assert result['status'] in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
                assert result['turns'] > 0


class TestConfigurationScenarios:
    """Test different configuration scenarios and edge cases."""
    
    def test_minimal_configuration(self):
        """Test system with minimal required configuration."""
        minimal_config = {
            "agents": {
                "agent_a": {"name": "A", "system_prompt": "You are A."},
                "agent_b": {"name": "B", "system_prompt": "You are B."}
            },
            "model": {"model_name": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 500},
            "conversation": {"max_turns": 2, "context_window_strategy": "truncate"},
            "logging": {"log_level": "ERROR", "output_directory": "./logs"}
        }
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(minimal_config)
        
        assert config.agent_a.name == "A"
        assert config.model.temperature == 0.5
        assert config.conversation.max_turns == 2
        assert config.logging.log_level == "ERROR"
    
    def test_maximal_configuration(self):
        """Test system with all optional configuration fields."""
        maximal_config = {
            "agents": {
                "agent_a": {"name": "Advanced Agent A", "system_prompt": "You are an advanced AI agent."},
                "agent_b": {"name": "Advanced Agent B", "system_prompt": "You are another advanced AI agent."}
            },
            "model": {
                "model_name": "gpt-4",
                "temperature": 0.8,
                "max_tokens": 2000,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            },
            "conversation": {
                "max_turns": 20,
                "context_window_strategy": "sliding",
                "initial_prompt": "Let's have an in-depth discussion about AI ethics.",
                "turn_timeout": 120.0
            },
            "logging": {
                "log_level": "DEBUG",
                "output_directory": "./detailed_logs",
                "real_time_display": True,
                "export_formats": ["json", "csv", "txt"],
                "save_conversation_history": True,
                "save_telemetry": True
            }
        }
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(maximal_config)
        
        assert config.model.model_name == "gpt-4"
        assert config.model.top_p == 0.9
        assert config.conversation.initial_prompt is not None
        assert config.conversation.turn_timeout == 120.0
        assert config.logging.real_time_display == True
        assert len(config.logging.export_formats) == 3
    
    def test_environment_variable_substitution_integration(self):
        """Test environment variable substitution in real configuration."""
        config_with_env_vars = {
            "agents": {
                "agent_a": {"name": "${AGENT_A_NAME}", "system_prompt": "${AGENT_A_PROMPT}"},
                "agent_b": {"name": "${AGENT_B_NAME}", "system_prompt": "${AGENT_B_PROMPT}"}
            },
            "model": {"model_name": "${MODEL_NAME}", "temperature": 0.7, "max_tokens": 1000},
            "conversation": {"max_turns": 4, "context_window_strategy": "sliding"},
            "logging": {"log_level": "${LOG_LEVEL}", "output_directory": "${LOG_DIR}"}
        }
        
        env_vars = {
            "AGENT_A_NAME": "Research Assistant",
            "AGENT_A_PROMPT": "You are a helpful research assistant.",
            "AGENT_B_NAME": "Analysis Expert", 
            "AGENT_B_PROMPT": "You are an expert at analyzing information.",
            "MODEL_NAME": "gpt-3.5-turbo",
            "LOG_LEVEL": "INFO",
            "LOG_DIR": "./env_test_logs"
        }
        
        with patch.dict(os.environ, env_vars):
            loader = ConfigurationLoader()
            config = loader.load_from_dict(config_with_env_vars)
            
            assert config.agent_a.name == "Research Assistant"
            assert config.agent_b.name == "Analysis Expert"
            assert config.model.model_name == "gpt-3.5-turbo"
            assert config.logging.log_level == "INFO"
            assert config.logging.output_directory == "./env_test_logs"


class TestErrorScenarios:
    """Test error handling and edge cases in integration scenarios."""
    
    @pytest.fixture
    def sample_config_dict(self):
        """Create a sample configuration dictionary."""
        return {
            "agents": {
                "agent_a": {"name": "Agent A", "system_prompt": "You are Agent A."},
                "agent_b": {"name": "Agent B", "system_prompt": "You are Agent B."}
            },
            "model": {"model_name": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 1000},
            "conversation": {"max_turns": 4, "context_window_strategy": "sliding"},
            "logging": {"log_level": "INFO", "output_directory": "./test_logs"}
        }
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, sample_config_dict):
        """Test handling of network errors during conversation."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(sample_config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config=config)
            
            # Simulate network errors
            def mock_network_error(*args, **kwargs):
                raise ConnectionError("Network connection failed")
            
            with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_network_error):
                with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_network_error):
                    
                    result = await orchestrator.run_conversation("network-error-test")
                    
                    # System should handle network errors gracefully
                    assert result is not None
                    assert result.status == ConversationStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_config_dict):
        """Test handling of request timeouts."""
        # Set short timeout for testing
        config_dict = sample_config_dict.copy()
        config_dict["conversation"]["turn_timeout"] = 0.1  # Very short timeout
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config=config)
            
            # Simulate slow responses
            async def mock_slow_response(*args, **kwargs):
                await asyncio.sleep(0.2)  # Longer than timeout
                return "This response took too long"
            
            with patch.object(orchestrator.agent_a, 'generate_response', side_effect=mock_slow_response):
                with patch.object(orchestrator.agent_b, 'generate_response', side_effect=mock_slow_response):
                    
                    result = await orchestrator.run_conversation("timeout-test")
                    
                    # System should handle timeouts gracefully
                    assert result is not None
                    # Result status depends on implementation - could be ERROR or COMPLETED with fewer turns
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        invalid_configs = [
            # Missing required sections
            {"agents": {"agent_a": {"name": "A", "system_prompt": "Prompt"}}},  # Missing agent_b
            {"model": {"model_name": "gpt-4"}},  # Missing agents
            # Invalid values
            {
                "agents": {
                    "agent_a": {"name": "", "system_prompt": "Prompt"},  # Empty name
                    "agent_b": {"name": "B", "system_prompt": ""}  # Empty prompt
                },
                "model": {"model_name": "gpt-4", "temperature": 2.0},  # Invalid temperature
                "conversation": {"max_turns": -1},  # Invalid max_turns
                "logging": {"log_level": "INVALID"}  # Invalid log level
            }
        ]
        
        loader = ConfigurationLoader()
        
        for invalid_config in invalid_configs:
            with pytest.raises(Exception):  # Should raise configuration error
                loader.load_from_dict(invalid_config)