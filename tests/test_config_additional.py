"""
Additional unit tests for configuration management to improve coverage.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.agentic_conversation.config import ConfigurationLoader
from src.agentic_conversation.models import SystemConfig


class TestConfigurationLoaderAdditional:
    """Additional tests for ConfigurationLoader to improve coverage."""
    
    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        loader = ConfigurationLoader()
        
        from src.agentic_conversation.config import ConfigurationError
        with pytest.raises(ConfigurationError):
            loader.load_from_file("nonexistent_config.yaml")
    
    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML syntax."""
        loader = ConfigurationLoader()
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with pytest.raises(Exception):  # YAML parsing error
                loader.load_from_file("invalid.yaml")
    
    def test_load_config_missing_required_sections(self):
        """Test loading configuration with missing required sections."""
        loader = ConfigurationLoader()
        incomplete_config = """
        agents:
          agent_a:
            name: "Agent A"
            system_prompt: "Test prompt"
        # Missing agent_b, model, conversation sections
        """
        
        with patch("builtins.open", mock_open(read_data=incomplete_config)):
            with pytest.raises(Exception):  # Configuration error
                loader.load_from_file("incomplete.yaml")
    
    def test_load_from_dict_success(self):
        """Test loading configuration from dictionary."""
        loader = ConfigurationLoader()
        
        config_dict = {
            "agents": {
                "agent_a": {"name": "Agent A", "system_prompt": "Prompt A"},
                "agent_b": {"name": "Agent B", "system_prompt": "Prompt B"}
            },
            "model": {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 2000},
            "conversation": {"max_turns": 10, "context_window_strategy": "sliding"},
            "logging": {"log_level": "INFO", "output_directory": "./logs"}
        }
        
        config = loader.load_from_dict(config_dict)
        
        assert config.agent_a.name == "Agent A"
        assert config.agent_b.name == "Agent B"
        assert config.model.model_name == "gpt-4"
        assert config.conversation.max_turns == 10
        assert config.logging.log_level == "INFO"
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution in configuration."""
        loader = ConfigurationLoader()
        
        config_with_env = """
        agents:
          agent_a:
            name: "Agent A"
            system_prompt: "${TEST_PROMPT}"
          agent_b:
            name: "Agent B"
            system_prompt: "Standard prompt"
        model:
          model_name: "gpt-4"
          temperature: 0.7
          max_tokens: 2000
        conversation:
          max_turns: 10
          context_window_strategy: "sliding"
        logging:
          log_level: "INFO"
          output_directory: "${TEST_LOG_DIR}"
        """
        
        with patch.dict(os.environ, {"TEST_PROMPT": "Test system prompt", "TEST_LOG_DIR": "/tmp/logs"}):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=config_with_env)):
                    config = loader.load_from_file("test.yaml")
                    
                    assert config.agent_a.system_prompt == "Test system prompt"
                    assert config.logging.output_directory == "/tmp/logs"
    
    def test_create_system_config_from_dict(self):
        """Test creating SystemConfig from dictionary."""
        loader = ConfigurationLoader()
        
        config_dict = {
            "agents": {
                "agent_a": {"name": "Agent A", "system_prompt": "Prompt A"},
                "agent_b": {"name": "Agent B", "system_prompt": "Prompt B"}
            },
            "model": {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 2000},
            "conversation": {"max_turns": 10, "context_window_strategy": "sliding"},
            "logging": {"log_level": "INFO", "output_directory": "./logs"}
        }
        
        system_config = loader._create_system_config(config_dict)
        
        assert isinstance(system_config, SystemConfig)
        assert system_config.agent_a.name == "Agent A"
        assert system_config.agent_b.name == "Agent B"
        assert system_config.model.model_name == "gpt-4"
        assert system_config.conversation.max_turns == 10
        assert system_config.logging.log_level == "INFO"
    
    def test_load_config_with_optional_fields(self):
        """Test loading configuration with optional fields."""
        loader = ConfigurationLoader()
        
        config_with_optionals = """
        agents:
          agent_a:
            name: "Agent A"
            system_prompt: "Prompt A"
            temperature: 0.8
            max_tokens: 1500
          agent_b:
            name: "Agent B"
            system_prompt: "Prompt B"
        model:
          model_name: "gpt-4"
          temperature: 0.7
          max_tokens: 2000
          top_p: 0.9
          frequency_penalty: 0.1
        conversation:
          max_turns: 10
          context_window_strategy: "sliding"
          initial_prompt: "Let's start the conversation"
          turn_timeout: 30.0
        logging:
          log_level: "DEBUG"
          output_directory: "./logs"
          real_time_display: true
          export_formats: ["json", "csv"]
        """
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=config_with_optionals)):
                config = loader.load_from_file("test.yaml")
            
            # Check optional agent fields
            assert hasattr(config.agent_a, 'temperature')
            assert hasattr(config.agent_a, 'max_tokens')
            
            # Check optional model fields
            assert hasattr(config.model, 'top_p')
            assert hasattr(config.model, 'frequency_penalty')
            
            # Check optional conversation fields
            assert hasattr(config.conversation, 'initial_prompt')
            assert hasattr(config.conversation, 'turn_timeout')
            
            # Check optional logging fields
            assert hasattr(config.logging, 'real_time_display')
            assert hasattr(config.logging, 'export_formats')