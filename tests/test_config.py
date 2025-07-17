"""
Unit tests for the configuration management system.
"""

import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from src.agentic_conversation.config import (
    ConfigurationLoader, 
    ConfigurationError, 
    load_config, 
    validate_config
)
from src.agentic_conversation.models import (
    SystemConfig, 
    AgentConfig, 
    ModelConfig, 
    ConversationConfig, 
    LoggingConfig
)


class TestConfigurationLoader:
    """Test cases for the ConfigurationLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ConfigurationLoader()
        self.valid_config = {
            "agents": {
                "agent_a": {
                    "name": "Researcher",
                    "system_prompt": "You are a thorough researcher who asks probing questions."
                },
                "agent_b": {
                    "name": "Analyst",
                    "system_prompt": "You are an analytical thinker who synthesizes information."
                }
            },
            "model": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "conversation": {
                "max_turns": 20,
                "initial_prompt": "Discuss the implications of artificial intelligence in education",
                "context_window_strategy": "sliding",
                "context_window_size": 8000,
                "turn_timeout": 30.0
            },
            "logging": {
                "log_level": "INFO",
                "output_directory": "./logs",
                "real_time_display": True,
                "export_formats": ["json", "csv"],
                "save_conversation_history": True,
                "save_telemetry": True
            }
        }
    
    def test_load_from_dict_valid_config(self):
        """Test loading a valid configuration from dictionary."""
        config = self.loader.load_from_dict(self.valid_config)
        
        assert isinstance(config, SystemConfig)
        assert config.agent_a.name == "Researcher"
        assert config.agent_b.name == "Analyst"
        assert config.model.model_name == "gpt-4"
        assert config.conversation.max_turns == 20
        assert config.logging.log_level == "INFO"
    
    def test_load_from_dict_minimal_config(self):
        """Test loading a minimal valid configuration."""
        minimal_config = {
            "agents": {
                "agent_a": {
                    "name": "Agent A",
                    "system_prompt": "You are agent A."
                },
                "agent_b": {
                    "name": "Agent B",
                    "system_prompt": "You are agent B."
                }
            },
            "model": {
                "model_name": "gpt-3.5-turbo"
            },
            "conversation": {
                "max_turns": 10
            }
        }
        
        config = self.loader.load_from_dict(minimal_config)
        
        assert isinstance(config, SystemConfig)
        assert config.model.temperature == 0.7  # Default value
        assert config.conversation.context_window_strategy == "sliding"  # Default value
        assert config.logging.log_level == "INFO"  # Default value
    
    def test_load_from_dict_invalid_type(self):
        """Test loading from non-dictionary raises error."""
        with pytest.raises(ConfigurationError, match="Configuration must be a dictionary"):
            self.loader.load_from_dict("not a dict")
    
    def test_load_from_dict_missing_required_sections(self):
        """Test loading configuration with missing required sections."""
        incomplete_config = {
            "agents": {
                "agent_a": {"name": "A", "system_prompt": "Prompt A"}
            }
            # Missing agent_b, model, and conversation sections
        }
        
        with pytest.raises(ConfigurationError, match="Missing required configuration sections"):
            self.loader.load_from_dict(incomplete_config)
    
    def test_load_from_dict_missing_agent_fields(self):
        """Test loading configuration with missing agent fields."""
        config_missing_name = {
            "agents": {
                "agent_a": {
                    "system_prompt": "You are agent A."
                    # Missing name
                },
                "agent_b": {
                    "name": "Agent B",
                    "system_prompt": "You are agent B."
                }
            },
            "model": {"model_name": "gpt-4"},
            "conversation": {"max_turns": 10}
        }
        
        with pytest.raises(ConfigurationError, match="Missing required field 'name'"):
            self.loader.load_from_dict(config_missing_name)
    
    def test_load_from_dict_invalid_model_config(self):
        """Test loading configuration with invalid model parameters."""
        invalid_model_config = self.valid_config.copy()
        invalid_model_config["model"]["temperature"] = 3.0  # Invalid temperature
        
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            self.loader.load_from_dict(invalid_model_config)
    
    def test_load_from_dict_invalid_conversation_config(self):
        """Test loading configuration with invalid conversation parameters."""
        invalid_conversation_config = self.valid_config.copy()
        invalid_conversation_config["conversation"]["max_turns"] = -1  # Invalid max_turns
        
        with pytest.raises(ConfigurationError, match="Configuration validation failed"):
            self.loader.load_from_dict(invalid_conversation_config)
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution in configuration."""
        config_with_env_vars = {
            "agents": {
                "agent_a": {
                    "name": "${AGENT_A_NAME:Default Agent A}",
                    "system_prompt": "You are ${AGENT_A_NAME:agent A}."
                },
                "agent_b": {
                    "name": "${AGENT_B_NAME}",
                    "system_prompt": "You are ${AGENT_B_NAME}."
                }
            },
            "model": {
                "model_name": "${MODEL_NAME:gpt-4}",
                "temperature": 0.7
            },
            "conversation": {
                "max_turns": 10
            }
        }
        
        with patch.dict(os.environ, {"AGENT_B_NAME": "Custom Agent B", "MODEL_NAME": "gpt-3.5-turbo"}):
            config = self.loader.load_from_dict(config_with_env_vars)
            
            assert config.agent_a.name == "Default Agent A"  # Used default value
            assert config.agent_b.name == "Custom Agent B"  # Used env var
            assert config.model.model_name == "gpt-3.5-turbo"  # Used env var
    
    def test_environment_variable_substitution_no_default(self):
        """Test environment variable substitution without default values."""
        config_with_env_vars = {
            "agents": {
                "agent_a": {
                    "name": "${UNDEFINED_VAR}",
                    "system_prompt": "You are an agent."
                },
                "agent_b": {
                    "name": "Agent B",
                    "system_prompt": "You are agent B."
                }
            },
            "model": {"model_name": "gpt-4"},
            "conversation": {"max_turns": 10}
        }
        
        # Should leave placeholder as-is when env var is not set and no default
        config = self.loader.load_from_dict(config_with_env_vars)
        assert config.agent_a.name == "${UNDEFINED_VAR}"
    
    def test_load_from_file_valid(self):
        """Test loading configuration from a valid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            temp_path = f.name
        
        try:
            config = self.loader.load_from_file(temp_path)
            assert isinstance(config, SystemConfig)
            assert config.agent_a.name == "Researcher"
        finally:
            os.unlink(temp_path)
    
    def test_load_from_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            self.loader.load_from_file("nonexistent.yaml")
    
    def test_load_from_file_invalid_yaml(self):
        """Test loading from file with invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Failed to parse YAML configuration"):
                self.loader.load_from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_from_file_empty(self):
        """Test loading from empty file raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Configuration file is empty"):
                self.loader.load_from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_from_file_not_dict(self):
        """Test loading from file that doesn't contain a dictionary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(["not", "a", "dict"], f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Configuration must be a YAML object/dictionary"):
                self.loader.load_from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_file_valid(self):
        """Test validating a valid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            temp_path = f.name
        
        try:
            result = self.loader.validate_config_file(temp_path)
            assert result["is_valid"] is True
            assert len(result["errors"]) == 0
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_file_invalid(self):
        """Test validating an invalid configuration file."""
        invalid_config = {
            "agents": {
                "agent_a": {"name": "", "system_prompt": ""},  # Invalid empty values
                "agent_b": {"name": "B", "system_prompt": "B prompt"}
            },
            "model": {"model_name": "gpt-4", "temperature": 3.0},  # Invalid temperature
            "conversation": {"max_turns": -1}  # Invalid max_turns
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            result = self.loader.validate_config_file(temp_path)
            assert result["is_valid"] is False
            assert len(result["errors"]) > 0
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_file_not_found(self):
        """Test validating non-existent file returns error."""
        result = self.loader.validate_config_file("nonexistent.yaml")
        assert result["is_valid"] is False
        assert len(result["errors"]) == 1
        assert "Configuration file not found" in result["errors"][0]


class TestEnvironmentVariableSubstitution:
    """Test cases for environment variable substitution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ConfigurationLoader()
    
    def test_substitute_simple_env_var(self):
        """Test substituting a simple environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = self.loader._substitute_env_vars_in_string("Hello ${TEST_VAR}")
            assert result == "Hello test_value"
    
    def test_substitute_env_var_with_default(self):
        """Test substituting environment variable with default value."""
        result = self.loader._substitute_env_vars_in_string("Hello ${UNDEFINED_VAR:default_value}")
        assert result == "Hello default_value"
    
    def test_substitute_env_var_with_empty_default(self):
        """Test substituting environment variable with empty default value."""
        result = self.loader._substitute_env_vars_in_string("Hello ${UNDEFINED_VAR:}")
        assert result == "Hello "
    
    def test_substitute_multiple_env_vars(self):
        """Test substituting multiple environment variables."""
        with patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"}):
            result = self.loader._substitute_env_vars_in_string("${VAR1} and ${VAR2}")
            assert result == "value1 and value2"
    
    def test_substitute_nested_structures(self):
        """Test substituting environment variables in nested data structures."""
        data = {
            "string": "${TEST_VAR:default}",
            "nested": {
                "list": ["${TEST_VAR:item1}", "${TEST_VAR:item2}"],
                "dict": {"key": "${TEST_VAR:nested_value}"}
            }
        }
        
        with patch.dict(os.environ, {"TEST_VAR": "actual_value"}):
            result = self.loader._substitute_environment_variables(data)
            
            assert result["string"] == "actual_value"
            assert result["nested"]["list"] == ["actual_value", "actual_value"]
            assert result["nested"]["dict"]["key"] == "actual_value"
    
    def test_substitute_non_string_values(self):
        """Test that non-string values are not affected by substitution."""
        data = {
            "number": 42,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3]
        }
        
        result = self.loader._substitute_environment_variables(data)
        assert result == data


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_config = {
            "agents": {
                "agent_a": {"name": "A", "system_prompt": "Prompt A"},
                "agent_b": {"name": "B", "system_prompt": "Prompt B"}
            },
            "model": {"model_name": "gpt-4"},
            "conversation": {"max_turns": 10}
        }
    
    def test_load_config_function(self):
        """Test the load_config convenience function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert isinstance(config, SystemConfig)
            assert config.agent_a.name == "A"
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_function(self):
        """Test the validate_config convenience function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            temp_path = f.name
        
        try:
            result = validate_config(temp_path)
            assert result["is_valid"] is True
            assert len(result["errors"]) == 0
        finally:
            os.unlink(temp_path)


class TestConfigurationValidation:
    """Test cases for configuration validation scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ConfigurationLoader()
    
    def test_duplicate_agent_names(self):
        """Test that duplicate agent names are caught during validation."""
        config_with_duplicate_names = {
            "agents": {
                "agent_a": {"name": "SameName", "system_prompt": "Prompt A"},
                "agent_b": {"name": "SameName", "system_prompt": "Prompt B"}
            },
            "model": {"model_name": "gpt-4"},
            "conversation": {"max_turns": 10}
        }
        
        with pytest.raises(ConfigurationError, match="Agent A and Agent B must have different names"):
            self.loader.load_from_dict(config_with_duplicate_names)
    
    def test_invalid_context_window_strategy(self):
        """Test that invalid context window strategy is caught."""
        config_with_invalid_strategy = {
            "agents": {
                "agent_a": {"name": "A", "system_prompt": "Prompt A"},
                "agent_b": {"name": "B", "system_prompt": "Prompt B"}
            },
            "model": {"model_name": "gpt-4"},
            "conversation": {
                "max_turns": 10,
                "context_window_strategy": "invalid_strategy"
            }
        }
        
        with pytest.raises(ConfigurationError, match="Context window strategy must be one of"):
            self.loader.load_from_dict(config_with_invalid_strategy)
    
    def test_invalid_log_level(self):
        """Test that invalid log level is caught."""
        config_with_invalid_log_level = {
            "agents": {
                "agent_a": {"name": "A", "system_prompt": "Prompt A"},
                "agent_b": {"name": "B", "system_prompt": "Prompt B"}
            },
            "model": {"model_name": "gpt-4"},
            "conversation": {"max_turns": 10},
            "logging": {"log_level": "INVALID_LEVEL"}
        }
        
        with pytest.raises(ConfigurationError, match="Log level must be one of"):
            self.loader.load_from_dict(config_with_invalid_log_level)
    
    def test_invalid_export_formats(self):
        """Test that invalid export formats are caught."""
        config_with_invalid_formats = {
            "agents": {
                "agent_a": {"name": "A", "system_prompt": "Prompt A"},
                "agent_b": {"name": "B", "system_prompt": "Prompt B"}
            },
            "model": {"model_name": "gpt-4"},
            "conversation": {"max_turns": 10},
            "logging": {"export_formats": ["json", "invalid_format"]}
        }
        
        with pytest.raises(ConfigurationError, match="Export format 'invalid_format' is not valid"):
            self.loader.load_from_dict(config_with_invalid_formats)


if __name__ == "__main__":
    pytest.main([__file__])