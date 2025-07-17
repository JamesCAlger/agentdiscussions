"""
Integration tests for the sample configuration file.

This module tests that the provided sample configuration file can be loaded
and validated successfully, ensuring it serves as a working example for users.
"""

import os
import pytest
from pathlib import Path

from src.agentic_conversation.config import ConfigurationLoader, load_config, validate_config
from src.agentic_conversation.models import SystemConfig


class TestSampleConfiguration:
    """Test cases for the sample configuration file."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_path = Path("config.yaml")
        self.loader = ConfigurationLoader()
    
    def test_sample_config_exists(self):
        """Test that the sample configuration file exists."""
        assert self.config_path.exists(), "Sample configuration file config.yaml should exist"
        assert self.config_path.is_file(), "config.yaml should be a file"
    
    def test_sample_config_loads_successfully(self):
        """Test that the sample configuration loads without errors."""
        config = self.loader.load_from_file(self.config_path)
        assert isinstance(config, SystemConfig)
    
    def test_sample_config_validation(self):
        """Test that the sample configuration passes all validation checks."""
        validation_result = validate_config(self.config_path)
        
        assert validation_result["is_valid"], f"Sample configuration should be valid. Errors: {validation_result['errors']}"
        assert len(validation_result["errors"]) == 0, f"Sample configuration should have no validation errors: {validation_result['errors']}"
    
    def test_sample_config_agent_configuration(self):
        """Test that agent configurations in the sample are properly structured."""
        config = load_config(self.config_path)
        
        # Test Agent A
        assert config.agent_a.name == "Researcher"
        assert len(config.agent_a.system_prompt.strip()) > 100, "Agent A should have a substantial system prompt"
        assert "Dr. Sarah Chen" in config.agent_a.system_prompt
        assert "research" in config.agent_a.system_prompt.lower()
        
        # Test Agent B
        assert config.agent_b.name == "Strategist"
        assert len(config.agent_b.system_prompt.strip()) > 100, "Agent B should have a substantial system prompt"
        assert "Marcus Rodriguez" in config.agent_b.system_prompt
        assert "strategic" in config.agent_b.system_prompt.lower()
        
        # Ensure agents have different names
        assert config.agent_a.name != config.agent_b.name
    
    def test_sample_config_model_configuration(self):
        """Test that model configuration in the sample is reasonable."""
        config = load_config(self.config_path)
        
        # Test model settings
        assert config.model.model_name in ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"], f"Model name should be a common model: {config.model.model_name}"
        assert 0.0 <= config.model.temperature <= 2.0, f"Temperature should be in valid range: {config.model.temperature}"
        assert config.model.max_tokens > 0, f"Max tokens should be positive: {config.model.max_tokens}"
        assert 0.0 <= config.model.top_p <= 1.0, f"Top-p should be in valid range: {config.model.top_p}"
        assert -2.0 <= config.model.frequency_penalty <= 2.0, f"Frequency penalty should be in valid range: {config.model.frequency_penalty}"
        assert -2.0 <= config.model.presence_penalty <= 2.0, f"Presence penalty should be in valid range: {config.model.presence_penalty}"
    
    def test_sample_config_conversation_configuration(self):
        """Test that conversation configuration in the sample is reasonable."""
        config = load_config(self.config_path)
        
        # Test conversation settings
        assert config.conversation.max_turns > 0, f"Max turns should be positive: {config.conversation.max_turns}"
        assert config.conversation.max_turns <= 100, f"Max turns should be reasonable: {config.conversation.max_turns}"
        assert config.conversation.initial_prompt is not None, "Sample should include an initial prompt"
        assert len(config.conversation.initial_prompt.strip()) > 50, "Initial prompt should be substantial"
        assert config.conversation.context_window_strategy in ["truncate", "summarize", "sliding"]
        assert config.conversation.context_window_size > 1000, f"Context window should be reasonable: {config.conversation.context_window_size}"
        assert config.conversation.turn_timeout > 0, f"Turn timeout should be positive: {config.conversation.turn_timeout}"
    
    def test_sample_config_logging_configuration(self):
        """Test that logging configuration in the sample is reasonable."""
        config = load_config(self.config_path)
        
        # Test logging settings
        assert config.logging.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert config.logging.output_directory is not None
        assert len(config.logging.output_directory.strip()) > 0
        assert isinstance(config.logging.real_time_display, bool)
        assert len(config.logging.export_formats) > 0
        for format_type in config.logging.export_formats:
            assert format_type in ["json", "csv", "txt"], f"Invalid export format: {format_type}"
        assert isinstance(config.logging.save_conversation_history, bool)
        assert isinstance(config.logging.save_telemetry, bool)
    
    def test_sample_config_environment_variable_support(self):
        """Test that the sample configuration properly uses environment variables."""
        # Read the raw YAML content to check for environment variable patterns
        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check that environment variables are used with defaults
        assert "${MODEL_NAME:gpt-4}" in content, "Sample should use MODEL_NAME environment variable with default"
        assert "${LOG_LEVEL:INFO}" in content, "Sample should use LOG_LEVEL environment variable with default"
        assert "${LOG_DIR:./logs}" in content, "Sample should use LOG_DIR environment variable with default"
    
    def test_sample_config_with_environment_variables(self):
        """Test loading sample configuration with environment variables set."""
        # Test with custom environment variables
        test_env = {
            "MODEL_NAME": "gpt-3.5-turbo",
            "LOG_LEVEL": "DEBUG",
            "LOG_DIR": "./custom_logs"
        }
        
        with pytest.MonkeyPatch().context() as m:
            for key, value in test_env.items():
                m.setenv(key, value)
            
            config = load_config(self.config_path)
            
            assert config.model.model_name == "gpt-3.5-turbo"
            assert config.logging.log_level == "DEBUG"
            assert config.logging.output_directory == "./custom_logs"
    
    def test_sample_config_comprehensive_validation(self):
        """Test comprehensive validation of the sample configuration."""
        config = load_config(self.config_path)
        validation_summary = config.get_validation_summary()
        
        assert validation_summary["is_valid"], f"Sample configuration should pass comprehensive validation: {validation_summary['errors']}"
        assert validation_summary["error_count"] == 0, f"Sample should have no validation errors: {validation_summary['errors']}"
        
        # Check that all sections are valid
        sections = validation_summary["sections"]
        assert sections["agent_a"], "Agent A configuration should be valid"
        assert sections["agent_b"], "Agent B configuration should be valid"
        assert sections["model"], "Model configuration should be valid"
        assert sections["conversation"], "Conversation configuration should be valid"
        assert sections["logging"], "Logging configuration should be valid"
    
    def test_sample_config_realistic_prompts(self):
        """Test that the sample configuration contains realistic, useful prompts."""
        config = load_config(self.config_path)
        
        # Check that system prompts are detailed and realistic
        agent_a_prompt = config.agent_a.system_prompt.lower()
        agent_b_prompt = config.agent_b.system_prompt.lower()
        
        # Agent A should be research-focused
        research_keywords = ["research", "question", "evidence", "data", "investigate", "analysis"]
        assert any(keyword in agent_a_prompt for keyword in research_keywords), "Agent A should have research-focused keywords"
        
        # Agent B should be strategy-focused
        strategy_keywords = ["strategic", "practical", "implementation", "business", "solution", "application"]
        assert any(keyword in agent_b_prompt for keyword in strategy_keywords), "Agent B should have strategy-focused keywords"
        
        # Initial prompt should be substantive and engaging
        initial_prompt = config.conversation.initial_prompt.lower()
        assert "artificial intelligence" in initial_prompt or "ai" in initial_prompt, "Initial prompt should mention AI"
        assert "education" in initial_prompt, "Initial prompt should mention education"
        assert len(initial_prompt.split()) > 20, "Initial prompt should be substantial (>20 words)"
    
    def test_sample_config_comments_and_documentation(self):
        """Test that the sample configuration file contains helpful comments."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for presence of explanatory comments
        assert "# Agent Configuration" in content, "Should have section headers"
        assert "# Language Model Configuration" in content, "Should have section headers"
        assert "# Conversation Behavior Configuration" in content, "Should have section headers"
        assert "# Logging and Telemetry Configuration" in content, "Should have section headers"
        
        # Check for parameter explanations
        assert "# Temperature controls randomness" in content, "Should explain temperature parameter"
        assert "# Maximum number of tokens" in content, "Should explain max_tokens parameter"
        assert "# Strategy for managing context window" in content, "Should explain context window strategy"
        
        # Check for environment variable documentation
        assert "# Environment variables can be substituted" in content, "Should document environment variable usage"
        assert "# Example of environment variable usage" in content, "Should provide environment variable examples"


if __name__ == "__main__":
    pytest.main([__file__])