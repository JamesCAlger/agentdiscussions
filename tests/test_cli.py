"""
Integration tests for the CLI application.

This module tests the command-line interface functionality including argument parsing,
configuration validation, and command execution.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner

from agentic_conversation.main import cli
from agentic_conversation.config import ConfigurationLoader
from agentic_conversation.models import (
    SystemConfig, AgentConfig, ModelConfig, ConversationConfig, LoggingConfig
)


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_config():
    """Create a sample system configuration."""
    return SystemConfig(
        agent_a=AgentConfig(
            name="Researcher",
            system_prompt="You are a thorough researcher who asks probing questions."
        ),
        agent_b=AgentConfig(
            name="Analyst",
            system_prompt="You are an analytical thinker who synthesizes information."
        ),
        model=ModelConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2000
        ),
        conversation=ConversationConfig(
            max_turns=10,
            initial_prompt="Discuss the implications of AI in education",
            context_window_strategy="sliding",
            context_window_size=8000,
            turn_timeout=30.0
        ),
        logging=LoggingConfig(
            log_level="INFO",
            output_directory="./test_logs",
            real_time_display=True,
            export_formats=["json"]
        )
    )


@pytest.fixture
def config_file(sample_config):
    """Create a temporary configuration file."""
    config_dict = sample_config.to_dict()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_dict, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def invalid_config_file():
    """Create a temporary invalid configuration file."""
    invalid_config = {
        "agents": {
            "agent_a": {
                "name": "Test Agent"
                # Missing system_prompt
            }
        }
        # Missing other required sections
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(invalid_config, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def sample_log_data():
    """Create sample conversation log data."""
    return {
        "conversation_id": "test_conv_123",
        "status": "completed",
        "total_turns": 5,
        "total_messages": 10,
        "total_tokens": 1500,
        "duration_seconds": 45.2,
        "agent_info": {
            "agent_a": {"name": "Researcher"},
            "agent_b": {"name": "Analyst"}
        },
        "configuration": {
            "model": {"model_name": "gpt-4", "temperature": 0.7},
            "conversation": {"max_turns": 10, "context_window_strategy": "sliding"}
        },
        "telemetry": {
            "agent_metrics": {
                "agent_a": {
                    "average_response_time": 2.1,
                    "total_tokens": 750,
                    "errors": []
                },
                "agent_b": {
                    "average_response_time": 1.8,
                    "total_tokens": 750,
                    "errors": ["timeout_error"]
                }
            }
        }
    }


@pytest.fixture
def log_directory(sample_log_data):
    """Create a temporary directory with sample log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir)
        
        # Create multiple log files
        for i in range(3):
            log_file = log_dir / f"conversation_{i}.json"
            log_data = sample_log_data.copy()
            log_data["conversation_id"] = f"test_conv_{i}"
            log_data["total_turns"] = 5 + i
            log_data["total_tokens"] = 1500 + (i * 100)
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f)
        
        yield str(log_dir)


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_cli_help(self, runner):
        """Test that CLI help is displayed correctly."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Agentic Conversation System" in result.output
        assert "CLI for managing AI agent conversations" in result.output
    
    def test_cli_with_invalid_log_level(self, runner):
        """Test CLI with invalid log level."""
        result = runner.invoke(cli, ['--log-level', 'INVALID'])
        assert result.exit_code != 0
        assert "Invalid value" in result.output
    
    def test_cli_with_nonexistent_config(self, runner):
        """Test CLI with non-existent configuration file."""
        result = runner.invoke(cli, ['--config', '/nonexistent/config.yaml'])
        assert result.exit_code != 0
        assert "does not exist" in result.output
    
    def test_cli_with_valid_config(self, runner, config_file):
        """Test CLI with valid configuration file."""
        result = runner.invoke(cli, ['--config', config_file, 'info'])
        assert result.exit_code == 0
        assert "Configuration Status: Loaded" in result.output


class TestRunCommand:
    """Test the 'run' command."""
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_run_without_config(self, mock_orchestrator, runner):
        """Test run command without configuration."""
        result = runner.invoke(cli, ['run'])
        assert result.exit_code == 1
        assert "Configuration file is required" in result.output
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_run_with_config(self, mock_orchestrator, runner, config_file):
        """Test run command with valid configuration."""
        # Mock the orchestrator and its methods
        mock_instance = Mock()
        mock_instance.run_conversation = AsyncMock(return_value={
            'conversation_id': 'test_123',
            'status': 'completed',
            'total_turns': 5,
            'total_tokens': 1500,
            'duration_seconds': 30.5
        })
        mock_instance.config.logging.output_directory = './test_logs'
        mock_orchestrator.return_value = mock_instance
        
        result = runner.invoke(cli, ['--config', config_file, 'run'])
        assert result.exit_code == 0
        assert "Starting conversation..." in result.output
        assert "Conversation completed successfully!" in result.output
        assert "test_123" in result.output
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_run_with_custom_conversation_id(self, mock_orchestrator, runner, config_file):
        """Test run command with custom conversation ID."""
        mock_instance = Mock()
        mock_instance.run_conversation = AsyncMock(return_value={
            'conversation_id': 'custom_id',
            'status': 'completed',
            'total_turns': 3,
            'total_tokens': 800,
            'duration_seconds': 15.2
        })
        mock_instance.config.logging.output_directory = './test_logs'
        mock_orchestrator.return_value = mock_instance
        
        result = runner.invoke(cli, [
            '--config', config_file, 
            'run', 
            '--conversation-id', 'custom_id'
        ])
        assert result.exit_code == 0
        assert "custom_id" in result.output
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_run_with_no_display(self, mock_orchestrator, runner, config_file):
        """Test run command with display disabled."""
        mock_instance = Mock()
        mock_instance.run_conversation = AsyncMock(return_value={
            'conversation_id': 'test_123',
            'status': 'completed',
            'total_turns': 5,
            'total_tokens': 1500,
            'duration_seconds': 30.5
        })
        mock_instance.config.logging.output_directory = './test_logs'
        mock_orchestrator.return_value = mock_instance
        
        result = runner.invoke(cli, [
            '--config', config_file, 
            'run', 
            '--no-display'
        ])
        assert result.exit_code == 0
        # Verify that real_time_display was set to False
        assert mock_instance.config.logging.real_time_display == False
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_run_with_no_save(self, mock_orchestrator, runner, config_file):
        """Test run command with save disabled."""
        mock_instance = Mock()
        mock_instance.run_conversation = AsyncMock(return_value={
            'conversation_id': 'test_123',
            'status': 'completed',
            'total_turns': 5,
            'total_tokens': 1500,
            'duration_seconds': 30.5
        })
        mock_orchestrator.return_value = mock_instance
        
        result = runner.invoke(cli, [
            '--config', config_file, 
            'run', 
            '--no-save'
        ])
        assert result.exit_code == 0
        # Verify that save_results was False in the call
        mock_instance.run_conversation.assert_called_once()
        call_args = mock_instance.run_conversation.call_args
        assert call_args.kwargs['save_results'] == False
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_run_with_orchestration_error(self, mock_orchestrator, runner, config_file):
        """Test run command with orchestration error."""
        from agentic_conversation.orchestrator import OrchestrationError
        
        mock_orchestrator.side_effect = OrchestrationError("Test error")
        
        result = runner.invoke(cli, ['--config', config_file, 'run'])
        assert result.exit_code == 1
        assert "Error: Test error" in result.output
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_run_with_keyboard_interrupt(self, mock_orchestrator, runner, config_file):
        """Test run command with keyboard interrupt."""
        mock_instance = Mock()
        mock_instance.run_conversation = AsyncMock(side_effect=KeyboardInterrupt())
        mock_orchestrator.return_value = mock_instance
        
        result = runner.invoke(cli, ['--config', config_file, 'run'])
        assert result.exit_code == 1
        assert "interrupted by user" in result.output


class TestValidateCommand:
    """Test the 'validate' command."""
    
    def test_validate_valid_config(self, runner, config_file):
        """Test validate command with valid configuration."""
        with patch('agentic_conversation.main.ConfigurationLoader') as mock_loader:
            mock_instance = Mock()
            mock_instance.validate_config_file.return_value = {'is_valid': True}
            mock_loader.return_value = mock_instance
            
            result = runner.invoke(cli, ['validate', config_file])
            assert result.exit_code == 0
            assert "is valid" in result.output
    
    def test_validate_invalid_config(self, runner, invalid_config_file):
        """Test validate command with invalid configuration."""
        with patch('agentic_conversation.main.ConfigurationLoader') as mock_loader:
            mock_instance = Mock()
            mock_instance.validate_config_file.return_value = {
                'is_valid': False,
                'errors': ['Missing required field: system_prompt'],
                'warnings': ['Deprecated option used']
            }
            mock_loader.return_value = mock_instance
            
            result = runner.invoke(cli, ['validate', invalid_config_file])
            assert result.exit_code == 1
            assert "is invalid" in result.output
            assert "Missing required field" in result.output
            assert "Deprecated option" in result.output
    
    def test_validate_with_detailed_flag(self, runner, config_file, sample_config):
        """Test validate command with detailed flag."""
        with patch('agentic_conversation.main.ConfigurationLoader') as mock_loader:
            mock_instance = Mock()
            mock_instance.validate_config_file.return_value = {'is_valid': True}
            mock_instance.load_from_file.return_value = sample_config
            mock_loader.return_value = mock_instance
            
            result = runner.invoke(cli, ['validate', config_file, '--detailed'])
            assert result.exit_code == 0
            assert "Configuration Details:" in result.output
            assert "gpt-4" in result.output
            assert "Researcher" in result.output
            assert "Analyst" in result.output
    
    def test_validate_nonexistent_file(self, runner):
        """Test validate command with non-existent file."""
        result = runner.invoke(cli, ['validate', '/nonexistent/config.yaml'])
        assert result.exit_code != 0
        assert "does not exist" in result.output


class TestAnalyzeCommand:
    """Test the 'analyze' command."""
    
    def test_analyze_summary_format(self, runner, log_directory):
        """Test analyze command with summary format."""
        result = runner.invoke(cli, ['analyze', log_directory])
        assert result.exit_code == 0
        assert "CONVERSATION ANALYSIS SUMMARY" in result.output
        assert "Total Conversations: 3" in result.output
        assert "Total Turns:" in result.output
        assert "Total Tokens:" in result.output
    
    def test_analyze_detailed_format(self, runner, log_directory):
        """Test analyze command with detailed format."""
        result = runner.invoke(cli, ['analyze', log_directory, '--format', 'detailed'])
        assert result.exit_code == 0
        assert "CONVERSATION 1:" in result.output
        assert "Status:" in result.output
        assert "Agent Performance:" in result.output
    
    def test_analyze_json_format(self, runner, log_directory):
        """Test analyze command with JSON format."""
        result = runner.invoke(cli, ['analyze', log_directory, '--format', 'json'])
        assert result.exit_code == 0
        
        # Verify that output is valid JSON
        try:
            json.loads(result.output)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
    
    def test_analyze_with_conversation_id_filter(self, runner, log_directory):
        """Test analyze command with conversation ID filter."""
        result = runner.invoke(cli, [
            'analyze', log_directory, 
            '--conversation-id', 'test_conv_1'
        ])
        assert result.exit_code == 0
        assert "test_conv_1" in result.output
    
    def test_analyze_with_limit(self, runner, log_directory):
        """Test analyze command with limit."""
        result = runner.invoke(cli, [
            'analyze', log_directory, 
            '--limit', '2'
        ])
        assert result.exit_code == 0
        assert "Analyzing 2 conversation log(s)" in result.output
    
    def test_analyze_nonexistent_directory(self, runner):
        """Test analyze command with non-existent directory."""
        result = runner.invoke(cli, ['analyze', '/nonexistent/directory'])
        assert result.exit_code != 0
        assert "does not exist" in result.output
    
    def test_analyze_empty_directory(self, runner):
        """Test analyze command with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, ['analyze', temp_dir])
            assert result.exit_code == 1
            assert "No log files found" in result.output
    
    def test_analyze_with_invalid_json(self, runner):
        """Test analyze command with invalid JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid JSON file
            invalid_file = Path(temp_dir) / "invalid.json"
            with open(invalid_file, 'w') as f:
                f.write("invalid json content")
            
            result = runner.invoke(cli, ['analyze', temp_dir])
            assert result.exit_code == 1
            assert "No valid conversation logs found" in result.output


class TestInfoCommand:
    """Test the 'info' command."""
    
    def test_info_without_config(self, runner):
        """Test info command without configuration."""
        result = runner.invoke(cli, ['info'])
        assert result.exit_code == 0
        assert "Agentic Conversation System - Information" in result.output
        assert "Configuration Status: Not loaded" in result.output
        assert "Python Version:" in result.output
    
    def test_info_with_config(self, runner, config_file):
        """Test info command with configuration."""
        result = runner.invoke(cli, ['--config', config_file, 'info'])
        assert result.exit_code == 0
        assert "Configuration Status: Loaded" in result.output
        assert "gpt-4" in result.output
        assert "Researcher" in result.output
        assert "Analyst" in result.output
    
    def test_info_with_verbose(self, runner):
        """Test info command with verbose flag."""
        result = runner.invoke(cli, ['--verbose', 'info'])
        assert result.exit_code == 0
        assert "Verbose Mode: True" in result.output
    
    def test_info_with_custom_log_level(self, runner):
        """Test info command with custom log level."""
        result = runner.invoke(cli, ['--log-level', 'DEBUG', 'info'])
        assert result.exit_code == 0
        assert "Log Level: DEBUG" in result.output


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def test_global_options_inheritance(self, runner, config_file):
        """Test that global options are inherited by subcommands."""
        result = runner.invoke(cli, [
            '--config', config_file,
            '--log-level', 'DEBUG',
            '--verbose',
            'info'
        ])
        assert result.exit_code == 0
        assert "Log Level: DEBUG" in result.output
        assert "Verbose Mode: True" in result.output
    
    def test_output_directory_override(self, runner, config_file):
        """Test that output directory can be overridden."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(cli, [
                '--config', config_file,
                '--output-dir', temp_dir,
                'info'
            ])
            assert result.exit_code == 0
            # The output directory should be updated in the config
    
    @patch('agentic_conversation.main.ConversationOrchestrator')
    def test_end_to_end_workflow(self, mock_orchestrator, runner, config_file):
        """Test a complete workflow: validate, run, analyze."""
        # Mock orchestrator for run command
        mock_instance = Mock()
        mock_instance.run_conversation = AsyncMock(return_value={
            'conversation_id': 'workflow_test',
            'status': 'completed',
            'total_turns': 3,
            'total_tokens': 900,
            'duration_seconds': 20.1
        })
        mock_instance.config.logging.output_directory = './test_logs'
        mock_orchestrator.return_value = mock_instance
        
        # Step 1: Validate configuration
        with patch('agentic_conversation.main.ConfigurationLoader') as mock_loader:
            mock_loader_instance = Mock()
            mock_loader_instance.validate_config_file.return_value = {'is_valid': True}
            mock_loader.return_value = mock_loader_instance
            
            result = runner.invoke(cli, ['validate', config_file])
            assert result.exit_code == 0
            assert "is valid" in result.output
        
        # Step 2: Run conversation
        result = runner.invoke(cli, ['--config', config_file, 'run'])
        assert result.exit_code == 0
        assert "Conversation completed successfully!" in result.output
        
        # Step 3: Show system info
        result = runner.invoke(cli, ['--config', config_file, 'info'])
        assert result.exit_code == 0
        assert "Configuration Status: Loaded" in result.output


class TestErrorHandling:
    """Test error handling in CLI commands."""
    
    def test_configuration_error_handling(self, runner):
        """Test handling of configuration errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_yaml_file = f.name
        
        try:
            result = runner.invoke(cli, ['--config', invalid_yaml_file, 'info'])
            assert result.exit_code == 1
            assert "Error loading configuration" in result.output
        finally:
            os.unlink(invalid_yaml_file)
    
    def test_permission_error_handling(self, runner):
        """Test handling of permission errors."""
        # This test might be platform-specific
        if os.name == 'posix':  # Unix-like systems
            result = runner.invoke(cli, ['--output-dir', '/root/restricted'])
            # Should handle permission error gracefully
            assert result.exit_code != 0
    
    def test_unexpected_error_with_verbose(self, runner, config_file):
        """Test unexpected error handling with verbose mode."""
        with patch('agentic_conversation.main.ConversationOrchestrator') as mock_orchestrator:
            mock_orchestrator.side_effect = RuntimeError("Unexpected error")
            
            result = runner.invoke(cli, [
                '--config', config_file,
                '--verbose',
                'run'
            ])
            assert result.exit_code == 1
            assert "Unexpected error" in result.output


if __name__ == '__main__':
    pytest.main([__file__])

class TestCreateConfigCommand:
    """Test the 'create-config' command."""
    
    def test_create_config_basic_template(self, runner):
        """Test create-config command with basic template."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'create-config', 'basic', config_path
            ])
            assert result.exit_code == 0
            assert "Created basic configuration" in result.output
            assert "Assistant A & Assistant B" in result.output
            
            # Verify the file was created and is valid YAML
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert 'agents' in config
            assert 'model' in config
            assert 'conversation' in config
            assert 'logging' in config
            assert config['agents']['agent_a']['name'] == 'Assistant A'
            assert config['agents']['agent_b']['name'] == 'Assistant B'
            assert config['model']['model_name'] == 'gpt-4'
            assert config['conversation']['max_turns'] == 10
            
        finally:
            os.unlink(config_path)
    
    def test_create_config_research_template(self, runner):
        """Test create-config command with research template."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'create-config', 'research', config_path,
                '--model', 'gpt-3.5-turbo',
                '--max-turns', '15'
            ])
            assert result.exit_code == 0
            assert "Created research configuration" in result.output
            assert "Researcher & Analyst" in result.output
            assert "gpt-3.5-turbo" in result.output
            assert "15" in result.output
            
            # Verify the configuration
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert config['agents']['agent_a']['name'] == 'Researcher'
            assert config['agents']['agent_b']['name'] == 'Analyst'
            assert config['model']['model_name'] == 'gpt-3.5-turbo'
            assert config['conversation']['max_turns'] == 15
            assert 'artificial intelligence' in config['conversation']['initial_prompt']
            
        finally:
            os.unlink(config_path)
    
    def test_create_config_creative_template(self, runner):
        """Test create-config command with creative template."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'create-config', 'creative', config_path
            ])
            assert result.exit_code == 0
            assert "Created creative configuration" in result.output
            assert "Creative Writer & Story Developer" in result.output
            
            # Verify the configuration
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert config['agents']['agent_a']['name'] == 'Creative Writer'
            assert config['agents']['agent_b']['name'] == 'Story Developer'
            assert 'story concept' in config['conversation']['initial_prompt']
            
        finally:
            os.unlink(config_path)
    
    def test_create_config_technical_template(self, runner):
        """Test create-config command with technical template."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            result = runner.invoke(cli, [
                'create-config', 'technical', config_path
            ])
            assert result.exit_code == 0
            assert "Created technical configuration" in result.output
            assert "Software Architect & Implementation Engineer" in result.output
            
            # Verify the configuration
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            assert config['agents']['agent_a']['name'] == 'Software Architect'
            assert config['agents']['agent_b']['name'] == 'Implementation Engineer'
            assert 'microservices architecture' in config['conversation']['initial_prompt']
            
        finally:
            os.unlink(config_path)
    
    def test_create_config_invalid_template(self, runner):
        """Test create-config command with invalid template."""
        result = runner.invoke(cli, [
            'create-config', 'invalid', 'output.yaml'
        ])
        assert result.exit_code != 0
        assert "Invalid value" in result.output
    
    def test_create_config_file_error(self, runner):
        """Test create-config command with file write error."""
        # Try to write to a directory that doesn't exist
        result = runner.invoke(cli, [
            'create-config', 'basic', '/nonexistent/directory/config.yaml'
        ])
        assert result.exit_code == 1
        assert "Error creating configuration" in result.output