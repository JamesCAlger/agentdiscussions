"""
Configuration management for the agentic conversation system.

This module provides functionality to load, validate, and manage system configuration
from YAML files with support for environment variable substitution.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from .models import SystemConfig, AgentConfig, ModelConfig, ConversationConfig, LoggingConfig


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ConfigurationLoader:
    """
    Loads and validates YAML configuration files for the agentic conversation system.
    
    Supports environment variable substitution using ${VAR_NAME} or ${VAR_NAME:default_value} syntax.
    """
    
    def __init__(self):
        """Initialize the configuration loader."""
        self._env_var_pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
    
    def load_from_file(self, config_path: Union[str, Path]) -> SystemConfig:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            SystemConfig: Validated system configuration
            
        Raises:
            ConfigurationError: If the file cannot be loaded or configuration is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration path is not a file: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration file: {e}")
        
        if raw_config is None:
            raise ConfigurationError("Configuration file is empty")
        
        if not isinstance(raw_config, dict):
            raise ConfigurationError("Configuration must be a YAML object/dictionary")
        
        # Substitute environment variables
        processed_config = self._substitute_environment_variables(raw_config)
        
        # Convert to SystemConfig and validate
        return self._create_system_config(processed_config)
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SystemConfig: Validated system configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not isinstance(config_dict, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        # Substitute environment variables
        processed_config = self._substitute_environment_variables(config_dict)
        
        # Convert to SystemConfig and validate
        return self._create_system_config(processed_config)
    
    def validate_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a configuration file without creating a SystemConfig object.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dict containing validation results with keys:
            - is_valid: bool
            - errors: List[str]
            - warnings: List[str]
            
        Raises:
            ConfigurationError: If the file cannot be loaded or parsed
        """
        try:
            config = self.load_from_file(config_path)
            validation_summary = config.get_validation_summary()
            return {
                "is_valid": validation_summary["is_valid"],
                "errors": validation_summary["errors"],
                "warnings": []
            }
        except ConfigurationError as e:
            return {
                "is_valid": False,
                "errors": [str(e)],
                "warnings": []
            }
    
    def _substitute_environment_variables(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration objects.
        
        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        
        Args:
            obj: Configuration object (dict, list, str, or other)
            
        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {key: self._substitute_environment_variables(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_environment_variables(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_vars_in_string(obj)
        else:
            return obj
    
    def _substitute_env_vars_in_string(self, text: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            text: String that may contain environment variable references
            
        Returns:
            String with environment variables substituted
        """
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2)
            
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default_value is not None:  # Check if default was provided (even if empty)
                return default_value
            else:
                # If no default value and environment variable is not set,
                # leave the placeholder as-is for validation to catch
                return match.group(0)
        
        return self._env_var_pattern.sub(replace_var, text)
    
    def _create_system_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """
        Create and validate a SystemConfig from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SystemConfig: Validated system configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Extract and validate required sections
            self._validate_required_sections(config_dict)
            
            # Create configuration objects
            agent_a_config = self._create_agent_config(config_dict["agents"]["agent_a"], "agent_a")
            agent_b_config = self._create_agent_config(config_dict["agents"]["agent_b"], "agent_b")
            model_config = self._create_model_config(config_dict["model"])
            conversation_config = self._create_conversation_config(config_dict["conversation"])
            logging_config = self._create_logging_config(config_dict.get("logging", {}))
            
            # Create system configuration
            system_config = SystemConfig(
                agent_a=agent_a_config,
                agent_b=agent_b_config,
                model=model_config,
                conversation=conversation_config,
                logging=logging_config
            )
            
            # Validate the complete configuration
            validation_errors = system_config.validate()
            if validation_errors:
                error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
                raise ConfigurationError(error_message)
            
            return system_config
            
        except KeyError as e:
            raise ConfigurationError(f"Missing required configuration section: {e}")
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to create system configuration: {e}")
    
    def _validate_required_sections(self, config_dict: Dict[str, Any]) -> None:
        """
        Validate that all required configuration sections are present.
        
        Args:
            config_dict: Configuration dictionary
            
        Raises:
            ConfigurationError: If required sections are missing
        """
        required_sections = ["agents", "model", "conversation"]
        missing_sections = []
        
        for section in required_sections:
            if section not in config_dict:
                missing_sections.append(section)
        
        if missing_sections:
            raise ConfigurationError(f"Missing required configuration sections: {', '.join(missing_sections)}")
        
        # Validate agents subsections
        agents_config = config_dict["agents"]
        if not isinstance(agents_config, dict):
            raise ConfigurationError("'agents' section must be a dictionary")
        
        required_agents = ["agent_a", "agent_b"]
        missing_agents = []
        
        for agent in required_agents:
            if agent not in agents_config:
                missing_agents.append(agent)
        
        if missing_agents:
            raise ConfigurationError(f"Missing required agent configurations: {', '.join(missing_agents)}")
    
    def _create_agent_config(self, agent_dict: Dict[str, Any], agent_name: str) -> AgentConfig:
        """
        Create an AgentConfig from a dictionary.
        
        Args:
            agent_dict: Agent configuration dictionary
            agent_name: Name of the agent (for error messages)
            
        Returns:
            AgentConfig: Validated agent configuration
            
        Raises:
            ConfigurationError: If agent configuration is invalid
        """
        try:
            required_fields = ["name", "system_prompt"]
            for field in required_fields:
                if field not in agent_dict:
                    raise ConfigurationError(f"Missing required field '{field}' in {agent_name} configuration")
            
            return AgentConfig(
                name=agent_dict["name"],
                system_prompt=agent_dict["system_prompt"]
            )
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to create {agent_name} configuration: {e}")
    
    def _create_model_config(self, model_dict: Dict[str, Any]) -> ModelConfig:
        """
        Create a ModelConfig from a dictionary.
        
        Args:
            model_dict: Model configuration dictionary
            
        Returns:
            ModelConfig: Validated model configuration
            
        Raises:
            ConfigurationError: If model configuration is invalid
        """
        try:
            if "model_name" not in model_dict:
                raise ConfigurationError("Missing required field 'model_name' in model configuration")
            
            return ModelConfig(
                model_name=model_dict["model_name"],
                temperature=model_dict.get("temperature", 0.7),
                max_tokens=model_dict.get("max_tokens", 2000),
                top_p=model_dict.get("top_p", 1.0),
                frequency_penalty=model_dict.get("frequency_penalty", 0.0),
                presence_penalty=model_dict.get("presence_penalty", 0.0)
            )
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to create model configuration: {e}")
    
    def _create_conversation_config(self, conversation_dict: Dict[str, Any]) -> ConversationConfig:
        """
        Create a ConversationConfig from a dictionary.
        
        Args:
            conversation_dict: Conversation configuration dictionary
            
        Returns:
            ConversationConfig: Validated conversation configuration
            
        Raises:
            ConfigurationError: If conversation configuration is invalid
        """
        try:
            if "max_turns" not in conversation_dict:
                raise ConfigurationError("Missing required field 'max_turns' in conversation configuration")
            
            return ConversationConfig(
                max_turns=conversation_dict["max_turns"],
                initial_prompt=conversation_dict.get("initial_prompt"),
                context_window_strategy=conversation_dict.get("context_window_strategy", "sliding"),
                context_window_size=conversation_dict.get("context_window_size", 8000),
                turn_timeout=conversation_dict.get("turn_timeout", 30.0)
            )
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to create conversation configuration: {e}")
    
    def _create_logging_config(self, logging_dict: Dict[str, Any]) -> LoggingConfig:
        """
        Create a LoggingConfig from a dictionary.
        
        Args:
            logging_dict: Logging configuration dictionary
            
        Returns:
            LoggingConfig: Validated logging configuration
            
        Raises:
            ConfigurationError: If logging configuration is invalid
        """
        try:
            return LoggingConfig(
                log_level=logging_dict.get("log_level", "INFO"),
                output_directory=logging_dict.get("output_directory", "./logs"),
                real_time_display=logging_dict.get("real_time_display", True),
                export_formats=logging_dict.get("export_formats", ["json"]),
                save_conversation_history=logging_dict.get("save_conversation_history", True),
                save_telemetry=logging_dict.get("save_telemetry", True)
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create logging configuration: {e}")


def load_config(config_path: Union[str, Path]) -> SystemConfig:
    """
    Convenience function to load configuration from a file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        SystemConfig: Validated system configuration
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
    """
    loader = ConfigurationLoader()
    return loader.load_from_file(config_path)


def validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing validation results
    """
    loader = ConfigurationLoader()
    return loader.validate_config_file(config_path)