"""
Tool registry and discovery system.

This module provides the ToolRegistry class for managing available tools,
including registration, validation, lifecycle management, and discovery.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union
from dataclasses import dataclass, field
from datetime import datetime

from .base import BaseTool, ToolInfo, ToolCapability
from .exceptions import ToolConfigurationError, ToolError


logger = logging.getLogger(__name__)


@dataclass
class ToolRegistration:
    """
    Information about a registered tool.
    
    This class holds metadata about a tool that has been registered
    with the tool registry, including its class, configuration, and status.
    
    Attributes:
        tool_class: The tool class type
        tool_info: Metadata about the tool
        config: Configuration for the tool instance
        enabled: Whether the tool is enabled
        instance: Cached tool instance (if created)
        registration_time: When the tool was registered
        last_used: When the tool was last used
        usage_count: Number of times the tool has been used
    """
    tool_class: Type[BaseTool]
    tool_info: ToolInfo
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    instance: Optional[BaseTool] = None
    registration_time: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    def create_instance(self) -> BaseTool:
        """Create a new instance of the tool with the configured settings."""
        if not self.enabled:
            raise ToolError(f"Tool {self.tool_info.name} is disabled")
        
        try:
            instance = self.tool_class(config=self.config)
            return instance
        except Exception as e:
            raise ToolConfigurationError(
                f"Failed to create instance of tool {self.tool_info.name}: {str(e)}",
                tool_name=self.tool_info.name,
                context={"config": self.config, "error": str(e)}
            )
    
    def get_or_create_instance(self) -> BaseTool:
        """Get the cached instance or create a new one if needed."""
        if self.instance is None:
            self.instance = self.create_instance()
        return self.instance
    
    def mark_used(self) -> None:
        """Mark the tool as used, updating usage statistics."""
        self.last_used = datetime.now()
        self.usage_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the registration to a dictionary for serialization."""
        return {
            "tool_class": f"{self.tool_class.__module__}.{self.tool_class.__name__}",
            "tool_info": self.tool_info.to_dict(),
            "config": self.config,
            "enabled": self.enabled,
            "registration_time": self.registration_time.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count
        }


@dataclass
class RegistryConfig:
    """
    Configuration for the tool registry.
    
    Attributes:
        auto_discovery_enabled: Whether to automatically discover tools
        discovery_paths: Paths to search for tools
        tool_configs: Configuration for individual tools
        enabled_tools: List of tool names that should be enabled
        disabled_tools: List of tool names that should be disabled
        max_instances_per_tool: Maximum cached instances per tool
        instance_timeout: How long to keep cached instances (seconds)
    """
    auto_discovery_enabled: bool = True
    discovery_paths: List[str] = field(default_factory=list)
    tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enabled_tools: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)
    max_instances_per_tool: int = 5
    instance_timeout: float = 3600.0  # 1 hour
    
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool should be enabled."""
        # If explicitly disabled, return False
        if tool_name in self.disabled_tools:
            return False
        
        # If enabled_tools is empty, all tools are enabled by default
        if not self.enabled_tools:
            return True
        
        # Otherwise, only enabled if explicitly listed
        return tool_name in self.enabled_tools
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool."""
        return self.tool_configs.get(tool_name, {})


class ToolRegistry:
    """
    Registry for managing available tools.
    
    The ToolRegistry is responsible for:
    - Discovering and registering tools
    - Managing tool configurations and lifecycle
    - Providing tools to agents and other components
    - Validating tool configurations
    - Tracking tool usage statistics
    """
    
    def __init__(self, config: Optional[RegistryConfig] = None):
        """
        Initialize the tool registry.
        
        Args:
            config: Configuration for the registry
        """
        self.config = config or RegistryConfig()
        self._registrations: Dict[str, ToolRegistration] = {}
        self._capabilities_index: Dict[ToolCapability, Set[str]] = {}
        self._initialized = False
        
        logger.info("Initializing ToolRegistry")
        
        # Initialize capabilities index
        for capability in ToolCapability:
            self._capabilities_index[capability] = set()
    
    def initialize(self) -> None:
        """
        Initialize the registry by discovering and registering tools.
        
        This method should be called after creating the registry to
        populate it with available tools.
        """
        if self._initialized:
            logger.warning("ToolRegistry already initialized")
            return
        
        logger.info("Starting tool registry initialization")
        
        try:
            if self.config.auto_discovery_enabled:
                self._discover_tools()
            
            self._validate_registrations()
            self._initialized = True
            
            logger.info(f"ToolRegistry initialized with {len(self._registrations)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize ToolRegistry: {str(e)}")
            raise ToolError(f"Registry initialization failed: {str(e)}")
    
    def register_tool(
        self,
        tool_class: Type[BaseTool],
        config: Optional[Dict[str, Any]] = None,
        enabled: Optional[bool] = None
    ) -> None:
        """
        Register a tool class with the registry.
        
        Args:
            tool_class: The tool class to register
            config: Configuration for the tool
            enabled: Whether the tool should be enabled (None = use default logic)
            
        Raises:
            ToolConfigurationError: If the tool configuration is invalid
            ToolError: If registration fails
        """
        try:
            # Create a temporary instance to get tool info
            # We need to handle cases where the tool requires configuration
            tool_info = None
            temp_instance = None
            
            # First, try to create instance with empty config to get tool info
            try:
                temp_instance = tool_class(config={})
                tool_info = temp_instance.get_tool_info()
            except ToolConfigurationError:
                # If that fails, we need to get the tool info another way
                # Try to create a minimal instance just to get the tool info
                # We'll bypass the validation temporarily
                try:
                    # Create instance without calling __init__ validation
                    temp_instance = tool_class.__new__(tool_class)
                    temp_instance.config = {}
                    tool_info = temp_instance.get_tool_info()
                except Exception:
                    # If all else fails, we can't register this tool
                    raise ToolError(f"Cannot get tool info from {tool_class.__name__}")
            
            if tool_info is None:
                raise ToolError(f"Failed to get tool info from {tool_class.__name__}")
            
            # Determine if tool should be enabled
            if enabled is None:
                enabled = self.config.is_tool_enabled(tool_info.name)
            
            # Get tool-specific configuration
            tool_config = config or self.config.get_tool_config(tool_info.name)
            
            # Validate configuration (only if tool is enabled)
            if enabled:
                config_errors = tool_info.validate_config(tool_config)
                if config_errors:
                    raise ToolConfigurationError(
                        f"Invalid configuration for tool {tool_info.name}",
                        tool_name=tool_info.name,
                        context={"errors": config_errors, "config": tool_config}
                    )
            
            # Create registration
            registration = ToolRegistration(
                tool_class=tool_class,
                tool_info=tool_info,
                config=tool_config,
                enabled=enabled
            )
            
            # Register the tool
            self._registrations[tool_info.name] = registration
            
            # Update capabilities index
            for capability in tool_info.capabilities:
                self._capabilities_index[capability].add(tool_info.name)
            
            logger.info(f"Registered tool: {tool_info.name} (enabled: {enabled})")
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool_class.__name__}: {str(e)}")
            raise ToolError(f"Tool registration failed: {str(e)}")
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name not in self._registrations:
            logger.warning(f"Tool {tool_name} not found in registry")
            return
        
        registration = self._registrations[tool_name]
        
        # Remove from capabilities index
        for capability in registration.tool_info.capabilities:
            self._capabilities_index[capability].discard(tool_name)
        
        # Remove registration
        del self._registrations[tool_name]
        
        logger.info(f"Unregistered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a tool instance by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            BaseTool instance or None if not found/disabled
        """
        if tool_name not in self._registrations:
            logger.warning(f"Tool {tool_name} not found in registry")
            return None
        
        registration = self._registrations[tool_name]
        
        if not registration.enabled:
            logger.warning(f"Tool {tool_name} is disabled")
            return None
        
        try:
            instance = registration.get_or_create_instance()
            registration.mark_used()
            return instance
        except Exception as e:
            logger.error(f"Failed to get tool instance {tool_name}: {str(e)}")
            return None
    
    def get_tools_by_capability(self, capability: ToolCapability) -> List[BaseTool]:
        """
        Get all enabled tools that have a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of tool instances with the specified capability
        """
        tool_names = self._capabilities_index.get(capability, set())
        tools = []
        
        for tool_name in tool_names:
            tool = self.get_tool(tool_name)
            if tool is not None:
                tools.append(tool)
        
        return tools
    
    def get_all_tools(self, enabled_only: bool = True) -> List[BaseTool]:
        """
        Get all tools in the registry.
        
        Args:
            enabled_only: Whether to return only enabled tools
            
        Returns:
            List of tool instances
        """
        tools = []
        
        for registration in self._registrations.values():
            if enabled_only and not registration.enabled:
                continue
            
            try:
                # For disabled tools when enabled_only=False, we still need to create instances
                # but we need to temporarily enable them to create the instance
                if not registration.enabled and not enabled_only:
                    # Temporarily enable to create instance
                    original_enabled = registration.enabled
                    registration.enabled = True
                    try:
                        instance = registration.get_or_create_instance()
                        registration.mark_used()
                        tools.append(instance)
                    finally:
                        registration.enabled = original_enabled
                else:
                    instance = registration.get_or_create_instance()
                    registration.mark_used()
                    tools.append(instance)
            except Exception as e:
                logger.error(f"Failed to get tool instance {registration.tool_info.name}: {str(e)}")
                continue
        
        return tools
    
    def get_tool_info(self, tool_name: str) -> Optional[ToolInfo]:
        """
        Get information about a tool without creating an instance.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolInfo or None if not found
        """
        registration = self._registrations.get(tool_name)
        return registration.tool_info if registration else None
    
    def list_tools(self, enabled_only: bool = True) -> List[str]:
        """
        List all tool names in the registry.
        
        Args:
            enabled_only: Whether to return only enabled tools
            
        Returns:
            List of tool names
        """
        if enabled_only:
            return [
                name for name, reg in self._registrations.items()
                if reg.enabled
            ]
        else:
            return list(self._registrations.keys())
    
    def enable_tool(self, tool_name: str) -> bool:
        """
        Enable a tool in the registry.
        
        Args:
            tool_name: Name of the tool to enable
            
        Returns:
            True if successful, False if tool not found
        """
        if tool_name not in self._registrations:
            return False
        
        self._registrations[tool_name].enabled = True
        logger.info(f"Enabled tool: {tool_name}")
        return True
    
    def disable_tool(self, tool_name: str) -> bool:
        """
        Disable a tool in the registry.
        
        Args:
            tool_name: Name of the tool to disable
            
        Returns:
            True if successful, False if tool not found
        """
        if tool_name not in self._registrations:
            return False
        
        registration = self._registrations[tool_name]
        registration.enabled = False
        registration.instance = None  # Clear cached instance
        
        logger.info(f"Disabled tool: {tool_name}")
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the tool registry.
        
        Returns:
            Dictionary with registry statistics
        """
        total_tools = len(self._registrations)
        enabled_tools = sum(1 for reg in self._registrations.values() if reg.enabled)
        
        capabilities_count = {}
        for capability, tool_names in self._capabilities_index.items():
            capabilities_count[capability.value] = len(tool_names)
        
        usage_stats = {}
        for name, reg in self._registrations.items():
            usage_stats[name] = {
                "usage_count": reg.usage_count,
                "last_used": reg.last_used.isoformat() if reg.last_used else None,
                "enabled": reg.enabled
            }
        
        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": total_tools - enabled_tools,
            "capabilities_count": capabilities_count,
            "usage_stats": usage_stats,
            "initialized": self._initialized
        }
    
    def _discover_tools(self) -> None:
        """
        Discover tools automatically from configured paths.
        
        This method searches for tool classes in the specified discovery paths
        and registers them automatically.
        """
        logger.info("Starting automatic tool discovery")
        
        # Default discovery paths
        discovery_paths = self.config.discovery_paths or [
            "agentic_conversation.tools",
            "agentic_conversation.tools.implementations"
        ]
        
        discovered_count = 0
        
        for path in discovery_paths:
            try:
                discovered_count += self._discover_tools_in_module(path)
            except Exception as e:
                logger.warning(f"Failed to discover tools in {path}: {str(e)}")
        
        logger.info(f"Discovered {discovered_count} tools")
    
    def _discover_tools_in_module(self, module_path: str) -> int:
        """
        Discover tools in a specific module path.
        
        Args:
            module_path: Python module path to search
            
        Returns:
            Number of tools discovered
        """
        discovered_count = 0
        
        try:
            # Try to import the module
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                logger.debug(f"Module {module_path} not found, skipping")
                return 0
            
            # Search for tool classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_tool_class(obj):
                    try:
                        self.register_tool(obj)
                        discovered_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to register discovered tool {name}: {str(e)}")
            
            # Search in submodules if it's a package
            if hasattr(module, '__path__'):
                for importer, modname, ispkg in pkgutil.iter_modules(module.__path__):
                    submodule_path = f"{module_path}.{modname}"
                    discovered_count += self._discover_tools_in_module(submodule_path)
            
        except Exception as e:
            logger.error(f"Error discovering tools in {module_path}: {str(e)}")
        
        return discovered_count
    
    def _is_tool_class(self, obj: Any) -> bool:
        """
        Check if an object is a valid tool class.
        
        Args:
            obj: Object to check
            
        Returns:
            True if it's a valid tool class
        """
        return (
            inspect.isclass(obj) and
            issubclass(obj, BaseTool) and
            obj is not BaseTool and
            not inspect.isabstract(obj)
        )
    
    def _validate_registrations(self) -> None:
        """
        Validate all registered tools.
        
        This method checks that all registered tools are properly configured
        and can be instantiated.
        """
        logger.info("Validating tool registrations")
        
        invalid_tools = []
        
        for name, registration in self._registrations.items():
            try:
                # Try to create an instance to validate the tool
                instance = registration.create_instance()
                
                # Validate that the tool info is consistent
                tool_info = instance.get_tool_info()
                if tool_info.name != name:
                    logger.warning(f"Tool {name} has inconsistent name in tool_info: {tool_info.name}")
                
            except Exception as e:
                logger.error(f"Tool {name} failed validation: {str(e)}")
                invalid_tools.append(name)
        
        # Remove invalid tools
        for tool_name in invalid_tools:
            self.unregister_tool(tool_name)
        
        if invalid_tools:
            logger.warning(f"Removed {len(invalid_tools)} invalid tools: {invalid_tools}")
    
    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._registrations)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._registrations
    
    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._registrations.keys())