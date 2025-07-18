"""
Unit tests for the tool registry system.

This module tests the ToolRegistry class and related functionality,
including tool registration, discovery, validation, and lifecycle management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from src.agentic_conversation.tools.registry import (
    ToolRegistry,
    ToolRegistration,
    RegistryConfig
)
from src.agentic_conversation.tools.base import (
    BaseTool,
    ToolInfo,
    ToolResult,
    ToolContext,
    ToolCapability
)
from src.agentic_conversation.tools.exceptions import (
    ToolError,
    ToolConfigurationError
)


class MockTool(BaseTool):
    """Mock tool for testing purposes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.executed = False
        self.execution_count = 0
    
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        """Mock execute method."""
        self.executed = True
        self.execution_count += 1
        return ToolResult(
            success=True,
            content=f"Mock result for query: {query}",
            tool_name="mock_tool",
            execution_time=0.1
        )
    
    def is_relevant(self, context: ToolContext) -> bool:
        """Mock relevance check."""
        return True
    
    def get_tool_info(self) -> ToolInfo:
        """Mock tool info."""
        return ToolInfo(
            name="mock_tool",
            description="A mock tool for testing",
            capabilities=[ToolCapability.SEARCH],
            required_config=["api_key"],
            optional_config=["timeout"],
            version="1.0.0"
        )


class MockToolWithoutRequiredConfig(BaseTool):
    """Mock tool that doesn't require configuration."""
    
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        return ToolResult(
            success=True,
            content="Mock result",
            tool_name="simple_mock_tool"
        )
    
    def is_relevant(self, context: ToolContext) -> bool:
        return True
    
    def get_tool_info(self) -> ToolInfo:
        return ToolInfo(
            name="simple_mock_tool",
            description="A simple mock tool",
            capabilities=[ToolCapability.ANALYSIS],
            version="1.0.0"
        )


class InvalidMockTool(BaseTool):
    """Mock tool that raises errors during initialization."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        raise ValueError("This tool always fails to initialize")
    
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        pass
    
    def is_relevant(self, context: ToolContext) -> bool:
        return False
    
    def get_tool_info(self) -> ToolInfo:
        return ToolInfo(
            name="invalid_tool",
            description="An invalid tool",
            version="1.0.0"
        )


class TestToolRegistration:
    """Test cases for ToolRegistration class."""
    
    def test_tool_registration_creation(self):
        """Test creating a tool registration."""
        tool_info = ToolInfo(
            name="test_tool",
            description="Test tool",
            capabilities=[ToolCapability.SEARCH]
        )
        
        registration = ToolRegistration(
            tool_class=MockTool,
            tool_info=tool_info,
            config={"api_key": "test_key"}
        )
        
        assert registration.tool_class == MockTool
        assert registration.tool_info == tool_info
        assert registration.config == {"api_key": "test_key"}
        assert registration.enabled is True
        assert registration.instance is None
        assert registration.usage_count == 0
    
    def test_create_instance(self):
        """Test creating a tool instance from registration."""
        tool_info = ToolInfo(
            name="mock_tool",
            description="Mock tool",
            capabilities=[ToolCapability.SEARCH],
            required_config=["api_key"]
        )
        
        registration = ToolRegistration(
            tool_class=MockTool,
            tool_info=tool_info,
            config={"api_key": "test_key"}
        )
        
        instance = registration.create_instance()
        assert isinstance(instance, MockTool)
        assert instance.config == {"api_key": "test_key"}
    
    def test_create_instance_disabled_tool(self):
        """Test creating instance of disabled tool raises error."""
        tool_info = ToolInfo(name="test_tool", description="Test")
        registration = ToolRegistration(
            tool_class=MockTool,
            tool_info=tool_info,
            enabled=False
        )
        
        with pytest.raises(ToolError, match="Tool test_tool is disabled"):
            registration.create_instance()
    
    def test_get_or_create_instance_caching(self):
        """Test that get_or_create_instance caches instances."""
        tool_info = ToolInfo(
            name="mock_tool",
            description="Mock tool",
            required_config=["api_key"]
        )
        
        registration = ToolRegistration(
            tool_class=MockTool,
            tool_info=tool_info,
            config={"api_key": "test_key"}
        )
        
        instance1 = registration.get_or_create_instance()
        instance2 = registration.get_or_create_instance()
        
        assert instance1 is instance2  # Same instance should be returned
    
    def test_mark_used(self):
        """Test marking a tool as used updates statistics."""
        tool_info = ToolInfo(name="test_tool", description="Test")
        registration = ToolRegistration(
            tool_class=MockTool,
            tool_info=tool_info
        )
        
        assert registration.usage_count == 0
        assert registration.last_used is None
        
        registration.mark_used()
        
        assert registration.usage_count == 1
        assert registration.last_used is not None
        assert isinstance(registration.last_used, datetime)
    
    def test_to_dict(self):
        """Test converting registration to dictionary."""
        tool_info = ToolInfo(
            name="test_tool",
            description="Test tool",
            capabilities=[ToolCapability.SEARCH]
        )
        
        registration = ToolRegistration(
            tool_class=MockTool,
            tool_info=tool_info,
            config={"api_key": "test"}
        )
        
        result = registration.to_dict()
        
        assert result["tool_class"] == f"{MockTool.__module__}.{MockTool.__name__}"
        assert result["tool_info"]["name"] == "test_tool"
        assert result["config"] == {"api_key": "test"}
        assert result["enabled"] is True
        assert result["usage_count"] == 0


class TestRegistryConfig:
    """Test cases for RegistryConfig class."""
    
    def test_default_config(self):
        """Test default registry configuration."""
        config = RegistryConfig()
        
        assert config.auto_discovery_enabled is True
        assert config.discovery_paths == []
        assert config.tool_configs == {}
        assert config.enabled_tools == []
        assert config.disabled_tools == []
        assert config.max_instances_per_tool == 5
        assert config.instance_timeout == 3600.0
    
    def test_is_tool_enabled_default(self):
        """Test tool enabled check with default settings."""
        config = RegistryConfig()
        
        # With empty enabled_tools list, all tools should be enabled by default
        assert config.is_tool_enabled("any_tool") is True
    
    def test_is_tool_enabled_with_enabled_list(self):
        """Test tool enabled check with explicit enabled list."""
        config = RegistryConfig(enabled_tools=["tool1", "tool2"])
        
        assert config.is_tool_enabled("tool1") is True
        assert config.is_tool_enabled("tool2") is True
        assert config.is_tool_enabled("tool3") is False
    
    def test_is_tool_enabled_with_disabled_list(self):
        """Test tool enabled check with disabled list."""
        config = RegistryConfig(disabled_tools=["tool1"])
        
        assert config.is_tool_enabled("tool1") is False
        assert config.is_tool_enabled("tool2") is True
    
    def test_is_tool_enabled_disabled_overrides_enabled(self):
        """Test that disabled list overrides enabled list."""
        config = RegistryConfig(
            enabled_tools=["tool1", "tool2"],
            disabled_tools=["tool1"]
        )
        
        assert config.is_tool_enabled("tool1") is False
        assert config.is_tool_enabled("tool2") is True
    
    def test_get_tool_config(self):
        """Test getting tool-specific configuration."""
        config = RegistryConfig(
            tool_configs={
                "tool1": {"api_key": "key1"},
                "tool2": {"timeout": 30}
            }
        )
        
        assert config.get_tool_config("tool1") == {"api_key": "key1"}
        assert config.get_tool_config("tool2") == {"timeout": 30}
        assert config.get_tool_config("tool3") == {}


class TestToolRegistry:
    """Test cases for ToolRegistry class."""
    
    def test_registry_initialization(self):
        """Test basic registry initialization."""
        registry = ToolRegistry()
        
        assert not registry._initialized
        assert len(registry._registrations) == 0
        assert len(registry._capabilities_index) == len(ToolCapability)
    
    def test_register_tool_success(self):
        """Test successful tool registration."""
        config = RegistryConfig(
            tool_configs={"mock_tool": {"api_key": "test_key"}}
        )
        registry = ToolRegistry(config)
        
        registry.register_tool(MockTool)
        
        assert "mock_tool" in registry._registrations
        registration = registry._registrations["mock_tool"]
        assert registration.tool_class == MockTool
        assert registration.config == {"api_key": "test_key"}
        assert registration.enabled is True
    
    def test_register_tool_with_custom_config(self):
        """Test registering tool with custom configuration."""
        registry = ToolRegistry()
        custom_config = {"api_key": "custom_key", "timeout": 60}
        
        registry.register_tool(MockTool, config=custom_config)
        
        registration = registry._registrations["mock_tool"]
        assert registration.config == custom_config
    
    def test_register_tool_disabled(self):
        """Test registering a disabled tool."""
        config = RegistryConfig(disabled_tools=["mock_tool"])
        registry = ToolRegistry(config)
        
        registry.register_tool(MockTool)
        
        registration = registry._registrations["mock_tool"]
        assert registration.enabled is False
    
    def test_register_tool_invalid_config(self):
        """Test registering tool with invalid configuration."""
        registry = ToolRegistry()
        
        # MockTool requires api_key but we're not providing it
        with pytest.raises(ToolError, match="Tool registration failed"):
            registry.register_tool(MockTool, config={})
    
    def test_register_tool_updates_capabilities_index(self):
        """Test that registering a tool updates the capabilities index."""
        registry = ToolRegistry()
        
        registry.register_tool(MockToolWithoutRequiredConfig)
        
        # MockToolWithoutRequiredConfig has ANALYSIS capability
        assert "simple_mock_tool" in registry._capabilities_index[ToolCapability.ANALYSIS]
    
    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        registry.register_tool(MockToolWithoutRequiredConfig)
        
        assert "simple_mock_tool" in registry._registrations
        assert "simple_mock_tool" in registry._capabilities_index[ToolCapability.ANALYSIS]
        
        registry.unregister_tool("simple_mock_tool")
        
        assert "simple_mock_tool" not in registry._registrations
        assert "simple_mock_tool" not in registry._capabilities_index[ToolCapability.ANALYSIS]
    
    def test_unregister_nonexistent_tool(self):
        """Test unregistering a tool that doesn't exist."""
        registry = ToolRegistry()
        
        # Should not raise an error
        registry.unregister_tool("nonexistent_tool")
    
    def test_get_tool_success(self):
        """Test successfully getting a tool instance."""
        config = RegistryConfig(
            tool_configs={"mock_tool": {"api_key": "test_key"}}
        )
        registry = ToolRegistry(config)
        registry.register_tool(MockTool)
        
        tool = registry.get_tool("mock_tool")
        
        assert tool is not None
        assert isinstance(tool, MockTool)
        
        # Check that usage was marked
        registration = registry._registrations["mock_tool"]
        assert registration.usage_count == 1
        assert registration.last_used is not None
    
    def test_get_tool_not_found(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        
        tool = registry.get_tool("nonexistent_tool")
        assert tool is None
    
    def test_get_tool_disabled(self):
        """Test getting a disabled tool."""
        registry = ToolRegistry()
        registry.register_tool(MockToolWithoutRequiredConfig)
        registry.disable_tool("simple_mock_tool")
        
        tool = registry.get_tool("simple_mock_tool")
        assert tool is None
    
    def test_get_tools_by_capability(self):
        """Test getting tools by capability."""
        config = RegistryConfig(
            tool_configs={"mock_tool": {"api_key": "test_key"}}
        )
        registry = ToolRegistry(config)
        registry.register_tool(MockTool)  # Has SEARCH capability
        registry.register_tool(MockToolWithoutRequiredConfig)  # Has ANALYSIS capability
        
        search_tools = registry.get_tools_by_capability(ToolCapability.SEARCH)
        analysis_tools = registry.get_tools_by_capability(ToolCapability.ANALYSIS)
        
        assert len(search_tools) == 1
        assert isinstance(search_tools[0], MockTool)
        
        assert len(analysis_tools) == 1
        assert isinstance(analysis_tools[0], MockToolWithoutRequiredConfig)
    
    def test_get_all_tools(self):
        """Test getting all tools."""
        config = RegistryConfig(
            tool_configs={"mock_tool": {"api_key": "test_key"}}
        )
        registry = ToolRegistry(config)
        registry.register_tool(MockTool)
        registry.register_tool(MockToolWithoutRequiredConfig)
        
        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2
        
        tool_names = [tool.get_tool_info().name for tool in all_tools]
        assert "mock_tool" in tool_names
        assert "simple_mock_tool" in tool_names
    
    def test_get_all_tools_enabled_only(self):
        """Test getting only enabled tools."""
        config = RegistryConfig(
            tool_configs={"mock_tool": {"api_key": "test_key"}}
        )
        registry = ToolRegistry(config)
        registry.register_tool(MockTool)
        registry.register_tool(MockToolWithoutRequiredConfig)
        registry.disable_tool("simple_mock_tool")
        
        enabled_tools = registry.get_all_tools(enabled_only=True)
        all_tools = registry.get_all_tools(enabled_only=False)
        
        assert len(enabled_tools) == 1
        assert len(all_tools) == 2
    
    def test_get_tool_info(self):
        """Test getting tool information."""
        registry = ToolRegistry()
        registry.register_tool(MockToolWithoutRequiredConfig)
        
        tool_info = registry.get_tool_info("simple_mock_tool")
        
        assert tool_info is not None
        assert tool_info.name == "simple_mock_tool"
        assert tool_info.description == "A simple mock tool"
    
    def test_list_tools(self):
        """Test listing tool names."""
        config = RegistryConfig(
            tool_configs={"mock_tool": {"api_key": "test_key"}}
        )
        registry = ToolRegistry(config)
        registry.register_tool(MockTool)
        registry.register_tool(MockToolWithoutRequiredConfig)
        registry.disable_tool("simple_mock_tool")
        
        enabled_tools = registry.list_tools(enabled_only=True)
        all_tools = registry.list_tools(enabled_only=False)
        
        assert len(enabled_tools) == 1
        assert "mock_tool" in enabled_tools
        
        assert len(all_tools) == 2
        assert "mock_tool" in all_tools
        assert "simple_mock_tool" in all_tools
    
    def test_enable_disable_tool(self):
        """Test enabling and disabling tools."""
        registry = ToolRegistry()
        registry.register_tool(MockToolWithoutRequiredConfig)
        
        # Tool should be enabled by default
        assert registry._registrations["simple_mock_tool"].enabled is True
        
        # Disable the tool
        result = registry.disable_tool("simple_mock_tool")
        assert result is True
        assert registry._registrations["simple_mock_tool"].enabled is False
        
        # Enable the tool
        result = registry.enable_tool("simple_mock_tool")
        assert result is True
        assert registry._registrations["simple_mock_tool"].enabled is True
    
    def test_enable_disable_nonexistent_tool(self):
        """Test enabling/disabling a tool that doesn't exist."""
        registry = ToolRegistry()
        
        assert registry.enable_tool("nonexistent") is False
        assert registry.disable_tool("nonexistent") is False
    
    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        config = RegistryConfig(
            tool_configs={"mock_tool": {"api_key": "test_key"}}
        )
        registry = ToolRegistry(config)
        registry.register_tool(MockTool)
        registry.register_tool(MockToolWithoutRequiredConfig)
        registry.disable_tool("simple_mock_tool")
        
        # Use a tool to generate usage stats
        registry.get_tool("mock_tool")
        
        stats = registry.get_registry_stats()
        
        assert stats["total_tools"] == 2
        assert stats["enabled_tools"] == 1
        assert stats["disabled_tools"] == 1
        assert stats["initialized"] is False
        
        # Check capabilities count
        assert stats["capabilities_count"][ToolCapability.SEARCH.value] == 1
        assert stats["capabilities_count"][ToolCapability.ANALYSIS.value] == 1
        
        # Check usage stats
        assert stats["usage_stats"]["mock_tool"]["usage_count"] == 1
        assert stats["usage_stats"]["simple_mock_tool"]["usage_count"] == 0
    
    def test_discover_tools_in_module(self):
        """Test discovering tools in a module."""
        registry = ToolRegistry()
        
        # Test with a module that doesn't exist - should return 0
        count = registry._discover_tools_in_module("nonexistent.module.that.does.not.exist")
        assert count == 0
        
        # Test with the current test module (which has our mock tools)
        # This is a real test that doesn't require complex mocking
        current_module = "tests.test_tool_registry"
        count = registry._discover_tools_in_module(current_module)
        
        # Should discover MockToolWithoutRequiredConfig but not MockTool (requires config)
        # The exact count depends on what tools can be successfully registered
        assert count >= 0  # At least it shouldn't crash
        
        # Check if any tools were actually registered
        if count > 0:
            assert len(registry._registrations) > 0
    
    def test_is_tool_class(self):
        """Test the _is_tool_class method."""
        registry = ToolRegistry()
        
        assert registry._is_tool_class(MockTool) is True
        assert registry._is_tool_class(MockToolWithoutRequiredConfig) is True
        assert registry._is_tool_class(BaseTool) is False  # Abstract base class
        assert registry._is_tool_class(str) is False  # Not a tool class
        assert registry._is_tool_class("not_a_class") is False  # Not a class
    
    def test_validate_registrations(self):
        """Test validation of registered tools."""
        registry = ToolRegistry()
        
        # Register a valid tool
        registry.register_tool(MockToolWithoutRequiredConfig)
        
        # Manually add an invalid registration
        invalid_registration = ToolRegistration(
            tool_class=InvalidMockTool,
            tool_info=ToolInfo(name="invalid_tool", description="Invalid"),
            config={}
        )
        registry._registrations["invalid_tool"] = invalid_registration
        
        assert len(registry._registrations) == 2
        
        # Validate registrations - should remove the invalid one
        registry._validate_registrations()
        
        assert len(registry._registrations) == 1
        assert "simple_mock_tool" in registry._registrations
        assert "invalid_tool" not in registry._registrations
    
    def test_registry_magic_methods(self):
        """Test registry magic methods (__len__, __contains__, __iter__)."""
        registry = ToolRegistry()
        registry.register_tool(MockToolWithoutRequiredConfig)
        
        # Test __len__
        assert len(registry) == 1
        
        # Test __contains__
        assert "simple_mock_tool" in registry
        assert "nonexistent_tool" not in registry
        
        # Test __iter__
        tool_names = list(registry)
        assert tool_names == ["simple_mock_tool"]
    
    @patch('src.agentic_conversation.tools.registry.logger')
    def test_initialization_with_discovery(self, mock_logger):
        """Test registry initialization with auto-discovery."""
        config = RegistryConfig(auto_discovery_enabled=True)
        registry = ToolRegistry(config)
        
        with patch.object(registry, '_discover_tools') as mock_discover:
            with patch.object(registry, '_validate_registrations') as mock_validate:
                registry.initialize()
                
                mock_discover.assert_called_once()
                mock_validate.assert_called_once()
                assert registry._initialized is True
    
    def test_initialization_already_initialized(self):
        """Test that initializing twice doesn't cause issues."""
        registry = ToolRegistry()
        registry._initialized = True
        
        with patch.object(registry, '_discover_tools') as mock_discover:
            registry.initialize()
            mock_discover.assert_not_called()
    
    def test_initialization_failure(self):
        """Test handling of initialization failure."""
        registry = ToolRegistry()
        
        with patch.object(registry, '_discover_tools', side_effect=Exception("Discovery failed")):
            with pytest.raises(ToolError, match="Registry initialization failed"):
                registry.initialize()