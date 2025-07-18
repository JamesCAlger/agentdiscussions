"""
Tool system for the agentic conversation system.

This module provides a modular tool system that allows agents to use external
tools like web search, file operations, and other capabilities.
"""

from .base import BaseTool, ToolResult, ToolContext, ToolInfo, ToolCapability
from .exceptions import (
    ToolError,
    ToolExecutionError,
    ToolTimeoutError,
    ToolConfigurationError,
    ToolRateLimitError
)
from .registry import ToolRegistry, ToolRegistration, RegistryConfig
from .executor import (
    ToolExecutor,
    ExecutorConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RateLimiter,
    RateLimitConfig,
    RetryConfig,
    ConnectionPool
)
from .manager import (
    ToolManager,
    ToolManagerConfig,
    ToolSelectionCriteria,
    QueryGenerationContext,
    ToolExecutionPlan
)
from .web_search import WebSearchTool

__all__ = [
    "BaseTool",
    "ToolResult", 
    "ToolContext",
    "ToolInfo",
    "ToolCapability",
    "ToolError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolConfigurationError",
    "ToolRateLimitError",
    "ToolRegistry",
    "ToolRegistration",
    "RegistryConfig",
    "ToolExecutor",
    "ExecutorConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "RateLimiter",
    "RateLimitConfig",
    "RetryConfig",
    "ConnectionPool",
    "ToolManager",
    "ToolManagerConfig",
    "ToolSelectionCriteria",
    "QueryGenerationContext",
    "ToolExecutionPlan",
    "WebSearchTool"
]