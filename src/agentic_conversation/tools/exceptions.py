"""
Tool-specific exception classes for error handling.

This module defines the exception hierarchy for tool-related errors,
providing specific error types for different failure scenarios.
"""


class ToolError(Exception):
    """
    Base exception for tool-related errors.
    
    This is the base class for all tool-specific exceptions. It provides
    common functionality for error handling and reporting.
    
    Attributes:
        message: Human-readable error message
        tool_name: Name of the tool that caused the error
        context: Additional context information about the error
    """
    
    def __init__(self, message: str, tool_name: str = None, context: dict = None):
        super().__init__(message)
        self.message = message
        self.tool_name = tool_name
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return a formatted error message."""
        if self.tool_name:
            return f"[{self.tool_name}] {self.message}"
        return self.message
    
    def to_dict(self) -> dict:
        """Convert the exception to a dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "tool_name": self.tool_name,
            "context": self.context
        }


class ToolExecutionError(ToolError):
    """
    Raised when tool execution fails.
    
    This exception is raised when a tool fails to execute properly,
    such as when an API call fails or the tool logic encounters an error.
    
    Attributes:
        original_error: The original exception that caused the failure (if any)
        retry_count: Number of retry attempts made before failing
    """
    
    def __init__(self, message: str, tool_name: str = None, context: dict = None, 
                 original_error: Exception = None, retry_count: int = 0):
        super().__init__(message, tool_name, context)
        self.original_error = original_error
        self.retry_count = retry_count
    
    def to_dict(self) -> dict:
        """Convert the exception to a dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "original_error": str(self.original_error) if self.original_error else None,
            "retry_count": self.retry_count
        })
        return result


class ToolTimeoutError(ToolError):
    """
    Raised when tool execution times out.
    
    This exception is raised when a tool takes longer than the configured
    timeout period to complete execution.
    
    Attributes:
        timeout_duration: The timeout duration in seconds
        elapsed_time: The actual time elapsed before timeout
    """
    
    def __init__(self, message: str, tool_name: str = None, context: dict = None,
                 timeout_duration: float = None, elapsed_time: float = None):
        super().__init__(message, tool_name, context)
        self.timeout_duration = timeout_duration
        self.elapsed_time = elapsed_time
    
    def to_dict(self) -> dict:
        """Convert the exception to a dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "timeout_duration": self.timeout_duration,
            "elapsed_time": self.elapsed_time
        })
        return result


class ToolConfigurationError(ToolError):
    """
    Raised when tool configuration is invalid.
    
    This exception is raised when a tool is configured incorrectly,
    such as missing required parameters or invalid configuration values.
    
    Attributes:
        config_field: The specific configuration field that is invalid
        expected_value: The expected value or format for the field
        actual_value: The actual value that was provided
    """
    
    def __init__(self, message: str, tool_name: str = None, context: dict = None,
                 config_field: str = None, expected_value: str = None, actual_value: str = None):
        super().__init__(message, tool_name, context)
        self.config_field = config_field
        self.expected_value = expected_value
        self.actual_value = actual_value
    
    def to_dict(self) -> dict:
        """Convert the exception to a dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "config_field": self.config_field,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value
        })
        return result


class ToolRateLimitError(ToolError):
    """
    Raised when rate limits are exceeded.
    
    This exception is raised when a tool exceeds the configured rate limits
    for API calls or other resource usage.
    
    Attributes:
        rate_limit: The configured rate limit
        current_usage: The current usage that exceeded the limit
        reset_time: When the rate limit will reset (if known)
    """
    
    def __init__(self, message: str, tool_name: str = None, context: dict = None,
                 rate_limit: int = None, current_usage: int = None, reset_time: float = None):
        super().__init__(message, tool_name, context)
        self.rate_limit = rate_limit
        self.current_usage = current_usage
        self.reset_time = reset_time
    
    def to_dict(self) -> dict:
        """Convert the exception to a dictionary for serialization."""
        result = super().to_dict()
        result.update({
            "rate_limit": self.rate_limit,
            "current_usage": self.current_usage,
            "reset_time": self.reset_time
        })
        return result