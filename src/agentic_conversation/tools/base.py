"""
Base classes and data models for the tool system.

This module defines the core interfaces and data structures that all tools
must implement, providing a standardized way for agents to interact with tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from ..models import Message


class ToolCapability(Enum):
    """Enumeration of tool capabilities."""
    SEARCH = "search"
    FILE_OPERATIONS = "file_operations"
    CALCULATION = "calculation"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"


@dataclass
class ToolResult:
    """
    Standardized tool execution result.
    
    This class represents the result of executing a tool, providing consistent
    structure for success/failure status, content, metadata, and performance metrics.
    
    Attributes:
        success: Whether the tool execution was successful
        content: The main result content from the tool
        metadata: Additional metadata about the execution
        execution_time: Time taken to execute the tool in seconds
        token_count: Number of tokens in the result content
        errors: List of error messages encountered during execution
        tool_name: Name of the tool that produced this result
        timestamp: When the result was generated
    """
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    token_count: int = 0
    errors: List[str] = field(default_factory=list)
    tool_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_error(self, error_message: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error_message)
        self.success = False
    
    def has_errors(self) -> bool:
        """Check if the result contains any errors."""
        return len(self.errors) > 0
    
    def get_summary(self) -> str:
        """Get a brief summary of the result."""
        status = "SUCCESS" if self.success else "FAILED"
        return f"[{self.tool_name}] {status} - {len(self.content)} chars, {self.execution_time:.2f}s"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization."""
        return {
            "success": self.success,
            "content": self.content,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "token_count": self.token_count,
            "errors": self.errors,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create a ToolResult from a dictionary."""
        return cls(
            success=data["success"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            execution_time=data.get("execution_time", 0.0),
            token_count=data.get("token_count", 0),
            errors=data.get("errors", []),
            tool_name=data.get("tool_name", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )


@dataclass
class ToolContext:
    """
    Context provided to tools for execution.
    
    This class provides tools with the necessary context about the current
    conversation state, allowing them to make informed decisions about
    how to execute and what information to return.
    
    Attributes:
        conversation_history: List of recent messages in the conversation
        current_turn: Current turn number in the conversation
        agent_id: ID of the agent requesting tool execution
        available_tokens: Number of tokens available in the context window
        metadata: Additional context metadata
        max_result_tokens: Maximum tokens the tool result should consume
        timeout: Maximum execution time allowed for the tool
    """
    conversation_history: List[Message] = field(default_factory=list)
    current_turn: int = 0
    agent_id: str = ""
    available_tokens: int = 8000
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_result_tokens: int = 1000
    timeout: float = 30.0
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get the most recent messages from the conversation history."""
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def get_messages_by_agent(self, agent_id: str) -> List[Message]:
        """Get all messages from a specific agent."""
        return [msg for msg in self.conversation_history if msg.agent_id == agent_id]
    
    def get_conversation_summary(self) -> str:
        """Get a brief summary of the conversation context."""
        if not self.conversation_history:
            return "No conversation history available"
        
        recent_messages = self.get_recent_messages(3)
        summary_parts = []
        
        for msg in recent_messages:
            if len(msg.content) > 100:
                content_preview = msg.content[:97] + "..."
            else:
                content_preview = msg.content
            summary_parts.append(f"{msg.agent_id}: {content_preview}")
        
        return " | ".join(summary_parts)
    
    def has_sufficient_tokens(self, required_tokens: int) -> bool:
        """Check if there are sufficient tokens available for the tool result."""
        return self.available_tokens >= required_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the context to a dictionary for serialization."""
        return {
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "current_turn": self.current_turn,
            "agent_id": self.agent_id,
            "available_tokens": self.available_tokens,
            "metadata": self.metadata,
            "max_result_tokens": self.max_result_tokens,
            "timeout": self.timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolContext":
        """Create a ToolContext from a dictionary."""
        conversation_history = [
            Message.from_dict(msg_data) 
            for msg_data in data.get("conversation_history", [])
        ]
        
        return cls(
            conversation_history=conversation_history,
            current_turn=data.get("current_turn", 0),
            agent_id=data.get("agent_id", ""),
            available_tokens=data.get("available_tokens", 8000),
            metadata=data.get("metadata", {}),
            max_result_tokens=data.get("max_result_tokens", 1000),
            timeout=data.get("timeout", 30.0)
        )


@dataclass
class ToolInfo:
    """
    Metadata about a tool's capabilities and configuration.
    
    This class provides information about what a tool can do, its requirements,
    and how it should be used by agents.
    
    Attributes:
        name: Unique name identifier for the tool
        description: Human-readable description of what the tool does
        capabilities: List of capabilities this tool provides
        required_config: List of required configuration parameters
        optional_config: List of optional configuration parameters
        version: Version of the tool implementation
        author: Author or maintainer of the tool
        max_execution_time: Maximum expected execution time in seconds
        token_cost_estimate: Estimated token cost for typical usage
    """
    name: str
    description: str
    capabilities: List[ToolCapability] = field(default_factory=list)
    required_config: List[str] = field(default_factory=list)
    optional_config: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    max_execution_time: float = 30.0
    token_cost_estimate: int = 100
    
    def has_capability(self, capability: ToolCapability) -> bool:
        """Check if the tool has a specific capability."""
        return capability in self.capabilities
    
    def get_config_requirements(self) -> Dict[str, bool]:
        """Get a mapping of configuration parameters and whether they're required."""
        config_map = {}
        for param in self.required_config:
            config_map[param] = True
        for param in self.optional_config:
            config_map[param] = False
        return config_map
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration against the tool's requirements."""
        errors = []
        
        # Check for missing required configuration
        for required_param in self.required_config:
            if required_param not in config:
                errors.append(f"Missing required configuration parameter: {required_param}")
            elif config[required_param] is None:
                errors.append(f"Required configuration parameter cannot be None: {required_param}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tool info to a dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "required_config": self.required_config,
            "optional_config": self.optional_config,
            "version": self.version,
            "author": self.author,
            "max_execution_time": self.max_execution_time,
            "token_cost_estimate": self.token_cost_estimate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolInfo":
        """Create a ToolInfo from a dictionary."""
        capabilities = [
            ToolCapability(cap) for cap in data.get("capabilities", [])
        ]
        
        return cls(
            name=data["name"],
            description=data["description"],
            capabilities=capabilities,
            required_config=data.get("required_config", []),
            optional_config=data.get("optional_config", []),
            version=data.get("version", "1.0.0"),
            author=data.get("author", ""),
            max_execution_time=data.get("max_execution_time", 30.0),
            token_cost_estimate=data.get("token_cost_estimate", 100)
        )


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    
    This class defines the standard interface that all tools must implement.
    It provides the contract for tool execution, relevance evaluation,
    and metadata retrieval.
    
    All concrete tool implementations must inherit from this class and
    implement the abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the tool with configuration.
        
        Args:
            config: Configuration dictionary for the tool
        """
        self.config = config or {}
        self._tool_info = None
        self._validate_config()
    
    @abstractmethod
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        """
        Execute the tool with the given query and context.
        
        This is the main method that performs the tool's functionality.
        It should be implemented by each concrete tool class.
        
        Args:
            query: The query or input for the tool
            context: The conversation context for the tool execution
            
        Returns:
            ToolResult: The result of the tool execution
            
        Raises:
            ToolExecutionError: If the tool execution fails
            ToolTimeoutError: If the tool execution times out
        """
        pass
    
    @abstractmethod
    def is_relevant(self, context: ToolContext) -> bool:
        """
        Determine if this tool is relevant for the current context.
        
        This method allows tools to evaluate whether they should be used
        based on the conversation context, helping agents make intelligent
        decisions about tool selection.
        
        Args:
            context: The conversation context to evaluate
            
        Returns:
            bool: True if the tool is relevant, False otherwise
        """
        pass
    
    @abstractmethod
    def get_tool_info(self) -> ToolInfo:
        """
        Get metadata about this tool.
        
        This method returns information about the tool's capabilities,
        configuration requirements, and other metadata.
        
        Returns:
            ToolInfo: Metadata about the tool
        """
        pass
    
    def validate_input(self, query: str, context: ToolContext) -> List[str]:
        """
        Validate the input parameters for tool execution.
        
        This method can be overridden by concrete tools to provide
        specific input validation logic.
        
        Args:
            query: The query to validate
            context: The context to validate
            
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        if not query or not query.strip():
            errors.append("Query cannot be empty")
        
        if not context:
            errors.append("Context cannot be None")
        
        if context and context.timeout <= 0:
            errors.append("Timeout must be greater than 0")
        
        return errors
    
    def _validate_config(self) -> None:
        """
        Validate the tool configuration.
        
        This method is called during initialization to ensure the tool
        is properly configured.
        
        Raises:
            ToolConfigurationError: If the configuration is invalid
        """
        try:
            tool_info = self.get_tool_info()
            if tool_info is None:
                return
            
            errors = tool_info.validate_config(self.config)
            
            if errors:
                from .exceptions import ToolConfigurationError
                raise ToolConfigurationError(
                    f"Invalid configuration for tool {tool_info.name}: {'; '.join(errors)}",
                    tool_name=tool_info.name,
                    context={"config": self.config, "errors": errors}
                )
        except NotImplementedError:
            # get_tool_info() not implemented yet, skip validation
            return
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.
        
        Args:
            key: The configuration key to retrieve
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        return self.config.get(key, default)
    
    def is_configured(self) -> bool:
        """
        Check if the tool is properly configured.
        
        Returns:
            bool: True if the tool is configured, False otherwise
        """
        try:
            self._validate_config()
            return True
        except Exception:
            return False
    
    def __str__(self) -> str:
        """Return a string representation of the tool."""
        tool_info = self.get_tool_info()
        return f"{tool_info.name} v{tool_info.version}"
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the tool."""
        tool_info = self.get_tool_info()
        return f"<{self.__class__.__name__}(name='{tool_info.name}', version='{tool_info.version}')>"