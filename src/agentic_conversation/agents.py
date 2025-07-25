"""
Agent interface and implementations for the agentic conversation system.

This module defines the abstract base agent interface and concrete implementations
for different types of agents that can participate in conversations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import logging
import time

from .models import Message, ConversationState, AgentMetrics


@dataclass
class AgentInfo:
    """
    Information about an agent's capabilities and configuration.
    
    Attributes:
        agent_id: Unique identifier for the agent
        name: Human-readable name for the agent
        description: Description of the agent's role or purpose
        model_name: Name of the underlying language model
        capabilities: List of capabilities this agent supports
        metadata: Additional metadata about the agent
    """
    agent_id: str
    name: str
    description: str
    model_name: str
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent info to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "model_name": self.model_name,
            "capabilities": self.capabilities,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentInfo":
        """Create agent info from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"],
            model_name=data["model_name"],
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentResponse:
    """
    Response generated by an agent.
    
    Attributes:
        content: The actual response content
        agent_id: ID of the agent that generated this response
        timestamp: When the response was generated
        token_count: Number of tokens in the response
        response_time: Time taken to generate the response (seconds)
        model_calls: Number of API calls made to generate this response
        confidence: Confidence score for the response (0.0 to 1.0)
        reasoning: Optional reasoning or explanation for the response
        metadata: Additional metadata about the response
        errors: List of any errors encountered during generation
    """
    content: str
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    response_time: float = 0.0
    model_calls: int = 1
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_message(self) -> Message:
        """Convert agent response to a Message object."""
        return Message(
            agent_id=self.agent_id,
            content=self.content,
            timestamp=self.timestamp,
            token_count=self.token_count,
            metadata={
                "response_time": self.response_time,
                "model_calls": self.model_calls,
                "confidence": self.confidence,
                "reasoning": self.reasoning,
                "errors": self.errors,
                **self.metadata
            }
        )
    
    def to_agent_metrics(self) -> AgentMetrics:
        """Convert agent response to AgentMetrics object."""
        return AgentMetrics(
            response_time=self.response_time,
            token_count=self.token_count,
            model_calls=self.model_calls,
            errors=self.errors.copy()
        )
    
    def has_errors(self) -> bool:
        """Check if the response has any errors."""
        return len(self.errors) > 0
    
    def add_error(self, error_message: str) -> None:
        """Add an error message to the response."""
        self.errors.append(error_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent response to dictionary for serialization."""
        return {
            "content": self.content,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "response_time": self.response_time,
            "model_calls": self.model_calls,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "errors": self.errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentResponse":
        """Create agent response from dictionary."""
        return cls(
            content=data["content"],
            agent_id=data["agent_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data.get("token_count", 0),
            response_time=data.get("response_time", 0.0),
            model_calls=data.get("model_calls", 1),
            confidence=data.get("confidence"),
            reasoning=data.get("reasoning"),
            metadata=data.get("metadata", {}),
            errors=data.get("errors", [])
        )


@dataclass
class ConversationContext:
    """
    Context provided to agents for generating responses.
    
    Attributes:
        conversation_state: Current state of the conversation
        system_prompt: System prompt for the agent
        available_tokens: Number of tokens available for the response
        turn_number: Current turn number in the conversation
        other_agent_id: ID of the other agent in the conversation
        metadata: Additional context metadata
    """
    conversation_state: ConversationState
    system_prompt: str
    available_tokens: int
    turn_number: int
    other_agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_conversation_history(self) -> List[Message]:
        """Get the complete conversation history."""
        return self.conversation_state.messages
    
    def get_recent_messages(self, count: int) -> List[Message]:
        """Get the most recent N messages from the conversation."""
        return self.conversation_state.messages[-count:] if count > 0 else []
    
    def get_messages_by_agent(self, agent_id: str) -> List[Message]:
        """Get all messages from a specific agent."""
        return self.conversation_state.get_messages_by_agent(agent_id)
    
    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the conversation."""
        messages = self.conversation_state.messages
        return messages[-1] if messages else None
    
    def get_context_utilization(self) -> float:
        """Get the current context window utilization percentage."""
        return self.conversation_state.get_context_utilization()
    
    def is_context_near_limit(self, threshold: float = 90.0) -> bool:
        """Check if context window is approaching capacity."""
        return self.conversation_state.is_context_full(threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation context to dictionary for serialization."""
        return {
            "conversation_state": self.conversation_state.to_dict(),
            "system_prompt": self.system_prompt,
            "available_tokens": self.available_tokens,
            "turn_number": self.turn_number,
            "other_agent_id": self.other_agent_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create conversation context from dictionary."""
        return cls(
            conversation_state=ConversationState.from_dict(data["conversation_state"]),
            system_prompt=data["system_prompt"],
            available_tokens=data["available_tokens"],
            turn_number=data["turn_number"],
            other_agent_id=data.get("other_agent_id"),
            metadata=data.get("metadata", {})
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the conversation system.
    
    This class defines the interface that all agent implementations must follow.
    Concrete implementations should inherit from this class and implement the
    abstract methods.
    """
    
    def __init__(self, agent_id: str, name: str, system_prompt: str):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for the agent
            system_prompt: System prompt that defines the agent's behavior
        """
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate required parameters
        if not agent_id or not agent_id.strip():
            raise ValueError("Agent ID cannot be empty")
        if not name or not name.strip():
            raise ValueError("Agent name cannot be empty")
        if not system_prompt or not system_prompt.strip():
            raise ValueError("System prompt cannot be empty")
    
    @abstractmethod
    async def generate_response(self, context: ConversationContext) -> AgentResponse:
        """
        Generate a response based on the conversation context.
        
        This method must be implemented by concrete agent classes to provide
        the core functionality of generating responses to conversation inputs.
        
        Args:
            context: The conversation context containing history and metadata
            
        Returns:
            AgentResponse: The generated response with metadata
            
        Raises:
            NotImplementedError: If not implemented by concrete class
            AgentError: If response generation fails
        """
        pass
    
    @abstractmethod
    def get_agent_info(self) -> AgentInfo:
        """
        Get information about this agent's capabilities and configuration.
        
        Returns:
            AgentInfo: Information about the agent
        """
        pass
    
    def validate_context(self, context: ConversationContext) -> List[str]:
        """
        Validate the conversation context for this agent.
        
        Args:
            context: The conversation context to validate
            
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        if context.available_tokens <= 0:
            errors.append("Available tokens must be greater than 0")
        
        if context.turn_number < 0:
            errors.append("Turn number cannot be negative")
        
        if not context.system_prompt or not context.system_prompt.strip():
            errors.append("System prompt cannot be empty")
        
        return errors
    
    def is_context_valid(self, context: ConversationContext) -> bool:
        """
        Check if the conversation context is valid for this agent.
        
        Args:
            context: The conversation context to validate
            
        Returns:
            bool: True if context is valid, False otherwise
        """
        return len(self.validate_context(context)) == 0
    
    async def prepare_context(self, context: ConversationContext) -> ConversationContext:
        """
        Prepare the conversation context before generating a response.
        
        This method can be overridden by concrete implementations to perform
        any necessary preprocessing of the context.
        
        Args:
            context: The original conversation context
            
        Returns:
            ConversationContext: The prepared context
        """
        # Default implementation returns context unchanged
        return context
    
    async def post_process_response(self, response: AgentResponse, context: ConversationContext) -> AgentResponse:
        """
        Post-process the generated response.
        
        This method can be overridden by concrete implementations to perform
        any necessary post-processing of the response.
        
        Args:
            response: The generated response
            context: The conversation context used to generate the response
            
        Returns:
            AgentResponse: The post-processed response
        """
        # Default implementation returns response unchanged
        return response
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(id='{self.agent_id}', name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"{self.__class__.__name__}("
                f"agent_id='{self.agent_id}', "
                f"name='{self.name}', "
                f"system_prompt='{self.system_prompt[:50]}...')")


class AgentError(Exception):
    """
    Base exception class for agent-related errors.
    
    Attributes:
        message: Error message
        agent_id: ID of the agent that encountered the error
        context: Optional conversation context when error occurred
        original_error: Original exception that caused this error (if any)
    """
    
    def __init__(
        self, 
        message: str, 
        agent_id: Optional[str] = None,
        context: Optional[ConversationContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.agent_id = agent_id
        self.context = context
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "original_error": str(self.original_error) if self.original_error else None
        }


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""
    pass


class AgentValidationError(AgentError):
    """Raised when agent input validation fails."""
    pass


class AgentResponseError(AgentError):
    """Raised when agent response generation fails."""
    pass


class AgentConfigurationError(AgentError):
    """Raised when agent configuration is invalid."""
    pass