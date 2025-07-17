"""
Core data models for the agentic conversation system.

This module contains the fundamental data structures used throughout the system
for representing messages, conversation state, and related entities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class ConversationStatus(Enum):
    """Enumeration of possible conversation states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    Attributes:
        agent_id: Identifier of the agent that generated this message
        content: The actual message content
        timestamp: When the message was created
        token_count: Number of tokens in the message content
        metadata: Additional metadata associated with the message
    """
    agent_id: str
    content: str
    timestamp: datetime
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data["token_count"],
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create message from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ConversationState:
    """
    Represents the complete state of a conversation.
    
    Attributes:
        messages: List of all messages in the conversation
        current_turn: Current turn number (0-indexed)
        start_time: When the conversation started
        status: Current status of the conversation
        current_context_tokens: Current number of tokens in context
        max_context_tokens: Maximum allowed tokens in context
        metadata: Additional metadata for the conversation
    """
    messages: List[Message] = field(default_factory=list)
    current_turn: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    status: ConversationStatus = ConversationStatus.INITIALIZED
    current_context_tokens: int = 0
    max_context_tokens: int = 8000
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation and update state."""
        self.messages.append(message)
        self.current_context_tokens += message.token_count
        self.current_turn += 1
    
    def get_total_messages(self) -> int:
        """Get the total number of messages in the conversation."""
        return len(self.messages)
    
    def get_total_tokens(self) -> int:
        """Get the total number of tokens across all messages."""
        return sum(msg.token_count for msg in self.messages)
    
    def get_messages_by_agent(self, agent_id: str) -> List[Message]:
        """Get all messages from a specific agent."""
        return [msg for msg in self.messages if msg.agent_id == agent_id]
    
    def get_context_utilization(self) -> float:
        """Get the current context window utilization as a percentage."""
        if self.max_context_tokens == 0:
            return 0.0
        return (self.current_context_tokens / self.max_context_tokens) * 100
    
    def is_context_full(self, threshold: float = 90.0) -> bool:
        """Check if context window is approaching capacity."""
        return self.get_context_utilization() >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation state to dictionary for serialization."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "current_turn": self.current_turn,
            "start_time": self.start_time.isoformat(),
            "status": self.status.value,
            "current_context_tokens": self.current_context_tokens,
            "max_context_tokens": self.max_context_tokens,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """Create conversation state from dictionary."""
        messages = [Message.from_dict(msg_data) for msg_data in data.get("messages", [])]
        return cls(
            messages=messages,
            current_turn=data.get("current_turn", 0),
            start_time=datetime.fromisoformat(data["start_time"]),
            status=ConversationStatus(data.get("status", "initialized")),
            current_context_tokens=data.get("current_context_tokens", 0),
            max_context_tokens=data.get("max_context_tokens", 8000),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        """Convert conversation state to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConversationState":
        """Create conversation state from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class AgentMetrics:
    """
    Metrics for tracking agent performance and behavior.
    
    Attributes:
        response_time: Time taken to generate response in seconds
        token_count: Number of tokens in the agent's response
        model_calls: Number of API calls made to the language model
        errors: List of error messages encountered during processing
    """
    response_time: float = 0.0
    token_count: int = 0
    model_calls: int = 0
    errors: List[str] = field(default_factory=list)
    
    def add_error(self, error_message: str) -> None:
        """Add an error message to the metrics."""
        self.errors.append(error_message)
    
    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return len(self.errors) > 0
    
    def get_error_count(self) -> int:
        """Get the total number of errors."""
        return len(self.errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent metrics to dictionary for serialization."""
        return {
            "response_time": self.response_time,
            "token_count": self.token_count,
            "model_calls": self.model_calls,
            "errors": self.errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMetrics":
        """Create agent metrics from dictionary."""
        return cls(
            response_time=data.get("response_time", 0.0),
            token_count=data.get("token_count", 0),
            model_calls=data.get("model_calls", 0),
            errors=data.get("errors", [])
        )


@dataclass
class ContextWindowSnapshot:
    """
    Snapshot of context window utilization at a specific point in time.
    
    Attributes:
        turn_number: The conversation turn when this snapshot was taken
        total_tokens: Total number of tokens in the context window
        available_tokens: Number of tokens still available in the context window
        utilization_percentage: Percentage of context window currently used
        strategy_applied: Strategy applied for context management (if any)
    """
    turn_number: int
    total_tokens: int
    available_tokens: int
    utilization_percentage: float
    strategy_applied: Optional[str] = None
    
    def is_near_capacity(self, threshold: float = 90.0) -> bool:
        """Check if context window is near capacity."""
        return self.utilization_percentage >= threshold
    
    def get_used_tokens(self) -> int:
        """Calculate the number of tokens currently used."""
        return self.total_tokens - self.available_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context window snapshot to dictionary for serialization."""
        return {
            "turn_number": self.turn_number,
            "total_tokens": self.total_tokens,
            "available_tokens": self.available_tokens,
            "utilization_percentage": self.utilization_percentage,
            "strategy_applied": self.strategy_applied
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextWindowSnapshot":
        """Create context window snapshot from dictionary."""
        return cls(
            turn_number=data["turn_number"],
            total_tokens=data["total_tokens"],
            available_tokens=data["available_tokens"],
            utilization_percentage=data["utilization_percentage"],
            strategy_applied=data.get("strategy_applied")
        )


@dataclass
class RunTelemetry:
    """
    Complete telemetry data for a conversation run.
    
    Attributes:
        run_id: Unique identifier for this conversation run
        start_time: When the conversation run started
        end_time: When the conversation run ended (None if still running)
        total_turns: Total number of turns in the conversation
        agent_metrics: Dictionary mapping agent IDs to their metrics
        conversation_history: Complete list of messages in the conversation
        context_window_snapshots: List of context window snapshots throughout the run
        configuration: System configuration used for this run
        metadata: Additional metadata for the run
    """
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_turns: int = 0
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    conversation_history: List[Message] = field(default_factory=list)
    context_window_snapshots: List[ContextWindowSnapshot] = field(default_factory=list)
    configuration: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_duration(self) -> Optional[float]:
        """Get the duration of the run in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    def is_completed(self) -> bool:
        """Check if the run has completed."""
        return self.end_time is not None
    
    def add_agent_metrics(self, agent_id: str, metrics: AgentMetrics) -> None:
        """Add or update metrics for a specific agent."""
        if agent_id in self.agent_metrics:
            # Aggregate metrics if agent already exists
            existing = self.agent_metrics[agent_id]
            existing.response_time += metrics.response_time
            existing.token_count += metrics.token_count
            existing.model_calls += metrics.model_calls
            existing.errors.extend(metrics.errors)
        else:
            self.agent_metrics[agent_id] = metrics
    
    def add_context_snapshot(self, snapshot: ContextWindowSnapshot) -> None:
        """Add a context window snapshot."""
        self.context_window_snapshots.append(snapshot)
    
    def get_total_tokens_used(self) -> int:
        """Get the total number of tokens used across all agents."""
        return sum(metrics.token_count for metrics in self.agent_metrics.values())
    
    def get_total_model_calls(self) -> int:
        """Get the total number of model calls across all agents."""
        return sum(metrics.model_calls for metrics in self.agent_metrics.values())
    
    def get_total_errors(self) -> int:
        """Get the total number of errors across all agents."""
        return sum(len(metrics.errors) for metrics in self.agent_metrics.values())
    
    def get_average_response_time(self) -> float:
        """Get the average response time across all agents."""
        if not self.agent_metrics:
            return 0.0
        
        total_time = sum(metrics.response_time for metrics in self.agent_metrics.values())
        total_calls = sum(metrics.model_calls for metrics in self.agent_metrics.values())
        
        return total_time / total_calls if total_calls > 0 else 0.0
    
    def get_peak_context_utilization(self) -> float:
        """Get the peak context window utilization percentage."""
        if not self.context_window_snapshots:
            return 0.0
        return max(snapshot.utilization_percentage for snapshot in self.context_window_snapshots)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert run telemetry to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_turns": self.total_turns,
            "agent_metrics": {
                agent_id: metrics.to_dict() 
                for agent_id, metrics in self.agent_metrics.items()
            },
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "context_window_snapshots": [
                snapshot.to_dict() for snapshot in self.context_window_snapshots
            ],
            "configuration": self.configuration,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunTelemetry":
        """Create run telemetry from dictionary."""
        agent_metrics = {
            agent_id: AgentMetrics.from_dict(metrics_data)
            for agent_id, metrics_data in data.get("agent_metrics", {}).items()
        }
        
        conversation_history = [
            Message.from_dict(msg_data) 
            for msg_data in data.get("conversation_history", [])
        ]
        
        context_snapshots = [
            ContextWindowSnapshot.from_dict(snapshot_data)
            for snapshot_data in data.get("context_window_snapshots", [])
        ]
        
        return cls(
            run_id=data["run_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            total_turns=data.get("total_turns", 0),
            agent_metrics=agent_metrics,
            conversation_history=conversation_history,
            context_window_snapshots=context_snapshots,
            configuration=data.get("configuration"),
            metadata=data.get("metadata", {})
        )
    
    def to_json(self) -> str:
        """Convert run telemetry to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "RunTelemetry":
        """Create run telemetry from JSON string."""
        return cls.from_dict(json.loads(json_str))
@dataclass

class AgentConfig:
    """
    Configuration for an individual agent.
    
    Attributes:
        name: Human-readable name for the agent
        system_prompt: System prompt that defines the agent's role and behavior
    """
    name: str
    system_prompt: str
    
    def validate(self) -> List[str]:
        """Validate the agent configuration and return any error messages."""
        errors = []
        
        if not self.name or not self.name.strip():
            errors.append("Agent name cannot be empty")
        
        if not self.system_prompt or not self.system_prompt.strip():
            errors.append("Agent system prompt cannot be empty")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if the agent configuration is valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent config to dictionary for serialization."""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create agent config from dictionary."""
        return cls(
            name=data["name"],
            system_prompt=data["system_prompt"]
        )


@dataclass
class ModelConfig:
    """
    Configuration for the language model.
    
    Attributes:
        model_name: Name of the language model to use
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter (0.0 to 1.0)
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
    """
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def validate(self) -> List[str]:
        """Validate the model configuration and return any error messages."""
        errors = []
        
        if not self.model_name or not self.model_name.strip():
            errors.append("Model name cannot be empty")
        
        if not (0.0 <= self.temperature <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            errors.append("Max tokens must be greater than 0")
        
        if not (0.0 <= self.top_p <= 1.0):
            errors.append("Top-p must be between 0.0 and 1.0")
        
        if not (-2.0 <= self.frequency_penalty <= 2.0):
            errors.append("Frequency penalty must be between -2.0 and 2.0")
        
        if not (-2.0 <= self.presence_penalty <= 2.0):
            errors.append("Presence penalty must be between -2.0 and 2.0")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if the model configuration is valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model config to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create model config from dictionary."""
        return cls(
            model_name=data["model_name"],
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2000),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0)
        )


@dataclass
class ConversationConfig:
    """
    Configuration for conversation behavior.
    
    Attributes:
        max_turns: Maximum number of turns in the conversation
        initial_prompt: Optional initial prompt to start the conversation
        context_window_strategy: Strategy for managing context window ("truncate", "summarize", "sliding")
        context_window_size: Maximum number of tokens in the context window
        turn_timeout: Maximum time to wait for an agent response (seconds)
    """
    max_turns: int
    initial_prompt: Optional[str] = None
    context_window_strategy: str = "sliding"
    context_window_size: int = 8000
    turn_timeout: float = 30.0
    
    def validate(self) -> List[str]:
        """Validate the conversation configuration and return any error messages."""
        errors = []
        
        if self.max_turns <= 0:
            errors.append("Max turns must be greater than 0")
        
        valid_strategies = ["truncate", "summarize", "sliding"]
        if self.context_window_strategy not in valid_strategies:
            errors.append(f"Context window strategy must be one of: {', '.join(valid_strategies)}")
        
        if self.context_window_size <= 0:
            errors.append("Context window size must be greater than 0")
        
        if self.turn_timeout <= 0:
            errors.append("Turn timeout must be greater than 0")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if the conversation configuration is valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation config to dictionary for serialization."""
        return {
            "max_turns": self.max_turns,
            "initial_prompt": self.initial_prompt,
            "context_window_strategy": self.context_window_strategy,
            "context_window_size": self.context_window_size,
            "turn_timeout": self.turn_timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationConfig":
        """Create conversation config from dictionary."""
        return cls(
            max_turns=data["max_turns"],
            initial_prompt=data.get("initial_prompt"),
            context_window_strategy=data.get("context_window_strategy", "sliding"),
            context_window_size=data.get("context_window_size", 8000),
            turn_timeout=data.get("turn_timeout", 30.0)
        )


@dataclass
class LoggingConfig:
    """
    Configuration for logging and telemetry.
    
    Attributes:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        output_directory: Directory to save log files
        real_time_display: Whether to display conversation in real-time
        export_formats: List of export formats for logs ("json", "csv", "txt")
        save_conversation_history: Whether to save complete conversation history
        save_telemetry: Whether to save telemetry data
    """
    log_level: str = "INFO"
    output_directory: str = "./logs"
    real_time_display: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json"])
    save_conversation_history: bool = True
    save_telemetry: bool = True
    
    def validate(self) -> List[str]:
        """Validate the logging configuration and return any error messages."""
        errors = []
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of: {', '.join(valid_log_levels)}")
        
        if not self.output_directory or not self.output_directory.strip():
            errors.append("Output directory cannot be empty")
        
        valid_formats = ["json", "csv", "txt"]
        for format_type in self.export_formats:
            if format_type not in valid_formats:
                errors.append(f"Export format '{format_type}' is not valid. Must be one of: {', '.join(valid_formats)}")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if the logging configuration is valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert logging config to dictionary for serialization."""
        return {
            "log_level": self.log_level,
            "output_directory": self.output_directory,
            "real_time_display": self.real_time_display,
            "export_formats": self.export_formats,
            "save_conversation_history": self.save_conversation_history,
            "save_telemetry": self.save_telemetry
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoggingConfig":
        """Create logging config from dictionary."""
        return cls(
            log_level=data.get("log_level", "INFO"),
            output_directory=data.get("output_directory", "./logs"),
            real_time_display=data.get("real_time_display", True),
            export_formats=data.get("export_formats", ["json"]),
            save_conversation_history=data.get("save_conversation_history", True),
            save_telemetry=data.get("save_telemetry", True)
        )


@dataclass
class SystemConfig:
    """
    Complete system configuration combining all configuration sections.
    
    Attributes:
        agent_a: Configuration for the first agent
        agent_b: Configuration for the second agent
        model: Language model configuration
        conversation: Conversation behavior configuration
        logging: Logging and telemetry configuration
    """
    agent_a: AgentConfig
    agent_b: AgentConfig
    model: ModelConfig
    conversation: ConversationConfig
    logging: LoggingConfig
    
    def validate(self) -> List[str]:
        """Validate the complete system configuration and return any error messages."""
        errors = []
        
        # Validate each configuration section
        errors.extend([f"Agent A: {error}" for error in self.agent_a.validate()])
        errors.extend([f"Agent B: {error}" for error in self.agent_b.validate()])
        errors.extend([f"Model: {error}" for error in self.model.validate()])
        errors.extend([f"Conversation: {error}" for error in self.conversation.validate()])
        errors.extend([f"Logging: {error}" for error in self.logging.validate()])
        
        # Cross-validation checks
        if self.agent_a.name == self.agent_b.name:
            errors.append("Agent A and Agent B must have different names")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if the complete system configuration is valid."""
        return len(self.validate()) == 0
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a detailed validation summary."""
        errors = self.validate()
        return {
            "is_valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": errors,
            "sections": {
                "agent_a": self.agent_a.is_valid(),
                "agent_b": self.agent_b.is_valid(),
                "model": self.model.is_valid(),
                "conversation": self.conversation.is_valid(),
                "logging": self.logging.is_valid()
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system config to dictionary for serialization."""
        return {
            "agents": {
                "agent_a": self.agent_a.to_dict(),
                "agent_b": self.agent_b.to_dict()
            },
            "model": self.model.to_dict(),
            "conversation": self.conversation.to_dict(),
            "logging": self.logging.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        """Create system config from dictionary."""
        agents_data = data.get("agents", {})
        
        return cls(
            agent_a=AgentConfig.from_dict(agents_data.get("agent_a", {})),
            agent_b=AgentConfig.from_dict(agents_data.get("agent_b", {})),
            model=ModelConfig.from_dict(data.get("model", {})),
            conversation=ConversationConfig.from_dict(data.get("conversation", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {}))
        )
    
    def to_json(self) -> str:
        """Convert system config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SystemConfig":
        """Create system config from JSON string."""
        return cls.from_dict(json.loads(json_str))