# API Reference

This document provides comprehensive API documentation for all public classes and methods in the Agentic Conversation System.

## Table of Contents

- [Configuration Management](#configuration-management)
- [Data Models](#data-models)
- [Agent Framework](#agent-framework)
- [Orchestration](#orchestration)
- [Context Management](#context-management)
- [Telemetry and Logging](#telemetry-and-logging)
- [Circuit Breaker](#circuit-breaker)
- [Utility Functions](#utility-functions)

## Configuration Management

### ConfigurationLoader

Main class for loading and validating YAML configuration files.

```python
from agentic_conversation import ConfigurationLoader
```

#### Methods

##### `__init__()`
Initialize the configuration loader.

##### `load_from_file(config_path: Union[str, Path]) -> SystemConfig`
Load configuration from a YAML file.

**Parameters:**
- `config_path`: Path to the YAML configuration file

**Returns:**
- `SystemConfig`: Validated system configuration

**Raises:**
- `ConfigurationError`: If the file cannot be loaded or configuration is invalid

**Example:**
```python
loader = ConfigurationLoader()
config = loader.load_from_file("config.yaml")
```

##### `load_from_dict(config_dict: Dict[str, Any]) -> SystemConfig`
Load configuration from a dictionary.

**Parameters:**
- `config_dict`: Configuration dictionary

**Returns:**
- `SystemConfig`: Validated system configuration

**Raises:**
- `ConfigurationError`: If configuration is invalid

##### `validate_config_file(config_path: Union[str, Path]) -> Dict[str, Any]`
Validate a configuration file without creating a SystemConfig object.

**Parameters:**
- `config_path`: Path to the YAML configuration file

**Returns:**
- Dictionary with validation results containing:
  - `is_valid`: bool
  - `errors`: List[str]
  - `warnings`: List[str]

### Utility Functions

##### `load_config(config_path: Union[str, Path]) -> SystemConfig`
Convenience function to load configuration from a file.

##### `validate_config(config_path: Union[str, Path]) -> Dict[str, Any]`
Convenience function to validate a configuration file.

## Data Models

### Message

Represents a single message in a conversation.

```python
from agentic_conversation import Message
```

#### Attributes

- `agent_id: str` - Identifier of the agent that generated this message
- `content: str` - The actual message content
- `timestamp: datetime` - When the message was created
- `token_count: int` - Number of tokens in the message content
- `metadata: Dict[str, Any]` - Additional metadata associated with the message

#### Methods

##### `to_dict() -> Dict[str, Any]`
Convert message to dictionary for serialization.

##### `from_dict(data: Dict[str, Any]) -> Message` (classmethod)
Create message from dictionary.

##### `to_json() -> str`
Convert message to JSON string.

##### `from_json(json_str: str) -> Message` (classmethod)
Create message from JSON string.

### ConversationState

Represents the complete state of a conversation.

```python
from agentic_conversation import ConversationState
```

#### Attributes

- `messages: List[Message]` - List of all messages in the conversation
- `current_turn: int` - Current turn number (0-indexed)
- `start_time: datetime` - When the conversation started
- `status: ConversationStatus` - Current status of the conversation
- `current_context_tokens: int` - Current number of tokens in context
- `max_context_tokens: int` - Maximum allowed tokens in context
- `metadata: Dict[str, Any]` - Additional metadata for the conversation

#### Methods

##### `add_message(message: Message) -> None`
Add a message to the conversation and update state.

##### `get_total_messages() -> int`
Get the total number of messages in the conversation.

##### `get_total_tokens() -> int`
Get the total number of tokens across all messages.

##### `get_messages_by_agent(agent_id: str) -> List[Message]`
Get all messages from a specific agent.

##### `get_context_utilization() -> float`
Get the current context window utilization as a percentage.

##### `is_context_full(threshold: float = 90.0) -> bool`
Check if context window is approaching capacity.

### Configuration Models

#### SystemConfig

Complete system configuration combining all configuration sections.

```python
from agentic_conversation import SystemConfig
```

**Attributes:**
- `agent_a: AgentConfig` - Configuration for the first agent
- `agent_b: AgentConfig` - Configuration for the second agent
- `model: ModelConfig` - Language model configuration
- `conversation: ConversationConfig` - Conversation behavior configuration
- `logging: LoggingConfig` - Logging and telemetry configuration

**Methods:**
- `validate() -> List[str]` - Validate the complete system configuration
- `is_valid() -> bool` - Check if the complete system configuration is valid
- `get_validation_summary() -> Dict[str, Any]` - Get a detailed validation summary

#### AgentConfig

Configuration for an individual agent.

**Attributes:**
- `name: str` - Human-readable name for the agent
- `system_prompt: str` - System prompt that defines the agent's role and behavior

#### ModelConfig

Configuration for the language model.

**Attributes:**
- `model_name: str` - Name of the language model to use
- `temperature: float` - Sampling temperature (0.0 to 2.0)
- `max_tokens: int` - Maximum number of tokens to generate
- `top_p: float` - Nucleus sampling parameter (0.0 to 1.0)
- `frequency_penalty: float` - Frequency penalty (-2.0 to 2.0)
- `presence_penalty: float` - Presence penalty (-2.0 to 2.0)

#### ConversationConfig

Configuration for conversation behavior.

**Attributes:**
- `max_turns: int` - Maximum number of turns in the conversation
- `initial_prompt: Optional[str]` - Optional initial prompt to start the conversation
- `context_window_strategy: str` - Strategy for managing context window
- `context_window_size: int` - Maximum number of tokens in the context window
- `turn_timeout: float` - Maximum time to wait for an agent response (seconds)

#### LoggingConfig

Configuration for logging and telemetry.

**Attributes:**
- `log_level: str` - Logging level
- `output_directory: str` - Directory to save log files
- `real_time_display: bool` - Whether to display conversation in real-time
- `export_formats: List[str]` - List of export formats for logs
- `save_conversation_history: bool` - Whether to save complete conversation history
- `save_telemetry: bool` - Whether to save telemetry data

## Agent Framework

### BaseAgent

Abstract base class for all agent implementations.

```python
from agentic_conversation import BaseAgent
```

#### Abstract Methods

##### `async generate_response(conversation_history: List[Message], context: ConversationContext) -> AgentResponse`
Generate a response based on conversation history and context.

**Parameters:**
- `conversation_history`: List of previous messages in the conversation
- `context`: Current conversation context

**Returns:**
- `AgentResponse`: The agent's response

##### `get_agent_info() -> AgentInfo`
Get information about this agent.

**Returns:**
- `AgentInfo`: Agent information including ID, name, and description

### LangChainAgent

Concrete implementation of BaseAgent using LangChain.

```python
from agentic_conversation import LangChainAgent
```

#### Constructor

##### `__init__(agent_id: str, config: AgentConfig, model_config: ModelConfig, circuit_breaker: Optional[CircuitBreaker] = None)`

**Parameters:**
- `agent_id`: Unique identifier for this agent
- `config`: Agent configuration
- `model_config`: Model configuration
- `circuit_breaker`: Optional circuit breaker for resilience

#### Methods

##### `async generate_response(conversation_history: List[Message], context: ConversationContext) -> AgentResponse`
Generate a response using LangChain LLM integration.

##### `get_agent_info() -> AgentInfo`
Get agent information.

### Supporting Classes

#### AgentResponse

Response from an agent.

**Attributes:**
- `content: str` - Response content
- `agent_id: str` - ID of the responding agent
- `metadata: Dict[str, Any]` - Additional response metadata

#### AgentInfo

Information about an agent.

**Attributes:**
- `agent_id: str` - Unique agent identifier
- `name: str` - Human-readable agent name
- `description: str` - Agent description

#### ConversationContext

Context provided to agents for response generation.

**Attributes:**
- `current_turn: int` - Current conversation turn
- `total_tokens: int` - Total tokens in conversation
- `context_utilization: float` - Context window utilization percentage

### Exception Classes

#### AgentError
Base exception for agent-related errors.

#### AgentTimeoutError
Raised when agent response times out.

#### AgentValidationError
Raised when agent input validation fails.

#### AgentResponseError
Raised when agent response is invalid.

#### AgentConfigurationError
Raised when agent configuration is invalid.

## Orchestration

### ConversationOrchestrator

Main orchestrator for managing conversations between agents.

```python
from agentic_conversation import ConversationOrchestrator
```

#### Constructor

##### `__init__(config: Optional[SystemConfig] = None, config_path: Optional[Union[str, Path]] = None)`

**Parameters:**
- `config`: System configuration object
- `config_path`: Path to configuration file (alternative to config)

#### Methods

##### `async run_conversation(conversation_id: Optional[str] = None, display_progress: bool = True, save_results: bool = True) -> Dict[str, Any]`
Run a complete conversation between the configured agents.

**Parameters:**
- `conversation_id`: Optional unique identifier for this conversation
- `display_progress`: Whether to display real-time progress
- `save_results`: Whether to save conversation results

**Returns:**
- Dictionary containing conversation results:
  - `conversation_id`: str
  - `status`: str
  - `total_turns`: int
  - `total_tokens`: int
  - `duration_seconds`: float

##### `get_conversation_state() -> Optional[ConversationState]`
Get the current conversation state.

##### `get_telemetry() -> Optional[RunTelemetry]`
Get telemetry data for the current/last conversation.

### Utility Functions

##### `create_orchestrator_from_config(config: SystemConfig) -> ConversationOrchestrator`
Create orchestrator from a configuration object.

##### `create_orchestrator_from_file(config_path: Union[str, Path]) -> ConversationOrchestrator`
Create orchestrator from a configuration file.

##### `async run_single_conversation(config: SystemConfig, **kwargs) -> Dict[str, Any]`
Run a single conversation with the given configuration.

## Context Management

### ContextManager

Manages conversation context and handles context window limitations.

```python
from agentic_conversation import ContextManager
```

#### Constructor

##### `__init__(strategy: ContextStrategy, max_tokens: int, token_counter: TokenCounter)`

**Parameters:**
- `strategy`: Context management strategy
- `max_tokens`: Maximum tokens allowed in context
- `token_counter`: Token counter instance

#### Methods

##### `manage_context(conversation_state: ConversationState) -> ContextManagementResult`
Apply context management strategy to conversation state.

**Parameters:**
- `conversation_state`: Current conversation state

**Returns:**
- `ContextManagementResult`: Result of context management operation

##### `get_context_for_agent(conversation_state: ConversationState, agent_id: str) -> List[Message]`
Get appropriate context messages for a specific agent.

### TokenCounter

Handles token counting for different model types.

```python
from agentic_conversation import TokenCounter
```

#### Constructor

##### `__init__(model_type: ModelType)`

**Parameters:**
- `model_type`: Type of model for token counting

#### Methods

##### `count_tokens(text: str) -> TokenCountResult`
Count tokens in the given text.

**Parameters:**
- `text`: Text to count tokens for

**Returns:**
- `TokenCountResult`: Token count result

##### `count_message_tokens(message: Message) -> int`
Count tokens in a message.

##### `count_conversation_tokens(messages: List[Message]) -> int`
Count total tokens in a list of messages.

### Enums and Supporting Classes

#### ContextStrategy

Enumeration of context management strategies:
- `TRUNCATE` - Remove oldest messages when limit exceeded
- `SLIDING` - Maintain sliding window of recent messages
- `SUMMARIZE` - Create summaries of older content (planned)

#### ModelType

Enumeration of supported model types:
- `GPT_3_5_TURBO`
- `GPT_4`
- `GPT_4_TURBO`
- `CLAUDE_3_OPUS`
- `CLAUDE_3_SONNET`

#### ContextManagementResult

Result of context management operation.

**Attributes:**
- `messages: List[Message]` - Resulting messages after context management
- `tokens_removed: int` - Number of tokens removed
- `strategy_applied: str` - Strategy that was applied

#### TokenCountResult

Result of token counting operation.

**Attributes:**
- `token_count: int` - Number of tokens
- `model_type: ModelType` - Model type used for counting

## Telemetry and Logging

### TelemetryLogger

Handles telemetry data collection and logging.

```python
from agentic_conversation import TelemetryLogger
```

#### Constructor

##### `__init__(config: LoggingConfig)`

**Parameters:**
- `config`: Logging configuration

#### Methods

##### `start_run(run_id: str, configuration: SystemConfig) -> None`
Start logging for a new conversation run.

##### `log_agent_interaction(agent_id: str, request_data: Dict[str, Any], response_data: Dict[str, Any], metrics: AgentMetrics) -> None`
Log an agent interaction.

##### `log_context_snapshot(snapshot: ContextWindowSnapshot) -> None`
Log a context window snapshot.

##### `end_run(final_state: ConversationState) -> RunTelemetry`
End the current run and return telemetry data.

### RunLogger

Manages individual conversation run logs.

```python
from agentic_conversation import RunLogger
```

#### Constructor

##### `__init__(output_directory: str, export_formats: List[str])`

**Parameters:**
- `output_directory`: Directory to save log files
- `export_formats`: List of export formats

#### Methods

##### `save_run_data(telemetry: RunTelemetry) -> Dict[str, str]`
Save run data to files.

**Returns:**
- Dictionary mapping format to file path

##### `load_run_data(run_id: str) -> Optional[RunTelemetry]`
Load run data from files.

### Telemetry Data Models

#### AgentMetrics

Metrics for tracking agent performance.

**Attributes:**
- `response_time: float` - Time taken to generate response
- `token_count: int` - Number of tokens in response
- `model_calls: int` - Number of API calls made
- `errors: List[str]` - List of error messages

#### ContextWindowSnapshot

Snapshot of context window utilization.

**Attributes:**
- `turn_number: int` - Conversation turn when snapshot was taken
- `total_tokens: int` - Total tokens in context window
- `available_tokens: int` - Available tokens remaining
- `utilization_percentage: float` - Percentage of context window used
- `strategy_applied: Optional[str]` - Strategy applied for context management

#### RunTelemetry

Complete telemetry data for a conversation run.

**Attributes:**
- `run_id: str` - Unique identifier for the run
- `start_time: datetime` - When the run started
- `end_time: Optional[datetime]` - When the run ended
- `total_turns: int` - Total number of turns
- `agent_metrics: Dict[str, AgentMetrics]` - Metrics for each agent
- `conversation_history: List[Message]` - Complete conversation history
- `context_window_snapshots: List[ContextWindowSnapshot]` - Context snapshots
- `configuration: Optional[Dict[str, Any]]` - System configuration used
- `metadata: Dict[str, Any]` - Additional metadata

## Circuit Breaker

### CircuitBreaker

Implements circuit breaker pattern for API resilience.

```python
from agentic_conversation import CircuitBreaker
```

#### Constructor

##### `__init__(config: CircuitBreakerConfig)`

**Parameters:**
- `config`: Circuit breaker configuration

#### Methods

##### `async call(func: Callable, *args, **kwargs) -> Any`
Execute a function through the circuit breaker.

##### `get_state() -> CircuitState`
Get the current circuit breaker state.

##### `reset() -> None`
Manually reset the circuit breaker.

### CircuitBreakerManager

Manages multiple circuit breakers.

```python
from agentic_conversation import CircuitBreakerManager, get_circuit_breaker
```

#### Functions

##### `get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker`
Get or create a circuit breaker instance.

### Supporting Classes

#### CircuitBreakerConfig

Configuration for circuit breaker behavior.

**Attributes:**
- `failure_threshold: int` - Number of failures before opening circuit
- `recovery_timeout: float` - Time to wait before attempting recovery
- `expected_exception: Type[Exception]` - Exception type that triggers circuit

#### CircuitState

Enumeration of circuit breaker states:
- `CLOSED` - Normal operation
- `OPEN` - Circuit is open, calls are rejected
- `HALF_OPEN` - Testing if service has recovered

#### CircuitBreakerError

Exception raised when circuit breaker is open.

## Utility Functions

### High-Level Functions

These functions provide simplified interfaces for common operations:

```python
from agentic_conversation import (
    run_single_conversation,
    create_orchestrator_from_config,
    create_orchestrator_from_file,
    load_config,
    validate_config
)
```

##### `async run_single_conversation(config: SystemConfig, conversation_id: Optional[str] = None, display_progress: bool = True, save_results: bool = True) -> Dict[str, Any]`
Run a single conversation with minimal setup.

##### `create_orchestrator_from_config(config: SystemConfig) -> ConversationOrchestrator`
Create an orchestrator from a configuration object.

##### `create_orchestrator_from_file(config_path: Union[str, Path]) -> ConversationOrchestrator`
Create an orchestrator from a configuration file.

## Error Handling

### Exception Hierarchy

```
Exception
├── ConfigurationError
├── OrchestrationError
├── AgentError
│   ├── AgentTimeoutError
│   ├── AgentValidationError
│   ├── AgentResponseError
│   └── AgentConfigurationError
└── CircuitBreakerError
```

### Best Practices

1. **Always handle ConfigurationError** when loading configurations
2. **Use try-catch blocks** around orchestrator operations
3. **Check circuit breaker state** before making API calls
4. **Validate configurations** before running conversations
5. **Monitor telemetry data** for performance insights

## Examples

### Basic Usage

```python
import asyncio
from agentic_conversation import load_config, ConversationOrchestrator

async def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Create orchestrator
    orchestrator = ConversationOrchestrator(config=config)
    
    # Run conversation
    results = await orchestrator.run_conversation()
    
    print(f"Conversation completed: {results['total_turns']} turns")

asyncio.run(main())
```

### Advanced Usage with Custom Error Handling

```python
import asyncio
from agentic_conversation import (
    ConversationOrchestrator, ConfigurationError, 
    OrchestrationError, load_config
)

async def run_conversation_with_error_handling():
    try:
        config = load_config("config.yaml")
        orchestrator = ConversationOrchestrator(config=config)
        
        results = await orchestrator.run_conversation(
            conversation_id="my-conversation",
            display_progress=True,
            save_results=True
        )
        
        return results
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        return None
    except OrchestrationError as e:
        print(f"Orchestration error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

results = asyncio.run(run_conversation_with_error_handling())
```

### Custom Agent Implementation

```python
from agentic_conversation import BaseAgent, AgentResponse, AgentInfo
from typing import List

class EchoAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
    
    async def generate_response(self, conversation_history, context):
        # Simple echo logic
        if conversation_history:
            last_message = conversation_history[-1]
            response_content = f"Echo: {last_message.content}"
        else:
            response_content = "Hello! I'm an echo agent."
        
        return AgentResponse(
            content=response_content,
            agent_id=self.agent_id,
            metadata={"echo": True}
        )
    
    def get_agent_info(self):
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description="An agent that echoes messages"
        )
```

This API reference provides comprehensive documentation for all public classes and methods in the Agentic Conversation System. For additional examples and usage patterns, refer to the main README.md file and the test suite.