# Agentic Conversation System

An advanced Python framework for orchestrating sequential dialogue between two AI agents, featuring comprehensive state management, telemetry, and flexible configuration options.

## Overview

The Agentic Conversation System enables two distinct AI agents to engage in meaningful dialogue while maintaining complete conversation history within their context windows. Built on LangGraph for robust state management and LangChain for AI integration, the system provides enterprise-grade features including comprehensive logging, performance monitoring, and flexible configuration management.

## Key Features

### Core Functionality
- **Dual Agent Architecture**: Two distinct AI agents with configurable roles, personalities, and system prompts
- **Complete Context Management**: Full conversation history maintained with intelligent context window management
- **LangGraph Orchestration**: State machine-based conversation flow with error handling and recovery
- **Multi-Provider Support**: Compatible with OpenAI, Anthropic, and other LangChain-supported providers

### Advanced Features
- **Intelligent Context Management**: Multiple strategies (truncation, sliding window) for handling context limits
- **Comprehensive Telemetry**: Real-time performance monitoring, token usage tracking, and error analytics
- **Circuit Breaker Pattern**: Automatic failure detection and recovery for API resilience
- **Flexible Configuration**: YAML-based configuration with environment variable support
- **Rich CLI Interface**: Full-featured command-line interface with validation and analysis tools

### Monitoring & Analytics
- **Real-time Display**: Live conversation monitoring with performance metrics
- **Structured Logging**: JSON-formatted logs with detailed telemetry data
- **Performance Analytics**: Response time tracking, token usage analysis, and error reporting
- **Export Capabilities**: Multiple export formats (JSON, CSV, TXT) for analysis

## Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

This includes additional tools for development:
- pytest (testing framework)
- black (code formatting)
- isort (import sorting)
- mypy (type checking)
- flake8 (linting)
- pre-commit (git hooks)

### System Requirements

- Python 3.8 or higher
- API keys for your chosen LLM provider (OpenAI, Anthropic, etc.)
- Minimum 4GB RAM recommended for optimal performance

## Quick Start

### 1. Set Up Environment Variables

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"

# Optional: Custom log directory
export LOG_DIR="./my-logs"
```

### 2. Create a Configuration File

Generate a basic configuration:

```bash
agentic-conversation create-config basic my-config.yaml
```

Or use the provided example:

```bash
cp config.yaml my-config.yaml
```

### 3. Run Your First Conversation

```bash
agentic-conversation run --config my-config.yaml
```

### 4. Analyze Results

```bash
agentic-conversation analyze ./logs
```

## Configuration

### Configuration File Structure

The system uses YAML configuration files with the following structure:

```yaml
agents:
  agent_a:
    name: "Agent Name"
    system_prompt: "System prompt defining agent behavior..."
  agent_b:
    name: "Another Agent"
    system_prompt: "Different system prompt..."

model:
  model_name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

conversation:
  max_turns: 20
  initial_prompt: "Conversation starter..."
  context_window_strategy: "sliding"
  context_window_size: 8000

logging:
  log_level: "INFO"
  output_directory: "./logs"
  real_time_display: true
  export_formats: ["json", "txt"]
```

### Environment Variable Support

All configuration values support environment variable substitution:

```yaml
model:
  model_name: "${MODEL_NAME:gpt-4}"  # Uses MODEL_NAME env var, defaults to gpt-4
  temperature: 0.7
```

### Configuration Templates

Generate pre-configured templates for common use cases:

```bash
# Research-focused conversation
agentic-conversation create-config research research-config.yaml

# Creative collaboration
agentic-conversation create-config creative creative-config.yaml

# Technical discussion
agentic-conversation create-config technical tech-config.yaml
```

## Command Line Interface

### Core Commands

#### Run a Conversation
```bash
agentic-conversation run --config config.yaml [OPTIONS]

Options:
  --conversation-id TEXT    Unique identifier for this conversation
  --no-display            Disable real-time conversation display
  --no-save               Do not save conversation results to files
```

#### Validate Configuration
```bash
agentic-conversation validate config.yaml [OPTIONS]

Options:
  --detailed              Show detailed validation information
```

#### Analyze Conversation Logs
```bash
agentic-conversation analyze ./logs [OPTIONS]

Options:
  --conversation-id TEXT  Filter logs by specific conversation ID
  --format [summary|detailed|json]  Output format for log analysis
  --limit INTEGER         Limit number of conversations to analyze
```

#### Create Configuration Templates
```bash
agentic-conversation create-config TEMPLATE output.yaml [OPTIONS]

Templates: basic, research, creative, technical

Options:
  --model TEXT           Model to use in the configuration
  --max-turns INTEGER    Maximum number of conversation turns
```

#### System Information
```bash
agentic-conversation info
```

### Global Options

```bash
Options:
  --config, -c PATH       Path to configuration file
  --log-level, -l LEVEL   Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  --verbose, -v           Enable verbose output with detailed logging
  --output-dir, -o PATH   Output directory for logs and results
```

## Programming API

### Basic Usage

```python
from agentic_conversation import (
    ConversationOrchestrator,
    load_config,
    run_single_conversation
)

# Load configuration
config = load_config("config.yaml")

# Create orchestrator
orchestrator = ConversationOrchestrator(config=config)

# Run conversation
import asyncio
results = asyncio.run(orchestrator.run_conversation())

print(f"Conversation completed with {results['total_turns']} turns")
```

### Advanced Usage

```python
from agentic_conversation import (
    SystemConfig, AgentConfig, ModelConfig, ConversationConfig,
    LangChainAgent, TelemetryLogger, ContextManager
)

# Create configuration programmatically
config = SystemConfig(
    agent_a=AgentConfig(
        name="Researcher",
        system_prompt="You are a thorough researcher..."
    ),
    agent_b=AgentConfig(
        name="Analyst", 
        system_prompt="You are an analytical thinker..."
    ),
    model=ModelConfig(
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=2000
    ),
    conversation=ConversationConfig(
        max_turns=10,
        context_window_strategy="sliding",
        context_window_size=8000
    ),
    logging=LoggingConfig(
        log_level="INFO",
        output_directory="./logs"
    )
)

# Create and run orchestrator
orchestrator = ConversationOrchestrator(config=config)
results = asyncio.run(orchestrator.run_conversation())
```

### Custom Agent Implementation

```python
from agentic_conversation import BaseAgent, AgentResponse, ConversationContext

class CustomAgent(BaseAgent):
    async def generate_response(
        self, 
        conversation_history: List[Message], 
        context: ConversationContext
    ) -> AgentResponse:
        # Custom agent logic here
        response_content = "Custom response based on conversation history"
        
        return AgentResponse(
            content=response_content,
            agent_id=self.agent_id,
            metadata={"custom_field": "value"}
        )
    
    def get_agent_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id=self.agent_id,
            name="Custom Agent",
            description="A custom agent implementation"
        )
```

## Architecture

### System Components

1. **Configuration Management** (`config.py`)
   - YAML configuration loading and validation
   - Environment variable substitution
   - Configuration schema validation

2. **Agent Framework** (`agents.py`, `langchain_agent.py`)
   - Abstract base agent interface
   - LangChain-based agent implementation
   - Error handling and retry logic

3. **Conversation Orchestration** (`orchestrator.py`, `conversation_graph.py`)
   - LangGraph state machine for conversation flow
   - Turn management and termination conditions
   - Error recovery and circuit breaker patterns

4. **Context Management** (`context_manager.py`, `token_counter.py`)
   - Intelligent context window management
   - Token counting and optimization
   - Multiple context strategies (truncation, sliding window)

5. **Telemetry & Logging** (`telemetry.py`)
   - Comprehensive performance monitoring
   - Structured logging with JSON output
   - Real-time metrics and analytics

6. **Data Models** (`models.py`)
   - Type-safe data structures
   - Serialization and validation
   - Configuration and telemetry models

### Data Flow

```
Configuration → Orchestrator → LangGraph State Machine
                                      ↓
Agent A ← Context Manager ← Conversation State → Context Manager → Agent B
   ↓                                                                  ↓
Telemetry Logger ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
   ↓
Log Files & Analytics
```

## Performance Optimization

### Context Window Management

The system provides multiple strategies for managing context windows:

- **Sliding Window**: Maintains recent conversation history within token limits
- **Truncation**: Removes oldest messages when limits are exceeded
- **Summarization**: Creates summaries of older conversation parts (planned feature)

### Token Optimization

- Accurate token counting using tiktoken
- Real-time context utilization monitoring
- Automatic context window management
- Performance metrics and optimization suggestions

### Error Handling

- Circuit breaker pattern for API resilience
- Exponential backoff for transient failures
- Graceful degradation strategies
- Comprehensive error logging and recovery

## Troubleshooting

### Common Issues

#### Configuration Errors
```bash
# Validate your configuration
agentic-conversation validate config.yaml --detailed

# Check for missing environment variables
agentic-conversation info
```

#### API Connection Issues
```bash
# Test with verbose logging
agentic-conversation run --config config.yaml --verbose --log-level DEBUG
```

#### Context Window Problems
```bash
# Monitor context utilization
agentic-conversation analyze ./logs --format detailed
```

### Error Messages

#### "Configuration file not found"
- Ensure the configuration file path is correct
- Use absolute paths if relative paths don't work
- Check file permissions

#### "Missing required field 'model_name'"
- Verify all required configuration sections are present
- Use `agentic-conversation validate` to check configuration

#### "API key not found"
- Set appropriate environment variables for your LLM provider
- Check that API keys are valid and have sufficient credits

#### "Context window exceeded"
- Reduce `context_window_size` in configuration
- Use "sliding" context window strategy
- Reduce `max_tokens` for model responses

### Debug Mode

Enable comprehensive debugging:

```bash
export LOG_LEVEL=DEBUG
agentic-conversation run --config config.yaml --verbose
```

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd agentic-conversation-system

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentic_conversation --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Type checking
mypy src

# Linting
flake8 src tests

# Run all quality checks
pre-commit run --all-files
```

### Project Structure

```
src/agentic_conversation/
├── __init__.py           # Public API exports
├── agents.py             # Base agent interface
├── circuit_breaker.py    # Circuit breaker implementation
├── config.py             # Configuration management
├── context_manager.py    # Context window management
├── conversation_graph.py # LangGraph state machine
├── graph_state.py        # State definitions
├── langchain_agent.py    # LangChain agent implementation
├── main.py               # CLI interface
├── models.py             # Data models
├── orchestrator.py       # Main orchestrator
├── telemetry.py          # Logging and telemetry
├── token_counter.py      # Token counting utilities
└── version.py            # Version information

tests/
├── fixtures/             # Test fixtures and sample data
├── test_*.py            # Test modules
└── __init__.py

config.yaml              # Example configuration
pyproject.toml          # Project configuration
README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Run code quality checks (`pre-commit run --all-files`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:

1. Check the [troubleshooting section](#troubleshooting) above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem
4. Include configuration files, error messages, and system information

## Changelog

### Version 0.1.0
- Initial release
- Core conversation orchestration
- LangGraph state machine implementation
- Comprehensive configuration system
- CLI interface with full feature set
- Telemetry and logging system
- Context window management
- Circuit breaker pattern for resilience