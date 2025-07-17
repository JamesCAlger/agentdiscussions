# Configuration Reference

This document provides a comprehensive reference for all configuration options available in the Agentic Conversation System.

## Table of Contents

- [Configuration File Structure](#configuration-file-structure)
- [Environment Variable Support](#environment-variable-support)
- [Agent Configuration](#agent-configuration)
- [Model Configuration](#model-configuration)
- [Conversation Configuration](#conversation-configuration)
- [Logging Configuration](#logging-configuration)
- [Configuration Examples](#configuration-examples)
- [Validation](#validation)
- [Best Practices](#best-practices)

## Configuration File Structure

The system uses YAML configuration files with the following top-level structure:

```yaml
agents:          # Agent configurations (required)
  agent_a: {...} # First agent configuration
  agent_b: {...} # Second agent configuration

model: {...}     # Language model settings (required)

conversation: {...} # Conversation behavior settings (required)

logging: {...}   # Logging and telemetry settings (optional)
```

## Environment Variable Support

All configuration values support environment variable substitution using the following syntax:

- `${VAR_NAME}` - Use environment variable value, fail if not set
- `${VAR_NAME:default_value}` - Use environment variable value, or default if not set

### Examples

```yaml
model:
  model_name: "${MODEL_NAME:gpt-4}"           # Defaults to gpt-4
  temperature: 0.7

logging:
  log_level: "${LOG_LEVEL}"                   # Required environment variable
  output_directory: "${LOG_DIR:./logs}"      # Defaults to ./logs
```

### Common Environment Variables

```bash
# Model Configuration
export MODEL_NAME="gpt-4"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Logging Configuration
export LOG_LEVEL="INFO"
export LOG_DIR="/path/to/logs"

# Conversation Configuration
export MAX_TURNS="20"
export CONTEXT_SIZE="8000"
```

## Agent Configuration

Each agent requires a configuration section under the `agents` key.

### Structure

```yaml
agents:
  agent_a:
    name: "Agent Name"           # Required: Human-readable name
    system_prompt: |             # Required: Multi-line system prompt
      You are an AI assistant with specific role...
      
      Your responsibilities include:
      1. Task one
      2. Task two
      
      Communication style:
      - Professional and helpful
      - Detailed and thorough
  
  agent_b:
    name: "Another Agent"
    system_prompt: "Single line system prompt"
```

### Required Fields

#### `name` (string)
- **Description**: Human-readable name for the agent
- **Usage**: Used in logs, displays, and telemetry
- **Constraints**: Cannot be empty, must be unique between agents
- **Example**: `"Research Assistant"`, `"Creative Writer"`

#### `system_prompt` (string)
- **Description**: System prompt that defines the agent's role, personality, and behavior
- **Usage**: Prepended to every conversation turn for this agent
- **Constraints**: Cannot be empty
- **Best Practices**:
  - Be specific about the agent's role and expertise
  - Include communication style guidelines
  - Specify how the agent should interact with the other agent
  - Keep prompts focused but comprehensive (typically 100-500 words)

### Example Agent Configurations

#### Research-Focused Agent
```yaml
agent_a:
  name: "Research Scientist"
  system_prompt: |
    You are Dr. Sarah Chen, a thorough research scientist with expertise in AI and ML.
    
    Your role:
    - Ask probing, insightful questions
    - Seek evidence and data to support claims
    - Identify gaps in reasoning
    - Reference relevant research and studies
    - Challenge assumptions constructively
    
    Communication style:
    - Analytical and detail-oriented
    - Curious and inquisitive
    - Professional but approachable
    - Focus on understanding "why" and "how"
    
    Always build upon previous discussion and reference specific points.
    Keep responses focused, typically 2-4 paragraphs.
```

#### Strategic Business Agent
```yaml
agent_b:
  name: "Business Strategist"
  system_prompt: |
    You are Marcus Rodriguez, a strategic business consultant with extensive
    experience in technology implementation and organizational change.
    
    Your role:
    - Synthesize information and identify patterns
    - Focus on practical applications and real-world implications
    - Consider scalability, feasibility, and implementation challenges
    - Bridge theory and actionable strategies
    - Evaluate risks, opportunities, and consequences
    
    Communication style:
    - Strategic and big-picture oriented
    - Pragmatic and solution-focused
    - Collaborative, building on others' ideas
    - Focused on measurable outcomes
    
    Acknowledge and build upon your conversation partner's insights.
    Provide concrete examples and frameworks when possible.
```

## Model Configuration

Controls the language model behavior and API settings.

### Structure

```yaml
model:
  model_name: "gpt-4"              # Required: Model identifier
  temperature: 0.7                 # Optional: Sampling temperature
  max_tokens: 2000                 # Optional: Maximum response tokens
  top_p: 0.95                      # Optional: Nucleus sampling
  frequency_penalty: 0.1           # Optional: Frequency penalty
  presence_penalty: 0.1            # Optional: Presence penalty
```

### Configuration Fields

#### `model_name` (string, required)
- **Description**: Identifier for the language model to use
- **Supported Values**:
  - OpenAI: `"gpt-4"`, `"gpt-4-turbo"`, `"gpt-3.5-turbo"`
  - Anthropic: `"claude-3-opus"`, `"claude-3-sonnet"`, `"claude-3-haiku"`
  - Other LangChain-supported models
- **Environment Variable**: `MODEL_NAME`
- **Example**: `"gpt-4"`

#### `temperature` (float, optional, default: 0.7)
- **Description**: Controls randomness in model responses
- **Range**: 0.0 to 2.0
- **Usage**:
  - `0.0-0.3`: More focused, consistent, deterministic responses
  - `0.4-0.7`: Balanced creativity and consistency (recommended)
  - `0.8-1.2`: More creative and varied responses
  - `1.3-2.0`: Highly creative but potentially inconsistent
- **Example**: `0.7`

#### `max_tokens` (integer, optional, default: 2000)
- **Description**: Maximum number of tokens the model can generate in a single response
- **Range**: 1 to model's maximum (varies by model)
- **Considerations**:
  - Higher values allow longer responses but use more tokens
  - Should be balanced with context window size
  - Typical values: 1000-4000
- **Example**: `2000`

#### `top_p` (float, optional, default: 1.0)
- **Description**: Nucleus sampling parameter controlling diversity
- **Range**: 0.0 to 1.0
- **Usage**:
  - `1.0`: Consider all possible tokens (default)
  - `0.9-0.95`: Good balance of quality and diversity
  - `0.1-0.5`: More focused, less diverse responses
- **Example**: `0.95`

#### `frequency_penalty` (float, optional, default: 0.0)
- **Description**: Reduces repetition based on token frequency in the text
- **Range**: -2.0 to 2.0
- **Usage**:
  - `0.0`: No penalty (default)
  - `0.1-0.5`: Mild reduction in repetition
  - `0.6-1.0`: Strong reduction in repetition
  - Negative values encourage repetition
- **Example**: `0.1`

#### `presence_penalty` (float, optional, default: 0.0)
- **Description**: Reduces repetition based on whether tokens appear in the text
- **Range**: -2.0 to 2.0
- **Usage**:
  - `0.0`: No penalty (default)
  - `0.1-0.5`: Encourage new topics and ideas
  - `0.6-1.0`: Strong encouragement for new topics
  - Negative values encourage staying on topic
- **Example**: `0.1`

### Model-Specific Recommendations

#### GPT-4 (Recommended for most use cases)
```yaml
model:
  model_name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
  top_p: 0.95
  frequency_penalty: 0.1
  presence_penalty: 0.1
```

#### GPT-3.5-Turbo (Cost-effective option)
```yaml
model:
  model_name: "gpt-3.5-turbo"
  temperature: 0.8
  max_tokens: 1500
  top_p: 0.9
  frequency_penalty: 0.2
  presence_penalty: 0.1
```

#### Claude-3-Opus (High-quality alternative)
```yaml
model:
  model_name: "claude-3-opus"
  temperature: 0.7
  max_tokens: 2000
  # Note: Claude models may not support all parameters
```

## Conversation Configuration

Controls conversation flow, context management, and termination conditions.

### Structure

```yaml
conversation:
  max_turns: 20                           # Required: Maximum conversation turns
  initial_prompt: |                       # Optional: Conversation starter
    Let's discuss artificial intelligence...
  context_window_strategy: "sliding"      # Optional: Context management strategy
  context_window_size: 8000              # Optional: Maximum context tokens
  turn_timeout: 30.0                     # Optional: Response timeout in seconds
```

### Configuration Fields

#### `max_turns` (integer, required)
- **Description**: Maximum number of conversation turns (back-and-forth exchanges)
- **Range**: 1 to unlimited (practical limit ~1000)
- **Considerations**:
  - Each turn consists of one response from each agent
  - Higher values allow longer conversations but use more resources
  - Consider context window limitations
- **Typical Values**: 10-50
- **Example**: `20`

#### `initial_prompt` (string, optional)
- **Description**: Optional prompt to start the conversation
- **Usage**:
  - If provided, this becomes the first message that kicks off dialogue
  - If not provided, Agent A starts with their system prompt context
  - Should be engaging and provide clear direction
- **Best Practices**:
  - Be specific about the topic or goal
  - Provide enough context for meaningful discussion
  - Ask open-ended questions to encourage dialogue
- **Example**:
```yaml
initial_prompt: |
  Let's explore the future of artificial intelligence in education. 
  Specifically, I'd like to discuss how AI tutoring systems might 
  transform personalized learning, what challenges we need to overcome 
  for widespread adoption, and what the implications might be for 
  traditional educational institutions and teaching roles.
  
  What are your initial thoughts on the most promising applications 
  of AI in education?
```

#### `context_window_strategy` (string, optional, default: "sliding")
- **Description**: Strategy for managing context when approaching token limits
- **Supported Values**:
  - `"truncate"`: Remove oldest messages when limit is reached
  - `"sliding"`: Maintain a sliding window of recent messages (recommended)
  - `"summarize"`: Create summaries of older conversation parts (planned feature)
- **Considerations**:
  - `"sliding"` provides best balance of context and performance
  - `"truncate"` is simpler but may lose important context
- **Example**: `"sliding"`

#### `context_window_size` (integer, optional, default: 8000)
- **Description**: Maximum number of tokens to maintain in the context window
- **Range**: 1000 to model's maximum context length
- **Considerations**:
  - Should be less than the model's maximum context length
  - Larger values preserve more context but use more resources
  - Must account for system prompts and response generation
- **Model Limits**:
  - GPT-3.5-Turbo: ~16,000 tokens
  - GPT-4: ~8,000 tokens (original), ~128,000 tokens (turbo)
  - Claude-3: ~200,000 tokens
- **Recommended Values**: 4000-16000
- **Example**: `8000`

#### `turn_timeout` (float, optional, default: 30.0)
- **Description**: Maximum time to wait for an agent response (in seconds)
- **Range**: 1.0 to unlimited (practical limit ~300)
- **Usage**:
  - Prevents hanging on slow API responses
  - Should account for model processing time and network latency
  - Longer timeouts for complex prompts or slower models
- **Typical Values**: 15-60 seconds
- **Example**: `30.0`

### Context Management Examples

#### Conservative Context Management
```yaml
conversation:
  max_turns: 10
  context_window_strategy: "truncate"
  context_window_size: 4000
  turn_timeout: 15.0
```

#### Extensive Context Management
```yaml
conversation:
  max_turns: 50
  context_window_strategy: "sliding"
  context_window_size: 16000
  turn_timeout: 45.0
```

## Logging Configuration

Controls telemetry collection, log output, and data persistence.

### Structure

```yaml
logging:
  log_level: "INFO"                       # Optional: Logging verbosity
  output_directory: "./logs"              # Optional: Log file directory
  real_time_display: true                 # Optional: Show live conversation
  export_formats: ["json", "txt"]         # Optional: Export file formats
  save_conversation_history: true         # Optional: Save full conversation
  save_telemetry: true                    # Optional: Save performance data
```

### Configuration Fields

#### `log_level` (string, optional, default: "INFO")
- **Description**: Controls verbosity of system logs
- **Supported Values**:
  - `"DEBUG"`: Very detailed information for troubleshooting
  - `"INFO"`: General information about system operation (recommended)
  - `"WARNING"`: Important warnings that don't stop execution
  - `"ERROR"`: Error conditions that may affect functionality
  - `"CRITICAL"`: Serious errors that may cause system failure
- **Environment Variable**: `LOG_LEVEL`
- **Example**: `"INFO"`

#### `output_directory` (string, optional, default: "./logs")
- **Description**: Directory where log files and conversation records are saved
- **Usage**:
  - Can be absolute or relative path
  - Directory will be created if it doesn't exist
  - Should be writable by the application
- **Environment Variable**: `LOG_DIR`
- **Example**: `"./logs"`, `"/var/log/agentic-conversation"`

#### `real_time_display` (boolean, optional, default: true)
- **Description**: Whether to display the conversation in real-time in the console
- **Usage**:
  - `true`: Show each message as it's generated (recommended for interactive use)
  - `false`: Only show summary information (better for automated runs)
- **Example**: `true`

#### `export_formats` (list of strings, optional, default: ["json"])
- **Description**: List of export formats for conversation logs
- **Supported Values**:
  - `"json"`: Structured data format with full metadata (recommended)
  - `"csv"`: Tabular format suitable for analysis
  - `"txt"`: Human-readable plain text format
- **Usage**: Can specify multiple formats
- **Example**: `["json", "txt"]`

#### `save_conversation_history` (boolean, optional, default: true)
- **Description**: Whether to save complete conversation history to files
- **Usage**:
  - `true`: Save all messages, timestamps, and metadata (recommended)
  - `false`: Don't save conversation history (saves disk space)
- **Example**: `true`

#### `save_telemetry` (boolean, optional, default: true)
- **Description**: Whether to save detailed telemetry data
- **Usage**:
  - `true`: Save performance metrics, token usage, error rates, etc. (recommended)
  - `false`: Don't save telemetry data (saves disk space)
- **Example**: `true`

### Logging Examples

#### Development/Debug Configuration
```yaml
logging:
  log_level: "DEBUG"
  output_directory: "./debug-logs"
  real_time_display: true
  export_formats: ["json", "txt"]
  save_conversation_history: true
  save_telemetry: true
```

#### Production Configuration
```yaml
logging:
  log_level: "WARNING"
  output_directory: "/var/log/agentic-conversation"
  real_time_display: false
  export_formats: ["json"]
  save_conversation_history: true
  save_telemetry: true
```

#### Minimal Configuration
```yaml
logging:
  log_level: "ERROR"
  output_directory: "./logs"
  real_time_display: false
  export_formats: []
  save_conversation_history: false
  save_telemetry: false
```

## Configuration Examples

### Complete Configuration Templates

#### Basic Research Conversation
```yaml
# Basic research-focused conversation configuration
agents:
  agent_a:
    name: "Researcher"
    system_prompt: |
      You are a thorough researcher who asks probing questions,
      seeks evidence, and analyzes information critically.
      Keep responses focused and build on previous discussion.
  
  agent_b:
    name: "Analyst"
    system_prompt: |
      You are an analytical thinker who synthesizes information,
      identifies patterns, and draws practical conclusions.
      Provide concrete examples and actionable insights.

model:
  model_name: "${MODEL_NAME:gpt-4}"
  temperature: 0.7
  max_tokens: 2000

conversation:
  max_turns: 15
  initial_prompt: |
    Let's research and analyze the impact of remote work on 
    team productivity and employee satisfaction. What are the 
    key factors we should investigate?
  context_window_strategy: "sliding"
  context_window_size: 8000

logging:
  log_level: "${LOG_LEVEL:INFO}"
  output_directory: "${LOG_DIR:./logs}"
  real_time_display: true
  export_formats: ["json", "txt"]
```

#### Creative Collaboration
```yaml
# Creative writing collaboration configuration
agents:
  agent_a:
    name: "Creative Writer"
    system_prompt: |
      You are a creative writer who thinks outside the box and 
      generates innovative ideas. Focus on imagination, storytelling,
      and creative problem-solving. Build on your partner's ideas
      with enthusiasm and originality.
  
  agent_b:
    name: "Story Developer"
    system_prompt: |
      You are a story developer who adds structure, depth, and 
      practical considerations to creative ideas. Help refine 
      concepts, develop characters, and create compelling narratives.
      Balance creativity with storytelling craft.

model:
  model_name: "gpt-4"
  temperature: 0.9  # Higher creativity
  max_tokens: 2500
  presence_penalty: 0.2  # Encourage new ideas

conversation:
  max_turns: 25
  initial_prompt: |
    Let's collaborate on creating an original story concept that 
    explores themes of technology and human connection. The story 
    should be engaging, thought-provoking, and suitable for a 
    general audience. Where should we start?
  context_window_strategy: "sliding"
  context_window_size: 10000

logging:
  log_level: "INFO"
  output_directory: "./creative-logs"
  real_time_display: true
  export_formats: ["json", "txt"]
```

#### Technical Architecture Discussion
```yaml
# Technical architecture discussion configuration
agents:
  agent_a:
    name: "Software Architect"
    system_prompt: |
      You are a senior software architect who designs scalable 
      systems and considers long-term maintainability. Focus on 
      architectural patterns, system design principles, and 
      technical trade-offs. Ask detailed technical questions.
  
  agent_b:
    name: "Implementation Engineer"
    system_prompt: |
      You are a senior implementation engineer who focuses on 
      practical coding solutions and performance optimization. 
      Consider implementation details, development workflows, 
      and operational concerns. Provide concrete technical solutions.

model:
  model_name: "gpt-4"
  temperature: 0.5  # More focused responses
  max_tokens: 3000
  frequency_penalty: 0.1

conversation:
  max_turns: 30
  initial_prompt: |
    Let's design a scalable microservices architecture for a 
    high-traffic e-commerce platform. We need to handle millions 
    of users, ensure high availability, and maintain fast response 
    times. What are the key architectural decisions we need to make?
  context_window_strategy: "sliding"
  context_window_size: 12000
  turn_timeout: 45.0

logging:
  log_level: "DEBUG"
  output_directory: "./tech-logs"
  real_time_display: true
  export_formats: ["json", "csv"]
  save_telemetry: true
```

## Validation

### Configuration Validation

The system automatically validates all configuration values when loading:

```bash
# Validate a configuration file
agentic-conversation validate config.yaml

# Validate with detailed output
agentic-conversation validate config.yaml --detailed
```

### Common Validation Errors

#### Missing Required Fields
```
Error: Missing required field 'model_name' in model configuration
Error: Missing required configuration sections: agents, model
```

#### Invalid Values
```
Error: Temperature must be between 0.0 and 2.0
Error: Context window strategy must be one of: truncate, summarize, sliding
Error: Agent A and Agent B must have different names
```

#### Environment Variable Issues
```
Error: Environment variable MODEL_NAME is not set and no default provided
Warning: LOG_DIR environment variable not set, using default: ./logs
```

### Validation Checklist

Before running conversations, ensure:

- [ ] All required sections are present (`agents`, `model`, `conversation`)
- [ ] Agent names are unique and not empty
- [ ] System prompts are not empty
- [ ] Model name is valid for your provider
- [ ] Temperature is between 0.0 and 2.0
- [ ] Max tokens is positive
- [ ] Context window size is reasonable for your model
- [ ] Output directory is writable
- [ ] Required environment variables are set

## Best Practices

### Agent Configuration
1. **Make agents distinct**: Give each agent a unique role, personality, and expertise area
2. **Be specific in prompts**: Detailed system prompts lead to better conversations
3. **Include interaction guidelines**: Specify how agents should build on each other's responses
4. **Keep prompts focused**: Avoid overly long or complex system prompts

### Model Configuration
1. **Start with defaults**: Use recommended settings and adjust based on results
2. **Balance creativity and consistency**: Higher temperature for creative tasks, lower for analytical
3. **Consider token limits**: Balance max_tokens with context_window_size
4. **Test different models**: Different models excel at different types of conversations

### Conversation Configuration
1. **Set reasonable turn limits**: Start with 10-20 turns and adjust based on needs
2. **Use sliding window strategy**: Provides best balance of context and performance
3. **Size context appropriately**: Leave room for system prompts and responses
4. **Set appropriate timeouts**: Account for model processing time and network latency

### Logging Configuration
1. **Use INFO level for production**: Provides good balance of information and performance
2. **Save telemetry data**: Essential for monitoring and optimization
3. **Choose appropriate export formats**: JSON for analysis, TXT for human reading
4. **Organize log directories**: Use descriptive directory names and structure

### Environment Variables
1. **Use environment variables for secrets**: Never hardcode API keys in configuration files
2. **Provide sensible defaults**: Use `${VAR:default}` syntax for optional settings
3. **Document required variables**: Clearly specify which environment variables are needed
4. **Validate environment setup**: Check that all required variables are set before running

### Security Considerations
1. **Protect API keys**: Use environment variables, never commit keys to version control
2. **Limit file permissions**: Ensure log directories have appropriate access controls
3. **Sanitize prompts**: Be careful with user-provided content in system prompts
4. **Monitor usage**: Track API usage and costs through telemetry data

This configuration reference provides comprehensive documentation for all available options in the Agentic Conversation System. For additional examples and usage patterns, refer to the main README.md file and API documentation.