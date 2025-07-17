# Example Configurations and Use Cases

This directory contains example configurations and use cases for the Agentic Conversation System, demonstrating various conversation scenarios and system capabilities.

## Configuration Examples

### Basic Templates
- `basic-conversation.yaml` - Simple conversation between two general assistants
- `research-discussion.yaml` - Research-focused conversation with evidence-based analysis
- `creative-collaboration.yaml` - Creative writing and brainstorming session
- `technical-architecture.yaml` - Technical system design discussion

### Specialized Use Cases
- `educational-tutoring.yaml` - Educational conversation with teacher and student roles
- `business-strategy.yaml` - Strategic business planning discussion
- `scientific-debate.yaml` - Scientific hypothesis discussion and peer review
- `philosophical-dialogue.yaml` - Philosophical exploration and ethical reasoning

### Advanced Configurations
- `high-context.yaml` - Configuration for long, detailed conversations
- `rapid-fire.yaml` - Quick, short-turn conversations
- `multilingual.yaml` - Configuration for multilingual conversations
- `performance-optimized.yaml` - Optimized for speed and efficiency

## Usage Examples

Each configuration file includes:
- Detailed agent personas and system prompts
- Appropriate model settings for the use case
- Conversation parameters optimized for the scenario
- Initial prompts to start meaningful dialogue
- Logging configuration for analysis

## Running Examples

```bash
# Run a basic conversation
agentic-conversation run --config examples/basic-conversation.yaml

# Run a research discussion
agentic-conversation run --config examples/research-discussion.yaml

# Run with custom conversation ID
agentic-conversation run --config examples/creative-collaboration.yaml --conversation-id "creative-session-1"
```

## Analyzing Results

```bash
# Analyze conversation logs
agentic-conversation analyze ./logs --format detailed

# Compare different conversation types
agentic-conversation analyze ./logs --format summary
```

## Customization

These examples serve as starting points. Customize them by:
- Modifying agent personas and system prompts
- Adjusting model parameters for your needs
- Changing conversation length and context settings
- Adding your own initial prompts

## Performance Benchmarking

See `scripts/` directory for performance benchmarking scripts that use these configurations to test system capabilities under various conditions.