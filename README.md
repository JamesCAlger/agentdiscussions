# Agent Discussions: Context Engineering & Context Rot Research

A framework for running multi-turn conversations between two AI agents, built to test different context engineering strategies and investigate **context rot** -- the tendency for LLM performance to degrade as the context window fills up.

## Research Questions

1. **How do different context management strategies affect conversation quality over extended dialogues?** Sliding window vs. truncation vs. full history -- when does each break down?
2. **At what point does context rot set in?** As conversation history grows, do agents start repeating themselves, losing coherence, or hallucinating more?
3. **Can context engineering mitigate degradation?** Does summarizing older context, compressing history, or selectively pruning messages preserve quality better than naive truncation?

## How It Works

Two AI agents (configurable roles, system prompts, and models) take turns responding to each other through a LangGraph-based orchestrator. The system tracks:

- **Token usage per turn** -- how context consumption grows over time
- **Response latency** -- does generation slow as context fills?
- **Context utilization** -- what percentage of the window is used at each turn?
- **Conversation quality signals** -- repetition, coherence drift, topic abandonment

```
Config (YAML) ──> Orchestrator ──> LangGraph State Machine
                                         │
              Agent A ◄── Context Manager ──► Agent B
                                │
                          Telemetry Logger
                                │
                    Perf Logs / Token Tracking / Analytics
```

## Context Strategies Tested

| Strategy | Description | Hypothesis |
|----------|-------------|------------|
| **Full history** | Send entire conversation each turn | Best quality early, degrades or fails at token limits |
| **Sliding window** | Keep only the last N tokens of history | Maintains recency but loses early context |
| **Truncation** | Drop oldest messages when limit is reached | Abrupt context loss may cause incoherence |
| **Summarization** | Compress older turns into summaries | Best quality retention, but summaries lose nuance (planned) |

Strategies are configured per-experiment via YAML:

```yaml
conversation:
  max_turns: 20
  context_window_strategy: "sliding"  # or "truncation", "full"
  context_window_size: 8000

model:
  model_name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
```

## Example Configurations

Pre-built experiment configs in `examples/`:

| Config | Use Case |
|--------|----------|
| `research-discussion.yaml` | Deep research dialogue -- tests sustained reasoning |
| `creative-collaboration.yaml` | Creative writing -- tests coherence and style drift |
| `technical-architecture.yaml` | System design debate -- tests factual consistency |
| `performance-optimized.yaml` | Tuned for throughput and token efficiency |

## Running an Experiment

```bash
# Install
pip install -e .

# Set API key
export OPENAI_API_KEY="your-key"

# Run a conversation
agentic-conversation run --config examples/research-discussion.yaml

# Analyze results
agentic-conversation analyze ./logs --format detailed
```

## Architecture

```
src/agentic_conversation/
├── orchestrator.py       # Turn management, termination conditions
├── conversation_graph.py # LangGraph state machine
├── context_manager.py    # Context window strategies (core research component)
├── token_counter.py      # Accurate token counting via tiktoken
├── agents.py             # Base agent interface
├── langchain_agent.py    # LangChain-based agent implementation
├── circuit_breaker.py    # API resilience (exponential backoff, failure detection)
├── telemetry.py          # Performance monitoring, structured logging
├── config.py             # YAML config loading, env var substitution
└── models.py             # Type-safe data structures

tests/                    # Unit, integration, and performance tests
perf_logs/                # Benchmarking results (token scaling, throughput, concurrency)
examples/                 # Pre-built experiment configurations
```

## Key Components

**Context Manager** (`context_manager.py`) -- The core research component. Implements pluggable strategies for managing what goes into the context window at each turn. This is where the context rot mitigation logic lives.

**Token Counter** (`token_counter.py`) -- Accurate token counting using tiktoken. Tracks per-turn and cumulative token usage to map when degradation correlates with context saturation.

**Telemetry** (`telemetry.py`) -- Structured JSON logs capturing response times, token counts, context utilization, and error rates per turn. Enables post-hoc analysis of conversation quality over time.

**Circuit Breaker** (`circuit_breaker.py`) -- Handles API failures gracefully so long-running multi-turn experiments don't crash mid-conversation.

## Performance Benchmarks

The `perf_logs/` directory contains benchmarking results across:
- Single conversation runs at varying context sizes
- Concurrent multi-conversation scaling (2, 5, 10, 20 simultaneous)
- Throughput under sustained load
- Context scaling tests (measuring latency as context grows)

## Tech Stack

Python, LangGraph, LangChain, tiktoken, OpenAI/Anthropic APIs, YAML configuration

## Status

This is an active research POC. The framework is functional and instrumented. Next steps:
- Implement summarization-based context strategy
- Build automated quality scoring (repetition detection, coherence metrics)
- Run systematic comparisons across context strategies with controlled topics
