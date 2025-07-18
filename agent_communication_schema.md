# Agentic Conversation System - Agent Communication Schema

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEM OVERVIEW                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                ┌─────────────────┐
                                │  Configuration  │
                                │     Loader      │
                                └─────────┬───────┘
                                          │
                                          ▼
                            ┌─────────────────────────────┐
                            │   ConversationOrchestrator  │
                            │                             │
                            │  • Initializes all components
                            │  • Manages conversation flow │
                            │  • Handles error recovery   │
                            │  • Collects telemetry       │
                            └─────────────┬───────────────┘
                                          │
                                          ▼
                            ┌─────────────────────────────┐
                            │     ConversationGraph       │
                            │      (LangGraph State       │
                            │        Machine)             │
                            └─────────────┬───────────────┘
                                          │
                                          ▼

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            AGENT COMMUNICATION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐                                    ┌─────────────────┐
    │    Agent A      │                                    │    Agent B      │
    │  (LangChain)    │                                    │  (LangChain)    │
    │                 │                                    │                 │
    │ • Name/Role     │                                    │ • Name/Role     │
    │ • System Prompt │                                    │ • System Prompt │
    │ • LLM Model     │                                    │ • LLM Model     │
    └─────────┬───────┘                                    └─────────┬───────┘
              │                                                      │
              │                                                      │
              ▼                                                      ▼
    ┌─────────────────┐                                    ┌─────────────────┐
    │ Agent Response  │                                    │ Agent Response  │
    │                 │                                    │                 │
    │ • Content       │                                    │ • Content       │
    │ • Token Count   │                                    │ • Token Count   │
    │ • Response Time │                                    │ • Response Time │
    │ • Metadata      │                                    │ • Metadata      │
    └─────────┬───────┘                                    └─────────┬───────┘
              │                                                      │
              └──────────────────────┬───────────────────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │    ConversationState        │
                        │                             │
                        │ • Messages List             │
                        │ • Current Turn              │
                        │ • Token Counts              │
                        │ • Status                    │
                        │ • Context Utilization       │
                        └─────────────┬───────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │    Context Manager          │
                        │                             │
                        │ • Manages context window    │
                        │ • Applies strategies        │
                        │   - Sliding window          │
                        │   - Truncation              │
                        │ • Token counting            │
                        └─────────────┬───────────────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │   Telemetry Logger          │
                        │                             │
                        │ • Logs all interactions     │
                        │ • Performance metrics       │
                        │ • Error tracking            │
                        │ • Real-time display         │
                        └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DETAILED COMMUNICATION FLOW                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

1. INITIALIZATION PHASE
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  Configuration  │───▶│   Orchestrator  │───▶│  Graph Builder  │
   │     Loading     │    │  Initialization │    │   (LangGraph)   │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Agent Creation  │
                          │ • Agent A       │
                          │ • Agent B       │
                          │ • Context Mgr   │
                          │ • Telemetry     │
                          └─────────────────┘

2. CONVERSATION EXECUTION
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │                          TURN-BY-TURN FLOW                                     │
   └─────────────────────────────────────────────────────────────────────────────────┘

   Turn N:
   ┌─────────────────┐
   │ State Manager   │ ──── Determines next agent
   │ • Check limits  │
   │ • Route to      │
   │   agent         │
   └─────────┬───────┘
             │
             ▼
   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │ Context Prep    │ ───▶ │ Agent Execution │ ───▶ │ Response        │
   │ • History       │      │ • LLM Call      │      │ Processing      │
   │ • Available     │      │ • Retry Logic   │      │ • Validation    │
   │   tokens        │      │ • Timeout       │      │ • Token Count   │
   │ • System prompt │      │   handling      │      │ • Metadata      │
   └─────────────────┘      └─────────────────┘      └─────────┬───────┘
                                                               │
                                                               ▼
                                                    ┌─────────────────┐
                                                    │ State Update    │
                                                    │ • Add message   │
                                                    │ • Update tokens │
                                                    │ • Switch agent  │
                                                    │ • Log telemetry │
                                                    └─────────────────┘

3. MESSAGE FLOW BETWEEN AGENTS
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │                        AGENT-TO-AGENT COMMUNICATION                            │
   └─────────────────────────────────────────────────────────────────────────────────┘

   Agent A Turn:
   ┌─────────────────┐
   │ Agent A         │
   │ • Receives      │ ◄─── ConversationContext
   │   context       │      • Full message history
   │ • Sees Agent B's│      • Available tokens
   │   messages      │      • Turn number
   │ • Generates     │      • System prompt
   │   response      │
   └─────────┬───────┘
             │
             ▼ (Response added to shared state)
   ┌─────────────────┐
   │ Shared State    │
   │ • Message added │
   │ • Tokens updated│
   │ • Turn counter  │
   │   incremented   │
   └─────────┬───────┘
             │
             ▼ (Context prepared for next agent)
   ┌─────────────────┐
   │ Agent B         │
   │ • Receives      │ ◄─── ConversationContext
   │   updated       │      • Includes Agent A's
   │   context       │        new message
   │ • Sees Agent A's│      • Updated token counts
   │   new message   │      • Next turn number
   │ • Generates     │
   │   response      │
   └─────────────────┘

4. CONTEXT MANAGEMENT
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │                         CONTEXT WINDOW HANDLING                                │
   └─────────────────────────────────────────────────────────────────────────────────┘

   Before Each Turn:
   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │ Check Context   │ ───▶ │ Apply Strategy  │ ───▶ │ Prepare Context │
   │ • Token count   │      │ • Sliding       │      │ • Managed       │
   │ • Utilization   │      │   window        │      │   history       │
   │ • Threshold     │      │ • Truncation    │      │ • Available     │
   │   (90%)         │      │ • Keep recent   │      │   tokens        │
   └─────────────────┘      └─────────────────┘      └─────────────────┘

5. ERROR HANDLING & RECOVERY
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │                           ERROR RECOVERY FLOW                                  │
   └─────────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │ Error Detected  │ ───▶ │ Circuit Breaker │ ───▶ │ Recovery        │
   │ • Timeout       │      │ • Track failures│      │ Strategy        │
   │ • API failure   │      │ • Open/close    │      │ • Retry         │
   │ • Context       │      │   circuit       │      │ • Backoff       │
   │   overflow      │      │ • Rate limiting │      │ • Simplify      │
   └─────────────────┘      └─────────────────┘      └─────────────────┘

6. TELEMETRY & MONITORING
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │                          TELEMETRY COLLECTION                                  │
   └─────────────────────────────────────────────────────────────────────────────────┘

   Every Turn:
   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │ Collect Metrics │ ───▶ │ Log Events      │ ───▶ │ Real-time       │
   │ • Response time │      │ • Agent turns   │      │ Display         │
   │ • Token usage   │      │ • State changes │      │ • Progress      │
   │ • Model calls   │      │ • Errors        │      │ • Metrics       │
   │ • Errors        │      │ • Performance   │      │ • Status        │
   └─────────────────┘      └─────────────────┘      └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              KEY COMPONENTS                                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

• ConversationOrchestrator: Main coordinator that manages the entire conversation
• ConversationGraph: LangGraph state machine that handles turn-by-turn flow
• LangChainAgent: Individual agents that generate responses using LLMs
• ConversationState: Shared state containing all messages and metadata
• ContextManager: Manages context window and applies strategies
• TelemetryLogger: Collects and logs all performance and interaction data
• CircuitBreaker: Handles failures and implements recovery strategies

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           COMMUNICATION PATTERNS                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

1. SEQUENTIAL TURNS: Agents take turns responding to each other
2. SHARED STATE: All communication goes through centralized ConversationState
3. CONTEXT AWARE: Each agent sees full conversation history (managed by context window)
4. ASYNCHRONOUS: All operations are async for better performance
5. MONITORED: Every interaction is logged and monitored
6. RESILIENT: Circuit breakers and retry logic handle failures
7. CONFIGURABLE: Behavior controlled by comprehensive configuration system