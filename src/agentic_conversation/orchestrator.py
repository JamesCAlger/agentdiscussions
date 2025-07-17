"""
Main conversation orchestrator for the agentic conversation system.

This module provides the ConversationOrchestrator class that coordinates all system
components including configuration loading, agent creation, state machine initialization,
and conversation execution with comprehensive telemetry collection.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from .config import ConfigurationLoader, ConfigurationError
from .models import SystemConfig, ConversationState, ConversationStatus, Message
from .agents import BaseAgent
from .langchain_agent import LangChainAgent
from .context_manager import ContextManager, ContextStrategy
from .conversation_graph import ConversationGraph, ConversationError
from .telemetry import TelemetryLogger, RunLogger
from .token_counter import TokenCounter, ModelType
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, get_circuit_breaker


class OrchestrationError(Exception):
    """
    Exception raised when conversation orchestration fails.
    
    Attributes:
        message: Error message
        orchestrator_id: ID of the orchestrator that failed
        original_error: Original exception that caused this error
    """
    
    def __init__(
        self,
        message: str,
        orchestrator_id: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.orchestrator_id = orchestrator_id
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "orchestrator_id": self.orchestrator_id,
            "timestamp": datetime.now().isoformat(),
            "original_error": str(self.original_error) if self.original_error else None
        }


class ConversationOrchestrator:
    """
    Main orchestrator that coordinates all system components for agent conversations.
    
    This class integrates configuration loading, agent creation, state machine initialization,
    and conversation execution with comprehensive telemetry collection and real-time
    console display of conversation progress.
    """
    
    def __init__(
        self,
        orchestrator_id: Optional[str] = None,
        config: Optional[SystemConfig] = None,
        config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the conversation orchestrator.
        
        Args:
            orchestrator_id: Optional unique identifier for this orchestrator
            config: Optional pre-loaded system configuration
            config_path: Optional path to configuration file
            
        Raises:
            OrchestrationError: If initialization fails
        """
        self.orchestrator_id = orchestrator_id or f"orchestrator_{int(time.time())}"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.config: Optional[SystemConfig] = None
        self.agent_a: Optional[BaseAgent] = None
        self.agent_b: Optional[BaseAgent] = None
        self.token_counter: Optional[TokenCounter] = None
        self.context_manager: Optional[ContextManager] = None
        self.conversation_graph: Optional[ConversationGraph] = None
        self.telemetry_logger: Optional[TelemetryLogger] = None
        self.run_logger: Optional[RunLogger] = None
        
        # Error handling and recovery
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_recovery_strategies: Dict[str, Any] = {}
        self.max_recovery_attempts = 3
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self._load_configuration(config_path)
        else:
            raise OrchestrationError(
                "Either config or config_path must be provided",
                orchestrator_id=self.orchestrator_id
            )
        
        # Initialize all components
        self._initialize_components()
        
        # Initialize error handling and recovery
        self._initialize_error_handling()
        
        self.logger.info(f"Initialized ConversationOrchestrator '{self.orchestrator_id}'")
    
    def _load_configuration(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            OrchestrationError: If configuration loading fails
        """
        try:
            loader = ConfigurationLoader()
            self.config = loader.load_from_file(config_path)
            self.logger.info(f"Loaded configuration from {config_path}")
        except ConfigurationError as e:
            raise OrchestrationError(
                f"Failed to load configuration: {str(e)}",
                orchestrator_id=self.orchestrator_id,
                original_error=e
            )
        except Exception as e:
            raise OrchestrationError(
                f"Unexpected error loading configuration: {str(e)}",
                orchestrator_id=self.orchestrator_id,
                original_error=e
            )
    
    def _initialize_components(self) -> None:
        """
        Initialize all system components based on configuration.
        
        Raises:
            OrchestrationError: If component initialization fails
        """
        if not self.config:
            raise OrchestrationError(
                "Configuration not loaded",
                orchestrator_id=self.orchestrator_id
            )
        
        try:
            # Initialize token counter
            self._initialize_token_counter()
            
            # Initialize context manager
            self._initialize_context_manager()
            
            # Initialize agents
            self._initialize_agents()
            
            # Initialize telemetry and logging
            self._initialize_telemetry()
            
            # Initialize conversation graph
            self._initialize_conversation_graph()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            raise OrchestrationError(
                f"Failed to initialize components: {str(e)}",
                orchestrator_id=self.orchestrator_id,
                original_error=e
            )
    
    def _initialize_token_counter(self) -> None:
        """Initialize the token counter based on model configuration."""
        model_name = self.config.model.model_name
        
        self.token_counter = TokenCounter(model_name=model_name)
        self.logger.debug(f"Initialized token counter with model: {model_name}")
    
    def _initialize_context_manager(self) -> None:
        """Initialize the context manager based on conversation configuration."""
        strategy_name = self.config.conversation.context_window_strategy
        
        try:
            strategy = ContextStrategy(strategy_name.lower())
        except ValueError:
            self.logger.warning(f"Unknown context strategy '{strategy_name}', using sliding")
            strategy = ContextStrategy.SLIDING
        
        self.context_manager = ContextManager(
            token_counter=self.token_counter,
            max_context_tokens=self.config.conversation.context_window_size,
            strategy=strategy,
            threshold_percentage=90.0  # Default threshold
        )
        
        self.logger.debug(
            f"Initialized context manager: {strategy.value} strategy, "
            f"{self.config.conversation.context_window_size} max tokens"
        )
    
    def _initialize_agents(self) -> None:
        """Initialize both agents based on configuration."""
        # Initialize Agent A
        self.agent_a = LangChainAgent(
            agent_id="agent_a",
            name=self.config.agent_a.name,
            system_prompt=self.config.agent_a.system_prompt,
            model_config=self.config.model,
            token_counter=self.token_counter,
            timeout=self.config.conversation.turn_timeout
        )
        
        # Initialize Agent B
        self.agent_b = LangChainAgent(
            agent_id="agent_b",
            name=self.config.agent_b.name,
            system_prompt=self.config.agent_b.system_prompt,
            model_config=self.config.model,
            token_counter=self.token_counter,
            timeout=self.config.conversation.turn_timeout
        )
        
        self.logger.debug(
            f"Initialized agents: '{self.agent_a.name}' and '{self.agent_b.name}'"
        )
    
    def _initialize_telemetry(self) -> None:
        """Initialize telemetry and logging components."""
        # Initialize run logger
        self.run_logger = RunLogger(
            config=self.config.logging,
            output_directory=self.config.logging.output_directory
        )
        
        self.logger.debug(f"Initialized run logger with output directory: {self.config.logging.output_directory}")
    
    def _initialize_conversation_graph(self) -> None:
        """Initialize the LangGraph conversation orchestrator."""
        self.conversation_graph = ConversationGraph(
            agent_a=self.agent_a,
            agent_b=self.agent_b,
            context_manager=self.context_manager,
            telemetry_logger=self.telemetry_logger,
            max_turns=self.config.conversation.max_turns,
            turn_timeout=self.config.conversation.turn_timeout,
            initial_prompt=self.config.conversation.initial_prompt
        )
        
        self.logger.debug(
            f"Initialized conversation graph with max {self.config.conversation.max_turns} turns"
        )
    
    def _initialize_error_handling(self) -> None:
        """Initialize error handling and recovery mechanisms."""
        # Initialize circuit breakers for different components
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,  # Open after 3 failures
            recovery_timeout=30.0,  # Wait 30s before trying again
            success_threshold=2,   # Need 2 successes to close
            timeout=self.config.conversation.turn_timeout
        )
        
        # Circuit breaker for agent A
        self.circuit_breakers["agent_a"] = get_circuit_breaker(
            f"{self.orchestrator_id}_agent_a", 
            circuit_config
        )
        
        # Circuit breaker for agent B
        self.circuit_breakers["agent_b"] = get_circuit_breaker(
            f"{self.orchestrator_id}_agent_b", 
            circuit_config
        )
        
        # Circuit breaker for conversation execution
        self.circuit_breakers["conversation"] = get_circuit_breaker(
            f"{self.orchestrator_id}_conversation", 
            circuit_config
        )
        
        # Initialize error recovery strategies
        self.error_recovery_strategies = {
            "context_overflow": self._recover_from_context_overflow,
            "agent_failure": self._recover_from_agent_failure,
            "timeout": self._recover_from_timeout,
            "api_failure": self._recover_from_api_failure
        }
        
        self.logger.debug("Initialized error handling and recovery mechanisms")
    
    async def _recover_from_context_overflow(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recover from context window overflow errors.
        
        Args:
            error: The original error
            context: Context information about the error
            
        Returns:
            Dict containing recovery information
        """
        self.logger.warning("Attempting recovery from context overflow")
        
        try:
            # Force context management with aggressive truncation
            if self.context_manager and "conversation_state" in context:
                conversation_state = context["conversation_state"]
                
                # Apply more aggressive context management
                original_threshold = self.context_manager.threshold_percentage
                self.context_manager.threshold_percentage = 50.0  # More aggressive
                
                result = self.context_manager.manage_conversation_context(
                    conversation_state, 
                    force_apply=True
                )
                
                # Restore original threshold
                self.context_manager.threshold_percentage = original_threshold
                
                self.logger.info(f"Context overflow recovery: removed {result.messages_removed} messages")
                
                return {
                    "recovery_successful": True,
                    "strategy": "aggressive_context_truncation",
                    "messages_removed": result.messages_removed,
                    "tokens_saved": result.tokens_saved
                }
        except Exception as recovery_error:
            self.logger.error(f"Context overflow recovery failed: {recovery_error}")
        
        return {
            "recovery_successful": False,
            "strategy": "aggressive_context_truncation",
            "error": str(error)
        }
    
    async def _recover_from_agent_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recover from agent failure errors.
        
        Args:
            error: The original error
            context: Context information about the error
            
        Returns:
            Dict containing recovery information
        """
        agent_id = context.get("agent_id", "unknown")
        self.logger.warning(f"Attempting recovery from agent {agent_id} failure")
        
        try:
            # Check if we can retry with simplified prompt
            if "conversation_context" in context:
                conv_context = context["conversation_context"]
                
                # Simplify the system prompt for retry
                original_prompt = conv_context.system_prompt
                simplified_prompt = "You are a helpful assistant. Please provide a brief response."
                conv_context.system_prompt = simplified_prompt
                
                self.logger.info(f"Agent {agent_id} recovery: using simplified prompt")
                
                return {
                    "recovery_successful": True,
                    "strategy": "simplified_prompt",
                    "original_prompt_length": len(original_prompt),
                    "simplified_prompt_length": len(simplified_prompt)
                }
        except Exception as recovery_error:
            self.logger.error(f"Agent failure recovery failed: {recovery_error}")
        
        return {
            "recovery_successful": False,
            "strategy": "simplified_prompt",
            "error": str(error)
        }
    
    async def _recover_from_timeout(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recover from timeout errors.
        
        Args:
            error: The original error
            context: Context information about the error
            
        Returns:
            Dict containing recovery information
        """
        self.logger.warning("Attempting recovery from timeout")
        
        try:
            # Increase timeout for next attempt
            agent_id = context.get("agent_id")
            if agent_id and agent_id in ["agent_a", "agent_b"]:
                agent = self.agent_a if agent_id == "agent_a" else self.agent_b
                
                # Increase timeout by 50%
                original_timeout = agent.timeout
                agent.timeout = min(original_timeout * 1.5, 120.0)  # Cap at 2 minutes
                
                self.logger.info(f"Timeout recovery: increased timeout from {original_timeout}s to {agent.timeout}s")
                
                return {
                    "recovery_successful": True,
                    "strategy": "increased_timeout",
                    "original_timeout": original_timeout,
                    "new_timeout": agent.timeout
                }
        except Exception as recovery_error:
            self.logger.error(f"Timeout recovery failed: {recovery_error}")
        
        return {
            "recovery_successful": False,
            "strategy": "increased_timeout",
            "error": str(error)
        }
    
    async def _recover_from_api_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recover from API failure errors.
        
        Args:
            error: The original error
            context: Context information about the error
            
        Returns:
            Dict containing recovery information
        """
        self.logger.warning("Attempting recovery from API failure")
        
        try:
            # Implement exponential backoff
            attempt = context.get("attempt", 1)
            backoff_delay = min(2 ** attempt, 30)  # Cap at 30 seconds
            
            self.logger.info(f"API failure recovery: waiting {backoff_delay}s before retry")
            await asyncio.sleep(backoff_delay)
            
            return {
                "recovery_successful": True,
                "strategy": "exponential_backoff",
                "backoff_delay": backoff_delay,
                "attempt": attempt
            }
        except Exception as recovery_error:
            self.logger.error(f"API failure recovery failed: {recovery_error}")
        
        return {
            "recovery_successful": False,
            "strategy": "exponential_backoff",
            "error": str(error)
        }
    
    async def _attempt_error_recovery(
        self, 
        error: Exception, 
        error_type: str, 
        context: Dict[str, Any],
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        Attempt to recover from an error using appropriate strategy.
        
        Args:
            error: The original error
            error_type: Type of error for strategy selection
            context: Context information about the error
            attempt: Current attempt number
            
        Returns:
            Dict containing recovery results
        """
        if attempt > self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts ({self.max_recovery_attempts}) exceeded for {error_type}")
            return {
                "recovery_successful": False,
                "strategy": "max_attempts_exceeded",
                "attempts": attempt,
                "error": str(error)
            }
        
        context["attempt"] = attempt
        
        # Select appropriate recovery strategy
        if error_type in self.error_recovery_strategies:
            recovery_func = self.error_recovery_strategies[error_type]
            try:
                return await recovery_func(error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy {error_type} failed: {recovery_error}")
                return {
                    "recovery_successful": False,
                    "strategy": error_type,
                    "recovery_error": str(recovery_error),
                    "original_error": str(error)
                }
        else:
            self.logger.warning(f"No recovery strategy available for error type: {error_type}")
            return {
                "recovery_successful": False,
                "strategy": "no_strategy_available",
                "error_type": error_type,
                "error": str(error)
            }
    
    def get_circuit_breaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all circuit breakers.
        
        Returns:
            Dict containing circuit breaker statistics
        """
        return {
            name: breaker.get_stats() 
            for name, breaker in self.circuit_breakers.items()
        }
    
    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        for breaker in self.circuit_breakers.values():
            breaker.reset()
        self.logger.info("All circuit breakers reset to closed state")
    
    async def run_conversation(
        self,
        conversation_id: Optional[str] = None,
        display_progress: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run a complete conversation between the two agents.
        
        Args:
            conversation_id: Optional unique identifier for this conversation
            display_progress: Whether to display real-time progress to console
            save_results: Whether to save conversation results to files
            
        Returns:
            Dict containing conversation results and telemetry data
            
        Raises:
            OrchestrationError: If conversation execution fails
        """
        if conversation_id is None:
            conversation_id = f"conv_{self.orchestrator_id}_{int(time.time())}"
        
        self.logger.info(f"Starting conversation: {conversation_id}")
        
        try:
            # Initialize telemetry logger for this run
            self.telemetry_logger = TelemetryLogger(
                config=self.config.logging,
                run_id=conversation_id
            )
            
            # Update conversation graph with telemetry logger
            self.conversation_graph.telemetry_logger = self.telemetry_logger
            
            # Start run tracking
            if save_results:
                self.run_logger.start_run(conversation_id, self.config.to_dict())
            
            # Display initial information
            if display_progress:
                self._display_conversation_start(conversation_id)
            
            # Execute the conversation
            start_time = time.time()
            final_state = await self.conversation_graph.run_conversation(
                conversation_id=conversation_id
            )
            end_time = time.time()
            
            # Finalize telemetry
            run_telemetry = self.telemetry_logger.finalize(self.config.to_dict())
            
            # Complete run tracking
            if save_results:
                self.run_logger.complete_run(conversation_id, run_telemetry)
            
            # Display final results
            if display_progress:
                self._display_conversation_results(final_state, end_time - start_time)
            
            # Prepare results
            results = {
                "conversation_id": conversation_id,
                "status": final_state.status.value,
                "total_turns": final_state.current_turn,
                "total_messages": len(final_state.messages),
                "total_tokens": final_state.get_total_tokens(),
                "duration_seconds": end_time - start_time,
                "final_state": final_state.to_dict(),
                "telemetry": run_telemetry.to_dict(),
                "agent_info": {
                    "agent_a": self.agent_a.get_agent_info().to_dict(),
                    "agent_b": self.agent_b.get_agent_info().to_dict()
                },
                "configuration": self.config.to_dict()
            }
            
            self.logger.info(
                f"Conversation {conversation_id} completed successfully: "
                f"{final_state.current_turn} turns, {final_state.get_total_tokens()} tokens, "
                f"{end_time - start_time:.2f}s"
            )
            
            return results
            
        except ConversationError as e:
            error_msg = f"Conversation {conversation_id} failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Try to finalize telemetry even on error
            if self.telemetry_logger:
                try:
                    run_telemetry = self.telemetry_logger.finalize(self.config.to_dict())
                    if save_results:
                        self.run_logger.complete_run(conversation_id, run_telemetry)
                except Exception:
                    pass  # Don't fail on telemetry errors
            
            raise OrchestrationError(
                error_msg,
                orchestrator_id=self.orchestrator_id,
                original_error=e
            )
            
        except Exception as e:
            error_msg = f"Unexpected error in conversation {conversation_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Try to finalize telemetry even on error
            if self.telemetry_logger:
                try:
                    run_telemetry = self.telemetry_logger.finalize(self.config.to_dict())
                    if save_results:
                        self.run_logger.complete_run(conversation_id, run_telemetry)
                except Exception:
                    pass  # Don't fail on telemetry errors
            
            raise OrchestrationError(
                error_msg,
                orchestrator_id=self.orchestrator_id,
                original_error=e
            )
    
    def _display_conversation_start(self, conversation_id: str) -> None:
        """Display conversation start information to console."""
        if not self.config.logging.real_time_display:
            return
        
        print(f"\n{'='*60}")
        print(f"CONVERSATION START: {conversation_id}")
        print(f"{'='*60}")
        print(f"Agent A: {self.agent_a.name}")
        print(f"Agent B: {self.agent_b.name}")
        print(f"Model: {self.config.model.model_name}")
        print(f"Max Turns: {self.config.conversation.max_turns}")
        print(f"Context Strategy: {self.config.conversation.context_window_strategy}")
        print(f"Context Size: {self.config.conversation.context_window_size} tokens")
        
        if self.config.conversation.initial_prompt:
            print(f"Initial Prompt: {self.config.conversation.initial_prompt[:100]}...")
        
        print(f"{'='*60}\n")
    
    def _display_conversation_results(self, final_state: ConversationState, duration: float) -> None:
        """Display conversation results to console."""
        if not self.config.logging.real_time_display:
            return
        
        print(f"\n{'='*60}")
        print(f"CONVERSATION COMPLETE")
        print(f"{'='*60}")
        print(f"Status: {final_state.status.value}")
        print(f"Total Turns: {final_state.current_turn}")
        print(f"Total Messages: {len(final_state.messages)}")
        print(f"Total Tokens: {final_state.get_total_tokens()}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Context Utilization: {final_state.get_context_utilization():.1f}%")
        
        # Display agent message counts
        agent_a_messages = len(final_state.get_messages_by_agent("agent_a"))
        agent_b_messages = len(final_state.get_messages_by_agent("agent_b"))
        print(f"Agent A Messages: {agent_a_messages}")
        print(f"Agent B Messages: {agent_b_messages}")
        
        print(f"{'='*60}\n")
        
        # Display recent conversation excerpt
        if final_state.messages:
            print("RECENT CONVERSATION:")
            print("-" * 40)
            recent_messages = final_state.messages[-4:]  # Show last 4 messages
            for msg in recent_messages:
                agent_name = self.agent_a.name if msg.agent_id == "agent_a" else self.agent_b.name
                print(f"{agent_name}: {msg.content[:200]}...")
                print()
    
    def get_orchestrator_info(self) -> Dict[str, Any]:
        """
        Get information about this orchestrator and its configuration.
        
        Returns:
            Dict containing orchestrator information
        """
        info = {
            "orchestrator_id": self.orchestrator_id,
            "status": "initialized" if self.config else "not_configured",
            "components": {
                "config_loaded": self.config is not None,
                "agents_initialized": self.agent_a is not None and self.agent_b is not None,
                "context_manager_initialized": self.context_manager is not None,
                "conversation_graph_initialized": self.conversation_graph is not None,
                "telemetry_initialized": self.run_logger is not None
            }
        }
        
        if self.config:
            info["configuration"] = {
                "model": self.config.model.model_name,
                "max_turns": self.config.conversation.max_turns,
                "context_strategy": self.config.conversation.context_window_strategy,
                "context_size": self.config.conversation.context_window_size,
                "agents": {
                    "agent_a": self.config.agent_a.name,
                    "agent_b": self.config.agent_b.name
                }
            }
        
        if self.conversation_graph:
            info["conversation_graph"] = self.conversation_graph.get_conversation_info()
        
        return info
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Dict containing validation results
        """
        if not self.config:
            return {
                "is_valid": False,
                "errors": ["No configuration loaded"],
                "warnings": []
            }
        
        validation_summary = self.config.get_validation_summary()
        
        # Add orchestrator-specific validations
        warnings = []
        
        # Check for potential issues
        if self.config.conversation.max_turns > 100:
            warnings.append("Very high max_turns setting may result in long conversations")
        
        if self.config.conversation.context_window_size < 1000:
            warnings.append("Very small context window may limit conversation quality")
        
        if self.config.model.temperature > 1.5:
            warnings.append("High temperature setting may result in unpredictable responses")
        
        validation_summary["warnings"] = warnings
        
        return validation_summary
    
    def get_conversation_history(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get conversation history for a specific conversation ID.
        
        Args:
            conversation_id: ID of the conversation to retrieve
            
        Returns:
            Dict containing conversation history, or None if not found
        """
        if not self.run_logger:
            return None
        
        run_data = self.run_logger.load_run_data(conversation_id)
        if not run_data:
            return None
        
        return {
            "conversation_id": conversation_id,
            "messages": [msg.to_dict() for msg in run_data.conversation_history],
            "telemetry": run_data.to_dict(),
            "status": "completed" if run_data.is_completed() else "incomplete"
        }
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all saved conversations.
        
        Returns:
            List of conversation summaries
        """
        if not self.run_logger:
            return []
        
        return self.run_logger.list_saved_runs()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status information.
        
        Returns:
            Dict containing system status
        """
        status = {
            "orchestrator": self.get_orchestrator_info(),
            "configuration": self.validate_configuration(),
            "active_conversations": [],
            "saved_conversations_count": len(self.list_conversations()),
            "system_health": "healthy"
        }
        
        # Check for any issues
        if not status["configuration"]["is_valid"]:
            status["system_health"] = "configuration_error"
        elif not status["orchestrator"]["components"]["agents_initialized"]:
            status["system_health"] = "initialization_error"
        
        return status
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the orchestrator and clean up resources.
        """
        self.logger.info(f"Shutting down orchestrator '{self.orchestrator_id}'")
        
        # Close telemetry logger if active
        if self.telemetry_logger:
            try:
                self.telemetry_logger.close()
            except Exception as e:
                self.logger.warning(f"Error closing telemetry logger: {e}")
        
        # Clear component references
        self.agent_a = None
        self.agent_b = None
        self.context_manager = None
        self.conversation_graph = None
        self.telemetry_logger = None
        
        self.logger.info("Orchestrator shutdown complete")


# Factory functions for creating orchestrators

def create_orchestrator_from_config(
    config: SystemConfig,
    orchestrator_id: Optional[str] = None
) -> ConversationOrchestrator:
    """
    Factory function to create an orchestrator from a SystemConfig object.
    
    Args:
        config: System configuration
        orchestrator_id: Optional orchestrator ID
        
    Returns:
        ConversationOrchestrator: Configured orchestrator
    """
    return ConversationOrchestrator(
        orchestrator_id=orchestrator_id,
        config=config
    )


def create_orchestrator_from_file(
    config_path: Union[str, Path],
    orchestrator_id: Optional[str] = None
) -> ConversationOrchestrator:
    """
    Factory function to create an orchestrator from a configuration file.
    
    Args:
        config_path: Path to configuration file
        orchestrator_id: Optional orchestrator ID
        
    Returns:
        ConversationOrchestrator: Configured orchestrator
    """
    return ConversationOrchestrator(
        orchestrator_id=orchestrator_id,
        config_path=config_path
    )


async def run_single_conversation(
    config_path: Union[str, Path],
    conversation_id: Optional[str] = None,
    display_progress: bool = True,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run a single conversation from a configuration file.
    
    Args:
        config_path: Path to configuration file
        conversation_id: Optional conversation ID
        display_progress: Whether to display progress
        save_results: Whether to save results
        
    Returns:
        Dict containing conversation results
    """
    orchestrator = create_orchestrator_from_file(config_path)
    
    try:
        return await orchestrator.run_conversation(
            conversation_id=conversation_id,
            display_progress=display_progress,
            save_results=save_results
        )
    finally:
        await orchestrator.shutdown()