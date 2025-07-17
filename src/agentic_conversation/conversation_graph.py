"""
LangGraph conversation orchestrator for managing agent-to-agent conversations.

This module implements the LangGraph-based state machine that orchestrates
conversations between two agents, managing state transitions, error handling,
and conversation flow control.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .graph_state import (
    GraphState,
    ControlAction,
    StateTransitionManager,
    create_initial_graph_state,
    update_state_with_response,
    update_state_with_error,
)
from .models import ConversationState, ConversationStatus, Message
from .agents import BaseAgent, ConversationContext, AgentError
from .context_manager import ContextManager
from .telemetry import TelemetryLogger


class ConversationGraph:
    """
    LangGraph-based conversation orchestrator.
    
    This class manages the conversation flow between two agents using LangGraph's
    StateGraph to handle state transitions, error recovery, and termination conditions.
    """
    
    def __init__(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        context_manager: ContextManager,
        telemetry_logger: Optional[TelemetryLogger] = None,
        max_turns: int = 20,
        turn_timeout: float = 30.0,
        initial_prompt: Optional[str] = None
    ):
        """
        Initialize the conversation graph.
        
        Args:
            agent_a: First agent in the conversation
            agent_b: Second agent in the conversation
            context_manager: Context window manager
            telemetry_logger: Optional telemetry logger
            max_turns: Maximum number of conversation turns
            turn_timeout: Maximum time per turn in seconds
            initial_prompt: Optional initial prompt to start the conversation
        """
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.context_manager = context_manager
        self.telemetry_logger = telemetry_logger
        self.max_turns = max_turns
        self.turn_timeout = turn_timeout
        self.initial_prompt = initial_prompt
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize state transition manager
        self.state_manager = StateTransitionManager(
            max_turns=max_turns,
            turn_timeout=turn_timeout
        )
        
        # Build the LangGraph state machine
        self.graph = self._build_graph()
        
        # Initialize memory saver for state persistence
        self.memory = MemorySaver()
        
        # Compile the graph with memory
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        self.logger.info(
            f"Initialized ConversationGraph with agents: {agent_a.agent_id}, {agent_b.agent_id}"
        )
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine for conversation orchestration.
        
        Returns:
            StateGraph: The compiled conversation state graph
        """
        # Create the state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes for each agent and conversation management
        workflow.add_node("agent_a_turn", self._agent_a_node)
        workflow.add_node("agent_b_turn", self._agent_b_node)
        workflow.add_node("conversation_manager", self._conversation_manager_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("conversation_manager")
        
        # Add edges for conversation flow
        workflow.add_conditional_edges(
            "conversation_manager",
            self._route_conversation,
            {
                "agent_a": "agent_a_turn",
                "agent_b": "agent_b_turn",
                "terminate": END,
                "error": "error_handler"
            }
        )
        
        # Add edges from agent nodes back to conversation manager
        workflow.add_edge("agent_a_turn", "conversation_manager")
        workflow.add_edge("agent_b_turn", "conversation_manager")
        workflow.add_edge("error_handler", END)
        
        return workflow
    
    def _route_conversation(self, state: GraphState) -> str:
        """
        Route the conversation based on current state.
        
        Args:
            state: Current graph state
            
        Returns:
            str: Next node to execute
        """
        # Check for error conditions first
        if state.get("error_message") or state["conversation_state"].status == ConversationStatus.ERROR:
            return "error"
        
        # Check for termination conditions
        current_time = time.time()
        transition_result = self.state_manager.execute_state_transition(state, current_time)
        
        if not transition_result.should_continue:
            self.logger.info(f"Terminating conversation: {transition_result.termination_reason}")
            return "terminate"
        
        # Route to appropriate agent
        current_agent = state["current_agent"]
        if current_agent == "agent_a":
            return "agent_a"
        elif current_agent == "agent_b":
            return "agent_b"
        else:
            self.logger.error(f"Unknown agent: {current_agent}")
            return "error"
    
    async def _agent_a_node(self, state: GraphState) -> GraphState:
        """
        Execute Agent A's turn.
        
        Args:
            state: Current graph state
            
        Returns:
            GraphState: Updated state after agent's turn
        """
        return await self._execute_agent_turn(state, self.agent_a)
    
    async def _agent_b_node(self, state: GraphState) -> GraphState:
        """
        Execute Agent B's turn.
        
        Args:
            state: Current graph state
            
        Returns:
            GraphState: Updated state after agent's turn
        """
        return await self._execute_agent_turn(state, self.agent_b)
    
    async def _execute_agent_turn(self, state: GraphState, agent: BaseAgent) -> GraphState:
        """
        Execute a turn for the specified agent.
        
        Args:
            state: Current graph state
            agent: Agent to execute the turn for
            
        Returns:
            GraphState: Updated state after agent's turn
        """
        turn_start_time = time.time()
        state["turn_start_time"] = turn_start_time
        
        try:
            self.logger.info(f"Executing turn for agent: {agent.agent_id}")
            
            # Prepare conversation context
            context = await self._prepare_agent_context(state, agent)
            
            # Generate response with timeout
            response = await asyncio.wait_for(
                agent.generate_response(context),
                timeout=self.turn_timeout
            )
            
            # Validate response
            if not response or not response.content.strip():
                raise AgentError(f"Agent {agent.agent_id} generated empty response")
            
            # Update state with response
            response_time = time.time() - turn_start_time
            updated_state = update_state_with_response(
                state,
                response.content,
                agent.agent_id,
                response_time,
                response.token_count
            )
            
            # Log telemetry if available
            if self.telemetry_logger:
                await self._log_agent_turn(agent, response, context, response_time)
            
            self.logger.info(
                f"Agent {agent.agent_id} completed turn in {response_time:.2f}s "
                f"({response.token_count} tokens)"
            )
            
            return updated_state
            
        except asyncio.TimeoutError:
            error_msg = f"Agent {agent.agent_id} turn timed out after {self.turn_timeout}s"
            self.logger.error(error_msg)
            return update_state_with_error(state, error_msg, agent.agent_id)
            
        except AgentError as e:
            error_msg = f"Agent {agent.agent_id} error: {e.message}"
            self.logger.error(error_msg)
            return update_state_with_error(state, error_msg, agent.agent_id)
            
        except Exception as e:
            error_msg = f"Unexpected error in agent {agent.agent_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return update_state_with_error(state, error_msg, agent.agent_id)
    
    async def _prepare_agent_context(self, state: GraphState, agent: BaseAgent) -> ConversationContext:
        """
        Prepare conversation context for an agent.
        
        Args:
            state: Current graph state
            agent: Agent to prepare context for
            
        Returns:
            ConversationContext: Prepared context for the agent
        """
        conversation_state = state["conversation_state"]
        
        # Apply context management if needed
        context_result = self.context_manager.manage_conversation_context(conversation_state)
        managed_state = conversation_state  # The context manager updates the state in place
        
        # Determine other agent ID
        other_agent_id = "agent_b" if agent.agent_id == "agent_a" else "agent_a"
        
        # Calculate available tokens
        available_tokens = max(0, self.context_manager.max_context_tokens - managed_state.current_context_tokens)
        
        # Create conversation context
        context = ConversationContext(
            conversation_state=managed_state,
            system_prompt=agent.system_prompt,
            available_tokens=available_tokens,
            turn_number=managed_state.current_turn,
            other_agent_id=other_agent_id,
            metadata={
                "max_turns": self.max_turns,
                "turn_timeout": self.turn_timeout,
                "context_utilization": managed_state.get_context_utilization()
            }
        )
        
        return context
    
    async def _conversation_manager_node(self, state: GraphState) -> GraphState:
        """
        Manage conversation flow and state transitions.
        
        Args:
            state: Current graph state
            
        Returns:
            GraphState: Updated state after management
        """
        current_time = time.time()
        
        # Execute state transition
        transition_result = self.state_manager.execute_state_transition(state, current_time)
        
        # Update state based on transition result
        new_state = transition_result.new_state
        
        # Log transition if telemetry is available
        if self.telemetry_logger:
            await self._log_state_transition(transition_result)
        
        self.logger.debug(f"Conversation manager: {transition_result.action_taken}")
        
        return new_state
    
    async def _error_handler_node(self, state: GraphState) -> GraphState:
        """
        Handle error conditions and attempt recovery.
        
        Args:
            state: Current graph state with error
            
        Returns:
            GraphState: State after error handling
        """
        error_message = state.get("error_message", "Unknown error")
        self.logger.error(f"Handling conversation error: {error_message}")
        
        # Log error if telemetry is available
        if self.telemetry_logger:
            await self._log_conversation_error(state, error_message)
        
        # Mark conversation as terminated due to error
        state["conversation_state"].status = ConversationStatus.ERROR
        state["should_continue"] = False
        state["control_action"] = ControlAction.TERMINATE_ERROR
        
        return state
    
    async def _log_agent_turn(
        self,
        agent: BaseAgent,
        response: Any,
        context: ConversationContext,
        response_time: float
    ) -> None:
        """
        Log agent turn telemetry.
        
        Args:
            agent: Agent that executed the turn
            response: Agent's response
            context: Conversation context
            response_time: Time taken for the turn
        """
        if not self.telemetry_logger:
            return
        
        try:
            # Log agent response
            message = response.to_message() if hasattr(response, 'to_message') else Message(
                agent_id=agent.agent_id,
                content=response.content if hasattr(response, 'content') else str(response),
                timestamp=datetime.now(),
                token_count=getattr(response, 'token_count', 0),
                metadata={
                    "turn_number": context.turn_number,
                    "context_utilization": context.get_context_utilization(),
                    "available_tokens": context.available_tokens,
                    "response_time": response_time,
                    "model_calls": getattr(response, 'model_calls', 1)
                }
            )
            self.telemetry_logger.log_agent_response(
                agent_id=agent.agent_id,
                message=message,
                response_time=response_time,
                model_calls=getattr(response, 'model_calls', 1),
                errors=getattr(response, 'errors', [])
            )
        except Exception as e:
            self.logger.warning(f"Failed to log agent turn telemetry: {e}")
    
    async def _log_state_transition(self, transition_result: Any) -> None:
        """
        Log state transition telemetry.
        
        Args:
            transition_result: Result of the state transition
        """
        if not self.telemetry_logger:
            return
        
        try:
            # Log state transition
            self.telemetry_logger.log_system_event(
                event_type="state_transition",
                message=f"State transition: {transition_result.action_taken}",
                data={
                    "action_taken": transition_result.action_taken,
                    "should_continue": transition_result.should_continue,
                    "termination_reason": transition_result.termination_reason
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to log state transition telemetry: {e}")
    
    async def _log_conversation_error(self, state: GraphState, error_message: str) -> None:
        """
        Log conversation error telemetry.
        
        Args:
            state: Current graph state
            error_message: Error message
        """
        if not self.telemetry_logger:
            return
        
        try:
            # Log error event
            self.telemetry_logger.log_error(
                error_message=error_message,
                context={
                    "current_agent": state.get("current_agent"),
                    "turn_number": state["conversation_state"].current_turn,
                    "conversation_status": state["conversation_state"].status.value
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to log conversation error telemetry: {e}")
    
    async def run_conversation(
        self,
        conversation_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> ConversationState:
        """
        Run a complete conversation between the two agents.
        
        Args:
            conversation_id: Optional unique identifier for this conversation
            config: Optional configuration for the conversation run
            
        Returns:
            ConversationState: Final state of the conversation
            
        Raises:
            ConversationError: If conversation execution fails
        """
        if conversation_id is None:
            conversation_id = f"conv_{int(time.time())}"
        
        self.logger.info(f"Starting conversation: {conversation_id}")
        
        try:
            # Initialize conversation state
            conversation_state = ConversationState(
                max_context_tokens=self.context_manager.max_context_tokens
            )
            conversation_state.status = ConversationStatus.RUNNING
            
            # Add initial prompt if provided
            if self.initial_prompt:
                initial_message = Message(
                    agent_id="system",
                    content=self.initial_prompt,
                    timestamp=datetime.now(),
                    token_count=len(self.initial_prompt.split()),  # Rough estimate
                    metadata={"type": "initial_prompt"}
                )
                conversation_state.add_message(initial_message)
            
            # Create initial graph state
            current_state = create_initial_graph_state(conversation_state, "agent_a")
            
            # Start telemetry logging if available
            if self.telemetry_logger:
                self.telemetry_logger.log_system_event(
                    event_type="conversation_start",
                    message=f"Starting conversation {conversation_id}",
                    data={
                        "run_id": conversation_id,
                        "agent_ids": [self.agent_a.agent_id, self.agent_b.agent_id],
                        "configuration": {
                            "max_turns": self.max_turns,
                            "turn_timeout": self.turn_timeout,
                            "initial_prompt": self.initial_prompt
                        }
                    }
                )
            
            # Execute conversation loop manually for better control
            while current_state["should_continue"]:
                # Determine next action
                current_time = time.time()
                transition_result = self.state_manager.execute_state_transition(current_state, current_time)
                
                if not transition_result.should_continue:
                    self.logger.info(f"Terminating conversation: {transition_result.termination_reason}")
                    current_state = transition_result.new_state
                    break
                
                # Check for error conditions
                if current_state.get("error_message") or current_state["conversation_state"].status == ConversationStatus.ERROR:
                    current_state = await self._error_handler_node(current_state)
                    error_msg = current_state.get("error_message", "Unknown error occurred")
                    raise ConversationError(f"Conversation {conversation_id} failed: {error_msg}")
                
                # Execute agent turn
                current_agent_id = current_state["current_agent"]
                if current_agent_id == "agent_a":
                    current_state = await self._execute_agent_turn(current_state, self.agent_a)
                elif current_agent_id == "agent_b":
                    current_state = await self._execute_agent_turn(current_state, self.agent_b)
                else:
                    error_msg = f"Unknown agent: {current_agent_id}"
                    self.logger.error(error_msg)
                    current_state = update_state_with_error(current_state, error_msg)
                    error_msg = current_state.get("error_message", "Unknown error occurred")
                    raise ConversationError(f"Conversation {conversation_id} failed: {error_msg}")
                
                # Check if agent turn resulted in error
                if current_state.get("error_message") or current_state["conversation_state"].status == ConversationStatus.ERROR:
                    current_state = await self._error_handler_node(current_state)
                    error_msg = current_state.get("error_message", "Unknown error occurred")
                    raise ConversationError(f"Conversation {conversation_id} failed: {error_msg}")
                
                # Update state for next iteration
                # Switch to next agent
                next_agent = "agent_b" if current_agent_id == "agent_a" else "agent_a"
                current_state["current_agent"] = next_agent
                
                # Log intermediate states if needed
                if self.logger.isEnabledFor(logging.DEBUG):
                    turn_number = current_state["conversation_state"].current_turn
                    self.logger.debug(
                        f"Conversation {conversation_id}: Turn {turn_number}, Next Agent: {current_state['current_agent']}"
                    )
            
            # Extract final conversation state
            final_conversation_state = current_state["conversation_state"]
            
            # Mark conversation as completed if not already marked as error
            if final_conversation_state.status == ConversationStatus.RUNNING:
                final_conversation_state.status = ConversationStatus.COMPLETED
            
            # End telemetry logging if available
            if self.telemetry_logger:
                self.telemetry_logger.log_system_event(
                    event_type="conversation_end",
                    message=f"Conversation {conversation_id} completed",
                    data={
                        "run_id": conversation_id,
                        "final_status": final_conversation_state.status.value,
                        "total_turns": final_conversation_state.current_turn,
                        "total_messages": len(final_conversation_state.messages),
                        "total_tokens": final_conversation_state.get_total_tokens()
                    }
                )
            
            self.logger.info(
                f"Conversation {conversation_id} completed with {final_conversation_state.current_turn} turns"
            )
            
            return final_conversation_state
            
        except Exception as e:
            error_msg = f"Conversation {conversation_id} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Log error if telemetry is available
            if self.telemetry_logger:
                try:
                    self.telemetry_logger.log_error(
                        error_message=f"Conversation {conversation_id} failed: {str(e)}",
                        context={
                            "conversation_id": conversation_id,
                            "error_type": type(e).__name__
                        }
                    )
                except Exception:
                    pass  # Don't fail on telemetry errors
            
            raise ConversationError(error_msg) from e
    
    def get_conversation_info(self) -> Dict[str, Any]:
        """
        Get information about this conversation graph configuration.
        
        Returns:
            Dict[str, Any]: Configuration information
        """
        return {
            "agent_a": self.agent_a.get_agent_info().to_dict(),
            "agent_b": self.agent_b.get_agent_info().to_dict(),
            "max_turns": self.max_turns,
            "turn_timeout": self.turn_timeout,
            "initial_prompt": self.initial_prompt,
            "context_manager": {
                "max_context_tokens": self.context_manager.max_context_tokens,
                "strategy": getattr(self.context_manager, 'strategy', 'unknown')
            }
        }


class ConversationError(Exception):
    """
    Exception raised when conversation execution fails.
    
    Attributes:
        message: Error message
        conversation_id: ID of the conversation that failed
        original_error: Original exception that caused this error
    """
    
    def __init__(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.conversation_id = conversation_id
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "conversation_id": self.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "original_error": str(self.original_error) if self.original_error else None
        }


# Factory function for creating conversation graphs
def create_conversation_graph(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    context_manager: ContextManager,
    telemetry_logger: Optional[TelemetryLogger] = None,
    **kwargs
) -> ConversationGraph:
    """
    Factory function to create a ConversationGraph instance.
    
    Args:
        agent_a: First agent in the conversation
        agent_b: Second agent in the conversation
        context_manager: Context window manager
        telemetry_logger: Optional telemetry logger
        **kwargs: Additional configuration options
        
    Returns:
        ConversationGraph: Configured conversation graph
    """
    return ConversationGraph(
        agent_a=agent_a,
        agent_b=agent_b,
        context_manager=context_manager,
        telemetry_logger=telemetry_logger,
        **kwargs
    )