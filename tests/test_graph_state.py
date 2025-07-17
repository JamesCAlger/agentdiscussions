"""
Unit tests for LangGraph state definitions and transition functions.

This module tests the graph state management, state transitions, and
termination conditions for the conversation orchestration system.
"""

import pytest
from datetime import datetime
from typing import Dict, Any
import time

from agentic_conversation.graph_state import (
    GraphState,
    ControlAction,
    StateTransitionManager,
    StateTransitionResult,
    create_initial_graph_state,
    update_state_with_response,
    update_state_with_error,
)
from agentic_conversation.models import (
    ConversationState,
    ConversationStatus,
    Message,
)


class TestGraphState:
    """Test cases for GraphState TypedDict structure."""
    
    def test_graph_state_structure(self):
        """Test that GraphState has all required fields."""
        conversation_state = ConversationState()
        
        state: GraphState = {
            "conversation_state": conversation_state,
            "current_agent": "agent_a",
            "should_continue": True,
            "control_action": ControlAction.CONTINUE,
            "error_message": None,
            "last_response": None,
            "turn_start_time": None,
            "metadata": {}
        }
        
        # Verify all required fields are present
        assert "conversation_state" in state
        assert "current_agent" in state
        assert "should_continue" in state
        assert "control_action" in state
        assert "error_message" in state
        assert "last_response" in state
        assert "turn_start_time" in state
        assert "metadata" in state
        
        # Verify field types
        assert isinstance(state["conversation_state"], ConversationState)
        assert isinstance(state["current_agent"], str)
        assert isinstance(state["should_continue"], bool)
        assert isinstance(state["control_action"], ControlAction)
        assert isinstance(state["metadata"], dict)


class TestControlAction:
    """Test cases for ControlAction enum."""
    
    def test_control_action_values(self):
        """Test that all expected control actions are defined."""
        expected_actions = {
            "continue",
            "switch_agent", 
            "terminate_success",
            "terminate_error",
            "terminate_max_turns",
            "terminate_timeout"
        }
        
        actual_actions = {action.value for action in ControlAction}
        assert actual_actions == expected_actions
    
    def test_control_action_enum_access(self):
        """Test accessing control actions by name."""
        assert ControlAction.CONTINUE.value == "continue"
        assert ControlAction.SWITCH_AGENT.value == "switch_agent"
        assert ControlAction.TERMINATE_SUCCESS.value == "terminate_success"
        assert ControlAction.TERMINATE_ERROR.value == "terminate_error"
        assert ControlAction.TERMINATE_MAX_TURNS.value == "terminate_max_turns"
        assert ControlAction.TERMINATE_TIMEOUT.value == "terminate_timeout"


class TestStateTransitionManager:
    """Test cases for StateTransitionManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a StateTransitionManager for testing."""
        return StateTransitionManager(max_turns=5, turn_timeout=10.0)
    
    @pytest.fixture
    def base_state(self):
        """Create a base GraphState for testing."""
        conversation_state = ConversationState()
        return create_initial_graph_state(conversation_state, "agent_a")
    
    def test_manager_initialization(self):
        """Test StateTransitionManager initialization."""
        manager = StateTransitionManager(max_turns=10, turn_timeout=30.0)
        assert manager.max_turns == 10
        assert manager.turn_timeout == 30.0
    
    def test_should_terminate_max_turns(self, manager, base_state):
        """Test termination condition for maximum turns."""
        # Should not terminate with fewer turns
        base_state["conversation_state"].current_turn = 3
        assert not manager.should_terminate_max_turns(base_state)
        
        # Should terminate when max turns reached
        base_state["conversation_state"].current_turn = 5
        assert manager.should_terminate_max_turns(base_state)
        
        # Should terminate when max turns exceeded
        base_state["conversation_state"].current_turn = 6
        assert manager.should_terminate_max_turns(base_state)
    
    def test_should_terminate_error(self, manager, base_state):
        """Test termination condition for error states."""
        # Should not terminate with no error
        assert not manager.should_terminate_error(base_state)
        
        # Should terminate with error message
        base_state["error_message"] = "Test error"
        assert manager.should_terminate_error(base_state)
        
        # Should terminate with error status
        base_state["error_message"] = None
        base_state["conversation_state"].status = ConversationStatus.ERROR
        assert manager.should_terminate_error(base_state)
    
    def test_should_terminate_timeout(self, manager, base_state):
        """Test termination condition for turn timeout."""
        current_time = time.time()
        
        # Should not terminate with no start time
        assert not manager.should_terminate_timeout(base_state, current_time)
        
        # Should not terminate within timeout
        base_state["turn_start_time"] = current_time - 5.0
        assert not manager.should_terminate_timeout(base_state, current_time)
        
        # Should terminate when timeout exceeded
        base_state["turn_start_time"] = current_time - 15.0
        assert manager.should_terminate_timeout(base_state, current_time)
    
    def test_get_next_agent(self, manager):
        """Test agent switching logic."""
        assert manager.get_next_agent("agent_a") == "agent_b"
        assert manager.get_next_agent("agent_b") == "agent_a"
    
    def test_determine_control_action_continue(self, manager, base_state):
        """Test control action determination for normal continuation."""
        current_time = time.time()
        action = manager.determine_control_action(base_state, current_time)
        assert action == ControlAction.SWITCH_AGENT
    
    def test_determine_control_action_error(self, manager, base_state):
        """Test control action determination for error conditions."""
        current_time = time.time()
        
        # Test error message
        base_state["error_message"] = "Test error"
        action = manager.determine_control_action(base_state, current_time)
        assert action == ControlAction.TERMINATE_ERROR
        
        # Test error status
        base_state["error_message"] = None
        base_state["conversation_state"].status = ConversationStatus.ERROR
        action = manager.determine_control_action(base_state, current_time)
        assert action == ControlAction.TERMINATE_ERROR
    
    def test_determine_control_action_timeout(self, manager, base_state):
        """Test control action determination for timeout."""
        current_time = time.time()
        base_state["turn_start_time"] = current_time - 15.0
        
        action = manager.determine_control_action(base_state, current_time)
        assert action == ControlAction.TERMINATE_TIMEOUT
    
    def test_determine_control_action_max_turns(self, manager, base_state):
        """Test control action determination for max turns."""
        current_time = time.time()
        base_state["conversation_state"].current_turn = 5
        
        action = manager.determine_control_action(base_state, current_time)
        assert action == ControlAction.TERMINATE_MAX_TURNS
    
    def test_determine_control_action_completed(self, manager, base_state):
        """Test control action determination for completed conversation."""
        current_time = time.time()
        base_state["conversation_state"].status = ConversationStatus.COMPLETED
        
        action = manager.determine_control_action(base_state, current_time)
        assert action == ControlAction.TERMINATE_SUCCESS
    
    def test_transition_to_next_agent(self, manager, base_state):
        """Test transitioning to the next agent."""
        result = manager.transition_to_next_agent(base_state)
        
        assert isinstance(result, StateTransitionResult)
        assert result.should_continue is True
        assert result.termination_reason is None
        assert "Switched to agent_b" in result.action_taken
        
        # Check state changes
        assert result.new_state["current_agent"] == "agent_b"
        assert result.new_state["control_action"] == ControlAction.CONTINUE
        assert result.new_state["should_continue"] is True
        assert result.new_state["turn_start_time"] is None
    
    def test_transition_to_termination_success(self, manager, base_state):
        """Test transitioning to successful termination."""
        result = manager.transition_to_termination(
            base_state, 
            ControlAction.TERMINATE_SUCCESS,
            "Test completion"
        )
        
        assert isinstance(result, StateTransitionResult)
        assert result.should_continue is False
        assert result.termination_reason == "Test completion"
        assert "Terminated conversation: Test completion" in result.action_taken
        
        # Check state changes
        assert result.new_state["should_continue"] is False
        assert result.new_state["control_action"] == ControlAction.TERMINATE_SUCCESS
        assert result.new_state["conversation_state"].status == ConversationStatus.COMPLETED
    
    def test_transition_to_termination_error(self, manager, base_state):
        """Test transitioning to error termination."""
        result = manager.transition_to_termination(
            base_state,
            ControlAction.TERMINATE_ERROR,
            "Test error"
        )
        
        assert isinstance(result, StateTransitionResult)
        assert result.should_continue is False
        assert result.termination_reason == "Test error"
        
        # Check state changes
        assert result.new_state["should_continue"] is False
        assert result.new_state["control_action"] == ControlAction.TERMINATE_ERROR
        assert result.new_state["conversation_state"].status == ConversationStatus.ERROR
    
    def test_transition_to_termination_other(self, manager, base_state):
        """Test transitioning to other termination types."""
        result = manager.transition_to_termination(
            base_state,
            ControlAction.TERMINATE_MAX_TURNS,
            "Max turns reached"
        )
        
        assert result.new_state["conversation_state"].status == ConversationStatus.TERMINATED
    
    def test_execute_state_transition_switch_agent(self, manager, base_state):
        """Test executing state transition for agent switching."""
        current_time = time.time()
        result = manager.execute_state_transition(base_state, current_time)
        
        assert result.should_continue is True
        assert result.new_state["current_agent"] == "agent_b"
    
    def test_execute_state_transition_terminate_error(self, manager, base_state):
        """Test executing state transition for error termination."""
        current_time = time.time()
        base_state["error_message"] = "Test error"
        
        result = manager.execute_state_transition(base_state, current_time)
        
        assert result.should_continue is False
        assert "Error: Test error" in result.termination_reason
    
    def test_execute_state_transition_terminate_timeout(self, manager, base_state):
        """Test executing state transition for timeout termination."""
        current_time = time.time()
        base_state["turn_start_time"] = current_time - 15.0
        
        result = manager.execute_state_transition(base_state, current_time)
        
        assert result.should_continue is False
        assert "timeout exceeded" in result.termination_reason.lower()
    
    def test_execute_state_transition_terminate_max_turns(self, manager, base_state):
        """Test executing state transition for max turns termination."""
        current_time = time.time()
        base_state["conversation_state"].current_turn = 5
        
        result = manager.execute_state_transition(base_state, current_time)
        
        assert result.should_continue is False
        assert "Maximum turns reached" in result.termination_reason


class TestStateUtilityFunctions:
    """Test cases for state utility functions."""
    
    def test_create_initial_graph_state(self):
        """Test creating initial graph state."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_b")
        
        assert state["conversation_state"] is conversation_state
        assert state["current_agent"] == "agent_b"
        assert state["should_continue"] is True
        assert state["control_action"] == ControlAction.CONTINUE
        assert state["error_message"] is None
        assert state["last_response"] is None
        assert state["turn_start_time"] is None
        assert isinstance(state["metadata"], dict)
    
    def test_create_initial_graph_state_default_agent(self):
        """Test creating initial graph state with default starting agent."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state)
        
        assert state["current_agent"] == "agent_a"
    
    def test_update_state_with_response(self):
        """Test updating state with agent response."""
        conversation_state = ConversationState()
        initial_state = create_initial_graph_state(conversation_state)
        
        response = "Test response"
        agent_id = "agent_a"
        response_time = 1.5
        token_count = 10
        
        updated_state = update_state_with_response(
            initial_state, response, agent_id, response_time, token_count
        )
        
        # Check that message was added
        assert len(updated_state["conversation_state"].messages) == 1
        message = updated_state["conversation_state"].messages[0]
        assert message.agent_id == agent_id
        assert message.content == response
        assert message.token_count == token_count
        assert message.metadata["response_time"] == response_time
        
        # Check state updates
        assert updated_state["last_response"] == response
        assert updated_state["metadata"]["last_response_time"] == response_time
        assert updated_state["metadata"]["last_token_count"] == token_count
    
    def test_update_state_with_error(self):
        """Test updating state with error condition."""
        conversation_state = ConversationState()
        initial_state = create_initial_graph_state(conversation_state)
        
        error_message = "Test error occurred"
        agent_id = "agent_a"
        
        updated_state = update_state_with_error(initial_state, error_message, agent_id)
        
        assert updated_state["error_message"] == error_message
        assert updated_state["conversation_state"].status == ConversationStatus.ERROR
        assert updated_state["control_action"] == ControlAction.TERMINATE_ERROR
        assert updated_state["should_continue"] is False
        assert updated_state["metadata"]["error_agent"] == agent_id
    
    def test_update_state_with_error_no_agent(self):
        """Test updating state with error condition without agent ID."""
        conversation_state = ConversationState()
        initial_state = create_initial_graph_state(conversation_state)
        
        error_message = "Test error occurred"
        
        updated_state = update_state_with_error(initial_state, error_message)
        
        assert updated_state["error_message"] == error_message
        assert updated_state["conversation_state"].status == ConversationStatus.ERROR
        assert "error_agent" not in updated_state["metadata"]


class TestStateTransitionResult:
    """Test cases for StateTransitionResult dataclass."""
    
    def test_state_transition_result_creation(self):
        """Test creating StateTransitionResult."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state)
        
        result = StateTransitionResult(
            new_state=state,
            action_taken="Test action",
            should_continue=True,
            termination_reason="Test reason"
        )
        
        assert result.new_state is state
        assert result.action_taken == "Test action"
        assert result.should_continue is True
        assert result.termination_reason == "Test reason"
    
    def test_state_transition_result_optional_termination_reason(self):
        """Test StateTransitionResult with optional termination reason."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state)
        
        result = StateTransitionResult(
            new_state=state,
            action_taken="Test action",
            should_continue=True
        )
        
        assert result.termination_reason is None


class TestIntegrationScenarios:
    """Integration test scenarios for state transitions."""
    
    def test_complete_conversation_flow(self):
        """Test a complete conversation flow through state transitions."""
        manager = StateTransitionManager(max_turns=3, turn_timeout=10.0)
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        current_time = time.time()
        
        # First transition - should switch to agent_b
        result1 = manager.execute_state_transition(state, current_time)
        assert result1.should_continue is True
        assert result1.new_state["current_agent"] == "agent_b"
        
        # Add a response and increment turn
        state = update_state_with_response(
            result1.new_state, "Response 1", "agent_a", 1.0, 5
        )
        
        # Second transition - should switch back to agent_a
        result2 = manager.execute_state_transition(state, current_time)
        assert result2.should_continue is True
        assert result2.new_state["current_agent"] == "agent_a"
        
        # Add another response and increment turn
        state = update_state_with_response(
            result2.new_state, "Response 2", "agent_b", 1.0, 5
        )
        
        # Add one more response to reach max turns
        state = update_state_with_response(
            state, "Response 3", "agent_a", 1.0, 5
        )
        
        # Third transition - should terminate due to max turns
        result3 = manager.execute_state_transition(state, current_time)
        assert result3.should_continue is False
        assert "Maximum turns reached" in result3.termination_reason
    
    def test_error_handling_flow(self):
        """Test error handling in state transitions."""
        manager = StateTransitionManager(max_turns=10, turn_timeout=10.0)
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        # Introduce an error
        state = update_state_with_error(state, "API failure", "agent_a")
        
        current_time = time.time()
        result = manager.execute_state_transition(state, current_time)
        
        assert result.should_continue is False
        assert "Error: API failure" in result.termination_reason
        assert result.new_state["conversation_state"].status == ConversationStatus.ERROR
    
    def test_timeout_handling_flow(self):
        """Test timeout handling in state transitions."""
        manager = StateTransitionManager(max_turns=10, turn_timeout=5.0)
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        # Set turn start time to simulate timeout
        current_time = time.time()
        state["turn_start_time"] = current_time - 10.0
        
        result = manager.execute_state_transition(state, current_time)
        
        assert result.should_continue is False
        assert "timeout exceeded" in result.termination_reason.lower()
        assert result.new_state["conversation_state"].status == ConversationStatus.TERMINATED