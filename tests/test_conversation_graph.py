"""
Integration tests for LangGraph conversation orchestrator.

This module tests the complete conversation flow orchestration using LangGraph,
including agent interactions, state management, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, Optional

from agentic_conversation.conversation_graph import (
    ConversationGraph,
    ConversationError,
    create_conversation_graph,
)
from agentic_conversation.graph_state import (
    GraphState,
    ControlAction,
    create_initial_graph_state,
)
from agentic_conversation.models import (
    ConversationState,
    ConversationStatus,
    Message,
)
from agentic_conversation.agents import (
    BaseAgent,
    AgentResponse,
    AgentInfo,
    ConversationContext,
    AgentError,
)
from agentic_conversation.context_manager import ContextManager
from agentic_conversation.telemetry import TelemetryLogger


class MockAgent(BaseAgent):
    """Mock agent for testing purposes."""
    
    def __init__(self, agent_id: str, name: str, system_prompt: str, responses: Optional[list] = None):
        super().__init__(agent_id, name, system_prompt)
        self.responses = responses or ["Default response"]
        self.response_index = 0
        self.call_count = 0
        self.contexts_received = []
        self.should_raise_error = False
        self.error_message = "Mock agent error"
        self.response_delay = 0.0
    
    async def generate_response(self, context: ConversationContext) -> AgentResponse:
        """Generate a mock response."""
        self.call_count += 1
        self.contexts_received.append(context)
        
        if self.should_raise_error:
            raise AgentError(self.error_message, self.agent_id)
        
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        # Get next response
        if self.response_index < len(self.responses):
            response_content = self.responses[self.response_index]
            self.response_index += 1
        else:
            response_content = f"Response {self.response_index + 1}"
            self.response_index += 1
        
        return AgentResponse(
            content=response_content,
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            token_count=len(response_content.split()),
            response_time=self.response_delay,
            model_calls=1
        )
    
    def get_agent_info(self) -> AgentInfo:
        """Get mock agent info."""
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description=f"Mock agent {self.name}",
            model_name="mock-model",
            capabilities=["text_generation"],
            metadata={"type": "mock"}
        )
    
    def reset(self):
        """Reset mock agent state."""
        self.response_index = 0
        self.call_count = 0
        self.contexts_received = []
        self.should_raise_error = False


class TestConversationGraph:
    """Test cases for ConversationGraph class."""
    
    @pytest.fixture
    def mock_agent_a(self):
        """Create mock agent A."""
        return MockAgent(
            agent_id="agent_a",
            name="Agent A",
            system_prompt="You are Agent A",
            responses=["Hello from A", "Response 2 from A", "Final from A"]
        )
    
    @pytest.fixture
    def mock_agent_b(self):
        """Create mock agent B."""
        return MockAgent(
            agent_id="agent_b",
            name="Agent B",
            system_prompt="You are Agent B",
            responses=["Hello from B", "Response 2 from B", "Final from B"]
        )
    
    @pytest.fixture
    def mock_context_manager(self):
        """Create mock context manager."""
        context_manager = Mock(spec=ContextManager)
        context_manager.max_context_tokens = 8000
        context_manager.manage_context = AsyncMock(side_effect=lambda state: state)
        context_manager.get_available_tokens = Mock(return_value=7000)
        return context_manager
    
    @pytest.fixture
    def mock_telemetry_logger(self):
        """Create mock telemetry logger."""
        logger = Mock(spec=TelemetryLogger)
        logger.start_conversation_run = AsyncMock()
        logger.end_conversation_run = AsyncMock()
        logger.log_agent_interaction = AsyncMock()
        logger.log_conversation_event = AsyncMock()
        return logger
    
    @pytest.fixture
    def conversation_graph(self, mock_agent_a, mock_agent_b, mock_context_manager):
        """Create conversation graph for testing."""
        return ConversationGraph(
            agent_a=mock_agent_a,
            agent_b=mock_agent_b,
            context_manager=mock_context_manager,
            max_turns=5,
            turn_timeout=10.0,
            initial_prompt="Start the conversation"
        )
    
    def test_conversation_graph_initialization(self, conversation_graph, mock_agent_a, mock_agent_b):
        """Test conversation graph initialization."""
        assert conversation_graph.agent_a is mock_agent_a
        assert conversation_graph.agent_b is mock_agent_b
        assert conversation_graph.max_turns == 5
        assert conversation_graph.turn_timeout == 10.0
        assert conversation_graph.initial_prompt == "Start the conversation"
        assert conversation_graph.graph is not None
        assert conversation_graph.compiled_graph is not None
    
    def test_conversation_graph_with_telemetry(
        self, mock_agent_a, mock_agent_b, mock_context_manager, mock_telemetry_logger
    ):
        """Test conversation graph initialization with telemetry."""
        graph = ConversationGraph(
            agent_a=mock_agent_a,
            agent_b=mock_agent_b,
            context_manager=mock_context_manager,
            telemetry_logger=mock_telemetry_logger
        )
        
        assert graph.telemetry_logger is mock_telemetry_logger
    
    @pytest.mark.asyncio
    async def test_prepare_agent_context(self, conversation_graph, mock_agent_a):
        """Test preparing agent context."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        context = await conversation_graph._prepare_agent_context(state, mock_agent_a)
        
        assert isinstance(context, ConversationContext)
        assert context.system_prompt == mock_agent_a.system_prompt
        assert context.available_tokens == 8000  # max_context_tokens - current_context_tokens (0)
        assert context.turn_number == 0
        assert context.other_agent_id == "agent_b"
        assert "max_turns" in context.metadata
        assert "turn_timeout" in context.metadata
    
    @pytest.mark.asyncio
    async def test_execute_agent_turn_success(self, conversation_graph, mock_agent_a):
        """Test successful agent turn execution."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        result_state = await conversation_graph._execute_agent_turn(state, mock_agent_a)
        
        assert len(result_state["conversation_state"].messages) == 1
        assert result_state["conversation_state"].messages[0].agent_id == "agent_a"
        assert result_state["conversation_state"].messages[0].content == "Hello from A"
        assert result_state["last_response"] == "Hello from A"
        assert mock_agent_a.call_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_agent_turn_error(self, conversation_graph, mock_agent_a):
        """Test agent turn execution with error."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        # Configure agent to raise error
        mock_agent_a.should_raise_error = True
        mock_agent_a.error_message = "Test agent error"
        
        result_state = await conversation_graph._execute_agent_turn(state, mock_agent_a)
        
        assert result_state["error_message"] is not None
        assert "Test agent error" in result_state["error_message"]
        assert result_state["conversation_state"].status == ConversationStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_execute_agent_turn_timeout(self, conversation_graph, mock_agent_a):
        """Test agent turn execution with timeout."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        # Configure agent to take longer than timeout
        mock_agent_a.response_delay = 15.0  # Longer than 10s timeout
        
        result_state = await conversation_graph._execute_agent_turn(state, mock_agent_a)
        
        assert result_state["error_message"] is not None
        assert "timed out" in result_state["error_message"].lower()
        assert result_state["conversation_state"].status == ConversationStatus.ERROR
    
    def test_route_conversation_agent_a(self, conversation_graph):
        """Test conversation routing to agent A."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        route = conversation_graph._route_conversation(state)
        assert route == "agent_a"
    
    def test_route_conversation_agent_b(self, conversation_graph):
        """Test conversation routing to agent B."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_b")
        
        route = conversation_graph._route_conversation(state)
        assert route == "agent_b"
    
    def test_route_conversation_error(self, conversation_graph):
        """Test conversation routing with error state."""
        conversation_state = ConversationState()
        conversation_state.status = ConversationStatus.ERROR
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        route = conversation_graph._route_conversation(state)
        assert route == "error"
    
    def test_route_conversation_terminate(self, conversation_graph):
        """Test conversation routing for termination."""
        conversation_state = ConversationState()
        conversation_state.current_turn = 10  # Exceeds max_turns of 5
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        route = conversation_graph._route_conversation(state)
        assert route == "terminate"
    
    @pytest.mark.asyncio
    async def test_conversation_manager_node(self, conversation_graph):
        """Test conversation manager node execution."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        
        result_state = await conversation_graph._conversation_manager_node(state)
        
        # Should switch to agent_b
        assert result_state["current_agent"] == "agent_b"
        assert result_state["control_action"] == ControlAction.CONTINUE
        assert result_state["should_continue"] is True
    
    @pytest.mark.asyncio
    async def test_error_handler_node(self, conversation_graph):
        """Test error handler node execution."""
        conversation_state = ConversationState()
        state = create_initial_graph_state(conversation_state, "agent_a")
        state["error_message"] = "Test error"
        
        result_state = await conversation_graph._error_handler_node(state)
        
        assert result_state["conversation_state"].status == ConversationStatus.ERROR
        assert result_state["should_continue"] is False
        assert result_state["control_action"] == ControlAction.TERMINATE_ERROR
    
    @pytest.mark.asyncio
    async def test_run_conversation_success(self, conversation_graph, mock_agent_a, mock_agent_b):
        """Test successful conversation execution."""
        # Configure agents with limited responses to ensure termination
        mock_agent_a.responses = ["Hello from A", "Goodbye from A"]
        mock_agent_b.responses = ["Hello from B", "Goodbye from B"]
        
        final_state = await conversation_graph.run_conversation("test_conv")
        
        assert isinstance(final_state, ConversationState)
        assert final_state.status in [ConversationStatus.COMPLETED, ConversationStatus.TERMINATED]
        assert len(final_state.messages) > 0
        
        # Check that both agents were called
        assert mock_agent_a.call_count > 0
        assert mock_agent_b.call_count > 0
    
    @pytest.mark.asyncio
    async def test_run_conversation_with_initial_prompt(self, conversation_graph):
        """Test conversation execution with initial prompt."""
        final_state = await conversation_graph.run_conversation("test_conv")
        
        # Should have initial prompt message
        assert len(final_state.messages) > 0
        assert final_state.messages[0].agent_id == "system"
        assert final_state.messages[0].content == "Start the conversation"
    
    @pytest.mark.asyncio
    async def test_run_conversation_with_telemetry(
        self, mock_agent_a, mock_agent_b, mock_context_manager, mock_telemetry_logger
    ):
        """Test conversation execution with telemetry logging."""
        graph = ConversationGraph(
            agent_a=mock_agent_a,
            agent_b=mock_agent_b,
            context_manager=mock_context_manager,
            telemetry_logger=mock_telemetry_logger,
            max_turns=2
        )
        
        await graph.run_conversation("test_conv")
        
        # Verify telemetry calls
        mock_telemetry_logger.log_system_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_run_conversation_error_handling(self, conversation_graph, mock_agent_a):
        """Test conversation error handling."""
        # Configure agent to raise error
        mock_agent_a.should_raise_error = True
        mock_agent_a.error_message = "Test conversation error"
        
        with pytest.raises(ConversationError) as exc_info:
            await conversation_graph.run_conversation("test_conv")
        
        assert "failed" in str(exc_info.value).lower()
    
    def test_get_conversation_info(self, conversation_graph):
        """Test getting conversation information."""
        info = conversation_graph.get_conversation_info()
        
        assert "agent_a" in info
        assert "agent_b" in info
        assert info["max_turns"] == 5
        assert info["turn_timeout"] == 10.0
        assert info["initial_prompt"] == "Start the conversation"
        assert "context_manager" in info
    
    @pytest.mark.asyncio
    async def test_log_agent_turn_with_telemetry(
        self, mock_agent_a, mock_agent_b, mock_context_manager, mock_telemetry_logger
    ):
        """Test logging agent turn with telemetry."""
        graph = ConversationGraph(
            agent_a=mock_agent_a,
            agent_b=mock_agent_b,
            context_manager=mock_context_manager,
            telemetry_logger=mock_telemetry_logger
        )
        
        response = AgentResponse(
            content="Test response",
            agent_id="agent_a",
            token_count=5,
            response_time=1.0
        )
        
        context = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Test prompt",
            available_tokens=1000,
            turn_number=1
        )
        
        await graph._log_agent_turn(mock_agent_a, response, context, 1.0)
        
        mock_telemetry_logger.log_agent_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_agent_turn_without_telemetry(self, conversation_graph, mock_agent_a):
        """Test logging agent turn without telemetry."""
        response = AgentResponse(
            content="Test response",
            agent_id="agent_a"
        )
        
        context = ConversationContext(
            conversation_state=ConversationState(),
            system_prompt="Test prompt",
            available_tokens=1000,
            turn_number=1
        )
        
        # Should not raise error even without telemetry logger
        await conversation_graph._log_agent_turn(mock_agent_a, response, context, 1.0)


class TestConversationError:
    """Test cases for ConversationError exception."""
    
    def test_conversation_error_creation(self):
        """Test creating ConversationError."""
        error = ConversationError("Test error", "conv_123")
        
        assert error.message == "Test error"
        assert error.conversation_id == "conv_123"
        assert error.original_error is None
    
    def test_conversation_error_with_original(self):
        """Test creating ConversationError with original exception."""
        original = ValueError("Original error")
        error = ConversationError("Test error", "conv_123", original)
        
        assert error.original_error is original
    
    def test_conversation_error_to_dict(self):
        """Test converting ConversationError to dictionary."""
        error = ConversationError("Test error", "conv_123")
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "ConversationError"
        assert error_dict["message"] == "Test error"
        assert error_dict["conversation_id"] == "conv_123"
        assert "timestamp" in error_dict


class TestFactoryFunction:
    """Test cases for factory function."""
    
    def test_create_conversation_graph(self):
        """Test creating conversation graph with factory function."""
        agent_a = MockAgent("agent_a", "Agent A", "Prompt A")
        agent_b = MockAgent("agent_b", "Agent B", "Prompt B")
        context_manager = Mock(spec=ContextManager)
        
        graph = create_conversation_graph(
            agent_a=agent_a,
            agent_b=agent_b,
            context_manager=context_manager,
            max_turns=10
        )
        
        assert isinstance(graph, ConversationGraph)
        assert graph.agent_a is agent_a
        assert graph.agent_b is agent_b
        assert graph.context_manager is context_manager
        assert graph.max_turns == 10
    
    def test_create_conversation_graph_with_telemetry(self):
        """Test creating conversation graph with telemetry using factory function."""
        agent_a = MockAgent("agent_a", "Agent A", "Prompt A")
        agent_b = MockAgent("agent_b", "Agent B", "Prompt B")
        context_manager = Mock(spec=ContextManager)
        telemetry_logger = Mock(spec=TelemetryLogger)
        
        graph = create_conversation_graph(
            agent_a=agent_a,
            agent_b=agent_b,
            context_manager=context_manager,
            telemetry_logger=telemetry_logger
        )
        
        assert graph.telemetry_logger is telemetry_logger


class TestIntegrationScenarios:
    """Integration test scenarios for complete conversation flows."""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self):
        """Test a complete conversation flow from start to finish."""
        # Create agents with specific responses
        agent_a = MockAgent(
            "agent_a", "Researcher", "You are a researcher",
            responses=["What should we research?", "That's interesting!", "Let's conclude."]
        )
        agent_b = MockAgent(
            "agent_b", "Analyst", "You are an analyst", 
            responses=["Let's study AI ethics", "I agree completely", "Good idea!"]
        )
        
        # Create context manager
        context_manager = Mock(spec=ContextManager)
        context_manager.max_context_tokens = 8000
        context_manager.manage_context = AsyncMock(side_effect=lambda state: state)
        context_manager.get_available_tokens = Mock(return_value=7000)
        
        # Create conversation graph
        graph = ConversationGraph(
            agent_a=agent_a,
            agent_b=agent_b,
            context_manager=context_manager,
            max_turns=6,
            turn_timeout=5.0
        )
        
        # Run conversation
        final_state = await graph.run_conversation("integration_test")
        
        # Verify results
        assert final_state.status in [ConversationStatus.COMPLETED, ConversationStatus.TERMINATED]
        assert len(final_state.messages) > 0
        assert agent_a.call_count > 0
        assert agent_b.call_count > 0
        
        # Verify alternating agents
        agent_messages = [msg for msg in final_state.messages if msg.agent_id in ["agent_a", "agent_b"]]
        if len(agent_messages) > 1:
            # Check that agents alternate (allowing for some flexibility)
            agent_ids = [msg.agent_id for msg in agent_messages]
            # Should not have the same agent responding consecutively too often
            consecutive_same = sum(1 for i in range(1, len(agent_ids)) if agent_ids[i] == agent_ids[i-1])
            assert consecutive_same < len(agent_ids) // 2  # Less than half should be consecutive
    
    @pytest.mark.asyncio
    async def test_conversation_with_context_management(self):
        """Test conversation with active context management."""
        agent_a = MockAgent("agent_a", "Agent A", "Prompt A")
        agent_b = MockAgent("agent_b", "Agent B", "Prompt B")
        
        # Create context manager that modifies state
        context_manager = Mock(spec=ContextManager)
        context_manager.max_context_tokens = 100  # Small limit to trigger management
        context_manager.get_available_tokens = Mock(return_value=50)
        
        # Mock context management to add metadata
        async def mock_manage_context(state):
            state.metadata["context_managed"] = True
            return state
        
        context_manager.manage_conversation_context = Mock(side_effect=mock_manage_context)
        
        graph = ConversationGraph(
            agent_a=agent_a,
            agent_b=agent_b,
            context_manager=context_manager,
            max_turns=3
        )
        
        final_state = await graph.run_conversation("context_test")
        
        # Verify context management was called
        assert context_manager.manage_conversation_context.call_count > 0
    
    @pytest.mark.asyncio
    async def test_conversation_error_recovery(self):
        """Test conversation behavior with agent errors."""
        agent_a = MockAgent("agent_a", "Agent A", "Prompt A")
        agent_b = MockAgent("agent_b", "Agent B", "Prompt B")
        
        # Configure agent A to fail on first call, then succeed
        agent_a.should_raise_error = True
        
        context_manager = Mock(spec=ContextManager)
        context_manager.max_context_tokens = 8000
        context_manager.manage_context = AsyncMock(side_effect=lambda state: state)
        context_manager.get_available_tokens = Mock(return_value=7000)
        
        graph = ConversationGraph(
            agent_a=agent_a,
            agent_b=agent_b,
            context_manager=context_manager,
            max_turns=5
        )
        
        # Should raise ConversationError due to agent error
        with pytest.raises(ConversationError):
            await graph.run_conversation("error_test")
    
    @pytest.mark.asyncio
    async def test_conversation_timeout_handling(self):
        """Test conversation behavior with turn timeouts."""
        agent_a = MockAgent("agent_a", "Agent A", "Prompt A")
        agent_b = MockAgent("agent_b", "Agent B", "Prompt B")
        
        # Configure agent A to take too long
        agent_a.response_delay = 2.0  # 2 seconds
        
        context_manager = Mock(spec=ContextManager)
        context_manager.max_context_tokens = 8000
        context_manager.manage_context = AsyncMock(side_effect=lambda state: state)
        context_manager.get_available_tokens = Mock(return_value=7000)
        
        graph = ConversationGraph(
            agent_a=agent_a,
            agent_b=agent_b,
            context_manager=context_manager,
            max_turns=5,
            turn_timeout=1.0  # 1 second timeout
        )
        
        # Should handle timeout gracefully
        with pytest.raises(ConversationError):
            await graph.run_conversation("timeout_test")