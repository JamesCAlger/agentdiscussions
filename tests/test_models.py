"""
Unit tests for core data models.
"""

import json
import pytest
from datetime import datetime
from src.agentic_conversation.models import (
    Message, ConversationState, ConversationStatus,
    AgentMetrics, ContextWindowSnapshot, RunTelemetry,
    AgentConfig, ModelConfig, ConversationConfig, LoggingConfig, SystemConfig
)


class TestMessage:
    """Test cases for the Message dataclass."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        timestamp = datetime.now()
        message = Message(
            agent_id="agent_a",
            content="Hello, world!",
            timestamp=timestamp,
            token_count=3,
            metadata={"test": "value"}
        )
        
        assert message.agent_id == "agent_a"
        assert message.content == "Hello, world!"
        assert message.timestamp == timestamp
        assert message.token_count == 3
        assert message.metadata == {"test": "value"}
    
    def test_message_default_metadata(self):
        """Test message creation with default metadata."""
        message = Message(
            agent_id="agent_a",
            content="Hello",
            timestamp=datetime.now(),
            token_count=1
        )
        
        assert message.metadata == {}
    
    def test_message_to_dict(self):
        """Test message serialization to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        message = Message(
            agent_id="agent_a",
            content="Test message",
            timestamp=timestamp,
            token_count=2,
            metadata={"key": "value"}
        )
        
        expected_dict = {
            "agent_id": "agent_a",
            "content": "Test message",
            "timestamp": "2024-01-01T12:00:00",
            "token_count": 2,
            "metadata": {"key": "value"}
        }
        
        assert message.to_dict() == expected_dict
    
    def test_message_from_dict(self):
        """Test message deserialization from dictionary."""
        data = {
            "agent_id": "agent_b",
            "content": "Response message",
            "timestamp": "2024-01-01T12:30:00",
            "token_count": 3,
            "metadata": {"response": True}
        }
        
        message = Message.from_dict(data)
        
        assert message.agent_id == "agent_b"
        assert message.content == "Response message"
        assert message.timestamp == datetime(2024, 1, 1, 12, 30, 0)
        assert message.token_count == 3
        assert message.metadata == {"response": True}
    
    def test_message_from_dict_no_metadata(self):
        """Test message deserialization without metadata."""
        data = {
            "agent_id": "agent_a",
            "content": "Simple message",
            "timestamp": "2024-01-01T12:00:00",
            "token_count": 2
        }
        
        message = Message.from_dict(data)
        assert message.metadata == {}
    
    def test_message_json_serialization(self):
        """Test message JSON serialization and deserialization."""
        original_message = Message(
            agent_id="agent_a",
            content="JSON test",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            token_count=2,
            metadata={"json": True}
        )
        
        json_str = original_message.to_json()
        restored_message = Message.from_json(json_str)
        
        assert restored_message.agent_id == original_message.agent_id
        assert restored_message.content == original_message.content
        assert restored_message.timestamp == original_message.timestamp
        assert restored_message.token_count == original_message.token_count
        assert restored_message.metadata == original_message.metadata


class TestConversationStatus:
    """Test cases for the ConversationStatus enum."""
    
    def test_conversation_status_values(self):
        """Test that all expected status values exist."""
        assert ConversationStatus.INITIALIZED.value == "initialized"
        assert ConversationStatus.RUNNING.value == "running"
        assert ConversationStatus.COMPLETED.value == "completed"
        assert ConversationStatus.ERROR.value == "error"
        assert ConversationStatus.TERMINATED.value == "terminated"


class TestConversationState:
    """Test cases for the ConversationState dataclass."""
    
    def test_conversation_state_creation(self):
        """Test basic conversation state creation."""
        state = ConversationState()
        
        assert state.messages == []
        assert state.current_turn == 0
        assert isinstance(state.start_time, datetime)
        assert state.status == ConversationStatus.INITIALIZED
        assert state.current_context_tokens == 0
        assert state.max_context_tokens == 8000
        assert state.metadata == {}
    
    def test_conversation_state_with_custom_values(self):
        """Test conversation state creation with custom values."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        state = ConversationState(
            current_turn=5,
            start_time=start_time,
            status=ConversationStatus.RUNNING,
            max_context_tokens=4000,
            metadata={"test": True}
        )
        
        assert state.current_turn == 5
        assert state.start_time == start_time
        assert state.status == ConversationStatus.RUNNING
        assert state.max_context_tokens == 4000
        assert state.metadata == {"test": True}
    
    def test_add_message(self):
        """Test adding messages to conversation state."""
        state = ConversationState()
        message = Message(
            agent_id="agent_a",
            content="Hello",
            timestamp=datetime.now(),
            token_count=5
        )
        
        state.add_message(message)
        
        assert len(state.messages) == 1
        assert state.messages[0] == message
        assert state.current_turn == 1
        assert state.current_context_tokens == 5
    
    def test_get_total_messages(self):
        """Test getting total message count."""
        state = ConversationState()
        
        # Add multiple messages
        for i in range(3):
            message = Message(
                agent_id=f"agent_{i}",
                content=f"Message {i}",
                timestamp=datetime.now(),
                token_count=1
            )
            state.add_message(message)
        
        assert state.get_total_messages() == 3
    
    def test_get_total_tokens(self):
        """Test getting total token count."""
        state = ConversationState()
        
        # Add messages with different token counts
        token_counts = [5, 10, 3]
        for i, count in enumerate(token_counts):
            message = Message(
                agent_id=f"agent_{i}",
                content=f"Message {i}",
                timestamp=datetime.now(),
                token_count=count
            )
            state.add_message(message)
        
        assert state.get_total_tokens() == sum(token_counts)
    
    def test_get_messages_by_agent(self):
        """Test filtering messages by agent."""
        state = ConversationState()
        
        # Add messages from different agents
        agents = ["agent_a", "agent_b", "agent_a"]
        for i, agent_id in enumerate(agents):
            message = Message(
                agent_id=agent_id,
                content=f"Message {i}",
                timestamp=datetime.now(),
                token_count=1
            )
            state.add_message(message)
        
        agent_a_messages = state.get_messages_by_agent("agent_a")
        agent_b_messages = state.get_messages_by_agent("agent_b")
        
        assert len(agent_a_messages) == 2
        assert len(agent_b_messages) == 1
        assert all(msg.agent_id == "agent_a" for msg in agent_a_messages)
        assert all(msg.agent_id == "agent_b" for msg in agent_b_messages)
    
    def test_get_context_utilization(self):
        """Test context utilization calculation."""
        state = ConversationState(max_context_tokens=100)
        
        # Add message with 50 tokens
        message = Message(
            agent_id="agent_a",
            content="Test",
            timestamp=datetime.now(),
            token_count=50
        )
        state.add_message(message)
        
        assert state.get_context_utilization() == 50.0
    
    def test_get_context_utilization_zero_max(self):
        """Test context utilization with zero max tokens."""
        state = ConversationState(max_context_tokens=0)
        assert state.get_context_utilization() == 0.0
    
    def test_is_context_full(self):
        """Test context full detection."""
        state = ConversationState(max_context_tokens=100)
        
        # Add message with 85 tokens (below threshold)
        message1 = Message(
            agent_id="agent_a",
            content="Test",
            timestamp=datetime.now(),
            token_count=85
        )
        state.add_message(message1)
        assert not state.is_context_full()
        
        # Add message with 10 more tokens (above threshold)
        message2 = Message(
            agent_id="agent_b",
            content="Test",
            timestamp=datetime.now(),
            token_count=10
        )
        state.add_message(message2)
        assert state.is_context_full()
    
    def test_is_context_full_custom_threshold(self):
        """Test context full detection with custom threshold."""
        state = ConversationState(max_context_tokens=100)
        
        # Add message with 80 tokens
        message = Message(
            agent_id="agent_a",
            content="Test",
            timestamp=datetime.now(),
            token_count=80
        )
        state.add_message(message)
        
        assert not state.is_context_full(threshold=85.0)
        assert state.is_context_full(threshold=75.0)
    
    def test_conversation_state_to_dict(self):
        """Test conversation state serialization to dictionary."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        state = ConversationState(
            current_turn=2,
            start_time=start_time,
            status=ConversationStatus.RUNNING,
            max_context_tokens=4000,
            metadata={"test": True}
        )
        
        # Add a message
        message = Message(
            agent_id="agent_a",
            content="Test message",
            timestamp=datetime(2024, 1, 1, 12, 5, 0),
            token_count=2
        )
        state.messages.append(message)
        state.current_context_tokens = 2
        
        result_dict = state.to_dict()
        
        expected_dict = {
            "messages": [{
                "agent_id": "agent_a",
                "content": "Test message",
                "timestamp": "2024-01-01T12:05:00",
                "token_count": 2,
                "metadata": {}
            }],
            "current_turn": 2,
            "start_time": "2024-01-01T12:00:00",
            "status": "running",
            "current_context_tokens": 2,
            "max_context_tokens": 4000,
            "metadata": {"test": True}
        }
        
        assert result_dict == expected_dict
    
    def test_conversation_state_from_dict(self):
        """Test conversation state deserialization from dictionary."""
        data = {
            "messages": [{
                "agent_id": "agent_b",
                "content": "Response",
                "timestamp": "2024-01-01T12:10:00",
                "token_count": 1,
                "metadata": {"response": True}
            }],
            "current_turn": 3,
            "start_time": "2024-01-01T12:00:00",
            "status": "completed",
            "current_context_tokens": 1,
            "max_context_tokens": 2000,
            "metadata": {"finished": True}
        }
        
        state = ConversationState.from_dict(data)
        
        assert len(state.messages) == 1
        assert state.messages[0].agent_id == "agent_b"
        assert state.messages[0].content == "Response"
        assert state.current_turn == 3
        assert state.start_time == datetime(2024, 1, 1, 12, 0, 0)
        assert state.status == ConversationStatus.COMPLETED
        assert state.current_context_tokens == 1
        assert state.max_context_tokens == 2000
        assert state.metadata == {"finished": True}
    
    def test_conversation_state_json_serialization(self):
        """Test conversation state JSON serialization and deserialization."""
        original_state = ConversationState(
            current_turn=1,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            status=ConversationStatus.RUNNING,
            max_context_tokens=4000
        )
        
        # Add a message
        message = Message(
            agent_id="agent_a",
            content="JSON test",
            timestamp=datetime(2024, 1, 1, 12, 5, 0),
            token_count=2
        )
        original_state.add_message(message)
        
        json_str = original_state.to_json()
        restored_state = ConversationState.from_json(json_str)
        
        assert len(restored_state.messages) == len(original_state.messages)
        assert restored_state.messages[0].agent_id == original_state.messages[0].agent_id
        assert restored_state.current_turn == original_state.current_turn
        assert restored_state.start_time == original_state.start_time
        assert restored_state.status == original_state.status
        assert restored_state.max_context_tokens == original_state.max_context_tokens


class TestAgentMetrics:
    """Test cases for the AgentMetrics dataclass."""
    
    def test_agent_metrics_creation(self):
        """Test basic agent metrics creation."""
        metrics = AgentMetrics(
            response_time=1.5,
            token_count=100,
            model_calls=2,
            errors=["Error 1", "Error 2"]
        )
        
        assert metrics.response_time == 1.5
        assert metrics.token_count == 100
        assert metrics.model_calls == 2
        assert metrics.errors == ["Error 1", "Error 2"]
    
    def test_agent_metrics_defaults(self):
        """Test agent metrics creation with default values."""
        metrics = AgentMetrics()
        
        assert metrics.response_time == 0.0
        assert metrics.token_count == 0
        assert metrics.model_calls == 0
        assert metrics.errors == []
    
    def test_add_error(self):
        """Test adding errors to agent metrics."""
        metrics = AgentMetrics()
        
        metrics.add_error("First error")
        metrics.add_error("Second error")
        
        assert len(metrics.errors) == 2
        assert "First error" in metrics.errors
        assert "Second error" in metrics.errors
    
    def test_has_errors(self):
        """Test error detection."""
        metrics = AgentMetrics()
        assert not metrics.has_errors()
        
        metrics.add_error("An error occurred")
        assert metrics.has_errors()
    
    def test_get_error_count(self):
        """Test error count retrieval."""
        metrics = AgentMetrics()
        assert metrics.get_error_count() == 0
        
        metrics.add_error("Error 1")
        metrics.add_error("Error 2")
        assert metrics.get_error_count() == 2
    
    def test_agent_metrics_serialization(self):
        """Test agent metrics serialization and deserialization."""
        original_metrics = AgentMetrics(
            response_time=2.5,
            token_count=150,
            model_calls=3,
            errors=["Test error"]
        )
        
        metrics_dict = original_metrics.to_dict()
        restored_metrics = AgentMetrics.from_dict(metrics_dict)
        
        assert restored_metrics.response_time == original_metrics.response_time
        assert restored_metrics.token_count == original_metrics.token_count
        assert restored_metrics.model_calls == original_metrics.model_calls
        assert restored_metrics.errors == original_metrics.errors


class TestContextWindowSnapshot:
    """Test cases for the ContextWindowSnapshot dataclass."""
    
    def test_context_window_snapshot_creation(self):
        """Test basic context window snapshot creation."""
        snapshot = ContextWindowSnapshot(
            turn_number=5,
            total_tokens=8000,
            available_tokens=2000,
            utilization_percentage=75.0,
            strategy_applied="truncate"
        )
        
        assert snapshot.turn_number == 5
        assert snapshot.total_tokens == 8000
        assert snapshot.available_tokens == 2000
        assert snapshot.utilization_percentage == 75.0
        assert snapshot.strategy_applied == "truncate"
    
    def test_context_window_snapshot_no_strategy(self):
        """Test context window snapshot creation without strategy."""
        snapshot = ContextWindowSnapshot(
            turn_number=1,
            total_tokens=4000,
            available_tokens=3000,
            utilization_percentage=25.0
        )
        
        assert snapshot.strategy_applied is None
    
    def test_is_near_capacity(self):
        """Test context window capacity detection."""
        snapshot = ContextWindowSnapshot(
            turn_number=1,
            total_tokens=1000,
            available_tokens=50,
            utilization_percentage=95.0
        )
        
        assert snapshot.is_near_capacity()
        assert snapshot.is_near_capacity(threshold=90.0)
        assert not snapshot.is_near_capacity(threshold=98.0)
    
    def test_get_used_tokens(self):
        """Test used tokens calculation."""
        snapshot = ContextWindowSnapshot(
            turn_number=1,
            total_tokens=1000,
            available_tokens=300,
            utilization_percentage=70.0
        )
        
        assert snapshot.get_used_tokens() == 700
    
    def test_context_window_snapshot_serialization(self):
        """Test context window snapshot serialization and deserialization."""
        original_snapshot = ContextWindowSnapshot(
            turn_number=3,
            total_tokens=5000,
            available_tokens=1500,
            utilization_percentage=70.0,
            strategy_applied="sliding"
        )
        
        snapshot_dict = original_snapshot.to_dict()
        restored_snapshot = ContextWindowSnapshot.from_dict(snapshot_dict)
        
        assert restored_snapshot.turn_number == original_snapshot.turn_number
        assert restored_snapshot.total_tokens == original_snapshot.total_tokens
        assert restored_snapshot.available_tokens == original_snapshot.available_tokens
        assert restored_snapshot.utilization_percentage == original_snapshot.utilization_percentage
        assert restored_snapshot.strategy_applied == original_snapshot.strategy_applied


class TestRunTelemetry:
    """Test cases for the RunTelemetry dataclass."""
    
    def test_run_telemetry_creation(self):
        """Test basic run telemetry creation."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        telemetry = RunTelemetry(
            run_id="test-run-123",
            start_time=start_time,
            total_turns=5
        )
        
        assert telemetry.run_id == "test-run-123"
        assert telemetry.start_time == start_time
        assert telemetry.end_time is None
        assert telemetry.total_turns == 5
        assert telemetry.agent_metrics == {}
        assert telemetry.conversation_history == []
        assert telemetry.context_window_snapshots == []
        assert telemetry.configuration is None
        assert telemetry.metadata == {}
    
    def test_get_duration_incomplete(self):
        """Test duration calculation for incomplete run."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        assert telemetry.get_duration() is None
    
    def test_get_duration_complete(self):
        """Test duration calculation for completed run."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 5, 30)
        
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=start_time,
            end_time=end_time
        )
        
        duration = telemetry.get_duration()
        assert duration == 330.0  # 5 minutes 30 seconds
    
    def test_is_completed(self):
        """Test completion status detection."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        assert not telemetry.is_completed()
        
        telemetry.end_time = datetime.now()
        assert telemetry.is_completed()
    
    def test_add_agent_metrics_new(self):
        """Test adding metrics for a new agent."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        metrics = AgentMetrics(
            response_time=1.5,
            token_count=100,
            model_calls=1
        )
        
        telemetry.add_agent_metrics("agent_a", metrics)
        
        assert "agent_a" in telemetry.agent_metrics
        assert telemetry.agent_metrics["agent_a"].response_time == 1.5
        assert telemetry.agent_metrics["agent_a"].token_count == 100
        assert telemetry.agent_metrics["agent_a"].model_calls == 1
    
    def test_add_agent_metrics_existing(self):
        """Test adding metrics for an existing agent (aggregation)."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        # Add initial metrics
        initial_metrics = AgentMetrics(
            response_time=1.0,
            token_count=50,
            model_calls=1,
            errors=["Error 1"]
        )
        telemetry.add_agent_metrics("agent_a", initial_metrics)
        
        # Add additional metrics
        additional_metrics = AgentMetrics(
            response_time=2.0,
            token_count=75,
            model_calls=1,
            errors=["Error 2"]
        )
        telemetry.add_agent_metrics("agent_a", additional_metrics)
        
        # Check aggregated values
        agent_metrics = telemetry.agent_metrics["agent_a"]
        assert agent_metrics.response_time == 3.0
        assert agent_metrics.token_count == 125
        assert agent_metrics.model_calls == 2
        assert agent_metrics.errors == ["Error 1", "Error 2"]
    
    def test_add_context_snapshot(self):
        """Test adding context window snapshots."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        snapshot = ContextWindowSnapshot(
            turn_number=1,
            total_tokens=1000,
            available_tokens=500,
            utilization_percentage=50.0
        )
        
        telemetry.add_context_snapshot(snapshot)
        
        assert len(telemetry.context_window_snapshots) == 1
        assert telemetry.context_window_snapshots[0] == snapshot
    
    def test_get_total_tokens_used(self):
        """Test total tokens calculation."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        telemetry.add_agent_metrics("agent_a", AgentMetrics(token_count=100))
        telemetry.add_agent_metrics("agent_b", AgentMetrics(token_count=150))
        
        assert telemetry.get_total_tokens_used() == 250
    
    def test_get_total_model_calls(self):
        """Test total model calls calculation."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        telemetry.add_agent_metrics("agent_a", AgentMetrics(model_calls=3))
        telemetry.add_agent_metrics("agent_b", AgentMetrics(model_calls=2))
        
        assert telemetry.get_total_model_calls() == 5
    
    def test_get_total_errors(self):
        """Test total errors calculation."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        telemetry.add_agent_metrics("agent_a", AgentMetrics(errors=["Error 1", "Error 2"]))
        telemetry.add_agent_metrics("agent_b", AgentMetrics(errors=["Error 3"]))
        
        assert telemetry.get_total_errors() == 3
    
    def test_get_average_response_time(self):
        """Test average response time calculation."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        telemetry.add_agent_metrics("agent_a", AgentMetrics(response_time=2.0, model_calls=2))
        telemetry.add_agent_metrics("agent_b", AgentMetrics(response_time=4.0, model_calls=1))
        
        # Total time: 6.0, Total calls: 3, Average: 2.0
        assert telemetry.get_average_response_time() == 2.0
    
    def test_get_average_response_time_no_calls(self):
        """Test average response time with no model calls."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        assert telemetry.get_average_response_time() == 0.0
    
    def test_get_peak_context_utilization(self):
        """Test peak context utilization calculation."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        # Add snapshots with different utilization levels
        telemetry.add_context_snapshot(ContextWindowSnapshot(1, 1000, 500, 50.0))
        telemetry.add_context_snapshot(ContextWindowSnapshot(2, 1000, 200, 80.0))
        telemetry.add_context_snapshot(ContextWindowSnapshot(3, 1000, 300, 70.0))
        
        assert telemetry.get_peak_context_utilization() == 80.0
    
    def test_get_peak_context_utilization_no_snapshots(self):
        """Test peak context utilization with no snapshots."""
        telemetry = RunTelemetry(
            run_id="test-run",
            start_time=datetime.now()
        )
        
        assert telemetry.get_peak_context_utilization() == 0.0
    
    def test_run_telemetry_serialization(self):
        """Test run telemetry serialization and deserialization."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 10, 0)
        
        original_telemetry = RunTelemetry(
            run_id="test-run-456",
            start_time=start_time,
            end_time=end_time,
            total_turns=3,
            configuration={"model": "gpt-4"},
            metadata={"test": True}
        )
        
        # Add some data
        original_telemetry.add_agent_metrics("agent_a", AgentMetrics(token_count=100))
        
        message = Message(
            agent_id="agent_a",
            content="Test message",
            timestamp=start_time,
            token_count=10
        )
        original_telemetry.conversation_history.append(message)
        
        snapshot = ContextWindowSnapshot(1, 1000, 500, 50.0)
        original_telemetry.add_context_snapshot(snapshot)
        
        # Test serialization
        telemetry_dict = original_telemetry.to_dict()
        restored_telemetry = RunTelemetry.from_dict(telemetry_dict)
        
        assert restored_telemetry.run_id == original_telemetry.run_id
        assert restored_telemetry.start_time == original_telemetry.start_time
        assert restored_telemetry.end_time == original_telemetry.end_time
        assert restored_telemetry.total_turns == original_telemetry.total_turns
        assert len(restored_telemetry.agent_metrics) == 1
        assert len(restored_telemetry.conversation_history) == 1
        assert len(restored_telemetry.context_window_snapshots) == 1
        assert restored_telemetry.configuration == original_telemetry.configuration
        assert restored_telemetry.metadata == original_telemetry.metadata
    
    def test_run_telemetry_json_serialization(self):
        """Test run telemetry JSON serialization and deserialization."""
        original_telemetry = RunTelemetry(
            run_id="json-test-run",
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            total_turns=1
        )
        
        json_str = original_telemetry.to_json()
        restored_telemetry = RunTelemetry.from_json(json_str)
        
        assert restored_telemetry.run_id == original_telemetry.run_id
        assert restored_telemetry.start_time == original_telemetry.start_time
        assert restored_telemetry.total_turns == original_telemetry.total_turns


class TestAgentConfig:
    """Test cases for the AgentConfig dataclass."""
    
    def test_agent_config_creation(self):
        """Test basic agent config creation."""
        config = AgentConfig(
            name="Test Agent",
            system_prompt="You are a helpful assistant."
        )
        
        assert config.name == "Test Agent"
        assert config.system_prompt == "You are a helpful assistant."
    
    def test_agent_config_validation_valid(self):
        """Test validation of valid agent config."""
        config = AgentConfig(
            name="Valid Agent",
            system_prompt="Valid system prompt"
        )
        
        assert config.is_valid()
        assert config.validate() == []
    
    def test_agent_config_validation_empty_name(self):
        """Test validation with empty name."""
        config = AgentConfig(
            name="",
            system_prompt="Valid prompt"
        )
        
        assert not config.is_valid()
        errors = config.validate()
        assert "Agent name cannot be empty" in errors
    
    def test_agent_config_validation_empty_prompt(self):
        """Test validation with empty system prompt."""
        config = AgentConfig(
            name="Valid Name",
            system_prompt=""
        )
        
        assert not config.is_valid()
        errors = config.validate()
        assert "Agent system prompt cannot be empty" in errors
    
    def test_agent_config_serialization(self):
        """Test agent config serialization and deserialization."""
        original_config = AgentConfig(
            name="Serialization Test",
            system_prompt="Test prompt for serialization"
        )
        
        config_dict = original_config.to_dict()
        restored_config = AgentConfig.from_dict(config_dict)
        
        assert restored_config.name == original_config.name
        assert restored_config.system_prompt == original_config.system_prompt


class TestModelConfig:
    """Test cases for the ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test basic model config creation."""
        config = ModelConfig(
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=1500
        )
        
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.8
        assert config.max_tokens == 1500
        assert config.top_p == 1.0  # default value
    
    def test_model_config_defaults(self):
        """Test model config creation with default values."""
        config = ModelConfig(model_name="gpt-3.5-turbo")
        
        assert config.model_name == "gpt-3.5-turbo"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
    
    def test_model_config_validation_valid(self):
        """Test validation of valid model config."""
        config = ModelConfig(
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=-0.1
        )
        
        assert config.is_valid()
        assert config.validate() == []
    
    def test_model_config_validation_invalid_temperature(self):
        """Test validation with invalid temperature."""
        config = ModelConfig(
            model_name="gpt-4",
            temperature=3.0  # Invalid: > 2.0
        )
        
        assert not config.is_valid()
        errors = config.validate()
        assert any("Temperature must be between 0.0 and 2.0" in error for error in errors)
    
    def test_model_config_validation_invalid_max_tokens(self):
        """Test validation with invalid max tokens."""
        config = ModelConfig(
            model_name="gpt-4",
            max_tokens=-100  # Invalid: <= 0
        )
        
        assert not config.is_valid()
        errors = config.validate()
        assert any("Max tokens must be greater than 0" in error for error in errors)
    
    def test_model_config_serialization(self):
        """Test model config serialization and deserialization."""
        original_config = ModelConfig(
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.95
        )
        
        config_dict = original_config.to_dict()
        restored_config = ModelConfig.from_dict(config_dict)
        
        assert restored_config.model_name == original_config.model_name
        assert restored_config.temperature == original_config.temperature
        assert restored_config.max_tokens == original_config.max_tokens
        assert restored_config.top_p == original_config.top_p


class TestConversationConfig:
    """Test cases for the ConversationConfig dataclass."""
    
    def test_conversation_config_creation(self):
        """Test basic conversation config creation."""
        config = ConversationConfig(
            max_turns=10,
            initial_prompt="Let's discuss AI",
            context_window_strategy="truncate"
        )
        
        assert config.max_turns == 10
        assert config.initial_prompt == "Let's discuss AI"
        assert config.context_window_strategy == "truncate"
        assert config.context_window_size == 8000  # default
    
    def test_conversation_config_defaults(self):
        """Test conversation config creation with default values."""
        config = ConversationConfig(max_turns=5)
        
        assert config.max_turns == 5
        assert config.initial_prompt is None
        assert config.context_window_strategy == "sliding"
        assert config.context_window_size == 8000
        assert config.turn_timeout == 30.0
    
    def test_conversation_config_validation_valid(self):
        """Test validation of valid conversation config."""
        config = ConversationConfig(
            max_turns=15,
            context_window_strategy="summarize",
            context_window_size=4000,
            turn_timeout=60.0
        )
        
        assert config.is_valid()
        assert config.validate() == []
    
    def test_conversation_config_validation_invalid_strategy(self):
        """Test validation with invalid context window strategy."""
        config = ConversationConfig(
            max_turns=10,
            context_window_strategy="invalid_strategy"
        )
        
        assert not config.is_valid()
        errors = config.validate()
        assert any("Context window strategy must be one of" in error for error in errors)
    
    def test_conversation_config_validation_invalid_max_turns(self):
        """Test validation with invalid max turns."""
        config = ConversationConfig(max_turns=0)
        
        assert not config.is_valid()
        errors = config.validate()
        assert any("Max turns must be greater than 0" in error for error in errors)
    
    def test_conversation_config_serialization(self):
        """Test conversation config serialization and deserialization."""
        original_config = ConversationConfig(
            max_turns=20,
            initial_prompt="Start conversation",
            context_window_strategy="truncate",
            context_window_size=6000
        )
        
        config_dict = original_config.to_dict()
        restored_config = ConversationConfig.from_dict(config_dict)
        
        assert restored_config.max_turns == original_config.max_turns
        assert restored_config.initial_prompt == original_config.initial_prompt
        assert restored_config.context_window_strategy == original_config.context_window_strategy
        assert restored_config.context_window_size == original_config.context_window_size


class TestLoggingConfig:
    """Test cases for the LoggingConfig dataclass."""
    
    def test_logging_config_creation(self):
        """Test basic logging config creation."""
        config = LoggingConfig(
            log_level="DEBUG",
            output_directory="./custom_logs",
            real_time_display=False
        )
        
        assert config.log_level == "DEBUG"
        assert config.output_directory == "./custom_logs"
        assert config.real_time_display is False
        assert config.export_formats == ["json"]  # default
    
    def test_logging_config_defaults(self):
        """Test logging config creation with default values."""
        config = LoggingConfig()
        
        assert config.log_level == "INFO"
        assert config.output_directory == "./logs"
        assert config.real_time_display is True
        assert config.export_formats == ["json"]
        assert config.save_conversation_history is True
        assert config.save_telemetry is True
    
    def test_logging_config_validation_valid(self):
        """Test validation of valid logging config."""
        config = LoggingConfig(
            log_level="WARNING",
            output_directory="./logs",
            export_formats=["json", "csv"]
        )
        
        assert config.is_valid()
        assert config.validate() == []
    
    def test_logging_config_validation_invalid_log_level(self):
        """Test validation with invalid log level."""
        config = LoggingConfig(log_level="INVALID_LEVEL")
        
        assert not config.is_valid()
        errors = config.validate()
        assert any("Log level must be one of" in error for error in errors)
    
    def test_logging_config_validation_invalid_export_format(self):
        """Test validation with invalid export format."""
        config = LoggingConfig(export_formats=["json", "invalid_format"])
        
        assert not config.is_valid()
        errors = config.validate()
        assert any("Export format 'invalid_format' is not valid" in error for error in errors)
    
    def test_logging_config_serialization(self):
        """Test logging config serialization and deserialization."""
        original_config = LoggingConfig(
            log_level="ERROR",
            output_directory="./test_logs",
            export_formats=["json", "csv", "txt"]
        )
        
        config_dict = original_config.to_dict()
        restored_config = LoggingConfig.from_dict(config_dict)
        
        assert restored_config.log_level == original_config.log_level
        assert restored_config.output_directory == original_config.output_directory
        assert restored_config.export_formats == original_config.export_formats


class TestSystemConfig:
    """Test cases for the SystemConfig dataclass."""
    
    def create_valid_system_config(self) -> SystemConfig:
        """Helper method to create a valid system config."""
        return SystemConfig(
            agent_a=AgentConfig(
                name="Agent A",
                system_prompt="You are agent A"
            ),
            agent_b=AgentConfig(
                name="Agent B",
                system_prompt="You are agent B"
            ),
            model=ModelConfig(model_name="gpt-4"),
            conversation=ConversationConfig(max_turns=10),
            logging=LoggingConfig()
        )
    
    def test_system_config_creation(self):
        """Test basic system config creation."""
        config = self.create_valid_system_config()
        
        assert config.agent_a.name == "Agent A"
        assert config.agent_b.name == "Agent B"
        assert config.model.model_name == "gpt-4"
        assert config.conversation.max_turns == 10
        assert config.logging.log_level == "INFO"
    
    def test_system_config_validation_valid(self):
        """Test validation of valid system config."""
        config = self.create_valid_system_config()
        
        assert config.is_valid()
        assert config.validate() == []
        
        summary = config.get_validation_summary()
        assert summary["is_valid"] is True
        assert summary["error_count"] == 0
        assert all(summary["sections"].values())
    
    def test_system_config_validation_duplicate_agent_names(self):
        """Test validation with duplicate agent names."""
        config = SystemConfig(
            agent_a=AgentConfig(name="Same Name", system_prompt="Prompt A"),
            agent_b=AgentConfig(name="Same Name", system_prompt="Prompt B"),
            model=ModelConfig(model_name="gpt-4"),
            conversation=ConversationConfig(max_turns=10),
            logging=LoggingConfig()
        )
        
        assert not config.is_valid()
        errors = config.validate()
        assert any("Agent A and Agent B must have different names" in error for error in errors)
    
    def test_system_config_validation_invalid_subsections(self):
        """Test validation with invalid subsections."""
        config = SystemConfig(
            agent_a=AgentConfig(name="", system_prompt="Valid prompt"),  # Invalid name
            agent_b=AgentConfig(name="Valid Name", system_prompt="Valid prompt"),
            model=ModelConfig(model_name="gpt-4", temperature=5.0),  # Invalid temperature
            conversation=ConversationConfig(max_turns=0),  # Invalid max_turns
            logging=LoggingConfig(log_level="INVALID")  # Invalid log level
        )
        
        assert not config.is_valid()
        errors = config.validate()
        
        # Should have errors from multiple sections
        assert any("Agent A:" in error for error in errors)
        assert any("Model:" in error for error in errors)
        assert any("Conversation:" in error for error in errors)
        assert any("Logging:" in error for error in errors)
        
        summary = config.get_validation_summary()
        assert summary["is_valid"] is False
        assert summary["error_count"] > 0
        assert not summary["sections"]["agent_a"]
        assert summary["sections"]["agent_b"]  # This one should be valid
        assert not summary["sections"]["model"]
        assert not summary["sections"]["conversation"]
        assert not summary["sections"]["logging"]
    
    def test_system_config_serialization(self):
        """Test system config serialization and deserialization."""
        original_config = self.create_valid_system_config()
        
        config_dict = original_config.to_dict()
        restored_config = SystemConfig.from_dict(config_dict)
        
        assert restored_config.agent_a.name == original_config.agent_a.name
        assert restored_config.agent_b.name == original_config.agent_b.name
        assert restored_config.model.model_name == original_config.model.model_name
        assert restored_config.conversation.max_turns == original_config.conversation.max_turns
        assert restored_config.logging.log_level == original_config.logging.log_level
    
    def test_system_config_json_serialization(self):
        """Test system config JSON serialization and deserialization."""
        original_config = self.create_valid_system_config()
        
        json_str = original_config.to_json()
        restored_config = SystemConfig.from_json(json_str)
        
        assert restored_config.agent_a.name == original_config.agent_a.name
        assert restored_config.agent_b.name == original_config.agent_b.name
        assert restored_config.model.model_name == original_config.model.model_name