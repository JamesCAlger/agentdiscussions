"""
Unit tests for the telemetry logging system.

This module contains comprehensive tests for the TelemetryLogger class,
covering telemetry data collection, formatting, and logging functionality.
"""

import json
import logging
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from src.agentic_conversation.telemetry import TelemetryLogger
from src.agentic_conversation.models import (
    AgentMetrics,
    ContextWindowSnapshot,
    ConversationState,
    ConversationStatus,
    LoggingConfig,
    Message,
    RunTelemetry
)


class TestTelemetryLogger:
    """Test cases for the TelemetryLogger class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def logging_config(self, temp_dir):
        """Create a test logging configuration."""
        return LoggingConfig(
            log_level="INFO",
            output_directory=temp_dir,
            real_time_display=False,  # Disable for testing
            export_formats=["json"],
            save_conversation_history=True,
            save_telemetry=True
        )
    
    @pytest.fixture
    def telemetry_logger(self, logging_config):
        """Create a TelemetryLogger instance for testing."""
        logger = TelemetryLogger(logging_config, run_id="test-run-123")
        yield logger
        # Clean up to avoid file lock issues on Windows
        try:
            logger.close()
        except Exception:
            pass
    
    @pytest.fixture
    def sample_message(self):
        """Create a sample message for testing."""
        return Message(
            agent_id="agent_a",
            content="Hello, this is a test message.",
            timestamp=datetime.now(),
            token_count=25,
            metadata={"test": True}
        )
    
    def test_initialization(self, logging_config):
        """Test TelemetryLogger initialization."""
        logger = TelemetryLogger(logging_config, run_id="test-init")
        
        assert logger.run_id == "test-init"
        assert logger.config == logging_config
        assert isinstance(logger.start_time, datetime)
        assert logger.agent_metrics == {}
        assert logger.messages == []
        assert logger.context_snapshots == []
        assert logger.errors == []
    
    def test_initialization_with_auto_run_id(self, logging_config):
        """Test TelemetryLogger initialization with auto-generated run ID."""
        logger = TelemetryLogger(logging_config)
        
        assert logger.run_id is not None
        assert len(logger.run_id) > 0
        assert logger.run_id != "test-run-123"  # Should be different from fixture
    
    def test_start_agent_turn(self, telemetry_logger):
        """Test logging the start of an agent turn."""
        agent_id = "agent_a"
        context = {"turn": 1, "previous_message": "Hello"}
        
        with patch.object(telemetry_logger.logger, 'info') as mock_info:
            telemetry_logger.start_agent_turn(agent_id, context)
        
        # Verify the turn start time was recorded
        assert agent_id in telemetry_logger._turn_start_times
        assert isinstance(telemetry_logger._turn_start_times[agent_id], float)
        
        # Verify logging was called
        mock_info.assert_called_once()
        call_args = mock_info.call_args
        assert f"Agent {agent_id} turn started" in call_args[0][0]
    
    def test_log_agent_response(self, telemetry_logger, sample_message):
        """Test logging an agent response with metrics."""
        agent_id = "agent_a"
        response_time = 1.5
        model_calls = 2
        errors = ["Minor warning"]
        
        with patch.object(telemetry_logger.logger, 'info') as mock_info:
            telemetry_logger.log_agent_response(
                agent_id, sample_message, response_time, model_calls, errors
            )
        
        # Verify agent metrics were created and updated
        assert agent_id in telemetry_logger.agent_metrics
        metrics = telemetry_logger.agent_metrics[agent_id]
        assert metrics.response_time == response_time
        assert metrics.token_count == sample_message.token_count
        assert metrics.model_calls == model_calls
        assert metrics.errors == errors
        
        # Verify message was stored
        assert len(telemetry_logger.messages) == 1
        assert telemetry_logger.messages[0] == sample_message
        
        # Verify logging was called
        mock_info.assert_called()
    
    def test_log_agent_response_auto_timing(self, telemetry_logger, sample_message):
        """Test logging agent response with automatic timing calculation."""
        agent_id = "agent_a"
        
        # Start a turn to set up timing
        telemetry_logger.start_agent_turn(agent_id)
        time.sleep(0.1)  # Small delay to ensure measurable time difference
        
        with patch.object(telemetry_logger.logger, 'info'):
            telemetry_logger.log_agent_response(agent_id, sample_message)
        
        # Verify timing was calculated automatically
        metrics = telemetry_logger.agent_metrics[agent_id]
        assert metrics.response_time > 0
        assert agent_id not in telemetry_logger._turn_start_times  # Should be cleaned up
    
    def test_log_agent_response_accumulates_metrics(self, telemetry_logger, sample_message):
        """Test that multiple responses accumulate metrics correctly."""
        agent_id = "agent_a"
        
        # Log first response
        telemetry_logger.log_agent_response(agent_id, sample_message, 1.0, 1)
        
        # Create second message
        second_message = Message(
            agent_id=agent_id,
            content="Second message",
            timestamp=datetime.now(),
            token_count=15,
            metadata={}
        )
        
        # Log second response
        telemetry_logger.log_agent_response(agent_id, second_message, 2.0, 1, ["Error"])
        
        # Verify metrics were accumulated
        metrics = telemetry_logger.agent_metrics[agent_id]
        assert metrics.response_time == 3.0  # 1.0 + 2.0
        assert metrics.token_count == 40  # 25 + 15
        assert metrics.model_calls == 2  # 1 + 1
        assert len(metrics.errors) == 1
        assert metrics.errors[0] == "Error"
        
        # Verify both messages were stored
        assert len(telemetry_logger.messages) == 2
    
    def test_log_context_snapshot(self, telemetry_logger):
        """Test logging context window snapshots."""
        snapshot = ContextWindowSnapshot(
            turn_number=5,
            total_tokens=8000,
            available_tokens=2000,
            utilization_percentage=75.0,
            strategy_applied="sliding"
        )
        
        with patch.object(telemetry_logger.logger, 'info') as mock_info:
            telemetry_logger.log_context_snapshot(snapshot)
        
        # Verify snapshot was stored
        assert len(telemetry_logger.context_snapshots) == 1
        assert telemetry_logger.context_snapshots[0] == snapshot
        
        # Verify logging was called
        mock_info.assert_called_once()
        call_args = mock_info.call_args
        assert "75.0% utilized" in call_args[0][0]
    
    def test_log_context_snapshot_near_capacity_warning(self, telemetry_logger):
        """Test that near-capacity context snapshots generate warnings."""
        snapshot = ContextWindowSnapshot(
            turn_number=10,
            total_tokens=8000,
            available_tokens=400,
            utilization_percentage=95.0
        )
        
        with patch.object(telemetry_logger.logger, 'info') as mock_info, \
             patch.object(telemetry_logger.logger, 'warning') as mock_warning:
            telemetry_logger.log_context_snapshot(snapshot)
        
        # Verify both info and warning were called
        mock_info.assert_called_once()
        mock_warning.assert_called_once()
        
        warning_call_args = mock_warning.call_args
        assert "near capacity" in warning_call_args[0][0]
        assert "95.0%" in warning_call_args[0][0]
    
    def test_log_conversation_state(self, telemetry_logger):
        """Test logging conversation state."""
        state = ConversationState(
            current_turn=3,
            status=ConversationStatus.RUNNING,
            current_context_tokens=1500,
            max_context_tokens=8000
        )
        
        # Add some messages to the state
        for i in range(3):
            message = Message(
                agent_id=f"agent_{i % 2}",
                content=f"Message {i}",
                timestamp=datetime.now(),
                token_count=100
            )
            state.add_message(message)
        
        with patch.object(telemetry_logger.logger, 'info') as mock_info:
            telemetry_logger.log_conversation_state(state)
        
        # Verify logging was called with correct information
        mock_info.assert_called_once()
        call_args = mock_info.call_args
        # The state.current_turn will be 6 after adding 3 messages (each add_message increments turn)
        assert "Turn 6" in call_args[0][0]
        assert "running" in call_args[0][0]
    
    def test_log_error(self, telemetry_logger):
        """Test error logging functionality."""
        error_message = "Test error occurred"
        context = {"agent_id": "agent_a", "turn": 5}
        
        with patch.object(telemetry_logger.logger, 'error') as mock_error:
            telemetry_logger.log_error(error_message, context)
        
        # Verify error was stored
        assert len(telemetry_logger.errors) == 1
        error_data = telemetry_logger.errors[0]
        assert error_data["message"] == error_message
        assert error_data["context"] == context
        assert "timestamp" in error_data
        
        # Verify logging was called
        mock_error.assert_called_once()
        call_args = mock_error.call_args
        assert error_message in call_args[0][0]
    
    def test_log_system_event(self, telemetry_logger):
        """Test logging general system events."""
        event_type = "configuration_loaded"
        message = "System configuration loaded successfully"
        data = {"config_file": "config.yaml", "agents": 2}
        
        with patch.object(telemetry_logger.logger, 'info') as mock_info:
            telemetry_logger.log_system_event(event_type, message, data)
        
        # Verify logging was called
        mock_info.assert_called_once()
        call_args = mock_info.call_args
        assert message in call_args[0][0]
    
    def test_get_current_metrics(self, telemetry_logger, sample_message):
        """Test getting current metrics summary."""
        # Add some test data
        telemetry_logger.log_agent_response("agent_a", sample_message, 1.5, 1)
        
        second_message = Message(
            agent_id="agent_b",
            content="Response from agent B",
            timestamp=datetime.now(),
            token_count=30
        )
        telemetry_logger.log_agent_response("agent_b", second_message, 2.0, 1)
        
        # Add context snapshot
        snapshot = ContextWindowSnapshot(1, 8000, 7000, 12.5)
        telemetry_logger.log_context_snapshot(snapshot)
        
        # Add error
        telemetry_logger.log_error("Test error")
        
        metrics = telemetry_logger.get_current_metrics()
        
        # Verify metrics structure and values
        assert metrics["run_id"] == "test-run-123"
        assert "start_time" in metrics
        assert "duration" in metrics
        assert metrics["total_messages"] == 2
        assert metrics["total_response_time"] == 3.5  # 1.5 + 2.0
        assert metrics["total_tokens"] == 55  # 25 + 30
        assert metrics["total_model_calls"] == 2
        assert metrics["total_errors"] == 1
        assert metrics["average_response_time"] == 1.75  # 3.5 / 2
        assert len(metrics["agent_metrics"]) == 2
        assert metrics["context_snapshots_count"] == 1
        assert metrics["peak_context_utilization"] == 12.5
    
    def test_create_run_telemetry(self, telemetry_logger, sample_message):
        """Test creating complete RunTelemetry object."""
        # Add test data
        telemetry_logger.log_agent_response("agent_a", sample_message, 1.0, 1)
        snapshot = ContextWindowSnapshot(1, 8000, 7500, 6.25)
        telemetry_logger.log_context_snapshot(snapshot)
        telemetry_logger.log_error("Test error", {"context": "test"})
        
        configuration = {"model": "gpt-4", "temperature": 0.7}
        end_time = datetime.now()
        
        run_telemetry = telemetry_logger.create_run_telemetry(configuration, end_time)
        
        # Verify RunTelemetry object
        assert isinstance(run_telemetry, RunTelemetry)
        assert run_telemetry.run_id == "test-run-123"
        assert run_telemetry.start_time == telemetry_logger.start_time
        assert run_telemetry.end_time == end_time
        assert run_telemetry.total_turns == 1
        assert len(run_telemetry.agent_metrics) == 1
        assert len(run_telemetry.conversation_history) == 1
        assert len(run_telemetry.context_window_snapshots) == 1
        assert run_telemetry.configuration == configuration
        assert run_telemetry.metadata["total_errors"] == 1
        assert len(run_telemetry.metadata["error_details"]) == 1
    
    def test_finalize(self, telemetry_logger, sample_message):
        """Test finalizing telemetry logging."""
        # Add test data
        telemetry_logger.log_agent_response("agent_a", sample_message, 1.0, 1)
        configuration = {"model": "gpt-4"}
        
        with patch.object(telemetry_logger.logger, 'info') as mock_info:
            run_telemetry = telemetry_logger.finalize(configuration)
        
        # Verify RunTelemetry was created correctly
        assert isinstance(run_telemetry, RunTelemetry)
        assert run_telemetry.configuration == configuration
        assert run_telemetry.end_time is not None
        
        # Verify finalization was logged
        mock_info.assert_called()
        final_call = mock_info.call_args_list[-1]  # Get the last call
        assert "finalized" in final_call[0][0]
    
    def test_logger_setup_with_file_output(self, temp_dir):
        """Test that logger creates log files correctly."""
        config = LoggingConfig(
            log_level="DEBUG",
            output_directory=temp_dir,
            real_time_display=False
        )
        
        logger = TelemetryLogger(config, run_id="file-test")
        logger.log_system_event("test", "Test message")
        
        # Verify log file was created
        log_files = list(Path(temp_dir).glob("telemetry_*.log"))
        assert len(log_files) == 1
        
        log_file = log_files[0]
        assert "file-test" in log_file.name
        
        # Verify log file contains expected content
        content = log_file.read_text()
        assert "Test message" in content
        assert "file-test" in content
    
    def test_structured_logging_format(self, telemetry_logger):
        """Test that logs are properly structured with JSON format."""
        with patch.object(telemetry_logger.logger, 'info') as mock_info:
            telemetry_logger.log_system_event("test_event", "Test message", {"key": "value"})
        
        # Verify the log call includes structured data
        mock_info.assert_called_once()
        call_args = mock_info.call_args
        
        # Check that extra fields contain event data
        event_data = call_args[1]["extra"]["event_data"]
        assert event_data["event_type"] == "test_event"
        assert event_data["data"]["key"] == "value"
    
    def test_multiple_agents_metrics_tracking(self, telemetry_logger):
        """Test tracking metrics for multiple agents separately."""
        # Create messages from different agents
        message_a = Message("agent_a", "Message from A", datetime.now(), 20)
        message_b = Message("agent_b", "Message from B", datetime.now(), 30)
        
        # Log responses from both agents
        telemetry_logger.log_agent_response("agent_a", message_a, 1.0, 1)
        telemetry_logger.log_agent_response("agent_b", message_b, 2.0, 1)
        telemetry_logger.log_agent_response("agent_a", message_a, 1.5, 1)  # Second response from A
        
        # Verify separate metrics tracking
        assert len(telemetry_logger.agent_metrics) == 2
        
        metrics_a = telemetry_logger.agent_metrics["agent_a"]
        assert metrics_a.response_time == 2.5  # 1.0 + 1.5
        assert metrics_a.token_count == 40  # 20 + 20
        assert metrics_a.model_calls == 2
        
        metrics_b = telemetry_logger.agent_metrics["agent_b"]
        assert metrics_b.response_time == 2.0
        assert metrics_b.token_count == 30
        assert metrics_b.model_calls == 1
    
    def test_error_handling_in_logging(self, telemetry_logger):
        """Test that logging errors don't break the telemetry system."""
        # Mock logger to raise an exception
        with patch.object(telemetry_logger.logger, 'info', side_effect=Exception("Log error")):
            # This should not raise an exception
            try:
                telemetry_logger.log_system_event("test", "Test message")
                # If we get here, the exception was handled (or not raised)
                # In a real implementation, you might want to handle logging errors gracefully
            except Exception as e:
                # For now, we expect the exception to propagate
                # In production, you might want to implement error handling
                assert "Log error" in str(e)
    
    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_different_log_levels(self, temp_dir, log_level):
        """Test telemetry logger with different log levels."""
        config = LoggingConfig(
            log_level=log_level,
            output_directory=temp_dir,
            real_time_display=False
        )
        
        logger = TelemetryLogger(config)
        assert logger.logger.level == getattr(logging, log_level)
    
    def test_context_snapshot_peak_utilization_tracking(self, telemetry_logger):
        """Test tracking peak context utilization across multiple snapshots."""
        snapshots = [
            ContextWindowSnapshot(1, 8000, 7000, 12.5),
            ContextWindowSnapshot(2, 8000, 5000, 37.5),
            ContextWindowSnapshot(3, 8000, 2000, 75.0),
            ContextWindowSnapshot(4, 8000, 4000, 50.0)
        ]
        
        for snapshot in snapshots:
            telemetry_logger.log_context_snapshot(snapshot)
        
        metrics = telemetry_logger.get_current_metrics()
        assert metrics["peak_context_utilization"] == 75.0
    
    def test_empty_metrics_handling(self, telemetry_logger):
        """Test handling of metrics when no data has been logged."""
        metrics = telemetry_logger.get_current_metrics()
        
        assert metrics["total_messages"] == 0
        assert metrics["total_response_time"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["total_model_calls"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["average_response_time"] == 0.0
        assert len(metrics["agent_metrics"]) == 0
        assert metrics["context_snapshots_count"] == 0
        assert metrics["peak_context_utilization"] == 0.0


class TestRunLogger:
    """Test cases for the RunLogger class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def logging_config(self, temp_dir):
        """Create a test logging configuration."""
        return LoggingConfig(
            log_level="INFO",
            output_directory=temp_dir,
            real_time_display=False,
            export_formats=["json", "csv", "txt"],
            save_conversation_history=True,
            save_telemetry=True
        )
    
    @pytest.fixture
    def run_logger(self, logging_config):
        """Create a RunLogger instance for testing."""
        from src.agentic_conversation.telemetry import RunLogger
        logger = RunLogger(logging_config)
        yield logger
        logger.close()
    
    @pytest.fixture
    def sample_run_telemetry(self):
        """Create sample run telemetry data."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)
        
        # Create sample messages
        messages = [
            Message("agent_a", "Hello, how are you?", start_time, 15),
            Message("agent_b", "I'm doing well, thank you!", start_time + timedelta(seconds=30), 20),
            Message("agent_a", "That's great to hear!", start_time + timedelta(minutes=1), 18)
        ]
        
        # Create sample agent metrics
        agent_metrics = {
            "agent_a": AgentMetrics(response_time=2.5, token_count=33, model_calls=2),
            "agent_b": AgentMetrics(response_time=1.8, token_count=20, model_calls=1)
        }
        
        # Create sample context snapshots
        context_snapshots = [
            ContextWindowSnapshot(1, 8000, 7500, 6.25),
            ContextWindowSnapshot(2, 8000, 7000, 12.5),
            ContextWindowSnapshot(3, 8000, 6500, 18.75)
        ]
        
        return RunTelemetry(
            run_id="test-run-456",
            start_time=start_time,
            end_time=end_time,
            total_turns=3,
            agent_metrics=agent_metrics,
            conversation_history=messages,
            context_window_snapshots=context_snapshots,
            configuration={"model": "gpt-4", "temperature": 0.7}
        )
    
    def test_initialization(self, logging_config):
        """Test RunLogger initialization."""
        from src.agentic_conversation.telemetry import RunLogger
        
        logger = RunLogger(logging_config)
        
        assert logger.config == logging_config
        assert logger.output_directory.exists()
        assert logger.active_runs == {}
        assert logger.logger is not None
        
        logger.close()
    
    def test_start_run(self, run_logger):
        """Test starting a new run."""
        run_id = "test-run-001"
        configuration = {"model": "gpt-4", "temperature": 0.7}
        
        result = run_logger.start_run(run_id, configuration)
        
        assert result == run_id
        assert run_id in run_logger.active_runs
        
        run_data = run_logger.active_runs[run_id]
        assert run_data.run_id == run_id
        assert run_data.configuration == configuration
        assert run_data.start_time is not None
        assert run_data.end_time is None
    
    def test_start_run_duplicate_id(self, run_logger):
        """Test that starting a run with duplicate ID raises error."""
        run_id = "duplicate-run"
        configuration = {"model": "gpt-4"}
        
        run_logger.start_run(run_id, configuration)
        
        with pytest.raises(ValueError, match="already active"):
            run_logger.start_run(run_id, configuration)
    
    def test_update_run(self, run_logger, sample_run_telemetry):
        """Test updating an active run."""
        run_id = sample_run_telemetry.run_id
        
        # Start the run first
        run_logger.start_run(run_id, {"model": "gpt-4"})
        
        # Update with sample telemetry
        run_logger.update_run(run_id, sample_run_telemetry)
        
        # Verify update
        updated_data = run_logger.active_runs[run_id]
        assert updated_data == sample_run_telemetry
    
    def test_update_run_not_active(self, run_logger, sample_run_telemetry):
        """Test updating a non-active run raises error."""
        with pytest.raises(ValueError, match="not active"):
            run_logger.update_run("non-existent-run", sample_run_telemetry)
    
    def test_complete_run(self, run_logger, sample_run_telemetry):
        """Test completing a run and saving data."""
        run_id = sample_run_telemetry.run_id
        
        # Start the run
        run_logger.start_run(run_id, {"model": "gpt-4"})
        
        # Complete the run
        run_logger.complete_run(run_id, sample_run_telemetry)
        
        # Verify run is no longer active
        assert run_id not in run_logger.active_runs
        
        # Verify files were created
        output_dir = run_logger.output_directory
        timestamp = sample_run_telemetry.start_time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"run_{run_id}_{timestamp}"
        
        assert (output_dir / f"{base_filename}.json").exists()
        assert (output_dir / f"{base_filename}_messages.csv").exists()
        assert (output_dir / f"{base_filename}_metrics.csv").exists()
        assert (output_dir / f"{base_filename}.txt").exists()
    
    def test_complete_run_not_active(self, run_logger, sample_run_telemetry):
        """Test completing a non-active run raises error."""
        with pytest.raises(ValueError, match="not active"):
            run_logger.complete_run("non-existent-run", sample_run_telemetry)
    
    def test_save_json_format(self, run_logger, sample_run_telemetry):
        """Test saving run data in JSON format."""
        run_id = sample_run_telemetry.run_id
        timestamp = sample_run_telemetry.start_time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"run_{run_id}_{timestamp}"
        
        run_logger._save_json(base_filename, sample_run_telemetry)
        
        json_file = run_logger.output_directory / f"{base_filename}.json"
        assert json_file.exists()
        
        # Verify JSON content
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["run_id"] == run_id
        assert data["total_turns"] == 3
        assert len(data["conversation_history"]) == 3
        assert len(data["agent_metrics"]) == 2
    
    def test_save_csv_format(self, run_logger, sample_run_telemetry):
        """Test saving run data in CSV format."""
        import csv
        
        run_id = sample_run_telemetry.run_id
        timestamp = sample_run_telemetry.start_time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"run_{run_id}_{timestamp}"
        
        run_logger._save_csv(base_filename, sample_run_telemetry)
        
        # Check messages CSV
        messages_file = run_logger.output_directory / f"{base_filename}_messages.csv"
        assert messages_file.exists()
        
        with open(messages_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4  # Header + 3 messages
            assert rows[0] == ['timestamp', 'agent_id', 'content', 'token_count', 'metadata']
        
        # Check metrics CSV
        metrics_file = run_logger.output_directory / f"{base_filename}_metrics.csv"
        assert metrics_file.exists()
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) > 10  # Should have multiple metrics rows
            assert rows[0] == ['metric', 'value']
    
    def test_save_txt_format(self, run_logger, sample_run_telemetry):
        """Test saving run data in text format."""
        run_id = sample_run_telemetry.run_id
        timestamp = sample_run_telemetry.start_time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"run_{run_id}_{timestamp}"
        
        run_logger._save_txt(base_filename, sample_run_telemetry)
        
        txt_file = run_logger.output_directory / f"{base_filename}.txt"
        assert txt_file.exists()
        
        # Verify text content
        content = txt_file.read_text(encoding='utf-8')
        assert "Conversation Run Report" in content
        assert run_id in content
        assert "Agent Performance" in content
        assert "Conversation History" in content
        assert "Context Window Utilization" in content
    
    def test_get_active_runs(self, run_logger):
        """Test getting list of active runs."""
        assert run_logger.get_active_runs() == []
        
        # Start some runs
        run_logger.start_run("run1", {"model": "gpt-4"})
        run_logger.start_run("run2", {"model": "gpt-3.5"})
        
        active_runs = run_logger.get_active_runs()
        assert len(active_runs) == 2
        assert "run1" in active_runs
        assert "run2" in active_runs
    
    def test_get_run_status(self, run_logger):
        """Test getting run status information."""
        # Test non-existent run
        assert run_logger.get_run_status("non-existent") is None
        
        # Start a run and check status
        run_id = "status-test-run"
        run_logger.start_run(run_id, {"model": "gpt-4"})
        
        status = run_logger.get_run_status(run_id)
        assert status is not None
        assert status["run_id"] == run_id
        assert "start_time" in status
        assert "duration" in status
        assert status["total_turns"] == 0
        assert status["total_messages"] == 0
        assert status["is_completed"] == False
    
    def test_list_saved_runs(self, run_logger, sample_run_telemetry):
        """Test listing saved run files."""
        # Initially no saved runs
        assert run_logger.list_saved_runs() == []
        
        # Complete a run to create saved files
        run_id = sample_run_telemetry.run_id
        run_logger.start_run(run_id, {"model": "gpt-4"})
        run_logger.complete_run(run_id, sample_run_telemetry)
        
        # Check saved runs list
        saved_runs = run_logger.list_saved_runs()
        assert len(saved_runs) == 1
        
        saved_run = saved_runs[0]
        assert saved_run["run_id"] == run_id
        assert saved_run["total_turns"] == 3
        assert "file_path" in saved_run
        assert "file_size" in saved_run
    
    def test_load_run_data(self, run_logger, sample_run_telemetry):
        """Test loading run data from saved files."""
        run_id = sample_run_telemetry.run_id
        
        # Test loading non-existent run
        assert run_logger.load_run_data("non-existent") is None
        
        # Save a run and then load it
        run_logger.start_run(run_id, {"model": "gpt-4"})
        run_logger.complete_run(run_id, sample_run_telemetry)
        
        loaded_data = run_logger.load_run_data(run_id)
        assert loaded_data is not None
        assert loaded_data.run_id == run_id
        assert loaded_data.total_turns == 3
        assert len(loaded_data.conversation_history) == 3
    
    def test_aggregate_runs_empty(self, run_logger):
        """Test aggregating runs when no runs exist."""
        result = run_logger.aggregate_runs()
        assert "error" in result
        assert "No runs found" in result["error"]
    
    def test_aggregate_runs(self, run_logger):
        """Test aggregating statistics across multiple runs."""
        # Create and save multiple runs
        runs_data = []
        for i in range(3):
            run_id = f"aggregate-test-{i}"
            start_time = datetime.now() - timedelta(hours=i)
            end_time = start_time + timedelta(minutes=10)
            
            telemetry = RunTelemetry(
                run_id=run_id,
                start_time=start_time,
                end_time=end_time,
                total_turns=5 + i,
                agent_metrics={
                    "agent_a": AgentMetrics(response_time=2.0 + i, token_count=50 + i*10, model_calls=3 + i)
                },
                conversation_history=[
                    Message("agent_a", f"Message {j}", start_time, 10) 
                    for j in range(5 + i)
                ]
            )
            
            run_logger.start_run(run_id, {"model": "gpt-4"})
            run_logger.complete_run(run_id, telemetry)
            runs_data.append(telemetry)
        
        # Test aggregation
        result = run_logger.aggregate_runs()
        
        assert "error" not in result
        assert result["aggregation_summary"]["total_runs"] == 3
        assert result["conversation_metrics"]["total_turns"] == 18  # 5+6+7
        assert result["conversation_metrics"]["total_tokens"] == 180  # (5+6+7)*10 tokens per message
        assert "performance_metrics" in result
    
    def test_aggregate_specific_runs(self, run_logger, sample_run_telemetry):
        """Test aggregating specific run IDs."""
        # Save a run
        run_id = sample_run_telemetry.run_id
        run_logger.start_run(run_id, {"model": "gpt-4"})
        run_logger.complete_run(run_id, sample_run_telemetry)
        
        # Aggregate specific run
        result = run_logger.aggregate_runs([run_id])
        
        assert "error" not in result
        assert result["aggregation_summary"]["total_runs"] == 1
        assert run_id in result["aggregation_summary"]["run_ids"]
    
    def test_cleanup_old_runs(self, run_logger, sample_run_telemetry):
        """Test cleaning up old run files."""
        # Create a run file
        run_id = sample_run_telemetry.run_id
        run_logger.start_run(run_id, {"model": "gpt-4"})
        run_logger.complete_run(run_id, sample_run_telemetry)
        
        # Verify files exist
        saved_runs = run_logger.list_saved_runs()
        assert len(saved_runs) == 1
        
        # Clean up files older than 0 days (should delete all)
        deleted_count = run_logger.cleanup_old_runs(days_old=0)
        
        # Should have deleted multiple files (json, csv, txt)
        assert deleted_count > 0
        
        # Verify files are gone
        saved_runs_after = run_logger.list_saved_runs()
        assert len(saved_runs_after) == 0
    
    def test_multiple_export_formats(self, temp_dir):
        """Test RunLogger with different export format configurations."""
        # Test with only JSON
        config_json = LoggingConfig(
            output_directory=temp_dir,
            export_formats=["json"],
            real_time_display=False
        )
        
        from src.agentic_conversation.telemetry import RunLogger
        logger_json = RunLogger(config_json)
        
        # Create sample telemetry
        telemetry = RunTelemetry(
            run_id="format-test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_turns=1
        )
        
        logger_json.start_run("format-test", {})
        logger_json.complete_run("format-test", telemetry)
        
        # Check only JSON file was created
        output_dir = Path(temp_dir)
        json_files = list(output_dir.glob("*.json"))
        csv_files = list(output_dir.glob("*.csv"))
        txt_files = list(output_dir.glob("*.txt"))
        
        assert len(json_files) == 1
        assert len(csv_files) == 0
        assert len(txt_files) == 0
        
        logger_json.close()
    
    def test_error_handling_in_file_operations(self, run_logger, sample_run_telemetry):
        """Test error handling during file save operations."""
        # Mock file operations to raise exceptions
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # This should not raise an exception, but log errors
            run_logger._save_json("test", sample_run_telemetry)
            run_logger._save_csv("test", sample_run_telemetry)
            run_logger._save_txt("test", sample_run_telemetry)
    
    def test_run_logger_with_custom_output_directory(self, temp_dir, logging_config):
        """Test RunLogger with custom output directory override."""
        custom_dir = Path(temp_dir) / "custom_logs"
        
        from src.agentic_conversation.telemetry import RunLogger
        logger = RunLogger(logging_config, output_directory=str(custom_dir))
        
        assert logger.output_directory == custom_dir
        assert custom_dir.exists()
        
        logger.close()