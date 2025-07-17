"""
Unit tests for the CircuitBreaker implementation.

These tests verify the circuit breaker pattern implementation including
state transitions, failure handling, and recovery mechanisms.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.agentic_conversation.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError,
    CircuitState, CircuitBreakerManager, get_circuit_breaker
)


class TestCircuitBreakerConfig:
    """Test suite for CircuitBreakerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=15.0
        )
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2
        assert config.timeout == 15.0


class TestCircuitBreaker:
    """Test suite for CircuitBreaker class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=1.0
        )
    
    @pytest.fixture
    def circuit_breaker(self, config):
        """Create a circuit breaker for testing."""
        return CircuitBreaker("test_circuit", config)
    
    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert circuit_breaker.total_calls == 0
        assert circuit_breaker.total_failures == 0
        assert circuit_breaker.total_successes == 0
        assert circuit_breaker.is_available() is True
    
    def test_successful_call(self, circuit_breaker):
        """Test successful function call through circuit breaker."""
        def success_func():
            return "success"
        
        result = circuit_breaker.call(success_func)
        
        assert result == "success"
        assert circuit_breaker.total_calls == 1
        assert circuit_breaker.total_successes == 1
        assert circuit_breaker.total_failures == 0
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == CircuitState.CLOSED
    
    def test_failed_call(self, circuit_breaker):
        """Test failed function call through circuit breaker."""
        def fail_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            circuit_breaker.call(fail_func)
        
        assert circuit_breaker.total_calls == 1
        assert circuit_breaker.total_successes == 0
        assert circuit_breaker.total_failures == 1
        assert circuit_breaker.failure_count == 1
        assert circuit_breaker.state == CircuitState.CLOSED  # Still closed, below threshold
    
    def test_circuit_opens_after_threshold_failures(self, circuit_breaker):
        """Test that circuit opens after reaching failure threshold."""
        def fail_func():
            raise ValueError("test error")
        
        # Make failures up to threshold
        for i in range(circuit_breaker.config.failure_threshold):
            with pytest.raises(ValueError):
                circuit_breaker.call(fail_func)
        
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == circuit_breaker.config.failure_threshold
        assert circuit_breaker.is_available() is False
    
    def test_circuit_blocks_calls_when_open(self, circuit_breaker):
        """Test that circuit breaker blocks calls when open."""
        # Force circuit to open and set recent failure time
        circuit_breaker._transition_to_open()
        circuit_breaker.last_failure_time = datetime.now()  # Recent failure
        
        def success_func():
            return "success"
        
        with pytest.raises(CircuitBreakerError) as exc_info:
            circuit_breaker.call(success_func)
        
        assert "Circuit breaker 'test_circuit' is open" in str(exc_info.value)
        assert exc_info.value.circuit_name == "test_circuit"
        assert exc_info.value.state == CircuitState.OPEN
    
    def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transition from open to half-open after timeout."""
        # Force circuit to open and set last failure time in the past
        circuit_breaker._transition_to_open()
        circuit_breaker.last_failure_time = datetime.now() - timedelta(seconds=2)
        
        def success_func():
            return "success"
        
        # Should transition to half-open and allow the call
        result = circuit_breaker.call(success_func)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_from_half_open_after_successes(self, circuit_breaker):
        """Test circuit closes from half-open after enough successes."""
        # Set circuit to half-open
        circuit_breaker._transition_to_half_open()
        
        def success_func():
            return "success"
        
        # Make successful calls up to success threshold
        for i in range(circuit_breaker.config.success_threshold):
            result = circuit_breaker.call(success_func)
            assert result == "success"
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.success_count == 0  # Reset after closing
        assert circuit_breaker.failure_count == 0  # Reset after closing
    
    def test_circuit_reopens_on_failure_in_half_open(self, circuit_breaker):
        """Test circuit reopens on failure while in half-open state."""
        # Set circuit to half-open
        circuit_breaker._transition_to_half_open()
        
        def fail_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            circuit_breaker.call(fail_func)
        
        assert circuit_breaker.state == CircuitState.OPEN
    
    def test_timeout_handling(self, circuit_breaker):
        """Test timeout handling in circuit breaker."""
        def slow_func():
            time.sleep(2)  # Longer than timeout
            return "success"
        
        with pytest.raises(TimeoutError):
            circuit_breaker.call(slow_func)
        
        assert circuit_breaker.total_failures == 1
        assert circuit_breaker.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_async_call_success(self, circuit_breaker):
        """Test successful async function call through circuit breaker."""
        async def async_success_func():
            return "async_success"
        
        result = await circuit_breaker.async_call(async_success_func)
        
        assert result == "async_success"
        assert circuit_breaker.total_successes == 1
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_async_call_failure(self, circuit_breaker):
        """Test failed async function call through circuit breaker."""
        async def async_fail_func():
            raise ValueError("async test error")
        
        with pytest.raises(ValueError, match="async test error"):
            await circuit_breaker.async_call(async_fail_func)
        
        assert circuit_breaker.total_failures == 1
        assert circuit_breaker.failure_count == 1
    
    @pytest.mark.asyncio
    async def test_async_timeout(self, circuit_breaker):
        """Test async timeout handling."""
        async def slow_async_func():
            await asyncio.sleep(2)  # Longer than timeout
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.async_call(slow_async_func)
        
        assert circuit_breaker.total_failures == 1
    
    def test_manual_reset(self, circuit_breaker):
        """Test manual reset of circuit breaker."""
        # Force circuit to open
        circuit_breaker._transition_to_open()
        circuit_breaker.failure_count = 5
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        circuit_breaker.reset()
        
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
    
    def test_force_open(self, circuit_breaker):
        """Test manually forcing circuit breaker open."""
        assert circuit_breaker.state == CircuitState.CLOSED
        
        circuit_breaker.force_open()
        
        assert circuit_breaker.state == CircuitState.OPEN
    
    def test_get_stats(self, circuit_breaker):
        """Test getting circuit breaker statistics."""
        # Make some calls to generate stats
        def success_func():
            return "success"
        
        def fail_func():
            raise ValueError("error")
        
        circuit_breaker.call(success_func)
        
        try:
            circuit_breaker.call(fail_func)
        except ValueError:
            pass
        
        stats = circuit_breaker.get_stats()
        
        assert stats["name"] == "test_circuit"
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["total_calls"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1
        assert stats["success_rate"] == 50.0
        assert "config" in stats
        assert "state_changes" in stats
    
    def test_string_representations(self, circuit_breaker):
        """Test string representations of circuit breaker."""
        str_repr = str(circuit_breaker)
        assert "CircuitBreaker" in str_repr
        assert "test_circuit" in str_repr
        assert "closed" in str_repr
        
        repr_str = repr(circuit_breaker)
        assert "CircuitBreaker" in repr_str
        assert "test_circuit" in repr_str


class TestCircuitBreakerManager:
    """Test suite for CircuitBreakerManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a circuit breaker manager for testing."""
        return CircuitBreakerManager()
    
    def test_get_circuit_creates_new(self, manager):
        """Test that get_circuit creates new circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        circuit = manager.get_circuit("test1", config)
        
        assert isinstance(circuit, CircuitBreaker)
        assert circuit.name == "test1"
        assert circuit.config.failure_threshold == 2
        assert "test1" in manager.circuits
    
    def test_get_circuit_returns_existing(self, manager):
        """Test that get_circuit returns existing circuit breaker."""
        circuit1 = manager.get_circuit("test1")
        circuit2 = manager.get_circuit("test1")
        
        assert circuit1 is circuit2
        assert len(manager.circuits) == 1
    
    def test_remove_circuit(self, manager):
        """Test removing circuit breaker."""
        manager.get_circuit("test1")
        assert "test1" in manager.circuits
        
        removed = manager.remove_circuit("test1")
        assert removed is True
        assert "test1" not in manager.circuits
        
        # Try to remove non-existent circuit
        removed = manager.remove_circuit("nonexistent")
        assert removed is False
    
    def test_reset_all(self, manager):
        """Test resetting all circuit breakers."""
        circuit1 = manager.get_circuit("test1")
        circuit2 = manager.get_circuit("test2")
        
        # Force circuits to open
        circuit1._transition_to_open()
        circuit2._transition_to_open()
        
        assert circuit1.state == CircuitState.OPEN
        assert circuit2.state == CircuitState.OPEN
        
        manager.reset_all()
        
        assert circuit1.state == CircuitState.CLOSED
        assert circuit2.state == CircuitState.CLOSED
    
    def test_get_all_stats(self, manager):
        """Test getting statistics for all circuit breakers."""
        circuit1 = manager.get_circuit("test1")
        circuit2 = manager.get_circuit("test2")
        
        # Make some calls to generate stats
        def success_func():
            return "success"
        
        circuit1.call(success_func)
        circuit2.call(success_func)
        
        all_stats = manager.get_all_stats()
        
        assert "test1" in all_stats
        assert "test2" in all_stats
        assert all_stats["test1"]["total_calls"] == 1
        assert all_stats["test2"]["total_calls"] == 1
    
    def test_get_summary_empty(self, manager):
        """Test getting summary with no circuit breakers."""
        summary = manager.get_summary()
        
        assert summary["total_circuits"] == 0
        assert summary["open_circuits"] == 0
        assert summary["half_open_circuits"] == 0
        assert summary["closed_circuits"] == 0
        assert summary["total_calls"] == 0
        assert summary["overall_success_rate"] == 0.0
    
    def test_get_summary_with_circuits(self, manager):
        """Test getting summary with circuit breakers."""
        circuit1 = manager.get_circuit("test1")
        circuit2 = manager.get_circuit("test2")
        
        # Make some calls
        def success_func():
            return "success"
        
        def fail_func():
            raise ValueError("error")
        
        circuit1.call(success_func)
        circuit2.call(success_func)
        
        try:
            circuit1.call(fail_func)
        except ValueError:
            pass
        
        # Force one circuit to open
        circuit2._transition_to_open()
        
        summary = manager.get_summary()
        
        assert summary["total_circuits"] == 2
        assert summary["open_circuits"] == 1
        assert summary["closed_circuits"] == 1
        assert summary["total_calls"] == 3
        assert summary["total_successes"] == 2
        assert summary["total_failures"] == 1
        assert summary["overall_success_rate"] == pytest.approx(66.67, rel=1e-2)
    
    def test_list_circuits(self, manager):
        """Test listing circuit breaker names."""
        assert manager.list_circuits() == []
        
        manager.get_circuit("test1")
        manager.get_circuit("test2")
        
        circuits = manager.list_circuits()
        assert len(circuits) == 2
        assert "test1" in circuits
        assert "test2" in circuits


class TestGlobalCircuitBreakerManager:
    """Test suite for global circuit breaker manager functions."""
    
    def test_get_circuit_breaker_function(self):
        """Test global get_circuit_breaker function."""
        config = CircuitBreakerConfig(failure_threshold=2)
        circuit = get_circuit_breaker("global_test", config)
        
        assert isinstance(circuit, CircuitBreaker)
        assert circuit.name == "global_test"
        assert circuit.config.failure_threshold == 2
        
        # Getting same circuit should return same instance
        circuit2 = get_circuit_breaker("global_test")
        assert circuit is circuit2


class TestCircuitBreakerError:
    """Test suite for CircuitBreakerError exception."""
    
    def test_circuit_breaker_error_creation(self):
        """Test creating CircuitBreakerError."""
        error = CircuitBreakerError("test message", "test_circuit", CircuitState.OPEN)
        
        assert str(error) == "test message"
        assert error.circuit_name == "test_circuit"
        assert error.state == CircuitState.OPEN
    
    def test_circuit_breaker_error_inheritance(self):
        """Test that CircuitBreakerError inherits from Exception."""
        error = CircuitBreakerError("test", "circuit", CircuitState.OPEN)
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__])