"""
Circuit breaker pattern implementation for the agentic conversation system.

This module provides a circuit breaker to handle persistent API failures
and prevent cascading failures in the conversation system.
"""

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, calls are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout: float = 60.0  # Seconds to wait before trying half-open
    success_threshold: int = 3  # Successful calls needed to close circuit from half-open
    timeout: float = 30.0  # Timeout for individual calls


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, circuit_name: str, state: CircuitState):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.state = state


class CircuitBreaker:
    """
    Circuit breaker implementation for handling persistent failures.
    
    The circuit breaker monitors failures and prevents calls when a service
    is consistently failing, allowing it time to recover.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker for identification
            config: Configuration for circuit breaker behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes: Dict[str, int] = {
            CircuitState.CLOSED.value: 0,
            CircuitState.OPEN.value: 0,
            CircuitState.HALF_OPEN.value: 0
        }
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function call through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        self.total_calls += 1
        
        # Check if circuit should transition from open to half-open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open",
                    self.name,
                    self.state
                )
        
        try:
            # Execute the function call
            start_time = time.time()
            result = func(*args, **kwargs)
            call_time = time.time() - start_time
            
            # Check for timeout
            if call_time > self.config.timeout:
                raise TimeoutError(f"Call timed out after {call_time:.2f}s")
            
            # Record success
            self._record_success()
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(e)
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an async function call through the circuit breaker.
        
        Args:
            func: Async function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function call
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        import asyncio
        
        self.total_calls += 1
        
        # Check if circuit should transition from open to half-open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open",
                    self.name,
                    self.state
                )
        
        try:
            # Execute the async function call with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            self._record_success()
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(e)
            raise
    
    def _record_success(self) -> None:
        """Record a successful call."""
        self.total_successes += 1
        self.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        self.total_failures += 1
        self.last_failure_time = datetime.now()
        self.failure_count += 1
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state should open the circuit
            self._transition_to_open()
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_changes[CircuitState.CLOSED.value] += 1
    
    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.state_changes[CircuitState.OPEN.value] += 1
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.state_changes[CircuitState.HALF_OPEN.value] += 1
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._transition_to_closed()
    
    def force_open(self) -> None:
        """Manually force the circuit breaker to open state."""
        self._transition_to_open()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "success_rate": (self.total_successes / self.total_calls * 100) if self.total_calls > 0 else 0,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "state_changes": self.state_changes.copy(),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }
    
    def is_available(self) -> bool:
        """Check if the circuit breaker allows calls."""
        if self.state == CircuitState.OPEN:
            return self._should_attempt_reset()
        return True
    
    def __str__(self) -> str:
        """String representation of the circuit breaker."""
        return f"CircuitBreaker(name='{self.name}', state={self.state.value}, failures={self.failure_count})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the circuit breaker."""
        return (f"CircuitBreaker(name='{self.name}', state={self.state.value}, "
                f"failures={self.failure_count}, successes={self.success_count})")


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    
    This class provides a centralized way to manage multiple circuit breakers
    and get aggregate statistics across all circuits.
    """
    
    def __init__(self):
        """Initialize the circuit breaker manager."""
        self.circuits: Dict[str, CircuitBreaker] = {}
    
    def get_circuit(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create a circuit breaker by name.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker (only used if creating new)
            
        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuits:
            self.circuits[name] = CircuitBreaker(name, config)
        return self.circuits[name]
    
    def remove_circuit(self, name: str) -> bool:
        """
        Remove a circuit breaker by name.
        
        Args:
            name: Name of the circuit breaker to remove
            
        Returns:
            True if circuit was removed, False if it didn't exist
        """
        if name in self.circuits:
            del self.circuits[name]
            return True
        return False
    
    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        for circuit in self.circuits.values():
            circuit.reset()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: circuit.get_stats() for name, circuit in self.circuits.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all circuit breakers."""
        if not self.circuits:
            return {
                "total_circuits": 0,
                "open_circuits": 0,
                "half_open_circuits": 0,
                "closed_circuits": 0,
                "total_calls": 0,
                "total_failures": 0,
                "overall_success_rate": 0.0
            }
        
        total_calls = sum(c.total_calls for c in self.circuits.values())
        total_failures = sum(c.total_failures for c in self.circuits.values())
        total_successes = sum(c.total_successes for c in self.circuits.values())
        
        state_counts = {
            CircuitState.OPEN.value: 0,
            CircuitState.HALF_OPEN.value: 0,
            CircuitState.CLOSED.value: 0
        }
        
        for circuit in self.circuits.values():
            state_counts[circuit.state.value] += 1
        
        return {
            "total_circuits": len(self.circuits),
            "open_circuits": state_counts[CircuitState.OPEN.value],
            "half_open_circuits": state_counts[CircuitState.HALF_OPEN.value],
            "closed_circuits": state_counts[CircuitState.CLOSED.value],
            "total_calls": total_calls,
            "total_failures": total_failures,
            "total_successes": total_successes,
            "overall_success_rate": (total_successes / total_calls * 100) if total_calls > 0 else 0.0
        }
    
    def list_circuits(self) -> List[str]:
        """Get list of all circuit breaker names."""
        return list(self.circuits.keys())


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Convenience function to get a circuit breaker from the global manager.
    
    Args:
        name: Name of the circuit breaker
        config: Configuration for the circuit breaker
        
    Returns:
        CircuitBreaker instance
    """
    return circuit_manager.get_circuit(name, config)