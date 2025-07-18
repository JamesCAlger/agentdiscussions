"""
Tool execution engine with resource management.

This module provides the ToolExecutor class which handles the low-level execution
of tools with advanced features like timeout handling, retry logic, rate limiting,
circuit breakers, and resource pooling.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import aiohttp
from datetime import datetime, timedelta
import random

from .base import BaseTool, ToolResult, ToolContext
from .exceptions import (
    ToolExecutionError, 
    ToolTimeoutError, 
    ToolRateLimitError,
    ToolError
)


class CircuitBreakerState(Enum):
    """States for the circuit breaker pattern."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    burst_size: int = 10  # Allow burst of requests
    window_size: float = 60.0  # Time window in seconds


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to delays


@dataclass
class ExecutorConfig:
    """Configuration for the tool executor."""
    default_timeout: float = 30.0
    max_concurrent_executions: int = 10
    connection_pool_size: int = 100
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


class CircuitBreaker:
    """
    Circuit breaker implementation for tool execution.
    
    Prevents cascading failures by temporarily disabling tools that are
    consistently failing, allowing them time to recover.
    """
    
    def __init__(self, config: CircuitBreakerConfig, tool_name: str):
        self.config = config
        self.tool_name = tool_name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker.{tool_name}")
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            ToolExecutionError: If the circuit breaker is open
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info(f"Circuit breaker for {self.tool_name} moving to HALF_OPEN")
            else:
                raise ToolExecutionError(
                    f"Circuit breaker is OPEN for tool {self.tool_name}",
                    tool_name=self.tool_name,
                    context={"state": self.state.value, "failure_count": self.failure_count}
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker for {self.tool_name} CLOSED after recovery")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self.logger.warning(f"Circuit breaker for {self.tool_name} back to OPEN after failure")
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker for {self.tool_name} OPENED after {self.failure_count} failures")


class RateLimiter:
    """
    Token bucket rate limiter for tool execution.
    
    Implements a token bucket algorithm to limit the rate of tool executions,
    preventing overwhelming of external services.
    """
    
    def __init__(self, config: RateLimitConfig, tool_name: str):
        self.config = config
        self.tool_name = tool_name
        self.tokens = config.burst_size
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.RateLimiter.{tool_name}")
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the rate limiter.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            bool: True if tokens were acquired, False if rate limited
            
        Raises:
            ToolRateLimitError: If rate limit is exceeded
        """
        async with self.lock:
            self._refill_tokens()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                raise ToolRateLimitError(
                    f"Rate limit exceeded for tool {self.tool_name}",
                    tool_name=self.tool_name,
                    context={
                        "requested_tokens": tokens,
                        "available_tokens": self.tokens,
                        "rate_limit": self.config.requests_per_minute
                    }
                )
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate tokens to add based on rate
        tokens_per_second = self.config.requests_per_minute / 60.0
        tokens_to_add = elapsed * tokens_per_second
        
        self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)
        self.last_refill = now
    
    @asynccontextmanager
    async def limit(self, tokens: int = 1):
        """Context manager for rate limiting."""
        await self.acquire(tokens)
        try:
            yield
        finally:
            pass  # Nothing to clean up


class ConnectionPool:
    """
    HTTP connection pool for tool execution.
    
    Manages a pool of HTTP connections to improve performance and resource
    utilization when making external API calls.
    """
    
    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.session = None
        self.logger = logging.getLogger(f"{__name__}.ConnectionPool")
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config.read_timeout,
                connect=self.config.connection_timeout
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'AgenticConversation-ToolExecutor/1.0'
                }
            )
            
            self.logger.info("Created new HTTP session with connection pool")
        
        return self.session
    
    async def close(self):
        """Close the connection pool."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Closed HTTP session and connection pool")


class ToolExecutor:
    """
    Advanced tool execution engine with resource management.
    
    This class provides sophisticated execution capabilities including:
    - Async execution with timeout handling
    - Retry logic with exponential backoff
    - Rate limiting and circuit breaker patterns
    - HTTP connection pooling
    - Comprehensive error handling and logging
    """
    
    def __init__(self, config: ExecutorConfig = None):
        """
        Initialize the tool executor.
        
        Args:
            config: Configuration for the executor
        """
        self.config = config or ExecutorConfig()
        self.connection_pool = ConnectionPool(self.config)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.execution_semaphore = asyncio.Semaphore(self.config.max_concurrent_executions)
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'timeout_executions': 0,
            'rate_limited_executions': 0,
            'circuit_breaker_rejections': 0
        }
    
    async def execute_tool(
        self, 
        tool: BaseTool, 
        query: str, 
        context: ToolContext
    ) -> ToolResult:
        """
        Execute a tool with full resource management and error handling.
        
        Args:
            tool: The tool to execute
            query: The query for the tool
            context: The execution context
            
        Returns:
            ToolResult: The result of the tool execution
        """
        tool_info = tool.get_tool_info()
        tool_name = tool_info.name
        
        # Update metrics
        self.execution_metrics['total_executions'] += 1
        
        # Validate input
        validation_errors = tool.validate_input(query, context)
        if validation_errors:
            result = ToolResult(
                success=False,
                content="",
                tool_name=tool_name,
                errors=validation_errors
            )
            self.execution_metrics['failed_executions'] += 1
            return result
        
        # Get or create circuit breaker and rate limiter
        circuit_breaker = self._get_circuit_breaker(tool_name)
        rate_limiter = self._get_rate_limiter(tool_name)
        
        try:
            # Acquire execution semaphore to limit concurrent executions
            async with self.execution_semaphore:
                # Apply rate limiting
                async with rate_limiter.limit():
                    # Execute through circuit breaker with retry logic
                    result = await circuit_breaker.call(
                        self._execute_with_retry,
                        tool,
                        query,
                        context
                    )
                    
                    self.execution_metrics['successful_executions'] += 1
                    return result
                    
        except ToolRateLimitError as e:
            self.execution_metrics['rate_limited_executions'] += 1
            self.logger.warning(f"Rate limit exceeded for tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                content="",
                tool_name=tool_name,
                errors=[str(e)]
            )
            
        except ToolExecutionError as e:
            if "Circuit breaker is OPEN" in str(e):
                self.execution_metrics['circuit_breaker_rejections'] += 1
            else:
                self.execution_metrics['failed_executions'] += 1
            
            self.logger.error(f"Tool execution failed for {tool_name}: {e}")
            return ToolResult(
                success=False,
                content="",
                tool_name=tool_name,
                errors=[str(e)]
            )
            
        except Exception as e:
            self.execution_metrics['failed_executions'] += 1
            self.logger.error(f"Unexpected error executing tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                content="",
                tool_name=tool_name,
                errors=[f"Unexpected error: {str(e)}"]
            )
    
    async def _execute_with_retry(
        self, 
        tool: BaseTool, 
        query: str, 
        context: ToolContext
    ) -> ToolResult:
        """
        Execute a tool with retry logic and exponential backoff.
        
        Args:
            tool: The tool to execute
            query: The query for the tool
            context: The execution context
            
        Returns:
            ToolResult: The result of the tool execution
            
        Raises:
            ToolExecutionError: If all retry attempts fail
            ToolTimeoutError: If execution times out
        """
        tool_info = tool.get_tool_info()
        tool_name = tool_info.name
        retry_config = self.config.retry_config
        
        last_exception = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                # Execute with timeout
                result = await self._execute_with_timeout(tool, query, context)
                
                if result.success:
                    if attempt > 0:
                        self.logger.info(f"Tool {tool_name} succeeded on attempt {attempt + 1}")
                    return result
                else:
                    # Tool returned failure, but no exception - don't retry
                    return result
                    
            except (ToolTimeoutError, ToolExecutionError) as e:
                last_exception = e
                
                if attempt < retry_config.max_attempts - 1:
                    delay = self._calculate_retry_delay(attempt, retry_config)
                    self.logger.warning(
                        f"Tool {tool_name} failed on attempt {attempt + 1}, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"Tool {tool_name} failed after {retry_config.max_attempts} attempts: {e}"
                    )
            
            except Exception as e:
                # Non-retryable error
                raise ToolExecutionError(
                    f"Non-retryable error in tool {tool_name}: {str(e)}",
                    tool_name=tool_name,
                    original_error=e
                )
        
        # All retries exhausted
        if isinstance(last_exception, ToolTimeoutError):
            self.execution_metrics['timeout_executions'] += 1
            raise last_exception
        else:
            raise ToolExecutionError(
                f"Tool {tool_name} failed after {retry_config.max_attempts} attempts",
                tool_name=tool_name,
                original_error=last_exception,
                retry_count=retry_config.max_attempts
            )
    
    async def _execute_with_timeout(
        self, 
        tool: BaseTool, 
        query: str, 
        context: ToolContext
    ) -> ToolResult:
        """
        Execute a tool with timeout handling.
        
        Args:
            tool: The tool to execute
            query: The query for the tool
            context: The execution context
            
        Returns:
            ToolResult: The result of the tool execution
            
        Raises:
            ToolTimeoutError: If execution times out
        """
        tool_info = tool.get_tool_info()
        tool_name = tool_info.name
        timeout = context.timeout or self.config.default_timeout
        
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                tool.execute(query, context),
                timeout=timeout
            )
            
            # Update execution time in result
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.tool_name = tool_name
            
            return result
            
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            raise ToolTimeoutError(
                f"Tool {tool_name} execution timed out after {elapsed_time:.2f}s",
                tool_name=tool_name,
                timeout_duration=timeout,
                elapsed_time=elapsed_time
            )
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """
        Calculate the delay for the next retry attempt.
        
        Args:
            attempt: The current attempt number (0-based)
            config: Retry configuration
            
        Returns:
            float: Delay in seconds
        """
        # Exponential backoff
        delay = config.base_delay * (config.exponential_base ** attempt)
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter = random.uniform(0, delay * 0.1)  # Up to 10% jitter
            delay += jitter
        
        return delay
    
    def _get_circuit_breaker(self, tool_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a tool."""
        if tool_name not in self.circuit_breakers:
            self.circuit_breakers[tool_name] = CircuitBreaker(
                self.config.circuit_breaker_config,
                tool_name
            )
        return self.circuit_breakers[tool_name]
    
    def _get_rate_limiter(self, tool_name: str) -> RateLimiter:
        """Get or create a rate limiter for a tool."""
        if tool_name not in self.rate_limiters:
            self.rate_limiters[tool_name] = RateLimiter(
                self.config.rate_limit_config,
                tool_name
            )
        return self.rate_limiters[tool_name]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics.
        
        Returns:
            Dict containing execution statistics
        """
        metrics = self.execution_metrics.copy()
        
        # Add circuit breaker states
        circuit_breaker_states = {}
        for tool_name, cb in self.circuit_breakers.items():
            circuit_breaker_states[tool_name] = {
                'state': cb.state.value,
                'failure_count': cb.failure_count,
                'success_count': cb.success_count
            }
        
        metrics['circuit_breakers'] = circuit_breaker_states
        
        # Add rate limiter states
        rate_limiter_states = {}
        for tool_name, rl in self.rate_limiters.items():
            rate_limiter_states[tool_name] = {
                'available_tokens': rl.tokens,
                'last_refill': rl.last_refill
            }
        
        metrics['rate_limiters'] = rate_limiter_states
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the executor.
        
        Returns:
            Dict containing health status information
        """
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'connection_pool': {
                'active': not (self.connection_pool.session is None or 
                              self.connection_pool.session.closed)
            },
            'semaphore': {
                'available': self.execution_semaphore._value,
                'total': self.config.max_concurrent_executions
            },
            'metrics': self.get_metrics()
        }
        
        # Check circuit breaker states
        open_breakers = [
            name for name, cb in self.circuit_breakers.items()
            if cb.state == CircuitBreakerState.OPEN
        ]
        
        if open_breakers:
            health['status'] = 'degraded'
            health['open_circuit_breakers'] = open_breakers
        
        return health
    
    async def close(self):
        """Clean up resources."""
        await self.connection_pool.close()
        self.logger.info("Tool executor closed")