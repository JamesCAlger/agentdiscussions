"""
Unit tests for the tool execution engine.

This module tests the ToolExecutor class and its components including
circuit breakers, rate limiters, retry logic, and error handling.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from src.agentic_conversation.tools.executor import (
    ToolExecutor,
    ExecutorConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    RateLimiter,
    RateLimitConfig,
    RetryConfig,
    ConnectionPool
)
from src.agentic_conversation.tools.base import (
    BaseTool,
    ToolResult,
    ToolContext,
    ToolInfo,
    ToolCapability
)
from src.agentic_conversation.tools.exceptions import (
    ToolExecutionError,
    ToolTimeoutError,
    ToolRateLimitError
)


class MockTool(BaseTool):
    """Mock tool for testing."""
    
    def __init__(self, config=None, should_fail=False, execution_time=0.1):
        super().__init__(config)
        self.should_fail = should_fail
        self.execution_time = execution_time
        self.call_count = 0
    
    async def execute(self, query: str, context: ToolContext) -> ToolResult:
        """Mock execute method."""
        self.call_count += 1
        
        if self.execution_time > 0:
            await asyncio.sleep(self.execution_time)
        
        if self.should_fail:
            raise ToolExecutionError("Mock tool failure", tool_name="mock_tool")
        
        return ToolResult(
            success=True,
            content=f"Mock result for: {query}",
            tool_name="mock_tool"
        )
    
    def is_relevant(self, context: ToolContext) -> bool:
        """Mock relevance check."""
        return True
    
    def get_tool_info(self) -> ToolInfo:
        """Mock tool info."""
        return ToolInfo(
            name="mock_tool",
            description="Mock tool for testing",
            capabilities=[ToolCapability.SEARCH]
        )


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    return MockTool()


@pytest.fixture
def failing_tool():
    """Create a failing mock tool for testing."""
    return MockTool(should_fail=True)


@pytest.fixture
def slow_tool():
    """Create a slow mock tool for testing."""
    return MockTool(execution_time=2.0)


@pytest.fixture
def tool_context():
    """Create a tool context for testing."""
    return ToolContext(
        agent_id="test_agent",
        current_turn=1,
        available_tokens=1000,
        timeout=5.0
    )


@pytest.fixture
def executor_config():
    """Create executor configuration for testing."""
    return ExecutorConfig(
        default_timeout=1.0,
        max_concurrent_executions=5,
        retry_config=RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0
        ),
        rate_limit_config=RateLimitConfig(
            requests_per_minute=60,
            burst_size=10
        ),
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            success_threshold=2
        )
    )


@pytest.fixture
def executor(executor_config):
    """Create a tool executor for testing."""
    return ToolExecutor(executor_config)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config, "test_tool")
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        config = CircuitBreakerConfig()
        cb = CircuitBreaker(config, "test_tool")
        
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker(config, "test_tool")
        
        async def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejection(self):
        """Test circuit breaker rejecting calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config, "test_tool")
        
        async def failing_func():
            raise Exception("Test failure")
        
        # Trigger circuit breaker to open
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitBreakerState.OPEN
        
        # Should reject subsequent calls
        with pytest.raises(ToolExecutionError) as exc_info:
            await cb.call(failing_func)
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=1
        )
        cb = CircuitBreaker(config, "test_tool")
        
        async def failing_func():
            raise Exception("Test failure")
        
        async def success_func():
            return "success"
        
        # Open the circuit
        with pytest.raises(Exception):
            await cb.call(failing_func)
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Should move to half-open and succeed
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitBreakerState.CLOSED


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        rl = RateLimiter(config, "test_tool")
        
        assert rl.tokens == 10  # Should start with burst size
        assert rl.config.requests_per_minute == 60
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_success(self):
        """Test successful token acquisition."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        rl = RateLimiter(config, "test_tool")
        
        # Should succeed with available tokens
        success = await rl.acquire(1)
        assert success is True
        assert rl.tokens == 9
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_failure(self):
        """Test rate limit exceeded."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=2)
        rl = RateLimiter(config, "test_tool")
        
        # Use up all tokens
        await rl.acquire(1)
        await rl.acquire(1)
        
        # Should fail on third attempt
        with pytest.raises(ToolRateLimitError) as exc_info:
            await rl.acquire(1)
        assert "Rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_token_refill(self):
        """Test token refill over time."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=2)
        rl = RateLimiter(config, "test_tool")
        
        # Use all tokens
        await rl.acquire(2)
        assert rl.tokens == 0
        
        # Manually advance time and refill
        rl.last_refill = time.time() - 60  # 1 minute ago
        rl._refill_tokens()
        
        # Should have refilled to burst size
        assert rl.tokens == 2
    
    @pytest.mark.asyncio
    async def test_rate_limiter_context_manager(self):
        """Test rate limiter context manager."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        rl = RateLimiter(config, "test_tool")
        
        async with rl.limit(1):
            assert rl.tokens == 9


class TestConnectionPool:
    """Test connection pool functionality."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_creation(self):
        """Test connection pool creation."""
        config = ExecutorConfig()
        pool = ConnectionPool(config)
        
        session = await pool.get_session()
        assert session is not None
        assert not session.closed
        
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self):
        """Test connection pool session reuse."""
        config = ExecutorConfig()
        pool = ConnectionPool(config)
        
        session1 = await pool.get_session()
        session2 = await pool.get_session()
        
        # Should reuse the same session
        assert session1 is session2
        
        await pool.close()
    
    @pytest.mark.asyncio
    async def test_connection_pool_recreation(self):
        """Test connection pool recreation after close."""
        config = ExecutorConfig()
        pool = ConnectionPool(config)
        
        session1 = await pool.get_session()
        await pool.close()
        
        session2 = await pool.get_session()
        
        # Should create a new session
        assert session1 is not session2
        assert session1.closed
        assert not session2.closed
        
        await pool.close()


class TestToolExecutor:
    """Test tool executor functionality."""
    
    @pytest.mark.asyncio
    async def test_executor_successful_execution(self, executor, mock_tool, tool_context):
        """Test successful tool execution."""
        result = await executor.execute_tool(mock_tool, "test query", tool_context)
        
        assert result.success is True
        assert "Mock result for: test query" in result.content
        assert result.tool_name == "mock_tool"
        assert result.execution_time > 0
        assert mock_tool.call_count == 1
    
    @pytest.mark.asyncio
    async def test_executor_failed_execution(self, executor, failing_tool, tool_context):
        """Test failed tool execution."""
        result = await executor.execute_tool(failing_tool, "test query", tool_context)
        
        assert result.success is False
        assert len(result.errors) > 0
        assert result.tool_name == "mock_tool"
        assert failing_tool.call_count == 3  # Should retry 3 times
    
    @pytest.mark.asyncio
    async def test_executor_timeout_handling(self, executor, slow_tool, tool_context):
        """Test timeout handling."""
        # Set a short timeout
        tool_context.timeout = 0.5
        
        result = await executor.execute_tool(slow_tool, "test query", tool_context)
        
        assert result.success is False
        assert any("timed out" in error.lower() for error in result.errors)
        assert executor.execution_metrics['timeout_executions'] > 0
    
    @pytest.mark.asyncio
    async def test_executor_retry_logic(self, executor, tool_context):
        """Test retry logic with eventual success."""
        # Create a tool that fails twice then succeeds
        call_count = 0
        
        class RetryTool(MockTool):
            async def execute(self, query: str, context: ToolContext) -> ToolResult:
                nonlocal call_count
                call_count += 1
                
                if call_count < 3:
                    raise ToolExecutionError("Temporary failure", tool_name="retry_tool")
                
                return ToolResult(
                    success=True,
                    content="Success after retries",
                    tool_name="retry_tool"
                )
        
        retry_tool = RetryTool()
        result = await executor.execute_tool(retry_tool, "test query", tool_context)
        
        assert result.success is True
        assert result.content == "Success after retries"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_executor_circuit_breaker_integration(self, executor, tool_context):
        """Test circuit breaker integration."""
        failing_tool = MockTool(should_fail=True)
        
        # Execute multiple times to trigger circuit breaker
        for _ in range(5):
            result = await executor.execute_tool(failing_tool, "test query", tool_context)
            assert result.success is False
        
        # Circuit breaker should now be open
        metrics = executor.get_metrics()
        assert executor.execution_metrics['circuit_breaker_rejections'] > 0
    
    @pytest.mark.asyncio
    async def test_executor_rate_limiting(self, tool_context):
        """Test rate limiting functionality."""
        # Create executor with very low rate limit
        config = ExecutorConfig(
            rate_limit_config=RateLimitConfig(
                requests_per_minute=1,
                burst_size=1
            )
        )
        executor = ToolExecutor(config)
        mock_tool = MockTool()
        
        # First request should succeed
        result1 = await executor.execute_tool(mock_tool, "query1", tool_context)
        assert result1.success is True
        
        # Second request should be rate limited
        result2 = await executor.execute_tool(mock_tool, "query2", tool_context)
        assert result2.success is False
        assert any("rate limit" in error.lower() for error in result2.errors)
        
        assert executor.execution_metrics['rate_limited_executions'] > 0
    
    @pytest.mark.asyncio
    async def test_executor_concurrent_execution_limit(self, executor, tool_context):
        """Test concurrent execution limiting."""
        # Create slow tools to test concurrency
        slow_tools = [MockTool(execution_time=0.5) for _ in range(10)]
        
        # Execute all tools concurrently
        tasks = [
            executor.execute_tool(tool, f"query{i}", tool_context)
            for i, tool in enumerate(slow_tools)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Should take longer than 0.5s due to concurrency limiting
        # but less than 10 * 0.5s (sequential execution)
        assert execution_time > 0.5
        assert execution_time < 5.0
        
        # All should succeed
        assert all(result.success for result in results)
    
    @pytest.mark.asyncio
    async def test_executor_input_validation(self, executor, mock_tool, tool_context):
        """Test input validation."""
        # Test with empty query
        result = await executor.execute_tool(mock_tool, "", tool_context)
        assert result.success is False
        assert any("empty" in error.lower() for error in result.errors)
        
        # Test with None context
        result = await executor.execute_tool(mock_tool, "query", None)
        assert result.success is False
        assert any("none" in error.lower() for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_executor_metrics_collection(self, executor, mock_tool, failing_tool, tool_context):
        """Test metrics collection."""
        # Execute successful tool
        await executor.execute_tool(mock_tool, "success query", tool_context)
        
        # Execute failing tool
        await executor.execute_tool(failing_tool, "fail query", tool_context)
        
        metrics = executor.get_metrics()
        
        assert metrics['total_executions'] >= 2
        assert metrics['successful_executions'] >= 1
        assert metrics['failed_executions'] >= 1
        assert 'circuit_breakers' in metrics
        assert 'rate_limiters' in metrics
    
    @pytest.mark.asyncio
    async def test_executor_health_check(self, executor):
        """Test health check functionality."""
        health = await executor.health_check()
        
        assert health['status'] in ['healthy', 'degraded']
        assert 'timestamp' in health
        assert 'connection_pool' in health
        assert 'semaphore' in health
        assert 'metrics' in health
    
    @pytest.mark.asyncio
    async def test_executor_cleanup(self, executor):
        """Test executor cleanup."""
        # Use the executor to create resources
        mock_tool = MockTool()
        tool_context = ToolContext()
        await executor.execute_tool(mock_tool, "test", tool_context)
        
        # Clean up
        await executor.close()
        
        # Connection pool should be closed
        if executor.connection_pool.session:
            assert executor.connection_pool.session.closed


class TestRetryLogic:
    """Test retry logic calculations."""
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        executor = ToolExecutor()
        
        # Test exponential backoff
        delay0 = executor._calculate_retry_delay(0, config)
        delay1 = executor._calculate_retry_delay(1, config)
        delay2 = executor._calculate_retry_delay(2, config)
        
        assert delay0 == 1.0  # base_delay * 2^0
        assert delay1 == 2.0  # base_delay * 2^1
        assert delay2 == 4.0  # base_delay * 2^2
    
    def test_retry_delay_max_limit(self):
        """Test retry delay maximum limit."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,
            jitter=False
        )
        executor = ToolExecutor()
        
        # Large attempt number should be capped at max_delay
        delay = executor._calculate_retry_delay(10, config)
        assert delay == 5.0
    
    def test_retry_delay_with_jitter(self):
        """Test retry delay with jitter."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=True
        )
        executor = ToolExecutor()
        
        # With jitter, delays should vary slightly
        delays = [executor._calculate_retry_delay(1, config) for _ in range(10)]
        
        # All delays should be around 2.0 but with some variation
        assert all(1.8 <= delay <= 2.2 for delay in delays)
        assert len(set(delays)) > 1  # Should have some variation


if __name__ == "__main__":
    pytest.main([__file__])