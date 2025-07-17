"""
Performance tests for the agentic conversation system.

These tests verify system performance under various load conditions
and measure response times, memory usage, and throughput.
"""

import asyncio
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, Mock
import pytest

from src.agentic_conversation.orchestrator import ConversationOrchestrator
from src.agentic_conversation.config import ConfigurationLoader
from src.agentic_conversation.models import ConversationStatus
from tests.fixtures.sample_conversations import ConversationFixtures


class TestPerformanceScenarios:
    """Performance tests for various system scenarios."""
    
    @pytest.fixture
    def performance_config(self):
        """Create a performance-optimized configuration."""
        return {
            "agents": {
                "agent_a": {"name": "Fast Agent A", "system_prompt": "You are a fast-responding agent."},
                "agent_b": {"name": "Fast Agent B", "system_prompt": "You are another fast-responding agent."}
            },
            "model": {"model_name": "gpt-3.5-turbo", "temperature": 0.3, "max_tokens": 500},
            "conversation": {"max_turns": 4, "context_window_strategy": "truncate", "turn_timeout": 10.0},
            "logging": {"log_level": "WARNING", "output_directory": "./perf_logs", "real_time_display": False}
        }
    
    @pytest.fixture
    def mock_fast_responses(self):
        """Create fast mock responses for performance testing."""
        responses = [
            "Quick response 1",
            "Fast reply 2", 
            "Rapid answer 3",
            "Swift response 4",
            "Immediate reply 5",
            "Instant answer 6"
        ]
        return responses
    
    @pytest.mark.asyncio
    async def test_single_conversation_performance(self, performance_config, mock_fast_responses):
        """Test performance of a single conversation."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(performance_config)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config)
            
            # Mock fast LLM responses
            response_iter = iter(mock_fast_responses)
            def mock_fast_llm(*args, **kwargs):
                try:
                    return next(response_iter)
                except StopIteration:
                    return "Final response"
            
            with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_fast_llm):
                with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_fast_llm):
                    
                    # Measure performance
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    result = await orchestrator.run_conversation("perf-test-single")
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    duration = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    # Performance assertions
                    assert duration < 5.0  # Should complete within 5 seconds
                    assert memory_used < 50  # Should use less than 50MB additional memory
                    assert result.status in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
                    assert result.current_turn > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_conversations_performance(self, performance_config, mock_fast_responses):
        """Test performance with multiple concurrent conversations."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(performance_config)
        
        async def run_single_conversation(conv_id: int):
            """Run a single conversation and return performance metrics."""
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                orchestrator = ConversationOrchestrator(config)
                
                # Mock responses specific to this conversation
                responses = [f"Response {i} for conversation {conv_id}" for i in range(6)]
                response_iter = iter(responses)
                
                def mock_llm(*args, **kwargs):
                    try:
                        return next(response_iter)
                    except StopIteration:
                        return f"Final response for conversation {conv_id}"
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_llm):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_llm):
                        
                        start_time = time.time()
                        result = await orchestrator.run_conversation(f"perf-concurrent-{conv_id}")
                        end_time = time.time()
                        
                        return {
                            'conversation_id': conv_id,
                            'duration': end_time - start_time,
                            'status': result.status,
                            'turns': result.current_turn,
                            'message_count': len(result.messages)
                        }
        
        # Run multiple conversations concurrently
        num_conversations = 8
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        tasks = [run_single_conversation(i) for i in range(num_conversations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        total_duration = end_time - start_time
        total_memory_used = end_memory - start_memory
        
        # Filter successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Performance assertions
        assert len(successful_results) == num_conversations  # All should succeed
        assert total_duration < 15.0  # Should complete within 15 seconds
        assert total_memory_used < 200  # Should use less than 200MB additional memory
        
        # Check individual conversation performance
        durations = [r['duration'] for r in successful_results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        
        assert avg_duration < 3.0  # Average conversation should be fast
        assert max_duration < 8.0  # No conversation should take too long
        
        # Verify all conversations completed successfully
        for result in successful_results:
            assert result['status'] in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
            assert result['turns'] > 0
    
    @pytest.mark.asyncio
    async def test_context_window_performance(self, performance_config):
        """Test performance with large context windows."""
        # Create config with larger context window
        config_dict = performance_config.copy()
        config_dict["conversation"]["max_turns"] = 20
        config_dict["conversation"]["context_window_strategy"] = "sliding"
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(config_dict)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config)
            
            # Create long responses to stress context management
            long_responses = [
                "This is a very detailed response that contains extensive information about the topic. " * 20,
                "Another comprehensive response with thorough analysis and detailed explanations. " * 20,
                "A third lengthy response providing in-depth coverage of all relevant aspects. " * 20,
                "Yet another detailed response with comprehensive information and analysis. " * 20
            ]
            
            response_cycle = iter(long_responses * 10)  # Repeat responses
            
            def mock_long_response(*args, **kwargs):
                try:
                    return next(response_cycle)
                except StopIteration:
                    return "Final comprehensive response."
            
            with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_long_response):
                with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_long_response):
                    
                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    result = await orchestrator.run_conversation("perf-context-test")
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    duration = end_time - start_time
                    memory_used = end_memory - start_memory
                    
                    # Performance assertions for context-heavy conversation
                    assert duration < 20.0  # Should handle context management efficiently
                    assert memory_used < 100  # Context management shouldn't use excessive memory
                    assert result.status in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
    
    def test_memory_usage_patterns(self, performance_config):
        """Test memory usage patterns during system operation."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(performance_config)
        
        # Measure baseline memory
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            # Create multiple orchestrators to test memory scaling
            orchestrators = []
            memory_measurements = []
            
            for i in range(5):
                orchestrator = ConversationOrchestrator(config)
                orchestrators.append(orchestrator)
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_measurements.append(current_memory - baseline_memory)
            
            # Memory should scale reasonably with number of orchestrators
            memory_per_orchestrator = memory_measurements[-1] / len(orchestrators)
            assert memory_per_orchestrator < 20  # Each orchestrator should use less than 20MB
            
            # Clean up
            del orchestrators
    
    @pytest.mark.asyncio
    async def test_throughput_measurement(self, performance_config):
        """Test system throughput (conversations per minute)."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(performance_config)
        
        # Reduce turns for faster throughput testing
        config_dict = performance_config.copy()
        config_dict["conversation"]["max_turns"] = 2
        config = loader.load_from_dict(config_dict)
        
        async def run_quick_conversation(conv_id: int):
            """Run a quick conversation for throughput testing."""
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                orchestrator = ConversationOrchestrator(config)
                
                def mock_quick_response(*args, **kwargs):
                    return f"Quick response {conv_id}"
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_quick_response):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_quick_response):
                        
                        result = await orchestrator.run_conversation(f"throughput-{conv_id}")
                        return result.status in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
        
        # Measure throughput over a time period
        start_time = time.time()
        num_conversations = 20
        
        # Run conversations in batches to avoid overwhelming the system
        batch_size = 5
        successful_conversations = 0
        
        for i in range(0, num_conversations, batch_size):
            batch_tasks = [
                run_quick_conversation(j) 
                for j in range(i, min(i + batch_size, num_conversations))
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Count successful conversations
            successful_conversations += sum(
                1 for result in batch_results 
                if not isinstance(result, Exception) and result
            )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate throughput
        conversations_per_minute = (successful_conversations / total_duration) * 60
        
        # Throughput assertions
        assert successful_conversations >= num_conversations * 0.9  # At least 90% success rate
        assert conversations_per_minute > 10  # Should handle at least 10 conversations per minute
        assert total_duration < 60  # Should complete within 1 minute
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, performance_config):
        """Test that resources are properly cleaned up after conversations."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(performance_config)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            # Run several conversations and ensure cleanup
            for i in range(10):
                orchestrator = ConversationOrchestrator(config)
                
                def mock_response(*args, **kwargs):
                    return f"Response for cleanup test {i}"
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_response):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_response):
                        
                        await orchestrator.run_conversation(f"cleanup-test-{i}")
                
                # Explicitly clean up
                del orchestrator
        
        # Allow some time for garbage collection
        import gc
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal after cleanup
        assert memory_growth < 30  # Should not grow by more than 30MB
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, performance_config):
        """Test performance impact of error handling."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(performance_config)
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            orchestrator = ConversationOrchestrator(config)
            
            # Mix of successful and failing responses
            def mock_mixed_responses(*args, **kwargs):
                import random
                if random.random() < 0.3:  # 30% failure rate
                    raise ConnectionError("Simulated network error")
                return "Successful response"
            
            with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_mixed_responses):
                with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_mixed_responses):
                    
                    start_time = time.time()
                    
                    # Run conversation with errors
                    result = await orchestrator.run_conversation("error-perf-test")
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Error handling shouldn't significantly impact performance
                    assert duration < 15.0  # Should complete within reasonable time even with errors
                    assert result is not None  # Should return a result even with errors


class TestScalabilityScenarios:
    """Test system scalability under various conditions."""
    
    @pytest.mark.asyncio
    async def test_scaling_with_conversation_length(self, performance_config):
        """Test how system scales with conversation length."""
        loader = ConfigurationLoader()
        
        # Test different conversation lengths
        turn_counts = [2, 5, 10, 20]
        performance_results = []
        
        for max_turns in turn_counts:
            config_dict = performance_config.copy()
            config_dict["conversation"]["max_turns"] = max_turns
            config = loader.load_from_dict(config_dict)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                orchestrator = ConversationOrchestrator(config)
                
                def mock_response(*args, **kwargs):
                    return f"Response for {max_turns}-turn conversation"
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_response):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_response):
                        
                        start_time = time.time()
                        result = await orchestrator.run_conversation(f"scaling-{max_turns}")
                        end_time = time.time()
                        
                        performance_results.append({
                            'turns': max_turns,
                            'duration': end_time - start_time,
                            'actual_turns': result.current_turn,
                            'status': result.status
                        })
        
        # Analyze scaling characteristics
        for i, result in enumerate(performance_results):
            # Performance should scale reasonably with conversation length
            expected_max_duration = result['turns'] * 0.5  # 0.5 seconds per turn
            assert result['duration'] < expected_max_duration
            
            # Verify conversations reached expected length
            assert result['actual_turns'] <= result['turns']
            assert result['status'] in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
    
    @pytest.mark.asyncio
    async def test_concurrent_user_simulation(self, performance_config):
        """Simulate multiple concurrent users with different conversation patterns."""
        loader = ConfigurationLoader()
        config = loader.load_from_dict(performance_config)
        
        # Define different user patterns
        user_patterns = [
            {"turns": 2, "delay": 0.1, "response_length": "short"},
            {"turns": 4, "delay": 0.2, "response_length": "medium"},
            {"turns": 6, "delay": 0.3, "response_length": "long"},
            {"turns": 3, "delay": 0.15, "response_length": "short"},
            {"turns": 5, "delay": 0.25, "response_length": "medium"}
        ]
        
        async def simulate_user(user_id: int, pattern: dict):
            """Simulate a single user's conversation pattern."""
            config_dict = performance_config.copy()
            config_dict["conversation"]["max_turns"] = pattern["turns"]
            user_config = loader.load_from_dict(config_dict)
            
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                orchestrator = ConversationOrchestrator(user_config)
                
                # Create responses based on pattern
                response_templates = {
                    "short": "Brief response",
                    "medium": "This is a medium-length response with some detail",
                    "long": "This is a comprehensive response that provides detailed information and thorough analysis of the topic at hand"
                }
                
                def mock_user_response(*args, **kwargs):
                    # Simulate user thinking time
                    time.sleep(pattern["delay"])
                    return response_templates[pattern["response_length"]]
                
                with patch.object(orchestrator.agent_a, '_call_llm', side_effect=mock_user_response):
                    with patch.object(orchestrator.agent_b, '_call_llm', side_effect=mock_user_response):
                        
                        start_time = time.time()
                        result = await orchestrator.run_conversation(f"user-{user_id}")
                        end_time = time.time()
                        
                        return {
                            'user_id': user_id,
                            'pattern': pattern,
                            'duration': end_time - start_time,
                            'status': result.status,
                            'turns': result.current_turn
                        }
        
        # Run concurrent user simulations
        start_time = time.time()
        tasks = [simulate_user(i, pattern) for i, pattern in enumerate(user_patterns)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        total_duration = end_time - start_time
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        # Scalability assertions
        assert len(successful_results) == len(user_patterns)  # All users should complete
        assert total_duration < 30.0  # Should handle concurrent users efficiently
        
        # Verify each user's conversation completed successfully
        for result in successful_results:
            assert result['status'] in [ConversationStatus.COMPLETED, ConversationStatus.MAX_TURNS_REACHED]
            assert result['turns'] > 0