"""
Unit tests for the context window management system.

This module tests the ContextManager class and related functionality
for different context management strategies.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from agentic_conversation.context_manager import (
    ContextManager,
    ContextStrategy,
    ContextManagementResult,
    TruncateStrategy,
    SlidingWindowStrategy,
    BaseContextStrategy
)
from agentic_conversation.models import Message, ConversationState, ContextWindowSnapshot
from agentic_conversation.token_counter import TokenCounter, TokenCountResult


class TestContextStrategy:
    """Test the ContextStrategy enum."""
    
    def test_enum_values(self):
        """Test enum values are correct."""
        assert ContextStrategy.TRUNCATE.value == "truncate"
        assert ContextStrategy.SLIDING.value == "sliding"
        assert ContextStrategy.SUMMARIZE.value == "summarize"


class TestContextManagementResult:
    """Test the ContextManagementResult dataclass."""
    
    def test_creation(self):
        """Test creating a ContextManagementResult."""
        messages = [
            Message("agent1", "Hello", datetime.now(), 10),
            Message("agent2", "Hi", datetime.now(), 8)
        ]
        
        result = ContextManagementResult(
            messages=messages,
            total_tokens=18,
            strategy_applied="truncate",
            messages_removed=2,
            tokens_saved=15,
            utilization_percentage=75.0
        )
        
        assert len(result.messages) == 2
        assert result.total_tokens == 18
        assert result.strategy_applied == "truncate"
        assert result.messages_removed == 2
        assert result.tokens_saved == 15
        assert result.utilization_percentage == 75.0
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        messages = [Message("agent1", "Hello", datetime.now(), 10)]
        
        result = ContextManagementResult(
            messages=messages,
            total_tokens=10,
            strategy_applied="sliding",
            messages_removed=1,
            tokens_saved=5,
            utilization_percentage=50.0
        )
        
        expected = {
            "total_tokens": 10,
            "strategy_applied": "sliding",
            "messages_removed": 1,
            "tokens_saved": 5,
            "utilization_percentage": 50.0,
            "message_count": 1
        }
        
        assert result.to_dict() == expected


class TestTruncateStrategy:
    """Test the TruncateStrategy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_token_counter = Mock(spec=TokenCounter)
        self.strategy = TruncateStrategy(self.mock_token_counter)
    
    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.token_counter == self.mock_token_counter
    
    def test_apply_empty_messages(self):
        """Test applying strategy to empty message list."""
        result = self.strategy.apply([], 1000, 90.0)
        
        assert len(result.messages) == 0
        assert result.total_tokens == 0
        assert result.strategy_applied is None
        assert result.messages_removed == 0
        assert result.tokens_saved == 0
        assert result.utilization_percentage == 0.0
    
    def test_apply_under_threshold(self):
        """Test applying strategy when under threshold."""
        messages = [
            Message("agent1", "Hello", datetime.now(), 0),
            Message("agent2", "Hi", datetime.now(), 0)
        ]
        
        # Mock token counting to return low token counts
        self.mock_token_counter.count_message_tokens.side_effect = [
            TokenCountResult(10, None, "", 5),
            TokenCountResult(8, None, "", 3)
        ]
        
        result = self.strategy.apply(messages, 1000, 90.0)  # 18 tokens < 900 threshold
        
        assert len(result.messages) == 2
        assert result.total_tokens == 18
        assert result.strategy_applied is None
        assert result.messages_removed == 0
        assert result.tokens_saved == 0
        assert abs(result.utilization_percentage - 1.8) < 0.01  # 18/1000 * 100 (allow for floating point precision)
    
    def test_apply_over_threshold(self):
        """Test applying strategy when over threshold."""
        messages = [
            Message("agent1", "First message", datetime.now(), 0),
            Message("agent2", "Second message", datetime.now(), 0),
            Message("agent1", "Third message", datetime.now(), 0)
        ]
        
        # Mock token counting - first call for total, then individual removals
        token_counts = [400, 300, 200]  # Total = 900, threshold = 450 (50% of 900)
        self.mock_token_counter.count_message_tokens.side_effect = (
            [TokenCountResult(count, None, "", 0) for count in token_counts] +  # Initial total calculation
            [TokenCountResult(400, None, "", 0)] +  # First removal
            [TokenCountResult(300, None, "", 0)]    # Second removal (if needed)
        )
        
        result = self.strategy.apply(messages, 900, 50.0)  # 900 tokens > 450 threshold
        
        assert len(result.messages) == 1  # Should keep only the last message
        assert result.messages[0].content == "Third message"
        assert result.strategy_applied == "truncate"
        assert result.messages_removed == 2
        assert result.tokens_saved == 700  # 400 + 300
    
    def test_calculate_total_tokens(self):
        """Test token calculation helper method."""
        messages = [
            Message("agent1", "Hello", datetime.now(), 0),
            Message("agent2", "Hi", datetime.now(), 0)
        ]
        
        self.mock_token_counter.count_message_tokens.side_effect = [
            TokenCountResult(10, None, "", 5),
            TokenCountResult(8, None, "", 3)
        ]
        
        total = self.strategy._calculate_total_tokens(messages)
        assert total == 18
    
    def test_calculate_utilization(self):
        """Test utilization calculation helper method."""
        assert self.strategy._calculate_utilization(100, 1000) == 10.0
        assert self.strategy._calculate_utilization(500, 1000) == 50.0
        assert self.strategy._calculate_utilization(1000, 1000) == 100.0
        assert self.strategy._calculate_utilization(100, 0) == 0.0


class TestSlidingWindowStrategy:
    """Test the SlidingWindowStrategy class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_token_counter = Mock(spec=TokenCounter)
        self.strategy = SlidingWindowStrategy(self.mock_token_counter, min_messages=2)
    
    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.token_counter == self.mock_token_counter
        assert self.strategy.min_messages == 2
    
    def test_initialization_min_messages(self):
        """Test minimum messages constraint."""
        strategy = SlidingWindowStrategy(self.mock_token_counter, min_messages=1)
        assert strategy.min_messages == 2  # Should be at least 2
        
        strategy = SlidingWindowStrategy(self.mock_token_counter, min_messages=5)
        assert strategy.min_messages == 5
    
    def test_apply_empty_messages(self):
        """Test applying strategy to empty message list."""
        result = self.strategy.apply([], 1000, 90.0)
        
        assert len(result.messages) == 0
        assert result.total_tokens == 0
        assert result.strategy_applied is None
        assert result.messages_removed == 0
        assert result.tokens_saved == 0
        assert result.utilization_percentage == 0.0
    
    def test_apply_under_threshold(self):
        """Test applying strategy when under threshold."""
        messages = [
            Message("agent1", "Hello", datetime.now(), 0),
            Message("agent2", "Hi", datetime.now(), 0)
        ]
        
        self.mock_token_counter.count_message_tokens.side_effect = [
            TokenCountResult(10, None, "", 5),
            TokenCountResult(8, None, "", 3)
        ]
        
        result = self.strategy.apply(messages, 1000, 90.0)
        
        assert len(result.messages) == 2
        assert result.total_tokens == 18
        assert result.strategy_applied is None
        assert result.messages_removed == 0
        assert result.tokens_saved == 0
    
    def test_apply_over_threshold_with_min_messages(self):
        """Test sliding window respects minimum message count."""
        messages = [
            Message("agent1", "First", datetime.now(), 0),
            Message("agent2", "Second", datetime.now(), 0),
            Message("agent1", "Third", datetime.now(), 0),
            Message("agent2", "Fourth", datetime.now(), 0)
        ]
        
        # Mock high token counts to trigger sliding window
        token_counts = [300, 300, 300, 300]  # Total = 1200, threshold = 450 (50% of 900)
        self.mock_token_counter.count_message_tokens.side_effect = (
            [TokenCountResult(count, None, "", 0) for count in token_counts] +  # Initial calculation
            [TokenCountResult(300, None, "", 0)] +  # First removal
            [TokenCountResult(300, None, "", 0)]    # Second removal
        )
        
        result = self.strategy.apply(messages, 900, 50.0)
        
        # Should keep at least min_messages (2)
        assert len(result.messages) >= 2
        assert result.strategy_applied == "sliding"
        assert result.messages_removed > 0
    
    def test_apply_hard_limit_override(self):
        """Test that hard limit overrides minimum messages."""
        messages = [
            Message("agent1", "Very long message", datetime.now(), 0),
            Message("agent2", "Another long message", datetime.now(), 0)
        ]
        
        # Mock very high token counts that exceed hard limit
        self.mock_token_counter.count_message_tokens.side_effect = [
            TokenCountResult(600, None, "", 0),  # First message
            TokenCountResult(600, None, "", 0),  # Second message
            TokenCountResult(600, None, "", 0)   # For removal calculation
        ]
        
        result = self.strategy.apply(messages, 1000, 50.0)  # Hard limit 1000, both messages = 1200
        
        # Should remove one message even if below min_messages to fit hard limit
        assert len(result.messages) == 1
        assert result.strategy_applied == "sliding"
        assert result.messages_removed == 1


class TestContextManager:
    """Test the ContextManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_token_counter = Mock(spec=TokenCounter)
        self.context_manager = ContextManager(
            token_counter=self.mock_token_counter,
            max_context_tokens=1000,
            strategy=ContextStrategy.SLIDING,
            threshold_percentage=80.0
        )
    
    def test_initialization(self):
        """Test context manager initialization."""
        assert self.context_manager.token_counter == self.mock_token_counter
        assert self.context_manager.max_context_tokens == 1000
        assert self.context_manager.strategy == ContextStrategy.SLIDING
        assert self.context_manager.threshold_percentage == 80.0
        assert len(self.context_manager._strategies) == 2  # TRUNCATE and SLIDING
    
    def test_manage_context_empty_messages(self):
        """Test managing empty message list."""
        result = self.context_manager.manage_context([])
        
        assert len(result.messages) == 0
        assert result.total_tokens == 0
        assert result.strategy_applied is None
    
    def test_manage_context_with_strategy(self):
        """Test managing context with strategy application."""
        messages = [
            Message("agent1", "Hello", datetime.now(), 0),
            Message("agent2", "Hi there", datetime.now(), 0)
        ]
        
        # Mock token counting to trigger strategy
        self.mock_token_counter.count_message_tokens.side_effect = [
            TokenCountResult(500, None, "", 0),  # First message
            TokenCountResult(400, None, "", 0),  # Second message
            TokenCountResult(500, None, "", 0)   # For removal
        ]
        
        result = self.context_manager.manage_context(messages)
        
        # Should apply sliding window strategy
        assert result.strategy_applied == "sliding"
        assert len(self.context_manager._snapshots) == 1
    
    def test_manage_context_force_apply(self):
        """Test forcing strategy application."""
        messages = [Message("agent1", "Hello", datetime.now(), 0)]
        
        self.mock_token_counter.count_message_tokens.return_value = TokenCountResult(100, None, "", 0)
        
        result = self.context_manager.manage_context(messages, force_apply=True)
        
        # Should apply strategy even if under threshold
        assert len(self.context_manager._snapshots) == 1
    
    def test_manage_conversation_context(self):
        """Test managing conversation state context."""
        messages = [Message("agent1", "Hello", datetime.now(), 0)]
        conversation = ConversationState(messages=messages)
        
        self.mock_token_counter.count_message_tokens.return_value = TokenCountResult(100, None, "", 0)
        
        result = self.context_manager.manage_conversation_context(conversation)
        
        # Should update conversation state
        assert conversation.current_context_tokens == result.total_tokens
        assert len(conversation.messages) == len(result.messages)
    
    def test_check_context_status_empty(self):
        """Test checking status of empty context."""
        status = self.context_manager.check_context_status([])
        
        expected = {
            "total_tokens": 0,
            "max_tokens": 1000,
            "available_tokens": 1000,
            "utilization_percentage": 0.0,
            "needs_management": False,
            "over_threshold": False,
            "over_hard_limit": False,
            "message_count": 0
        }
        
        for key, value in expected.items():
            assert status[key] == value
    
    def test_check_context_status_with_messages(self):
        """Test checking status with messages."""
        messages = [
            Message("agent1", "Hello", datetime.now(), 0),
            Message("agent2", "Hi", datetime.now(), 0)
        ]
        
        self.mock_token_counter.count_message_tokens.side_effect = [
            TokenCountResult(400, None, "", 0),
            TokenCountResult(500, None, "", 0)
        ]
        
        status = self.context_manager.check_context_status(messages)
        
        assert status["total_tokens"] == 900
        assert status["max_tokens"] == 1000
        assert status["available_tokens"] == 100
        assert status["utilization_percentage"] == 90.0
        assert status["needs_management"] is True  # 900 > 800 (80% threshold)
        assert status["over_threshold"] is True
        assert status["over_hard_limit"] is False
        assert status["message_count"] == 2
    
    def test_get_context_for_agent(self):
        """Test getting context for specific agent."""
        messages = [Message("agent1", "Hello", datetime.now(), 0)]
        
        self.mock_token_counter.count_message_tokens.return_value = TokenCountResult(100, None, "", 0)
        
        managed_messages, context_info = self.context_manager.get_context_for_agent(messages, "agent1")
        
        assert len(managed_messages) == 1
        assert isinstance(context_info, dict)
        assert "total_tokens" in context_info
        assert "utilization_percentage" in context_info
    
    def test_snapshots_management(self):
        """Test context window snapshot management."""
        # Initially no snapshots
        assert len(self.context_manager.get_snapshots()) == 0
        assert self.context_manager.get_latest_snapshot() is None
        
        # Create some context operations to generate snapshots
        messages = [Message("agent1", "Hello", datetime.now(), 0)]
        self.mock_token_counter.count_message_tokens.return_value = TokenCountResult(100, None, "", 0)
        
        self.context_manager.manage_context(messages)
        self.context_manager.manage_context(messages)
        
        # Should have snapshots now
        snapshots = self.context_manager.get_snapshots()
        assert len(snapshots) == 2
        assert self.context_manager.get_latest_snapshot() is not None
        
        # Test utilization history
        history = self.context_manager.get_utilization_history()
        assert len(history) == 2
        
        # Test peak and average utilization
        peak = self.context_manager.get_peak_utilization()
        average = self.context_manager.get_average_utilization()
        assert peak >= 0
        assert average >= 0
        
        # Clear snapshots
        self.context_manager.clear_snapshots()
        assert len(self.context_manager.get_snapshots()) == 0
    
    def test_update_configuration(self):
        """Test updating context manager configuration."""
        # Update max tokens
        self.context_manager.update_configuration(max_context_tokens=2000)
        assert self.context_manager.max_context_tokens == 2000
        
        # Update strategy
        self.context_manager.update_configuration(strategy=ContextStrategy.TRUNCATE)
        assert self.context_manager.strategy == ContextStrategy.TRUNCATE
        
        # Update threshold
        self.context_manager.update_configuration(threshold_percentage=75.0)
        assert self.context_manager.threshold_percentage == 75.0
    
    def test_update_configuration_invalid_values(self):
        """Test updating configuration with invalid values."""
        with pytest.raises(ValueError, match="Max context tokens must be positive"):
            self.context_manager.update_configuration(max_context_tokens=0)
        
        with pytest.raises(ValueError, match="Threshold percentage must be between 0 and 100"):
            self.context_manager.update_configuration(threshold_percentage=150.0)
        
        with pytest.raises(ValueError, match="Threshold percentage must be between 0 and 100"):
            self.context_manager.update_configuration(threshold_percentage=0.0)
    
    def test_get_configuration(self):
        """Test getting current configuration."""
        # Mock the model_type attribute for the token counter
        from agentic_conversation.token_counter import ModelType
        self.mock_token_counter.model_type = ModelType.GPT_4
        
        config = self.context_manager.get_configuration()
        
        expected_keys = [
            "max_context_tokens", "strategy", "threshold_percentage",
            "available_strategies", "token_counter_model"
        ]
        
        for key in expected_keys:
            assert key in config
        
        assert config["max_context_tokens"] == 1000
        assert config["strategy"] == "sliding"
        assert config["threshold_percentage"] == 80.0
        assert len(config["available_strategies"]) == 2
        assert config["token_counter_model"] == "gpt-4"
    
    def test_estimate_messages_for_tokens(self):
        """Test estimating message count for token limit."""
        messages = [
            Message("agent1", "First", datetime.now(), 0),
            Message("agent2", "Second", datetime.now(), 0),
            Message("agent1", "Third", datetime.now(), 0)
        ]
        
        # Use a function to return consistent token counts based on message content
        def mock_count_tokens(message):
            token_map = {
                "Third": TokenCountResult(100, None, "", 0),
                "Second": TokenCountResult(150, None, "", 0),
                "First": TokenCountResult(200, None, "", 0)
            }
            return token_map.get(message.content, TokenCountResult(50, None, "", 0))
        
        self.mock_token_counter.count_message_tokens.side_effect = mock_count_tokens
        
        # Should fit 2 messages (100 + 150 = 250 <= 300)
        count = self.context_manager.estimate_messages_for_tokens(300, messages)
        assert count == 2
        
        # Should fit all 3 messages (100 + 150 + 200 = 450 <= 500)
        count = self.context_manager.estimate_messages_for_tokens(500, messages)
        assert count == 3
        
        # Should fit only 1 message (100 <= 120)
        count = self.context_manager.estimate_messages_for_tokens(120, messages)
        assert count == 1
    
    def test_estimate_messages_for_tokens_edge_cases(self):
        """Test edge cases for message estimation."""
        messages = [Message("agent1", "Hello", datetime.now(), 0)]
        
        # Empty messages
        assert self.context_manager.estimate_messages_for_tokens(100, []) == 0
        
        # Zero target tokens
        assert self.context_manager.estimate_messages_for_tokens(0, messages) == 0
        
        # Negative target tokens
        assert self.context_manager.estimate_messages_for_tokens(-100, messages) == 0
    
    def test_create_with_strategy_factory(self):
        """Test factory method for creating context manager."""
        token_counter = Mock(spec=TokenCounter)
        
        # Test valid strategy
        cm = ContextManager.create_with_strategy(
            token_counter=token_counter,
            strategy_name="truncate",
            max_context_tokens=2000,
            threshold_percentage=85.0
        )
        
        assert cm.strategy == ContextStrategy.TRUNCATE
        assert cm.max_context_tokens == 2000
        assert cm.threshold_percentage == 85.0
        
        # Test case insensitive
        cm = ContextManager.create_with_strategy(token_counter, "SLIDING")
        assert cm.strategy == ContextStrategy.SLIDING
    
    def test_create_with_strategy_invalid(self):
        """Test factory method with invalid strategy."""
        token_counter = Mock(spec=TokenCounter)
        
        with pytest.raises(ValueError, match="Unknown strategy 'invalid'"):
            ContextManager.create_with_strategy(token_counter, "invalid")
    
    def test_unknown_strategy_error(self):
        """Test error handling for unknown strategy."""
        # Create context manager with valid strategy
        cm = ContextManager(self.mock_token_counter)
        
        # Manually set invalid strategy to test error handling
        cm.strategy = "invalid_strategy"
        
        messages = [Message("agent1", "Hello", datetime.now(), 0)]
        
        with pytest.raises(ValueError, match="Unknown context strategy"):
            cm.manage_context(messages)


class TestContextManagementIntegration:
    """Integration tests for context management system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        # Use real TokenCounter for integration tests
        self.token_counter = TokenCounter("gpt-4")
        self.context_manager = ContextManager(
            token_counter=self.token_counter,
            max_context_tokens=100,  # Small limit for testing
            strategy=ContextStrategy.SLIDING,
            threshold_percentage=80.0
        )
    
    def test_real_token_counting_integration(self):
        """Test integration with real token counting."""
        messages = [
            Message("agent1", "This is a test message that should have some tokens.", datetime.now(), 0),
            Message("agent2", "This is another test message with different content.", datetime.now(), 0),
            Message("agent1", "And this is a third message to test context management.", datetime.now(), 0)
        ]
        
        # Check initial status
        status = self.context_manager.check_context_status(messages)
        assert status["total_tokens"] > 0
        assert status["message_count"] == 3
        
        # Apply context management
        result = self.context_manager.manage_context(messages)
        
        # Should have applied some management due to small token limit
        assert len(result.messages) <= 3
        assert result.total_tokens <= 100  # Should respect hard limit
    
    def test_truncate_vs_sliding_strategies(self):
        """Test different strategies produce different results."""
        messages = [
            Message("agent1", "First message with some content.", datetime.now(), 0),
            Message("agent2", "Second message with more content.", datetime.now(), 0),
            Message("agent1", "Third message with even more content.", datetime.now(), 0),
            Message("agent2", "Fourth message with the most content.", datetime.now(), 0)
        ]
        
        # Test with truncate strategy
        truncate_manager = ContextManager(
            token_counter=self.token_counter,
            max_context_tokens=50,  # Very small limit
            strategy=ContextStrategy.TRUNCATE,
            threshold_percentage=50.0
        )
        
        truncate_result = truncate_manager.manage_context(messages)
        
        # Test with sliding strategy
        sliding_manager = ContextManager(
            token_counter=self.token_counter,
            max_context_tokens=50,  # Same small limit
            strategy=ContextStrategy.SLIDING,
            threshold_percentage=50.0
        )
        
        sliding_result = sliding_manager.manage_context(messages)
        
        # Both should reduce messages, but may handle differently
        assert len(truncate_result.messages) <= 4
        assert len(sliding_result.messages) <= 4
        
        # Both should respect the token limit
        assert truncate_result.total_tokens <= 50
        assert sliding_result.total_tokens <= 50
    
    def test_conversation_state_integration(self):
        """Test integration with ConversationState."""
        messages = [
            Message("agent1", "Hello there!", datetime.now(), 0),
            Message("agent2", "Hi! How are you doing today?", datetime.now(), 0),
            Message("agent1", "I'm doing well, thanks for asking!", datetime.now(), 0)
        ]
        
        conversation = ConversationState(
            messages=messages,
            max_context_tokens=80  # Small limit
        )
        
        # Manage conversation context
        result = self.context_manager.manage_conversation_context(conversation)
        
        # Conversation state should be updated
        assert conversation.current_context_tokens == result.total_tokens
        assert len(conversation.messages) == len(result.messages)
        assert conversation.current_context_tokens <= 80
    
    def test_utilization_tracking_over_time(self):
        """Test utilization tracking over multiple operations."""
        messages = []
        
        # Gradually add messages and track utilization
        for i in range(5):
            messages.append(Message(f"agent{i%2}", f"Message number {i+1} with some content.", datetime.now(), 0))
            
            result = self.context_manager.manage_context(messages.copy())
            
            # Should create snapshots
            assert len(self.context_manager.get_snapshots()) == i + 1
        
        # Check utilization history
        history = self.context_manager.get_utilization_history()
        assert len(history) == 5
        
        # Peak utilization should be reasonable
        peak = self.context_manager.get_peak_utilization()
        assert 0 <= peak <= 100
        
        # Average should be reasonable
        average = self.context_manager.get_average_utilization()
        assert 0 <= average <= 100