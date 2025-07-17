"""
Unit tests for the token counting utilities.

This module tests the TokenCounter class and related functionality
for accurate token counting across different content types and models.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from agentic_conversation.token_counter import (
    TokenCounter,
    ModelType,
    TokenCountResult
)
from agentic_conversation.models import Message, ConversationState, ConversationStatus


class TestModelType:
    """Test the ModelType enum and its methods."""
    
    def test_from_string_exact_matches(self):
        """Test exact model name matches."""
        assert ModelType.from_string("gpt-3.5-turbo") == ModelType.GPT_3_5_TURBO
        assert ModelType.from_string("gpt-4") == ModelType.GPT_4
        assert ModelType.from_string("gpt-4-turbo") == ModelType.GPT_4_TURBO
        assert ModelType.from_string("gpt-4o") == ModelType.GPT_4O
        assert ModelType.from_string("gpt-4o-mini") == ModelType.GPT_4O_MINI
    
    def test_from_string_case_insensitive(self):
        """Test case insensitive model name matching."""
        assert ModelType.from_string("GPT-3.5-TURBO") == ModelType.GPT_3_5_TURBO
        assert ModelType.from_string("Gpt-4") == ModelType.GPT_4
        assert ModelType.from_string("GPT-4O") == ModelType.GPT_4O
    
    def test_from_string_variations(self):
        """Test model name variations."""
        assert ModelType.from_string("gpt-3.5") == ModelType.GPT_3_5_TURBO
        assert ModelType.from_string("gpt3.5") == ModelType.GPT_3_5_TURBO
        assert ModelType.from_string("gpt4") == ModelType.GPT_4
        assert ModelType.from_string("gpt-4-turbo-preview") == ModelType.GPT_4_TURBO
    
    def test_from_string_partial_matches(self):
        """Test partial model name matches."""
        assert ModelType.from_string("gpt-3.5-turbo-0613") == ModelType.GPT_3_5_TURBO
        assert ModelType.from_string("gpt-4-0314") == ModelType.GPT_4
        assert ModelType.from_string("gpt-4o-2024-05-13") == ModelType.GPT_4O
        assert ModelType.from_string("gpt-4o-mini-2024-07-18") == ModelType.GPT_4O_MINI
    
    def test_from_string_unknown_model(self):
        """Test handling of unknown model names."""
        with patch('agentic_conversation.token_counter.logger') as mock_logger:
            result = ModelType.from_string("unknown-model")
            assert result == ModelType.GPT_4
            mock_logger.warning.assert_called_once()


class TestTokenCountResult:
    """Test the TokenCountResult dataclass."""
    
    def test_creation(self):
        """Test creating a TokenCountResult."""
        result = TokenCountResult(
            token_count=100,
            model_type=ModelType.GPT_4,
            encoding_name="cl100k_base",
            content_length=400
        )
        
        assert result.token_count == 100
        assert result.model_type == ModelType.GPT_4
        assert result.encoding_name == "cl100k_base"
        assert result.content_length == 400
    
    def test_to_dict(self):
        """Test converting TokenCountResult to dictionary."""
        result = TokenCountResult(
            token_count=100,
            model_type=ModelType.GPT_4,
            encoding_name="cl100k_base",
            content_length=400
        )
        
        expected = {
            "token_count": 100,
            "model_type": "gpt-4",
            "encoding_name": "cl100k_base",
            "content_length": 400
        }
        
        assert result.to_dict() == expected


class TestTokenCounter:
    """Test the TokenCounter class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        counter = TokenCounter()
        assert counter.model_type == ModelType.GPT_4
        assert counter.encoding_name == "cl100k_base"
        assert counter._encoding is not None
    
    def test_initialization_with_model(self):
        """Test initialization with specific model."""
        counter = TokenCounter("gpt-3.5-turbo")
        assert counter.model_type == ModelType.GPT_3_5_TURBO
        assert counter.encoding_name == "cl100k_base"
    
    def test_initialization_gpt4o(self):
        """Test initialization with GPT-4o model."""
        counter = TokenCounter("gpt-4o")
        assert counter.model_type == ModelType.GPT_4O
        assert counter.encoding_name == "o200k_base"
    
    @patch('agentic_conversation.token_counter.tiktoken.get_encoding')
    def test_encoding_load_failure(self, mock_get_encoding):
        """Test handling of encoding load failure."""
        # First call fails, second call (fallback) succeeds
        mock_get_encoding.side_effect = [Exception("Load failed"), MagicMock()]
        
        with patch('agentic_conversation.token_counter.logger') as mock_logger:
            counter = TokenCounter("gpt-4")
            mock_logger.error.assert_called_once()
            mock_logger.warning.assert_called_once()
    
    def test_count_tokens_simple_text(self):
        """Test counting tokens for simple text."""
        counter = TokenCounter()
        result = counter.count_tokens("Hello, world!")
        
        assert isinstance(result, TokenCountResult)
        assert result.token_count > 0
        assert result.model_type == ModelType.GPT_4
        assert result.encoding_name == "cl100k_base"
        assert result.content_length == 13
    
    def test_count_tokens_empty_string(self):
        """Test counting tokens for empty string."""
        counter = TokenCounter()
        result = counter.count_tokens("")
        
        assert result.token_count == 0
        assert result.content_length == 0
    
    def test_count_tokens_invalid_input(self):
        """Test counting tokens with invalid input."""
        counter = TokenCounter()
        
        with pytest.raises(ValueError, match="Input must be a string"):
            counter.count_tokens(123)
        
        with pytest.raises(ValueError, match="Input must be a string"):
            counter.count_tokens(None)
    
    def test_count_tokens_long_text(self):
        """Test counting tokens for longer text."""
        counter = TokenCounter()
        long_text = "This is a longer piece of text that should have more tokens. " * 10
        result = counter.count_tokens(long_text)
        
        assert result.token_count > 50  # Should have many tokens
        assert result.content_length == len(long_text)
    
    def test_count_tokens_special_characters(self):
        """Test counting tokens with special characters."""
        counter = TokenCounter()
        special_text = "Hello! @#$%^&*()_+ 你好 🌟 emoji test"
        result = counter.count_tokens(special_text)
        
        assert result.token_count > 0
        assert result.content_length == len(special_text)
    
    def test_count_tokens_encoding_error(self):
        """Test handling of encoding errors."""
        counter = TokenCounter()
        
        # Mock the encoding to raise an exception
        with patch.object(counter, '_encoding') as mock_encoding:
            mock_encoding.encode.side_effect = Exception("Encoding failed")
            
            with patch('agentic_conversation.token_counter.logger') as mock_logger:
                result = counter.count_tokens("test text")
                
                # Should use fallback estimation
                assert result.token_count > 0
                mock_logger.error.assert_called_once()
                mock_logger.warning.assert_called_once()
    
    def test_count_message_tokens(self):
        """Test counting tokens for a Message object."""
        counter = TokenCounter()
        message = Message(
            agent_id="test_agent",
            content="Hello, this is a test message!",
            timestamp=datetime.now(),
            token_count=0  # Will be calculated
        )
        
        result = counter.count_message_tokens(message)
        
        assert isinstance(result, TokenCountResult)
        assert result.token_count > 0
        # Should be more than just content tokens due to overhead
        content_only = counter.count_tokens(message.content)
        assert result.token_count > content_only.token_count
    
    def test_count_message_tokens_invalid_input(self):
        """Test counting message tokens with invalid input."""
        counter = TokenCounter()
        
        with pytest.raises(ValueError, match="Input must be a Message object"):
            counter.count_message_tokens("not a message")
    
    def test_count_message_tokens_with_metadata(self):
        """Test counting tokens for message with metadata."""
        counter = TokenCounter()
        message = Message(
            agent_id="agent_with_long_name",
            content="Test message content",
            timestamp=datetime.now(),
            token_count=0,
            metadata={"key": "value", "another": "metadata"}
        )
        
        result = counter.count_message_tokens(message)
        assert result.token_count > 0
    
    def test_count_conversation_tokens(self):
        """Test counting tokens for a conversation."""
        counter = TokenCounter()
        
        messages = [
            Message("agent_a", "Hello!", datetime.now(), 0),
            Message("agent_b", "Hi there! How are you?", datetime.now(), 0),
            Message("agent_a", "I'm doing well, thanks for asking!", datetime.now(), 0)
        ]
        
        conversation = ConversationState(
            messages=messages,
            max_context_tokens=1000
        )
        
        result = counter.count_conversation_tokens(conversation)
        
        assert isinstance(result, dict)
        assert "total_tokens" in result
        assert "message_count" in result
        assert "average_tokens_per_message" in result
        assert "agent_token_counts" in result
        assert "message_tokens" in result
        assert "context_utilization" in result
        
        assert result["total_tokens"] > 0
        assert result["message_count"] == 3
        assert len(result["agent_token_counts"]) == 2
        assert "agent_a" in result["agent_token_counts"]
        assert "agent_b" in result["agent_token_counts"]
    
    def test_count_conversation_tokens_empty(self):
        """Test counting tokens for empty conversation."""
        counter = TokenCounter()
        conversation = ConversationState()
        
        result = counter.count_conversation_tokens(conversation)
        
        assert result["total_tokens"] == 0
        assert result["message_count"] == 0
        assert result["average_tokens_per_message"] == 0
        assert len(result["agent_token_counts"]) == 0
    
    def test_count_conversation_tokens_invalid_input(self):
        """Test counting conversation tokens with invalid input."""
        counter = TokenCounter()
        
        with pytest.raises(ValueError, match="Input must be a ConversationState object"):
            counter.count_conversation_tokens("not a conversation")
    
    def test_count_messages_tokens(self):
        """Test counting tokens for a list of messages."""
        counter = TokenCounter()
        
        messages = [
            Message("agent_a", "First message", datetime.now(), 0),
            Message("agent_b", "Second message", datetime.now(), 0),
            Message("agent_a", "Third message", datetime.now(), 0)
        ]
        
        result = counter.count_messages_tokens(messages)
        
        assert isinstance(result, dict)
        assert result["total_tokens"] > 0
        assert result["message_count"] == 3
        assert len(result["agent_token_counts"]) == 2
    
    def test_count_messages_tokens_invalid_input(self):
        """Test counting messages tokens with invalid input."""
        counter = TokenCounter()
        
        with pytest.raises(ValueError, match="Input must be a list of Message objects"):
            counter.count_messages_tokens("not a list")
        
        with pytest.raises(ValueError, match="All items in the list must be Message objects"):
            counter.count_messages_tokens(["not", "messages"])
    
    def test_estimate_tokens_for_text_list(self):
        """Test estimating tokens for a list of texts."""
        counter = TokenCounter()
        texts = [
            "First text string",
            "Second text string",
            "Third text string with more content"
        ]
        
        total_tokens = counter.estimate_tokens_for_text_list(texts)
        
        assert isinstance(total_tokens, int)
        assert total_tokens > 0
    
    def test_estimate_tokens_for_text_list_invalid_input(self):
        """Test estimating tokens with invalid input."""
        counter = TokenCounter()
        
        with pytest.raises(ValueError, match="Input must be a list of strings"):
            counter.estimate_tokens_for_text_list("not a list")
        
        with pytest.raises(ValueError, match="All items in the list must be strings"):
            counter.estimate_tokens_for_text_list([1, 2, 3])
    
    def test_get_model_info(self):
        """Test getting model information."""
        counter = TokenCounter("gpt-4")
        info = counter.get_model_info()
        
        assert isinstance(info, dict)
        assert info["model_type"] == "gpt-4"
        assert info["encoding_name"] == "cl100k_base"
        assert info["supports_chat_format"] is True
        assert info["tiktoken_available"] is True
    
    def test_get_model_info_davinci(self):
        """Test getting model info for non-chat model."""
        counter = TokenCounter("text-davinci-003")
        info = counter.get_model_info()
        
        assert info["model_type"] == "text-davinci-003"
        assert info["encoding_name"] == "p50k_base"
        assert info["supports_chat_format"] is False
    
    def test_create_for_model(self):
        """Test factory method."""
        counter = TokenCounter.create_for_model("gpt-3.5-turbo")
        
        assert isinstance(counter, TokenCounter)
        assert counter.model_type == ModelType.GPT_3_5_TURBO
    
    def test_get_supported_models(self):
        """Test getting supported models list."""
        models = TokenCounter.get_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert "gpt-4o" in models


class TestTokenCountingAccuracy:
    """Test token counting accuracy across different content types."""
    
    def test_consistent_counting(self):
        """Test that counting is consistent for the same text."""
        counter = TokenCounter()
        text = "This is a test message for consistency checking."
        
        result1 = counter.count_tokens(text)
        result2 = counter.count_tokens(text)
        
        assert result1.token_count == result2.token_count
    
    def test_different_models_same_encoding(self):
        """Test models with same encoding give same results."""
        text = "Test message for encoding comparison"
        
        counter_gpt4 = TokenCounter("gpt-4")
        counter_gpt35 = TokenCounter("gpt-3.5-turbo")
        
        result_gpt4 = counter_gpt4.count_tokens(text)
        result_gpt35 = counter_gpt35.count_tokens(text)
        
        # Both use cl100k_base encoding, so should be the same
        assert result_gpt4.token_count == result_gpt35.token_count
    
    def test_different_encodings(self):
        """Test that different encodings may give different results."""
        text = "Test message for encoding comparison"
        
        counter_gpt4 = TokenCounter("gpt-4")  # cl100k_base
        counter_davinci = TokenCounter("text-davinci-003")  # p50k_base
        
        result_gpt4 = counter_gpt4.count_tokens(text)
        result_davinci = counter_davinci.count_tokens(text)
        
        # Different encodings may give different results
        # We just check that both give reasonable results
        assert result_gpt4.token_count > 0
        assert result_davinci.token_count > 0
    
    def test_message_overhead_calculation(self):
        """Test that message overhead is calculated correctly."""
        counter = TokenCounter()
        
        # Test with short agent ID
        message1 = Message("a", "Hello", datetime.now(), 0)
        result1 = counter.count_message_tokens(message1)
        
        # Test with longer agent ID
        message2 = Message("very_long_agent_identifier", "Hello", datetime.now(), 0)
        result2 = counter.count_message_tokens(message2)
        
        # Message with longer agent ID should have more tokens
        assert result2.token_count > result1.token_count
    
    def test_context_utilization_calculation(self):
        """Test context utilization calculation in conversation analysis."""
        counter = TokenCounter()
        
        messages = [
            Message("agent_a", "Short message", datetime.now(), 0),
            Message("agent_b", "Another short message", datetime.now(), 0)
        ]
        
        conversation = ConversationState(
            messages=messages,
            max_context_tokens=100  # Small limit for testing
        )
        
        result = counter.count_conversation_tokens(conversation)
        
        assert 0 <= result["context_utilization"] <= 100
        assert result["context_utilization"] > 0  # Should have some utilization
    
    def test_empty_content_handling(self):
        """Test handling of messages with empty content."""
        counter = TokenCounter()
        
        message = Message("agent", "", datetime.now(), 0)
        result = counter.count_message_tokens(message)
        
        # Should still have some tokens due to message overhead
        assert result.token_count > 0
    
    def test_unicode_content(self):
        """Test token counting with Unicode content."""
        counter = TokenCounter()
        
        unicode_text = "Hello 世界! 🌍 Émojis and ñoñó characters"
        result = counter.count_tokens(unicode_text)
        
        assert result.token_count > 0
        assert result.content_length == len(unicode_text)
    
    def test_very_long_content(self):
        """Test token counting with very long content."""
        counter = TokenCounter()
        
        # Create a very long text
        long_text = "This is a test sentence. " * 1000
        result = counter.count_tokens(long_text)
        
        assert result.token_count > 1000  # Should have many tokens
        assert result.content_length == len(long_text)
    
    def test_code_content(self):
        """Test token counting with code content."""
        counter = TokenCounter()
        
        code_text = '''
def hello_world():
    print("Hello, world!")
    return True

if __name__ == "__main__":
    hello_world()
'''
        
        result = counter.count_tokens(code_text)
        
        assert result.token_count > 0
        assert result.content_length == len(code_text)