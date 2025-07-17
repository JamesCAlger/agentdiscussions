"""
Token counting utilities for the agentic conversation system.

This module provides accurate token counting for different language models
using tiktoken and other tokenization libraries.
"""

import tiktoken
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging
from dataclasses import dataclass

from .models import Message, ConversationState


logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported model types for token counting."""
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    TEXT_DAVINCI_003 = "text-davinci-003"
    TEXT_DAVINCI_002 = "text-davinci-002"
    
    @classmethod
    def from_string(cls, model_name: str) -> "ModelType":
        """Convert a model name string to ModelType enum."""
        # Handle common model name variations
        model_name_lower = model_name.lower().strip()
        
        # Map common variations to standard names
        model_mappings = {
            "gpt-3.5-turbo": cls.GPT_3_5_TURBO,
            "gpt-3.5": cls.GPT_3_5_TURBO,
            "gpt3.5": cls.GPT_3_5_TURBO,
            "gpt-4": cls.GPT_4,
            "gpt4": cls.GPT_4,
            "gpt-4-turbo": cls.GPT_4_TURBO,
            "gpt-4-turbo-preview": cls.GPT_4_TURBO,
            "gpt-4o": cls.GPT_4O,
            "gpt-4o-mini": cls.GPT_4O_MINI,
            "text-davinci-003": cls.TEXT_DAVINCI_003,
            "text-davinci-002": cls.TEXT_DAVINCI_002,
        }
        
        # Try exact match first
        for key, value in model_mappings.items():
            if model_name_lower == key:
                return value
        
        # Try partial matches for versioned models
        if "gpt-3.5" in model_name_lower:
            return cls.GPT_3_5_TURBO
        elif "gpt-4o-mini" in model_name_lower:
            return cls.GPT_4O_MINI
        elif "gpt-4o" in model_name_lower:
            return cls.GPT_4O
        elif "gpt-4-turbo" in model_name_lower:
            return cls.GPT_4_TURBO
        elif "gpt-4" in model_name_lower:
            return cls.GPT_4
        elif "davinci-003" in model_name_lower:
            return cls.TEXT_DAVINCI_003
        elif "davinci-002" in model_name_lower:
            return cls.TEXT_DAVINCI_002
        
        # Default to GPT-4 for unknown models
        logger.warning(f"Unknown model '{model_name}', defaulting to GPT-4 tokenizer")
        return cls.GPT_4


@dataclass
class TokenCountResult:
    """Result of token counting operation."""
    token_count: int
    model_type: ModelType
    encoding_name: str
    content_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "token_count": self.token_count,
            "model_type": self.model_type.value,
            "encoding_name": self.encoding_name,
            "content_length": self.content_length
        }


class TokenCounter:
    """
    Utility class for counting tokens using different model tokenizers.
    
    This class provides accurate token counting for various language models
    using tiktoken library and handles different tokenization strategies.
    """
    
    # Mapping of model types to their tiktoken encoding names
    MODEL_ENCODINGS = {
        ModelType.GPT_3_5_TURBO: "cl100k_base",
        ModelType.GPT_4: "cl100k_base",
        ModelType.GPT_4_TURBO: "cl100k_base",
        ModelType.GPT_4O: "o200k_base",
        ModelType.GPT_4O_MINI: "o200k_base",
        ModelType.TEXT_DAVINCI_003: "p50k_base",
        ModelType.TEXT_DAVINCI_002: "p50k_base",
    }
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the token counter for a specific model.
        
        Args:
            model_name: Name of the language model to use for tokenization
        """
        self.model_type = ModelType.from_string(model_name)
        self.encoding_name = self.MODEL_ENCODINGS[self.model_type]
        self._encoding = None
        self._load_encoding()
    
    def _load_encoding(self) -> None:
        """Load the tiktoken encoding for the model."""
        try:
            self._encoding = tiktoken.get_encoding(self.encoding_name)
            logger.debug(f"Loaded encoding '{self.encoding_name}' for model '{self.model_type.value}'")
        except Exception as e:
            logger.error(f"Failed to load encoding '{self.encoding_name}': {e}")
            # Fallback to a default encoding
            self._encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning("Using fallback encoding 'cl100k_base'")
    
    def count_tokens(self, text: str) -> TokenCountResult:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            TokenCountResult with detailed counting information
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        if not text:
            return TokenCountResult(
                token_count=0,
                model_type=self.model_type,
                encoding_name=self.encoding_name,
                content_length=0
            )
        
        try:
            tokens = self._encoding.encode(text)
            token_count = len(tokens)
            
            logger.debug(f"Counted {token_count} tokens for text of length {len(text)}")
            
            return TokenCountResult(
                token_count=token_count,
                model_type=self.model_type,
                encoding_name=self.encoding_name,
                content_length=len(text)
            )
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to rough estimation (4 characters per token)
            estimated_tokens = max(1, len(text) // 4)
            logger.warning(f"Using fallback estimation: {estimated_tokens} tokens")
            
            return TokenCountResult(
                token_count=estimated_tokens,
                model_type=self.model_type,
                encoding_name=self.encoding_name,
                content_length=len(text)
            )
    
    def count_message_tokens(self, message: Message) -> TokenCountResult:
        """
        Count tokens in a Message object.
        
        Args:
            message: The Message object to count tokens for
            
        Returns:
            TokenCountResult with detailed counting information
        """
        if not isinstance(message, Message):
            raise ValueError("Input must be a Message object")
        
        # For chat models, we need to account for message formatting overhead
        # This includes role tokens, formatting tokens, etc.
        content_result = self.count_tokens(message.content)
        
        # Add overhead for message structure (role, formatting, etc.)
        # This is an approximation based on OpenAI's token counting guidelines
        overhead_tokens = self._calculate_message_overhead(message)
        
        total_tokens = content_result.token_count + overhead_tokens
        
        return TokenCountResult(
            token_count=total_tokens,
            model_type=self.model_type,
            encoding_name=self.encoding_name,
            content_length=content_result.content_length
        )
    
    def _calculate_message_overhead(self, message: Message) -> int:
        """
        Calculate the token overhead for message formatting.
        
        Args:
            message: The Message object
            
        Returns:
            Number of overhead tokens
        """
        # Base overhead for message structure
        overhead = 4  # Typical overhead for role and formatting
        
        # Add tokens for agent_id (treated as role)
        if message.agent_id:
            agent_id_tokens = self.count_tokens(message.agent_id).token_count
            overhead += agent_id_tokens
        
        # Add small overhead for metadata if present
        if message.metadata:
            overhead += 2  # Small overhead for metadata structure
        
        return overhead
    
    def count_conversation_tokens(self, conversation_state: ConversationState) -> Dict[str, Any]:
        """
        Count tokens for an entire conversation.
        
        Args:
            conversation_state: The ConversationState to analyze
            
        Returns:
            Dictionary with detailed token analysis
        """
        if not isinstance(conversation_state, ConversationState):
            raise ValueError("Input must be a ConversationState object")
        
        total_tokens = 0
        message_tokens = []
        agent_token_counts = {}
        
        for message in conversation_state.messages:
            result = self.count_message_tokens(message)
            total_tokens += result.token_count
            
            message_tokens.append({
                "agent_id": message.agent_id,
                "token_count": result.token_count,
                "content_length": result.content_length,
                "timestamp": message.timestamp.isoformat()
            })
            
            # Track tokens per agent
            if message.agent_id not in agent_token_counts:
                agent_token_counts[message.agent_id] = 0
            agent_token_counts[message.agent_id] += result.token_count
        
        return {
            "total_tokens": total_tokens,
            "message_count": len(conversation_state.messages),
            "average_tokens_per_message": total_tokens / max(1, len(conversation_state.messages)),
            "agent_token_counts": agent_token_counts,
            "message_tokens": message_tokens,
            "model_type": self.model_type.value,
            "encoding_name": self.encoding_name,
            "context_utilization": (total_tokens / conversation_state.max_context_tokens) * 100 if conversation_state.max_context_tokens > 0 else 0
        }
    
    def count_messages_tokens(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Count tokens for a list of messages.
        
        Args:
            messages: List of Message objects
            
        Returns:
            Dictionary with detailed token analysis
        """
        if not isinstance(messages, list):
            raise ValueError("Input must be a list of Message objects")
        
        total_tokens = 0
        message_tokens = []
        agent_token_counts = {}
        
        for message in messages:
            if not isinstance(message, Message):
                raise ValueError("All items in the list must be Message objects")
            
            result = self.count_message_tokens(message)
            total_tokens += result.token_count
            
            message_tokens.append({
                "agent_id": message.agent_id,
                "token_count": result.token_count,
                "content_length": result.content_length,
                "timestamp": message.timestamp.isoformat()
            })
            
            # Track tokens per agent
            if message.agent_id not in agent_token_counts:
                agent_token_counts[message.agent_id] = 0
            agent_token_counts[message.agent_id] += result.token_count
        
        return {
            "total_tokens": total_tokens,
            "message_count": len(messages),
            "average_tokens_per_message": total_tokens / max(1, len(messages)),
            "agent_token_counts": agent_token_counts,
            "message_tokens": message_tokens,
            "model_type": self.model_type.value,
            "encoding_name": self.encoding_name
        }
    
    def estimate_tokens_for_text_list(self, texts: List[str]) -> int:
        """
        Estimate total tokens for a list of text strings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Total estimated token count
        """
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings")
        
        total_tokens = 0
        for text in texts:
            if not isinstance(text, str):
                raise ValueError("All items in the list must be strings")
            result = self.count_tokens(text)
            total_tokens += result.token_count
        
        return total_tokens
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model and tokenizer.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type.value,
            "encoding_name": self.encoding_name,
            "supports_chat_format": self.model_type in [
                ModelType.GPT_3_5_TURBO,
                ModelType.GPT_4,
                ModelType.GPT_4_TURBO,
                ModelType.GPT_4O,
                ModelType.GPT_4O_MINI
            ],
            "tiktoken_available": self._encoding is not None
        }
    
    @classmethod
    def create_for_model(cls, model_name: str) -> "TokenCounter":
        """
        Factory method to create a TokenCounter for a specific model.
        
        Args:
            model_name: Name of the language model
            
        Returns:
            TokenCounter instance configured for the model
        """
        return cls(model_name=model_name)
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """
        Get a list of supported model names.
        
        Returns:
            List of supported model names
        """
        return [model_type.value for model_type in ModelType]