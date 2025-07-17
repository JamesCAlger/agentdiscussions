"""
Context window management for the agentic conversation system.

This module provides context window management capabilities including
truncation, sliding window, and utilization tracking strategies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from .models import Message, ConversationState, ContextWindowSnapshot
from .token_counter import TokenCounter, TokenCountResult


logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Available context window management strategies."""
    TRUNCATE = "truncate"
    SLIDING = "sliding"
    SUMMARIZE = "summarize"  # Future implementation


@dataclass
class ContextManagementResult:
    """Result of context window management operation."""
    messages: List[Message]
    total_tokens: int
    strategy_applied: Optional[str]
    messages_removed: int
    tokens_saved: int
    utilization_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tokens": self.total_tokens,
            "strategy_applied": self.strategy_applied,
            "messages_removed": self.messages_removed,
            "tokens_saved": self.tokens_saved,
            "utilization_percentage": self.utilization_percentage,
            "message_count": len(self.messages)
        }


class BaseContextStrategy(ABC):
    """Abstract base class for context management strategies."""
    
    def __init__(self, token_counter: TokenCounter):
        """
        Initialize the context strategy.
        
        Args:
            token_counter: TokenCounter instance for token calculations
        """
        self.token_counter = token_counter
    
    @abstractmethod
    def apply(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        threshold_percentage: float = 90.0
    ) -> ContextManagementResult:
        """
        Apply the context management strategy.
        
        Args:
            messages: List of messages to manage
            max_tokens: Maximum allowed tokens in context
            threshold_percentage: Threshold percentage to trigger management
            
        Returns:
            ContextManagementResult with managed messages and metadata
        """
        pass
    
    def _calculate_total_tokens(self, messages: List[Message]) -> int:
        """Calculate total tokens for a list of messages."""
        return sum(
            self.token_counter.count_message_tokens(msg).token_count 
            for msg in messages
        )
    
    def _calculate_utilization(self, total_tokens: int, max_tokens: int) -> float:
        """Calculate context window utilization percentage."""
        if max_tokens <= 0:
            return 0.0
        return (total_tokens / max_tokens) * 100


class TruncateStrategy(BaseContextStrategy):
    """
    Truncation strategy that removes oldest messages when limit is exceeded.
    
    This strategy removes messages from the beginning of the conversation
    until the token count is below the threshold.
    """
    
    def apply(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        threshold_percentage: float = 90.0
    ) -> ContextManagementResult:
        """
        Apply truncation strategy by removing oldest messages.
        
        Args:
            messages: List of messages to manage
            max_tokens: Maximum allowed tokens in context
            threshold_percentage: Threshold percentage to trigger truncation
            
        Returns:
            ContextManagementResult with truncated messages
        """
        if not messages:
            return ContextManagementResult(
                messages=[],
                total_tokens=0,
                strategy_applied=None,
                messages_removed=0,
                tokens_saved=0,
                utilization_percentage=0.0
            )
        
        original_count = len(messages)
        current_messages = messages.copy()
        total_tokens = self._calculate_total_tokens(current_messages)
        threshold_tokens = max_tokens * (threshold_percentage / 100)
        
        # Check if truncation is needed
        if total_tokens <= threshold_tokens:
            return ContextManagementResult(
                messages=current_messages,
                total_tokens=total_tokens,
                strategy_applied=None,
                messages_removed=0,
                tokens_saved=0,
                utilization_percentage=self._calculate_utilization(total_tokens, max_tokens)
            )
        
        logger.info(f"Applying truncation strategy: {total_tokens} tokens > {threshold_tokens} threshold")
        
        # Remove messages from the beginning until under threshold
        messages_removed = 0
        tokens_saved = 0
        
        while current_messages and total_tokens > threshold_tokens:
            removed_message = current_messages.pop(0)
            removed_tokens = self.token_counter.count_message_tokens(removed_message).token_count
            total_tokens -= removed_tokens
            tokens_saved += removed_tokens
            messages_removed += 1
            
            logger.debug(f"Removed message from {removed_message.agent_id}, saved {removed_tokens} tokens")
        
        logger.info(f"Truncation complete: removed {messages_removed} messages, saved {tokens_saved} tokens")
        
        return ContextManagementResult(
            messages=current_messages,
            total_tokens=total_tokens,
            strategy_applied=ContextStrategy.TRUNCATE.value,
            messages_removed=messages_removed,
            tokens_saved=tokens_saved,
            utilization_percentage=self._calculate_utilization(total_tokens, max_tokens)
        )


class SlidingWindowStrategy(BaseContextStrategy):
    """
    Sliding window strategy that maintains recent context by removing older messages.
    
    This strategy keeps a sliding window of recent messages, ensuring that
    the most recent interactions are preserved while older context is removed.
    """
    
    def __init__(self, token_counter: TokenCounter, min_messages: int = 4):
        """
        Initialize sliding window strategy.
        
        Args:
            token_counter: TokenCounter instance for token calculations
            min_messages: Minimum number of messages to keep in the window
        """
        super().__init__(token_counter)
        self.min_messages = max(2, min_messages)  # Ensure at least 2 messages
    
    def apply(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        threshold_percentage: float = 90.0
    ) -> ContextManagementResult:
        """
        Apply sliding window strategy by maintaining recent context.
        
        Args:
            messages: List of messages to manage
            max_tokens: Maximum allowed tokens in context
            threshold_percentage: Threshold percentage to trigger sliding
            
        Returns:
            ContextManagementResult with sliding window applied
        """
        if not messages:
            return ContextManagementResult(
                messages=[],
                total_tokens=0,
                strategy_applied=None,
                messages_removed=0,
                tokens_saved=0,
                utilization_percentage=0.0
            )
        
        original_count = len(messages)
        current_messages = messages.copy()
        total_tokens = self._calculate_total_tokens(current_messages)
        threshold_tokens = max_tokens * (threshold_percentage / 100)
        
        # Check if sliding window is needed
        if total_tokens <= threshold_tokens:
            return ContextManagementResult(
                messages=current_messages,
                total_tokens=total_tokens,
                strategy_applied=None,
                messages_removed=0,
                tokens_saved=0,
                utilization_percentage=self._calculate_utilization(total_tokens, max_tokens)
            )
        
        logger.info(f"Applying sliding window strategy: {total_tokens} tokens > {threshold_tokens} threshold")
        
        # Keep removing oldest messages while maintaining minimum count
        messages_removed = 0
        tokens_saved = 0
        
        while (len(current_messages) > self.min_messages and 
               total_tokens > threshold_tokens):
            
            removed_message = current_messages.pop(0)
            removed_tokens = self.token_counter.count_message_tokens(removed_message).token_count
            total_tokens -= removed_tokens
            tokens_saved += removed_tokens
            messages_removed += 1
            
            logger.debug(f"Sliding window: removed message from {removed_message.agent_id}, saved {removed_tokens} tokens")
        
        # If still over threshold but at minimum messages, try to fit within hard limit
        if total_tokens > max_tokens and len(current_messages) > 1:
            logger.warning(f"Still over hard limit ({total_tokens} > {max_tokens}), removing more messages")
            
            while len(current_messages) > 1 and total_tokens > max_tokens:
                removed_message = current_messages.pop(0)
                removed_tokens = self.token_counter.count_message_tokens(removed_message).token_count
                total_tokens -= removed_tokens
                tokens_saved += removed_tokens
                messages_removed += 1
        
        logger.info(f"Sliding window complete: removed {messages_removed} messages, saved {tokens_saved} tokens")
        
        return ContextManagementResult(
            messages=current_messages,
            total_tokens=total_tokens,
            strategy_applied=ContextStrategy.SLIDING.value,
            messages_removed=messages_removed,
            tokens_saved=tokens_saved,
            utilization_percentage=self._calculate_utilization(total_tokens, max_tokens)
        )


class ContextManager:
    """
    Main context window manager that handles context limitations and strategies.
    
    This class provides comprehensive context window management including
    real-time token counting, strategy application, and utilization tracking.
    """
    
    def __init__(
        self, 
        token_counter: TokenCounter,
        max_context_tokens: int = 8000,
        strategy: ContextStrategy = ContextStrategy.SLIDING,
        threshold_percentage: float = 90.0
    ):
        """
        Initialize the context manager.
        
        Args:
            token_counter: TokenCounter instance for token calculations
            max_context_tokens: Maximum allowed tokens in context window
            strategy: Context management strategy to use
            threshold_percentage: Threshold percentage to trigger management
        """
        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens
        self.strategy = strategy
        self.threshold_percentage = threshold_percentage
        self._strategies = self._initialize_strategies()
        self._snapshots: List[ContextWindowSnapshot] = []
    
    def _initialize_strategies(self) -> Dict[ContextStrategy, BaseContextStrategy]:
        """Initialize available context management strategies."""
        return {
            ContextStrategy.TRUNCATE: TruncateStrategy(self.token_counter),
            ContextStrategy.SLIDING: SlidingWindowStrategy(self.token_counter)
        }
    
    def manage_context(
        self, 
        messages: List[Message], 
        force_apply: bool = False
    ) -> ContextManagementResult:
        """
        Manage context window for a list of messages.
        
        Args:
            messages: List of messages to manage
            force_apply: Force application of strategy regardless of threshold
            
        Returns:
            ContextManagementResult with managed messages and metadata
        """
        if not messages:
            return ContextManagementResult(
                messages=[],
                total_tokens=0,
                strategy_applied=None,
                messages_removed=0,
                tokens_saved=0,
                utilization_percentage=0.0
            )
        
        strategy_impl = self._strategies.get(self.strategy)
        if not strategy_impl:
            raise ValueError(f"Unknown context strategy: {self.strategy}")
        
        # Apply strategy with force option
        threshold = 0.0 if force_apply else self.threshold_percentage
        result = strategy_impl.apply(messages, self.max_context_tokens, threshold)
        
        # Create snapshot for tracking
        self._create_snapshot(result, len(messages))
        
        return result
    
    def manage_conversation_context(
        self, 
        conversation_state: ConversationState,
        force_apply: bool = False
    ) -> ContextManagementResult:
        """
        Manage context window for a conversation state.
        
        Args:
            conversation_state: ConversationState to manage
            force_apply: Force application of strategy regardless of threshold
            
        Returns:
            ContextManagementResult with managed messages and updated state
        """
        result = self.manage_context(conversation_state.messages, force_apply)
        
        # Update conversation state with managed messages
        conversation_state.messages = result.messages
        conversation_state.current_context_tokens = result.total_tokens
        
        return result
    
    def check_context_status(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Check the current context window status without applying management.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dictionary with context status information
        """
        if not messages:
            return {
                "total_tokens": 0,
                "max_tokens": self.max_context_tokens,
                "available_tokens": self.max_context_tokens,
                "utilization_percentage": 0.0,
                "needs_management": False,
                "over_threshold": False,
                "over_hard_limit": False,
                "message_count": 0
            }
        
        total_tokens = sum(
            self.token_counter.count_message_tokens(msg).token_count 
            for msg in messages
        )
        
        available_tokens = max(0, self.max_context_tokens - total_tokens)
        utilization_percentage = (total_tokens / self.max_context_tokens) * 100 if self.max_context_tokens > 0 else 0
        threshold_tokens = self.max_context_tokens * (self.threshold_percentage / 100)
        
        return {
            "total_tokens": total_tokens,
            "max_tokens": self.max_context_tokens,
            "available_tokens": available_tokens,
            "utilization_percentage": utilization_percentage,
            "needs_management": total_tokens > threshold_tokens,
            "over_threshold": total_tokens > threshold_tokens,
            "over_hard_limit": total_tokens > self.max_context_tokens,
            "message_count": len(messages),
            "threshold_tokens": threshold_tokens,
            "strategy": self.strategy.value
        }
    
    def get_context_for_agent(
        self, 
        messages: List[Message], 
        agent_id: Optional[str] = None
    ) -> Tuple[List[Message], Dict[str, Any]]:
        """
        Get managed context for a specific agent.
        
        Args:
            messages: List of messages to manage
            agent_id: Optional agent ID for agent-specific context
            
        Returns:
            Tuple of (managed_messages, context_info)
        """
        # Apply context management
        result = self.manage_context(messages)
        
        # Get context status
        context_info = self.check_context_status(result.messages)
        context_info.update(result.to_dict())
        
        return result.messages, context_info
    
    def _create_snapshot(self, result: ContextManagementResult, original_message_count: int) -> None:
        """Create a context window snapshot for tracking."""
        snapshot = ContextWindowSnapshot(
            turn_number=len(self._snapshots) + 1,
            total_tokens=result.total_tokens,
            available_tokens=max(0, self.max_context_tokens - result.total_tokens),
            utilization_percentage=result.utilization_percentage,
            strategy_applied=result.strategy_applied
        )
        
        self._snapshots.append(snapshot)
        
        logger.debug(f"Created context snapshot: {snapshot.utilization_percentage:.1f}% utilization")
    
    def get_snapshots(self) -> List[ContextWindowSnapshot]:
        """Get all context window snapshots."""
        return self._snapshots.copy()
    
    def get_latest_snapshot(self) -> Optional[ContextWindowSnapshot]:
        """Get the most recent context window snapshot."""
        return self._snapshots[-1] if self._snapshots else None
    
    def clear_snapshots(self) -> None:
        """Clear all context window snapshots."""
        self._snapshots.clear()
    
    def get_utilization_history(self) -> List[float]:
        """Get history of context window utilization percentages."""
        return [snapshot.utilization_percentage for snapshot in self._snapshots]
    
    def get_peak_utilization(self) -> float:
        """Get the peak context window utilization percentage."""
        if not self._snapshots:
            return 0.0
        return max(snapshot.utilization_percentage for snapshot in self._snapshots)
    
    def get_average_utilization(self) -> float:
        """Get the average context window utilization percentage."""
        if not self._snapshots:
            return 0.0
        return sum(snapshot.utilization_percentage for snapshot in self._snapshots) / len(self._snapshots)
    
    def update_configuration(
        self, 
        max_context_tokens: Optional[int] = None,
        strategy: Optional[ContextStrategy] = None,
        threshold_percentage: Optional[float] = None
    ) -> None:
        """
        Update context manager configuration.
        
        Args:
            max_context_tokens: New maximum context tokens
            strategy: New context management strategy
            threshold_percentage: New threshold percentage
        """
        if max_context_tokens is not None:
            if max_context_tokens <= 0:
                raise ValueError("Max context tokens must be positive")
            self.max_context_tokens = max_context_tokens
            logger.info(f"Updated max context tokens to {max_context_tokens}")
        
        if strategy is not None:
            if strategy not in self._strategies:
                raise ValueError(f"Unknown strategy: {strategy}")
            self.strategy = strategy
            logger.info(f"Updated context strategy to {strategy.value}")
        
        if threshold_percentage is not None:
            if not (0 < threshold_percentage <= 100):
                raise ValueError("Threshold percentage must be between 0 and 100")
            self.threshold_percentage = threshold_percentage
            logger.info(f"Updated threshold percentage to {threshold_percentage}%")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current context manager configuration."""
        return {
            "max_context_tokens": self.max_context_tokens,
            "strategy": self.strategy.value,
            "threshold_percentage": self.threshold_percentage,
            "available_strategies": [s.value for s in self._strategies.keys()],
            "token_counter_model": self.token_counter.model_type.value
        }
    
    def estimate_messages_for_tokens(self, target_tokens: int, messages: List[Message]) -> int:
        """
        Estimate how many messages can fit within a token limit.
        
        Args:
            target_tokens: Target token count
            messages: List of messages to analyze (from most recent)
            
        Returns:
            Number of messages that can fit within the token limit
        """
        if not messages or target_tokens <= 0:
            return 0
        
        current_tokens = 0
        message_count = 0
        
        # Start from the most recent messages (end of list)
        for message in reversed(messages):
            message_tokens = self.token_counter.count_message_tokens(message).token_count
            
            if current_tokens + message_tokens <= target_tokens:
                current_tokens += message_tokens
                message_count += 1
            else:
                break
        
        return message_count
    
    @classmethod
    def create_with_strategy(
        cls,
        token_counter: TokenCounter,
        strategy_name: str,
        max_context_tokens: int = 8000,
        threshold_percentage: float = 90.0
    ) -> "ContextManager":
        """
        Factory method to create ContextManager with specific strategy.
        
        Args:
            token_counter: TokenCounter instance
            strategy_name: Name of the strategy ("truncate", "sliding", "summarize")
            max_context_tokens: Maximum context tokens
            threshold_percentage: Threshold percentage
            
        Returns:
            ContextManager instance with specified strategy
        """
        try:
            strategy = ContextStrategy(strategy_name.lower())
        except ValueError:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {[s.value for s in ContextStrategy]}")
        
        return cls(
            token_counter=token_counter,
            max_context_tokens=max_context_tokens,
            strategy=strategy,
            threshold_percentage=threshold_percentage
        )