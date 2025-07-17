"""
LangChain-based agent implementation for the agentic conversation system.

This module provides a concrete implementation of the BaseAgent interface
using LangChain for LLM interactions and response generation.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .agents import (
    BaseAgent, AgentInfo, AgentResponse, ConversationContext,
    AgentError, AgentTimeoutError, AgentValidationError,
    AgentResponseError, AgentConfigurationError
)
from .models import ModelConfig, Message
from .token_counter import TokenCounter, ModelType


class LangChainAgent(BaseAgent):
    """
    LangChain-based agent implementation.
    
    This agent uses LangChain's chat models to generate responses and includes
    comprehensive error handling, retry logic, and telemetry collection.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        model_config: ModelConfig,
        token_counter: Optional[TokenCounter] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0
    ):
        """
        Initialize the LangChain agent.
        
        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for the agent
            system_prompt: System prompt that defines the agent's behavior
            model_config: Configuration for the language model
            token_counter: Token counter for tracking usage (optional)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
            timeout: Timeout for individual LLM requests in seconds
        """
        super().__init__(agent_id, name, system_prompt)
        
        self.model_config = model_config
        self.token_counter = token_counter or TokenCounter()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Initialize the LLM
        self.llm = self._create_llm()
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info(f"Initialized LangChain agent '{name}' with model '{model_config.model_name}'")
    
    def _create_llm(self) -> BaseChatModel:
        """
        Create the appropriate LangChain LLM based on model configuration.
        
        Returns:
            BaseChatModel: Configured LangChain chat model
            
        Raises:
            AgentConfigurationError: If model configuration is invalid
        """
        model_name = self.model_config.model_name.lower()
        
        try:
            if "gpt" in model_name or "openai" in model_name:
                return ChatOpenAI(
                    model_name=self.model_config.model_name,
                    temperature=self.model_config.temperature,
                    max_tokens=self.model_config.max_tokens,
                    top_p=self.model_config.top_p,
                    frequency_penalty=self.model_config.frequency_penalty,
                    presence_penalty=self.model_config.presence_penalty,
                    request_timeout=self.timeout
                )
            elif "claude" in model_name or "anthropic" in model_name:
                return ChatAnthropic(
                    model=self.model_config.model_name,
                    temperature=self.model_config.temperature,
                    max_tokens=self.model_config.max_tokens,
                    top_p=self.model_config.top_p,
                    timeout=self.timeout
                )
            else:
                # Default to OpenAI for unknown models
                self.logger.warning(f"Unknown model '{model_name}', defaulting to OpenAI")
                return ChatOpenAI(
                    model_name=self.model_config.model_name,
                    temperature=self.model_config.temperature,
                    max_tokens=self.model_config.max_tokens,
                    top_p=self.model_config.top_p,
                    frequency_penalty=self.model_config.frequency_penalty,
                    presence_penalty=self.model_config.presence_penalty,
                    request_timeout=self.timeout
                )
        except Exception as e:
            raise AgentConfigurationError(
                f"Failed to create LLM for model '{self.model_config.model_name}': {str(e)}",
                agent_id=self.agent_id,
                original_error=e
            )
    
    def _validate_configuration(self) -> None:
        """
        Validate the agent configuration.
        
        Raises:
            AgentConfigurationError: If configuration is invalid
        """
        errors = self.model_config.validate()
        if errors:
            raise AgentConfigurationError(
                f"Invalid model configuration: {'; '.join(errors)}",
                agent_id=self.agent_id
            )
        
        if self.max_retries < 0:
            raise AgentConfigurationError(
                "Max retries must be non-negative",
                agent_id=self.agent_id
            )
        
        if self.retry_delay <= 0:
            raise AgentConfigurationError(
                "Retry delay must be positive",
                agent_id=self.agent_id
            )
        
        if self.timeout <= 0:
            raise AgentConfigurationError(
                "Timeout must be positive",
                agent_id=self.agent_id
            )
    
    def _convert_messages_to_langchain(self, messages: List[Message]) -> List[BaseMessage]:
        """
        Convert conversation messages to LangChain message format.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            List[BaseMessage]: LangChain-formatted messages
        """
        langchain_messages = []
        
        for message in messages:
            if message.agent_id == self.agent_id:
                # This agent's previous messages
                langchain_messages.append(AIMessage(content=message.content))
            else:
                # Other agent's messages
                langchain_messages.append(HumanMessage(content=message.content))
        
        return langchain_messages
    
    def _prepare_messages(self, context: ConversationContext) -> List[BaseMessage]:
        """
        Prepare the message sequence for the LLM.
        
        Args:
            context: Conversation context
            
        Returns:
            List[BaseMessage]: Prepared message sequence
        """
        messages = []
        
        # Add system message
        messages.append(SystemMessage(content=context.system_prompt))
        
        # Convert conversation history to LangChain format
        conversation_messages = self._convert_messages_to_langchain(
            context.get_conversation_history()
        )
        messages.extend(conversation_messages)
        
        return messages
    
    async def _generate_response_with_retry(
        self, 
        messages: List[BaseMessage]
    ) -> tuple[str, int, float]:
        """
        Generate response with retry logic and exponential backoff.
        
        Args:
            messages: Prepared message sequence
            
        Returns:
            tuple: (response_content, model_calls, total_time)
            
        Raises:
            AgentResponseError: If all retry attempts fail
            AgentTimeoutError: If request times out
        """
        last_error = None
        total_time = 0.0
        model_calls = 0
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                # Make the LLM call with timeout
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=self.timeout
                )
                
                end_time = time.time()
                call_time = end_time - start_time
                total_time += call_time
                model_calls += 1
                
                # Extract content from response
                if hasattr(response, 'content'):
                    content = response.content
                elif hasattr(response, 'text'):
                    content = response.text
                else:
                    content = str(response)
                
                if not content or not content.strip():
                    raise AgentResponseError(
                        "LLM returned empty response",
                        agent_id=self.agent_id
                    )
                
                self.logger.debug(
                    f"Agent '{self.agent_id}' generated response in {call_time:.2f}s "
                    f"(attempt {attempt + 1})"
                )
                
                return content.strip(), model_calls, total_time
                
            except asyncio.TimeoutError as e:
                model_calls += 1  # Count failed attempts too
                last_error = AgentTimeoutError(
                    f"LLM request timed out after {self.timeout}s",
                    agent_id=self.agent_id,
                    original_error=e
                )
                self.logger.warning(f"Timeout on attempt {attempt + 1}: {str(e)}")
                
            except Exception as e:
                model_calls += 1  # Count failed attempts too
                last_error = AgentResponseError(
                    f"LLM request failed: {str(e)}",
                    agent_id=self.agent_id,
                    original_error=e
                )
                self.logger.warning(f"Error on attempt {attempt + 1}: {str(e)}")
            
            # Don't sleep after the last attempt
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                self.logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise last_error or AgentResponseError(
            "All retry attempts failed",
            agent_id=self.agent_id
        )
    
    async def generate_response(self, context: ConversationContext) -> AgentResponse:
        """
        Generate a response based on the conversation context.
        
        Args:
            context: The conversation context containing history and metadata
            
        Returns:
            AgentResponse: The generated response with metadata
            
        Raises:
            AgentValidationError: If context validation fails
            AgentResponseError: If response generation fails
            AgentTimeoutError: If request times out
        """
        start_time = time.time()
        errors = []
        
        try:
            # Validate context
            validation_errors = self.validate_context(context)
            if validation_errors:
                raise AgentValidationError(
                    f"Context validation failed: {'; '.join(validation_errors)}",
                    agent_id=self.agent_id,
                    context=context
                )
            
            # Prepare context
            prepared_context = await self.prepare_context(context)
            
            # Prepare messages for LLM
            messages = self._prepare_messages(prepared_context)
            
            self.logger.debug(f"Agent '{self.agent_id}' generating response...")
            
            # Generate response with retry logic
            content, model_calls, response_time = await self._generate_response_with_retry(messages)
            
            # Count tokens in the response
            token_count = 0
            try:
                # Determine model type for token counting
                model_type = ModelType.GPT_4
                if "gpt-3.5" in self.model_config.model_name.lower():
                    model_type = ModelType.GPT_3_5_TURBO
                elif "claude" in self.model_config.model_name.lower():
                    # For now, use GPT-4 tokenizer for Claude (approximation)
                    model_type = ModelType.GPT_4
                
                token_result = self.token_counter.count_tokens(content)
                token_count = token_result.token_count
                
            except Exception as e:
                self.logger.warning(f"Token counting failed: {str(e)}")
                errors.append(f"Token counting failed: {str(e)}")
                # Estimate token count as word count * 1.3 (rough approximation)
                token_count = int(len(content.split()) * 1.3)
            
            # Create response object
            response = AgentResponse(
                content=content,
                agent_id=self.agent_id,
                timestamp=datetime.now(),
                token_count=token_count,
                response_time=response_time,
                model_calls=model_calls,
                confidence=None,  # LangChain doesn't provide confidence scores by default
                reasoning=None,
                metadata={
                    "model_name": self.model_config.model_name,
                    "temperature": self.model_config.temperature,
                    "max_tokens": self.model_config.max_tokens,
                    "context_tokens": prepared_context.conversation_state.current_context_tokens,
                    "available_tokens": prepared_context.available_tokens
                },
                errors=errors
            )
            
            # Post-process response
            final_response = await self.post_process_response(response, prepared_context)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.logger.info(
                f"Agent '{self.agent_id}' generated response: "
                f"{len(content)} chars, {token_count} tokens, "
                f"{total_time:.2f}s total, {model_calls} API calls"
            )
            
            return final_response
            
        except (AgentValidationError, AgentResponseError, AgentTimeoutError):
            # Re-raise agent-specific errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise AgentResponseError(
                f"Unexpected error during response generation: {str(e)}",
                agent_id=self.agent_id,
                context=context,
                original_error=e
            )
    
    def get_agent_info(self) -> AgentInfo:
        """
        Get information about this agent's capabilities and configuration.
        
        Returns:
            AgentInfo: Information about the agent
        """
        capabilities = [
            "text_generation",
            "conversation",
            "context_aware",
            "retry_logic",
            "error_recovery"
        ]
        
        # Add model-specific capabilities
        model_name = self.model_config.model_name.lower()
        if "gpt" in model_name:
            capabilities.extend(["openai_models", "function_calling"])
        elif "claude" in model_name:
            capabilities.extend(["anthropic_models", "long_context"])
        
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description=f"LangChain-based agent using {self.model_config.model_name}",
            model_name=self.model_config.model_name,
            capabilities=capabilities,
            metadata={
                "temperature": self.model_config.temperature,
                "max_tokens": self.model_config.max_tokens,
                "max_retries": self.max_retries,
                "timeout": self.timeout,
                "langchain_version": "0.1.0+",  # Approximate version
                "system_prompt_length": len(self.system_prompt)
            }
        )
    
    async def prepare_context(self, context: ConversationContext) -> ConversationContext:
        """
        Prepare the conversation context before generating a response.
        
        This implementation adds model-specific context preparation.
        
        Args:
            context: The original conversation context
            
        Returns:
            ConversationContext: The prepared context
        """
        # For now, return the context unchanged
        # Future enhancements could include:
        # - Context window management
        # - Message filtering based on relevance
        # - Dynamic system prompt adjustment
        return context
    
    async def post_process_response(
        self, 
        response: AgentResponse, 
        context: ConversationContext
    ) -> AgentResponse:
        """
        Post-process the generated response.
        
        This implementation adds basic post-processing like content validation.
        
        Args:
            response: The generated response
            context: The conversation context used to generate the response
            
        Returns:
            AgentResponse: The post-processed response
        """
        # Basic content validation
        if not response.content or not response.content.strip():
            response.add_error("Generated empty response content")
        
        # Check for potential issues
        if len(response.content) < 10:
            response.add_error("Response content is very short")
        
        if len(response.content) > self.model_config.max_tokens * 4:  # Rough estimate
            response.add_error("Response content is unusually long")
        
        return response
    
    def update_model_config(self, new_config: ModelConfig) -> None:
        """
        Update the model configuration and recreate the LLM.
        
        Args:
            new_config: New model configuration
            
        Raises:
            AgentConfigurationError: If new configuration is invalid
        """
        # Validate new configuration
        errors = new_config.validate()
        if errors:
            raise AgentConfigurationError(
                f"Invalid model configuration: {'; '.join(errors)}",
                agent_id=self.agent_id
            )
        
        old_model = self.model_config.model_name
        self.model_config = new_config
        
        try:
            self.llm = self._create_llm()
            self.logger.info(f"Updated model configuration from '{old_model}' to '{new_config.model_name}'")
        except Exception as e:
            # Revert to old configuration if update fails
            self.model_config = ModelConfig(model_name=old_model)  # Simplified revert
            self.llm = self._create_llm()
            raise AgentConfigurationError(
                f"Failed to update model configuration: {str(e)}",
                agent_id=self.agent_id,
                original_error=e
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration details
        """
        return {
            "model_name": self.model_config.model_name,
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            "top_p": self.model_config.top_p,
            "frequency_penalty": self.model_config.frequency_penalty,
            "presence_penalty": self.model_config.presence_penalty,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return (f"LangChainAgent(id='{self.agent_id}', name='{self.name}', "
                f"model='{self.model_config.model_name}')")
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"LangChainAgent("
                f"agent_id='{self.agent_id}', "
                f"name='{self.name}', "
                f"model='{self.model_config.model_name}', "
                f"temperature={self.model_config.temperature}, "
                f"max_tokens={self.model_config.max_tokens})")