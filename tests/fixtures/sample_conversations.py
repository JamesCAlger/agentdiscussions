"""
Sample conversation fixtures for testing.

This module provides predefined conversation scenarios and test data
for integration and end-to-end testing.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.agentic_conversation.models import Message, ConversationState, ConversationStatus


class ConversationFixtures:
    """Collection of sample conversation fixtures for testing."""
    
    @staticmethod
    def create_sample_conversation_1() -> ConversationState:
        """
        Create a sample conversation about remote work.
        
        Returns:
            ConversationState with a complete conversation flow
        """
        state = ConversationState()
        base_time = datetime.now()
        
        messages = [
            Message(
                agent_id="agent_a",
                content="What are the main benefits of remote work for employees?",
                timestamp=base_time,
                token_count=12,
                metadata={"turn": 1, "agent_name": "Researcher"}
            ),
            Message(
                agent_id="agent_b", 
                content="Remote work offers flexibility, better work-life balance, and eliminates commuting. It also allows access to a global talent pool. What challenges do you see?",
                timestamp=base_time + timedelta(seconds=30),
                token_count=35,
                metadata={"turn": 2, "agent_name": "Analyst"}
            ),
            Message(
                agent_id="agent_a",
                content="Key challenges include communication barriers, potential isolation, and difficulty maintaining company culture. How do successful remote companies address these?",
                timestamp=base_time + timedelta(minutes=1),
                token_count=28,
                metadata={"turn": 3, "agent_name": "Researcher"}
            ),
            Message(
                agent_id="agent_b",
                content="They use regular video calls, virtual team building, clear communication protocols, and invest in collaboration tools. Structured check-ins are crucial.",
                timestamp=base_time + timedelta(minutes=1, seconds=45),
                token_count=32,
                metadata={"turn": 4, "agent_name": "Analyst"}
            )
        ]
        
        for message in messages:
            state.add_message(message)
        
        state.status = ConversationStatus.COMPLETED
        return state
    
    @staticmethod
    def create_sample_conversation_2() -> ConversationState:
        """
        Create a sample conversation about AI ethics.
        
        Returns:
            ConversationState with an AI ethics discussion
        """
        state = ConversationState()
        base_time = datetime.now()
        
        messages = [
            Message(
                agent_id="agent_a",
                content="What are the most pressing ethical concerns in AI development today?",
                timestamp=base_time,
                token_count=14,
                metadata={"turn": 1, "topic": "AI Ethics"}
            ),
            Message(
                agent_id="agent_b",
                content="Bias in algorithms, privacy concerns, job displacement, and the need for transparency in AI decision-making are major issues. Which concerns you most?",
                timestamp=base_time + timedelta(seconds=25),
                token_count=31,
                metadata={"turn": 2, "topic": "AI Ethics"}
            ),
            Message(
                agent_id="agent_a",
                content="Algorithmic bias is particularly concerning as it can perpetuate societal inequalities. How can we ensure AI systems are fair and unbiased?",
                timestamp=base_time + timedelta(minutes=1, seconds=10),
                token_count=29,
                metadata={"turn": 3, "topic": "AI Ethics"}
            ),
            Message(
                agent_id="agent_b",
                content="Diverse development teams, comprehensive testing with varied datasets, regular audits, and inclusive design processes are essential. Regulation may also be needed.",
                timestamp=base_time + timedelta(minutes=1, seconds=50),
                token_count=33,
                metadata={"turn": 4, "topic": "AI Ethics"}
            ),
            Message(
                agent_id="agent_a",
                content="Regulation is tricky - we need to balance innovation with protection. What role should governments play in AI governance?",
                timestamp=base_time + timedelta(minutes=2, seconds=30),
                token_count=26,
                metadata={"turn": 5, "topic": "AI Ethics"}
            ),
            Message(
                agent_id="agent_b",
                content="Governments should set ethical frameworks, fund research into AI safety, and ensure accountability without stifling innovation. International cooperation is key.",
                timestamp=base_time + timedelta(minutes=3, seconds=15),
                token_count=30,
                metadata={"turn": 6, "topic": "AI Ethics"}
            )
        ]
        
        for message in messages:
            state.add_message(message)
        
        state.status = ConversationStatus.MAX_TURNS_REACHED
        return state
    
    @staticmethod
    def create_error_scenario_conversation() -> ConversationState:
        """
        Create a conversation that demonstrates error handling.
        
        Returns:
            ConversationState with error scenarios
        """
        state = ConversationState()
        base_time = datetime.now()
        
        messages = [
            Message(
                agent_id="agent_a",
                content="Let's discuss climate change solutions.",
                timestamp=base_time,
                token_count=8,
                metadata={"turn": 1}
            ),
            Message(
                agent_id="agent_b",
                content="Renewable energy, carbon capture, and policy changes are key approaches. What's your priority?",
                timestamp=base_time + timedelta(seconds=20),
                token_count=18,
                metadata={"turn": 2}
            ),
            # Simulate an error scenario
            Message(
                agent_id="agent_a",
                content="I apologize, but I'm experiencing technical difficulties and cannot provide a proper response at this time.",
                timestamp=base_time + timedelta(minutes=1),
                token_count=22,
                metadata={"turn": 3, "error": "API_ERROR", "retry_count": 3}
            )
        ]
        
        for message in messages:
            state.add_message(message)
        
        state.status = ConversationStatus.ERROR
        return state
    
    @staticmethod
    def create_long_conversation() -> ConversationState:
        """
        Create a longer conversation for context window testing.
        
        Returns:
            ConversationState with many messages
        """
        state = ConversationState()
        base_time = datetime.now()
        
        topics = [
            "artificial intelligence", "machine learning", "data science",
            "software engineering", "cloud computing", "cybersecurity",
            "blockchain", "quantum computing", "robotics", "IoT"
        ]
        
        for i, topic in enumerate(topics):
            # Agent A asks about the topic
            message_a = Message(
                agent_id="agent_a",
                content=f"Can you explain the current state and future prospects of {topic}?",
                timestamp=base_time + timedelta(minutes=i*2),
                token_count=15 + len(topic.split()),
                metadata={"turn": i*2 + 1, "topic": topic}
            )
            state.add_message(message_a)
            
            # Agent B responds with detailed information
            response_content = f"{topic.title()} is a rapidly evolving field with significant implications for various industries. " \
                             f"Current developments in {topic} include advanced algorithms, improved efficiency, and broader adoption. " \
                             f"Future prospects for {topic} look promising with continued research and investment driving innovation."
            
            message_b = Message(
                agent_id="agent_b",
                content=response_content,
                timestamp=base_time + timedelta(minutes=i*2 + 1),
                token_count=len(response_content.split()),
                metadata={"turn": i*2 + 2, "topic": topic}
            )
            state.add_message(message_b)
        
        state.status = ConversationStatus.COMPLETED
        return state
    
    @staticmethod
    def get_sample_configurations() -> List[Dict[str, Any]]:
        """
        Get a list of sample configurations for testing different scenarios.
        
        Returns:
            List of configuration dictionaries
        """
        return [
            # Basic configuration
            {
                "name": "basic_config",
                "config": {
                    "agents": {
                        "agent_a": {"name": "Assistant A", "system_prompt": "You are a helpful assistant."},
                        "agent_b": {"name": "Assistant B", "system_prompt": "You are another helpful assistant."}
                    },
                    "model": {"model_name": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 1000},
                    "conversation": {"max_turns": 4, "context_window_strategy": "truncate"},
                    "logging": {"log_level": "INFO", "output_directory": "./logs"}
                }
            },
            # High-performance configuration
            {
                "name": "high_performance_config",
                "config": {
                    "agents": {
                        "agent_a": {"name": "Expert A", "system_prompt": "You are a domain expert with deep knowledge."},
                        "agent_b": {"name": "Expert B", "system_prompt": "You are another domain expert."}
                    },
                    "model": {"model_name": "gpt-4", "temperature": 0.3, "max_tokens": 2000},
                    "conversation": {"max_turns": 10, "context_window_strategy": "sliding", "turn_timeout": 60.0},
                    "logging": {"log_level": "DEBUG", "output_directory": "./detailed_logs", "export_formats": ["json", "csv"]}
                }
            },
            # Minimal configuration
            {
                "name": "minimal_config",
                "config": {
                    "agents": {
                        "agent_a": {"name": "A", "system_prompt": "You are A."},
                        "agent_b": {"name": "B", "system_prompt": "You are B."}
                    },
                    "model": {"model_name": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 500},
                    "conversation": {"max_turns": 2, "context_window_strategy": "truncate"},
                    "logging": {"log_level": "ERROR", "output_directory": "./logs"}
                }
            },
            # Research-focused configuration
            {
                "name": "research_config",
                "config": {
                    "agents": {
                        "agent_a": {
                            "name": "Researcher",
                            "system_prompt": "You are a thorough researcher who asks probing questions and seeks detailed information."
                        },
                        "agent_b": {
                            "name": "Analyst", 
                            "system_prompt": "You are an analytical thinker who synthesizes information and provides insights."
                        }
                    },
                    "model": {"model_name": "gpt-4", "temperature": 0.6, "max_tokens": 1500},
                    "conversation": {
                        "max_turns": 8,
                        "context_window_strategy": "sliding",
                        "initial_prompt": "Let's explore this topic in depth with thorough analysis."
                    },
                    "logging": {"log_level": "INFO", "output_directory": "./research_logs", "save_conversation_history": True}
                }
            }
        ]
    
    @staticmethod
    def get_performance_test_scenarios() -> List[Dict[str, Any]]:
        """
        Get performance test scenarios with different loads and configurations.
        
        Returns:
            List of performance test scenario definitions
        """
        return [
            {
                "name": "light_load",
                "concurrent_conversations": 3,
                "max_turns_per_conversation": 4,
                "expected_max_duration": 30.0,
                "config_name": "basic_config"
            },
            {
                "name": "medium_load", 
                "concurrent_conversations": 5,
                "max_turns_per_conversation": 6,
                "expected_max_duration": 60.0,
                "config_name": "high_performance_config"
            },
            {
                "name": "high_load",
                "concurrent_conversations": 10,
                "max_turns_per_conversation": 2,
                "expected_max_duration": 45.0,
                "config_name": "minimal_config"
            },
            {
                "name": "stress_test",
                "concurrent_conversations": 15,
                "max_turns_per_conversation": 3,
                "expected_max_duration": 90.0,
                "config_name": "basic_config"
            }
        ]
    
    @staticmethod
    def get_error_test_scenarios() -> List[Dict[str, Any]]:
        """
        Get error test scenarios for testing error handling and recovery.
        
        Returns:
            List of error test scenario definitions
        """
        return [
            {
                "name": "network_error",
                "error_type": "ConnectionError",
                "error_message": "Network connection failed",
                "expected_status": ConversationStatus.ERROR,
                "recovery_expected": False
            },
            {
                "name": "timeout_error",
                "error_type": "TimeoutError", 
                "error_message": "Request timed out",
                "expected_status": ConversationStatus.ERROR,
                "recovery_expected": True
            },
            {
                "name": "api_rate_limit",
                "error_type": "RateLimitError",
                "error_message": "API rate limit exceeded",
                "expected_status": ConversationStatus.ERROR,
                "recovery_expected": True
            },
            {
                "name": "invalid_response",
                "error_type": "ValueError",
                "error_message": "Invalid response format",
                "expected_status": ConversationStatus.ERROR,
                "recovery_expected": True
            },
            {
                "name": "context_overflow",
                "error_type": "ContextWindowError",
                "error_message": "Context window exceeded",
                "expected_status": ConversationStatus.COMPLETED,
                "recovery_expected": True
            }
        ]