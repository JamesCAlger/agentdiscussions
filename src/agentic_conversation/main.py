#!/usr/bin/env python3
"""
Command-line interface for the Agentic Conversation System.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from dotenv import load_dotenv

from .config import ConfigurationLoader, ConfigurationError
from .orchestrator import ConversationOrchestrator, OrchestrationError
from .models import SystemConfig

load_dotenv()


def setup_logging(log_level: str, verbose: bool = False) -> None:
    """Set up logging configuration for the CLI."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if verbose else '%(levelname)s: %(message)s'
    
    logging.basicConfig(level=numeric_level, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    if numeric_level > logging.DEBUG:
        for logger_name in ['urllib3', 'httpx', 'openai', 'anthropic']:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def validate_file_path(ctx, param, value: Optional[str]) -> Optional[Path]:
    """Validate that a file path exists and is readable."""
    if value is None:
        return None
    
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"File does not exist: {value}")
    if not path.is_file():
        raise click.BadParameter(f"Path is not a file: {value}")
    if not os.access(path, os.R_OK):
        raise click.BadParameter(f"File is not readable: {value}")
    
    return path


def validate_directory_path(ctx, param, value: Optional[str]) -> Optional[Path]:
    """Validate that a directory path exists and is writable."""
    if value is None:
        return None
    
    path = Path(value)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise click.BadParameter(f"Cannot create directory {value}: {e}")
    
    if not path.is_dir():
        raise click.BadParameter(f"Path is not a directory: {value}")
    if not os.access(path, os.W_OK):
        raise click.BadParameter(f"Directory is not writable: {value}")
    
    return path


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
              callback=validate_file_path, help='Path to configuration file (YAML format)')
@click.option('--log-level', '-l', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False),
              default='INFO', help='Set the logging level')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output with detailed logging')
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True, writable=True),
              callback=validate_directory_path, help='Output directory for logs and results')
@click.pass_context
def cli(ctx, config: Optional[Path], log_level: str, verbose: bool, output_dir: Optional[Path]):
    """Agentic Conversation System - CLI for managing AI agent conversations."""
    ctx.ensure_object(dict)
    setup_logging(log_level, verbose)
    
    ctx.obj.update({
        'config_path': config,
        'log_level': log_level,
        'verbose': verbose,
        'output_dir': output_dir
    })
    
    if config:
        try:
            loader = ConfigurationLoader()
            system_config = loader.load_from_file(config)
            ctx.obj['system_config'] = system_config
            
            if output_dir:
                system_config.logging.output_directory = str(output_dir)
                
        except ConfigurationError as e:
            click.echo(f"Error loading configuration: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.option('--conversation-id', help='Unique identifier for this conversation')
@click.option('--no-display', is_flag=True, help='Disable real-time conversation display')
@click.option('--no-save', is_flag=True, help='Do not save conversation results to files')
@click.pass_context
def run(ctx, conversation_id: Optional[str], no_display: bool, no_save: bool):
    """Start a new conversation between two AI agents."""
    config_path = ctx.obj.get('config_path')
    system_config = ctx.obj.get('system_config')
    
    if not config_path and not system_config:
        click.echo("Error: Configuration file is required. Use --config option.", err=True)
        sys.exit(1)
    
    try:
        orchestrator = ConversationOrchestrator(config=system_config) if system_config else ConversationOrchestrator(config_path=config_path)
        
        if no_display:
            orchestrator.config.logging.real_time_display = False
        
        click.echo("Starting conversation...")
        
        results = asyncio.run(orchestrator.run_conversation(
            conversation_id=conversation_id,
            display_progress=not no_display,
            save_results=not no_save
        ))
        
        click.echo(f"\nConversation completed successfully!")
        click.echo(f"Conversation ID: {results['conversation_id']}")
        click.echo(f"Status: {results['status']}")
        click.echo(f"Total turns: {results['total_turns']}")
        click.echo(f"Total tokens: {results['total_tokens']}")
        click.echo(f"Duration: {results['duration_seconds']:.2f} seconds")
        
        if not no_save:
            click.echo(f"Results saved to: {orchestrator.config.logging.output_directory}")
        
    except (OrchestrationError, ConfigurationError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nConversation interrupted by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('--detailed', is_flag=True, help='Show detailed validation information')
@click.pass_context
def validate(ctx, config_file: str, detailed: bool):
    """Validate a configuration file without running a conversation."""
    try:
        loader = ConfigurationLoader()
        validation_result = loader.validate_config_file(config_file)
        
        if validation_result['is_valid']:
            click.echo(f"✓ Configuration file '{config_file}' is valid")
            
            if detailed:
                system_config = loader.load_from_file(config_file)
                click.echo("\nConfiguration Details:")
                click.echo(f"  Model: {system_config.model.model_name}")
                click.echo(f"  Temperature: {system_config.model.temperature}")
                click.echo(f"  Max Tokens: {system_config.model.max_tokens}")
                click.echo(f"  Max Turns: {system_config.conversation.max_turns}")
                click.echo(f"  Context Strategy: {system_config.conversation.context_window_strategy}")
                click.echo(f"  Context Size: {system_config.conversation.context_window_size}")
                
                click.echo(f"\nAgent A:")
                click.echo(f"  Name: {system_config.agent_a.name}")
                click.echo(f"  System Prompt: {system_config.agent_a.system_prompt[:100]}...")
                
                click.echo(f"\nAgent B:")
                click.echo(f"  Name: {system_config.agent_b.name}")
                click.echo(f"  System Prompt: {system_config.agent_b.system_prompt[:100]}...")
                
                if system_config.conversation.initial_prompt:
                    click.echo(f"\nInitial Prompt: {system_config.conversation.initial_prompt[:100]}...")
        else:
            click.echo(f"✗ Configuration file '{config_file}' is invalid", err=True)
            
            if 'errors' in validation_result:
                click.echo("\nErrors found:")
                for error in validation_result['errors']:
                    click.echo(f"  - {error}", err=True)
            
            if 'warnings' in validation_result:
                click.echo("\nWarnings:")
                for warning in validation_result['warnings']:
                    click.echo(f"  - {warning}")
            
            sys.exit(1)
            
    except ConfigurationError as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('log_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.option('--conversation-id', help='Filter logs by specific conversation ID')
@click.option('--format', 'output_format', type=click.Choice(['summary', 'detailed', 'json'], case_sensitive=False),
              default='summary', help='Output format for log analysis')
@click.option('--limit', '-n', type=int, default=10, help='Limit number of conversations to analyze')
@click.pass_context
def analyze(ctx, log_directory: str, conversation_id: Optional[str], output_format: str, limit: int):
    """Analyze and display logs from previous conversation runs."""
    log_dir = Path(log_directory)
    
    try:
        log_files = list(log_dir.glob("*.json"))
        if not log_files:
            click.echo(f"No log files found in directory: {log_directory}", err=True)
            sys.exit(1)
        
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        log_files = log_files[:limit]
        
        conversations = []
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    
                    if conversation_id:
                        file_conv_id = log_data.get('conversation_id', '')
                        if conversation_id not in file_conv_id:
                            continue
                    
                    conversations.append(log_data)
            except Exception as e:
                if output_format != 'json':
                    click.echo(f"Warning: Could not read {log_file}: {e}", err=True)
                continue
        
        if not conversations:
            if conversation_id:
                click.echo(f"No conversation logs found for conversation ID: {conversation_id}", err=True)
            else:
                click.echo("No valid conversation logs found.", err=True)
            sys.exit(1)
        
        if output_format == 'json':
            click.echo(json.dumps(conversations, indent=2))
        elif output_format == 'detailed':
            click.echo(f"Analyzing {len(log_files)} conversation log(s)...")
            _display_detailed_analysis(conversations)
        else:  # summary
            click.echo(f"Analyzing {len(log_files)} conversation log(s)...")
            _display_summary_analysis(conversations)
            
    except Exception as e:
        click.echo(f"Error analyzing logs: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('config_template', type=click.Choice(['basic', 'research', 'creative', 'technical']))
@click.argument('output_file', type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.option('--model', default='gpt-4', help='Model to use in the configuration')
@click.option('--max-turns', type=int, default=10, help='Maximum number of conversation turns')
@click.pass_context
def create_config(ctx, config_template: str, output_file: str, model: str, max_turns: int):
    """Create a new configuration file from a template."""
    templates = {
        'basic': {
            'agents': {
                'agent_a': {'name': 'Assistant A', 'system_prompt': 'You are a helpful assistant who engages in thoughtful conversation.'},
                'agent_b': {'name': 'Assistant B', 'system_prompt': 'You are a helpful assistant who provides different perspectives and asks clarifying questions.'}
            },
            'conversation': {'initial_prompt': 'Let\'s have a discussion about an interesting topic.'}
        },
        'research': {
            'agents': {
                'agent_a': {'name': 'Researcher', 'system_prompt': 'You are a thorough researcher who asks probing questions, seeks evidence, and analyzes information critically.'},
                'agent_b': {'name': 'Analyst', 'system_prompt': 'You are an analytical thinker who synthesizes information, identifies patterns, and draws conclusions.'}
            },
            'conversation': {'initial_prompt': 'Let\'s research and analyze the impact of artificial intelligence on modern education systems.'}
        },
        'creative': {
            'agents': {
                'agent_a': {'name': 'Creative Writer', 'system_prompt': 'You are a creative writer who thinks outside the box and generates innovative ideas.'},
                'agent_b': {'name': 'Story Developer', 'system_prompt': 'You are a story developer who builds upon creative ideas and adds structure and depth.'}
            },
            'conversation': {'initial_prompt': 'Let\'s collaborate on creating an original story concept that explores themes of technology and human connection.'}
        },
        'technical': {
            'agents': {
                'agent_a': {'name': 'Software Architect', 'system_prompt': 'You are a software architect who designs systems and considers scalability and maintainability.'},
                'agent_b': {'name': 'Implementation Engineer', 'system_prompt': 'You are an implementation engineer who focuses on practical coding solutions and performance optimization.'}
            },
            'conversation': {'initial_prompt': 'Let\'s design a scalable microservices architecture for a high-traffic e-commerce platform.'}
        }
    }
    
    try:
        template_config = templates[config_template]
        
        config = {
            'agents': template_config['agents'],
            'model': {'model_name': model, 'temperature': 0.7, 'max_tokens': 2000},
            'conversation': {
                'max_turns': max_turns,
                'initial_prompt': template_config['conversation']['initial_prompt'],
                'context_window_strategy': 'sliding',
                'context_window_size': 8000,
                'turn_timeout': 30.0
            },
            'logging': {
                'log_level': 'INFO',
                'output_directory': './logs',
                'real_time_display': True,
                'export_formats': ['json'],
                'save_conversation_history': True,
                'save_telemetry': True
            }
        }
        
        import yaml
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        click.echo(f"✓ Created {config_template} configuration: {output_file}")
        click.echo(f"  Model: {model}")
        click.echo(f"  Max turns: {max_turns}")
        click.echo(f"  Agents: {template_config['agents']['agent_a']['name']} & {template_config['agents']['agent_b']['name']}")
        
    except Exception as e:
        click.echo(f"Error creating configuration: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Display information about the current configuration and system status."""
    config_path = ctx.obj.get('config_path')
    system_config = ctx.obj.get('system_config')
    
    click.echo("Agentic Conversation System - Information")
    click.echo("=" * 50)
    
    if config_path:
        click.echo(f"Configuration File: {config_path}")
    else:
        click.echo("Configuration File: Not specified")
    
    if system_config:
        click.echo("Configuration Status: Loaded")
        click.echo(f"Model: {system_config.model.model_name}")
        click.echo(f"Agent A: {system_config.agent_a.name}")
        click.echo(f"Agent B: {system_config.agent_b.name}")
        click.echo(f"Max Turns: {system_config.conversation.max_turns}")
        click.echo(f"Output Directory: {system_config.logging.output_directory}")
    else:
        click.echo("Configuration Status: Not loaded")
    
    click.echo(f"\nSystem Information:")
    click.echo(f"Python Version: {sys.version.split()[0]}")
    click.echo(f"Log Level: {ctx.obj.get('log_level', 'INFO')}")
    click.echo(f"Verbose Mode: {ctx.obj.get('verbose', False)}")
    
    try:
        import langchain
        click.echo(f"LangChain Version: {langchain.__version__}")
    except (ImportError, AttributeError):
        click.echo("LangChain: Not available or version unknown")
    
    try:
        import langgraph
        version = getattr(langgraph, '__version__', 'unknown')
        click.echo(f"LangGraph Version: {version}")
    except ImportError:
        click.echo("LangGraph: Not available")


def _display_summary_analysis(conversations: List[Dict[str, Any]]) -> None:
    """Display summary analysis of conversations."""
    click.echo(f"\n{'='*60}")
    click.echo("CONVERSATION ANALYSIS SUMMARY")
    click.echo(f"{'='*60}")
    
    total_conversations = len(conversations)
    total_turns = sum(conv.get('total_turns', 0) for conv in conversations)
    total_tokens = sum(conv.get('total_tokens', 0) for conv in conversations)
    total_duration = sum(conv.get('duration_seconds', 0) for conv in conversations)
    
    click.echo(f"Total Conversations: {total_conversations}")
    click.echo(f"Total Turns: {total_turns}")
    click.echo(f"Total Tokens: {total_tokens:,}")
    click.echo(f"Total Duration: {total_duration:.2f} seconds")
    
    if total_conversations > 0:
        avg_turns = total_turns / total_conversations
        avg_tokens = total_tokens / total_conversations
        avg_duration = total_duration / total_conversations
        
        click.echo(f"\nAverages per Conversation:")
        click.echo(f"  Turns: {avg_turns:.1f}")
        click.echo(f"  Tokens: {avg_tokens:,.0f}")
        click.echo(f"  Duration: {avg_duration:.2f} seconds")
    
    statuses = {}
    for conv in conversations:
        status = conv.get('status', 'unknown')
        statuses[status] = statuses.get(status, 0) + 1
    
    click.echo(f"\nStatus Distribution:")
    for status, count in statuses.items():
        percentage = (count / total_conversations) * 100
        click.echo(f"  {status}: {count} ({percentage:.1f}%)")
    
    click.echo(f"\nRecent Conversations:")
    for i, conv in enumerate(conversations[:5]):
        conv_id = conv.get('conversation_id', 'unknown')
        status = conv.get('status', 'unknown')
        turns = conv.get('total_turns', 0)
        tokens = conv.get('total_tokens', 0)
        click.echo(f"  {i+1}. {conv_id} - {status} - {turns} turns, {tokens:,} tokens")


def _display_detailed_analysis(conversations: List[Dict[str, Any]]) -> None:
    """Display detailed analysis of conversations."""
    for i, conv in enumerate(conversations):
        click.echo(f"\n{'='*60}")
        click.echo(f"CONVERSATION {i+1}: {conv.get('conversation_id', 'unknown')}")
        click.echo(f"{'='*60}")
        
        click.echo(f"Status: {conv.get('status', 'unknown')}")
        click.echo(f"Turns: {conv.get('total_turns', 0)}")
        click.echo(f"Messages: {conv.get('total_messages', 0)}")
        click.echo(f"Tokens: {conv.get('total_tokens', 0):,}")
        click.echo(f"Duration: {conv.get('duration_seconds', 0):.2f} seconds")
        
        agent_info = conv.get('agent_info', {})
        if agent_info:
            click.echo(f"\nAgents:")
            for agent_id, info in agent_info.items():
                name = info.get('name', 'Unknown')
                click.echo(f"  {agent_id}: {name}")
        
        config = conv.get('configuration', {})
        if config:
            model_config = config.get('model', {})
            conv_config = config.get('conversation', {})
            
            click.echo(f"\nConfiguration:")
            click.echo(f"  Model: {model_config.get('model_name', 'unknown')}")
            click.echo(f"  Temperature: {model_config.get('temperature', 'unknown')}")
            click.echo(f"  Max Turns: {conv_config.get('max_turns', 'unknown')}")
            click.echo(f"  Context Strategy: {conv_config.get('context_window_strategy', 'unknown')}")
        
        telemetry = conv.get('telemetry', {})
        if telemetry:
            agent_metrics = telemetry.get('agent_metrics', {})
            if agent_metrics:
                click.echo(f"\nAgent Performance:")
                for agent_id, metrics in agent_metrics.items():
                    avg_response_time = metrics.get('average_response_time', 0)
                    total_tokens = metrics.get('total_tokens', 0)
                    error_count = len(metrics.get('errors', []))
                    click.echo(f"  {agent_id}: {avg_response_time:.2f}s avg, {total_tokens} tokens, {error_count} errors")


def main():
    """Main entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()