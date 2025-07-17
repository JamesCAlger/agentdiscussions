#!/usr/bin/env python3
"""
Example script demonstrating various system capabilities.
This script runs different conversation scenarios and analyzes the results.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentic_conversation import (
    ConversationOrchestrator,
    load_config,
    ConfigurationError,
    OrchestrationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExampleRunner:
    """Runs example conversations and collects results."""
    
    def __init__(self, examples_dir: Path):
        self.examples_dir = examples_dir
        self.results: List[Dict[str, Any]] = []
    
    async def run_example(self, config_file: str, conversation_id: str = None) -> Dict[str, Any]:
        """Run a single example conversation."""
        config_path = self.examples_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Running example: {config_file}")
        
        try:
            # Load configuration
            config = load_config(config_path)
            
            # Create orchestrator
            orchestrator = ConversationOrchestrator(config=config)
            
            # Run conversation
            results = await orchestrator.run_conversation(
                conversation_id=conversation_id or f"example-{config_file.replace('.yaml', '')}",
                display_progress=False,  # Don't display for batch runs
                save_results=True
            )
            
            # Add configuration info to results
            results['config_file'] = config_file
            results['config_path'] = str(config_path)
            
            logger.info(f"Completed {config_file}: {results['total_turns']} turns, {results['total_tokens']} tokens")
            
            return results
            
        except (ConfigurationError, OrchestrationError) as e:
            logger.error(f"Error running {config_file}: {e}")
            return {
                'config_file': config_file,
                'error': str(e),
                'status': 'failed'
            }
    
    async def run_all_examples(self) -> List[Dict[str, Any]]:
        """Run all example configurations."""
        example_configs = [
            'basic-conversation.yaml',
            'research-discussion.yaml',
            'creative-collaboration.yaml',
            'technical-architecture.yaml',
            'educational-tutoring.yaml'
        ]
        
        results = []
        
        for config_file in example_configs:
            try:
                result = await self.run_example(config_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run {config_file}: {e}")
                results.append({
                    'config_file': config_file,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results from multiple conversation runs."""
        successful_runs = [r for r in results if r.get('status') != 'failed']
        failed_runs = [r for r in results if r.get('status') == 'failed']
        
        if not successful_runs:
            return {
                'summary': 'No successful runs to analyze',
                'failed_count': len(failed_runs),
                'errors': [r.get('error') for r in failed_runs]
            }
        
        # Calculate statistics
        total_turns = sum(r.get('total_turns', 0) for r in successful_runs)
        total_tokens = sum(r.get('total_tokens', 0) for r in successful_runs)
        total_duration = sum(r.get('duration_seconds', 0) for r in successful_runs)
        
        avg_turns = total_turns / len(successful_runs) if successful_runs else 0
        avg_tokens = total_tokens / len(successful_runs) if successful_runs else 0
        avg_duration = total_duration / len(successful_runs) if successful_runs else 0
        
        analysis = {
            'summary': {
                'total_runs': len(results),
                'successful_runs': len(successful_runs),
                'failed_runs': len(failed_runs),
                'success_rate': len(successful_runs) / len(results) * 100 if results else 0
            },
            'aggregated_metrics': {
                'total_turns': total_turns,
                'total_tokens': total_tokens,
                'total_duration_seconds': total_duration,
                'average_turns_per_conversation': avg_turns,
                'average_tokens_per_conversation': avg_tokens,
                'average_duration_per_conversation': avg_duration
            },
            'individual_results': []
        }
        
        # Add individual results
        for result in successful_runs:
            analysis['individual_results'].append({
                'config_file': result.get('config_file'),
                'conversation_id': result.get('conversation_id'),
                'turns': result.get('total_turns'),
                'tokens': result.get('total_tokens'),
                'duration': result.get('duration_seconds'),
                'status': result.get('status')
            })
        
        # Add error information
        if failed_runs:
            analysis['errors'] = [
                {
                    'config_file': r.get('config_file'),
                    'error': r.get('error')
                }
                for r in failed_runs
            ]
        
        return analysis


async def main():
    """Main function to run examples and analyze results."""
    # Get the examples directory
    script_dir = Path(__file__).parent
    examples_dir = script_dir.parent
    
    # Create runner
    runner = ExampleRunner(examples_dir)
    
    print("🚀 Starting example conversation runs...")
    print(f"Examples directory: {examples_dir}")
    print()
    
    # Run all examples
    start_time = datetime.now()
    results = await runner.run_all_examples()
    end_time = datetime.now()
    
    # Analyze results
    analysis = runner.analyze_results(results)
    
    # Print summary
    print("📊 Example Run Summary")
    print("=" * 50)
    print(f"Total runs: {analysis['summary']['total_runs']}")
    print(f"Successful: {analysis['summary']['successful_runs']}")
    print(f"Failed: {analysis['summary']['failed_runs']}")
    print(f"Success rate: {analysis['summary']['success_rate']:.1f}%")
    print(f"Total execution time: {(end_time - start_time).total_seconds():.2f} seconds")
    print()
    
    if analysis['summary']['successful_runs'] > 0:
        metrics = analysis['aggregated_metrics']
        print("📈 Aggregated Metrics")
        print("=" * 50)
        print(f"Total conversation turns: {metrics['total_turns']}")
        print(f"Total tokens processed: {metrics['total_tokens']:,}")
        print(f"Total conversation time: {metrics['total_duration_seconds']:.2f} seconds")
        print(f"Average turns per conversation: {metrics['average_turns_per_conversation']:.1f}")
        print(f"Average tokens per conversation: {metrics['average_tokens_per_conversation']:,.0f}")
        print(f"Average duration per conversation: {metrics['average_duration_per_conversation']:.2f} seconds")
        print()
        
        print("🔍 Individual Results")
        print("=" * 50)
        for result in analysis['individual_results']:
            print(f"{result['config_file']:<30} | {result['turns']:>3} turns | {result['tokens']:>6,} tokens | {result['duration']:>6.1f}s")
        print()
    
    if 'errors' in analysis:
        print("❌ Errors")
        print("=" * 50)
        for error in analysis['errors']:
            print(f"{error['config_file']}: {error['error']}")
        print()
    
    # Save detailed results
    results_file = script_dir / f"example_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': (end_time - start_time).total_seconds(),
            'analysis': analysis,
            'raw_results': results
        }, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    return analysis['summary']['success_rate'] == 100.0


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Example run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)