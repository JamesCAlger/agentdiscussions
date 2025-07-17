#!/usr/bin/env python3
"""
Performance benchmarking script for the Agentic Conversation System.
Tests system performance under various conditions and configurations.
"""

import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the src directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentic_conversation import (
    ConversationOrchestrator,
    SystemConfig, AgentConfig, ModelConfig, ConversationConfig, LoggingConfig,
    ConfigurationError,
    OrchestrationError
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during benchmarking
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Performance benchmarking suite for the conversation system."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[Dict[str, Any]] = []
    
    def create_test_config(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_turns: int = 10,
        context_window_size: int = 4000
    ) -> SystemConfig:
        """Create a test configuration with specified parameters."""
        return SystemConfig(
            agent_a=AgentConfig(
                name="Benchmark Agent A",
                system_prompt="You are a helpful assistant participating in a benchmark test. Keep responses concise and relevant."
            ),
            agent_b=AgentConfig(
                name="Benchmark Agent B", 
                system_prompt="You are a helpful assistant participating in a benchmark test. Engage constructively with your partner."
            ),
            model=ModelConfig(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            ),
            conversation=ConversationConfig(
                max_turns=max_turns,
                initial_prompt="Let's have a brief discussion about the benefits of renewable energy.",
                context_window_strategy="sliding",
                context_window_size=context_window_size,
                turn_timeout=30.0
            ),
            logging=LoggingConfig(
                log_level="WARNING",
                output_directory=str(self.output_dir),
                real_time_display=False,
                export_formats=["json"],
                save_conversation_history=True,
                save_telemetry=True
            )
        )
    
    async def run_single_benchmark(
        self,
        config: SystemConfig,
        test_name: str,
        run_id: int = 1
    ) -> Dict[str, Any]:
        """Run a single benchmark test."""
        start_time = time.time()
        
        try:
            orchestrator = ConversationOrchestrator(config=config)
            
            conversation_start = time.time()
            results = await orchestrator.run_conversation(
                conversation_id=f"{test_name}-run-{run_id}",
                display_progress=False,
                save_results=True
            )
            conversation_end = time.time()
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            conversation_time = conversation_end - conversation_start
            setup_time = conversation_start - start_time
            
            return {
                'test_name': test_name,
                'run_id': run_id,
                'status': 'success',
                'total_time': total_time,
                'setup_time': setup_time,
                'conversation_time': conversation_time,
                'total_turns': results.get('total_turns', 0),
                'total_tokens': results.get('total_tokens', 0),
                'tokens_per_second': results.get('total_tokens', 0) / conversation_time if conversation_time > 0 else 0,
                'turns_per_second': results.get('total_turns', 0) / conversation_time if conversation_time > 0 else 0,
                'avg_time_per_turn': conversation_time / results.get('total_turns', 1),
                'config': {
                    'model_name': config.model.model_name,
                    'temperature': config.model.temperature,
                    'max_tokens': config.model.max_tokens,
                    'max_turns': config.conversation.max_turns,
                    'context_window_size': config.conversation.context_window_size
                }
            }
            
        except Exception as e:
            return {
                'test_name': test_name,
                'run_id': run_id,
                'status': 'failed',
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    async def run_benchmark_suite(
        self,
        config: SystemConfig,
        test_name: str,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """Run multiple benchmark runs and calculate statistics."""
        print(f"🔄 Running {test_name} benchmark ({num_runs} runs)...")
        
        runs = []
        for i in range(num_runs):
            result = await self.run_single_benchmark(config, test_name, i + 1)
            runs.append(result)
            
            if result['status'] == 'success':
                print(f"  Run {i+1}: {result['total_turns']} turns, {result['total_tokens']} tokens, {result['conversation_time']:.2f}s")
            else:
                print(f"  Run {i+1}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Calculate statistics for successful runs
        successful_runs = [r for r in runs if r['status'] == 'success']
        
        if not successful_runs:
            return {
                'test_name': test_name,
                'num_runs': num_runs,
                'successful_runs': 0,
                'success_rate': 0.0,
                'runs': runs
            }
        
        # Calculate performance statistics
        metrics = ['total_time', 'conversation_time', 'total_turns', 'total_tokens', 
                  'tokens_per_second', 'turns_per_second', 'avg_time_per_turn']
        
        stats = {}
        for metric in metrics:
            values = [r[metric] for r in successful_runs]
            stats[metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0
            }
        
        return {
            'test_name': test_name,
            'num_runs': num_runs,
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / num_runs * 100,
            'statistics': stats,
            'runs': runs,
            'config': successful_runs[0]['config'] if successful_runs else None
        }
    
    async def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run comprehensive performance benchmarks."""
        benchmarks = []
        
        print("🚀 Starting Performance Benchmarks")
        print("=" * 50)
        
        # Test 1: Basic performance with different models
        if os.getenv('OPENAI_API_KEY'):
            print("\n📊 Testing GPT-3.5-Turbo Performance...")
            config = self.create_test_config(
                model_name="gpt-3.5-turbo",
                max_turns=10,
                max_tokens=1000
            )
            result = await self.run_benchmark_suite(config, "gpt-3.5-turbo-basic", 3)
            benchmarks.append(result)
        
        # Test 2: Context window scaling
        print("\n📊 Testing Context Window Scaling...")
        for context_size in [2000, 4000, 8000]:
            config = self.create_test_config(
                model_name="gpt-3.5-turbo",
                max_turns=15,
                context_window_size=context_size
            )
            result = await self.run_benchmark_suite(config, f"context-{context_size}", 2)
            benchmarks.append(result)
        
        # Test 3: Turn count scaling
        print("\n📊 Testing Turn Count Scaling...")
        for turn_count in [5, 10, 20]:
            config = self.create_test_config(
                model_name="gpt-3.5-turbo",
                max_turns=turn_count,
                max_tokens=800
            )
            result = await self.run_benchmark_suite(config, f"turns-{turn_count}", 2)
            benchmarks.append(result)
        
        # Test 4: Token limit scaling
        print("\n📊 Testing Token Limit Scaling...")
        for token_limit in [500, 1000, 2000]:
            config = self.create_test_config(
                model_name="gpt-3.5-turbo",
                max_turns=10,
                max_tokens=token_limit
            )
            result = await self.run_benchmark_suite(config, f"tokens-{token_limit}", 2)
            benchmarks.append(result)
        
        # Test 5: Temperature variation
        print("\n📊 Testing Temperature Variation...")
        for temp in [0.3, 0.7, 1.0]:
            config = self.create_test_config(
                model_name="gpt-3.5-turbo",
                temperature=temp,
                max_turns=8
            )
            result = await self.run_benchmark_suite(config, f"temp-{temp}", 2)
            benchmarks.append(result)
        
        return benchmarks
    
    def analyze_benchmarks(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze benchmark results and generate insights."""
        successful_benchmarks = [b for b in benchmarks if b['success_rate'] > 0]
        
        if not successful_benchmarks:
            return {'error': 'No successful benchmarks to analyze'}
        
        # Overall statistics
        total_runs = sum(b['num_runs'] for b in benchmarks)
        total_successful = sum(b['successful_runs'] for b in benchmarks)
        overall_success_rate = total_successful / total_runs * 100 if total_runs > 0 else 0
        
        # Performance insights
        insights = {
            'overall_summary': {
                'total_benchmarks': len(benchmarks),
                'successful_benchmarks': len(successful_benchmarks),
                'total_runs': total_runs,
                'successful_runs': total_successful,
                'overall_success_rate': overall_success_rate
            },
            'performance_insights': [],
            'recommendations': []
        }
        
        # Analyze performance patterns
        for benchmark in successful_benchmarks:
            stats = benchmark['statistics']
            insights['performance_insights'].append({
                'test_name': benchmark['test_name'],
                'avg_conversation_time': stats['conversation_time']['mean'],
                'avg_tokens_per_second': stats['tokens_per_second']['mean'],
                'avg_time_per_turn': stats['avg_time_per_turn']['mean'],
                'consistency': 1 - (stats['conversation_time']['stdev'] / stats['conversation_time']['mean']) if stats['conversation_time']['mean'] > 0 else 0
            })
        
        # Generate recommendations
        if successful_benchmarks:
            fastest_test = min(successful_benchmarks, key=lambda x: x['statistics']['conversation_time']['mean'])
            most_efficient = max(successful_benchmarks, key=lambda x: x['statistics']['tokens_per_second']['mean'])
            
            insights['recommendations'].extend([
                f"Fastest configuration: {fastest_test['test_name']} ({fastest_test['statistics']['conversation_time']['mean']:.2f}s avg)",
                f"Most token-efficient: {most_efficient['test_name']} ({most_efficient['statistics']['tokens_per_second']['mean']:.1f} tokens/s)",
                "Consider using lower token limits for faster responses",
                "Smaller context windows generally provide better performance",
                "Temperature settings have minimal impact on performance"
            ])
        
        return insights
    
    def save_results(self, benchmarks: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Path:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f"performance_benchmark_{timestamp}.json"
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            },
            'analysis': analysis,
            'detailed_results': benchmarks
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results_file


async def main():
    """Main function to run performance benchmarks."""
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("❌ Error: No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.")
        return False
    
    # Create benchmark runner
    output_dir = Path(__file__).parent / "benchmark_results"
    benchmark = PerformanceBenchmark(output_dir)
    
    print("🎯 Agentic Conversation System - Performance Benchmark")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Run benchmarks
        results = await benchmark.run_all_benchmarks()
        
        # Analyze results
        analysis = benchmark.analyze_benchmarks(results)
        
        # Save results
        results_file = benchmark.save_results(results, analysis)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        if 'error' not in analysis:
            summary = analysis['overall_summary']
            print(f"Total benchmarks: {summary['total_benchmarks']}")
            print(f"Successful benchmarks: {summary['successful_benchmarks']}")
            print(f"Overall success rate: {summary['overall_success_rate']:.1f}%")
            print(f"Total execution time: {total_time:.2f} seconds")
            
            print("\n🔍 Performance Insights:")
            for insight in analysis['performance_insights'][:5]:  # Show top 5
                print(f"  {insight['test_name']}: {insight['avg_conversation_time']:.2f}s avg, {insight['avg_tokens_per_second']:.1f} tokens/s")
            
            print("\n💡 Recommendations:")
            for rec in analysis['recommendations'][:3]:  # Show top 3
                print(f"  • {rec}")
        else:
            print(f"❌ Analysis failed: {analysis['error']}")
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return analysis.get('overall_summary', {}).get('overall_success_rate', 0) > 50
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


if __name__ == "__main__":
    import os
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)