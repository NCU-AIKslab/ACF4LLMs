#!/usr/bin/env python3
"""
Run All Experiments

Comprehensive experiment runner for all research questions (RQ1-RQ4).
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_compression.optimization.agent_driven import run_rq2_experiment
from agentic_compression.optimization.dynamic_vs_static import run_rq1_experiment
from agentic_compression.optimization.resource_adaptation import run_rq4_experiment
from agentic_compression.optimization.weighting import run_rq3_experiment


async def run_rq1():
    """Run RQ1: Dynamic vs Static Compression"""
    print("\n" + "=" * 80)
    print("📈 RQ1: Dynamic vs Static Compression Comparison")
    print("=" * 80)

    results = await run_rq1_experiment(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", accuracy_threshold=0.93, carbon_budget=5.0, max_iterations=20
    )

    comparison = results["comparison"]
    print(f"\n✅ RQ1 Complete: {results['conclusion']}")
    print("\n💡 Key Findings:")
    for finding in comparison.get("key_findings", []):
        print(f"   • {finding}")

    return results


async def run_rq2():
    """Run RQ2: Agent-Driven Optimization"""
    print("\n" + "=" * 80)
    print("📊 RQ2: Agent-Driven Pruning and Quantization")
    print("=" * 80)

    results = await run_rq2_experiment(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", accuracy_threshold=0.93, carbon_budget=5.0
    )

    print(f"\n✅ RQ2 Complete: {results['pareto_frontier_size']} Pareto-optimal solutions found")
    print("\n💡 Key Findings:")
    for finding in results.get("key_findings", []):
        print(f"   • {finding}")

    return results


async def run_rq3():
    """Run RQ3: Weighting Scheme Analysis"""
    print("\n" + "=" * 80)
    print("⚖️  RQ3: Weighting Scheme Analysis")
    print("=" * 80)

    results = await run_rq3_experiment(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", carbon_budget=5.0)
    analysis = results["analysis"]

    print(
        f"\n✅ RQ3 Complete: Tested {len(analysis.get('weight_schemes_tested', []))} weight schemes "
        f"across {analysis.get('total_solutions_explored', 0)} solutions"
    )
    print("\n💡 Key Findings:")
    for finding in analysis.get("key_findings", []):
        print(f"   • {finding}")

    return results


async def run_rq4():
    """Run RQ4: Resource-Constrained Adaptation"""
    print("\n" + "=" * 80)
    print("🌍 RQ4: Resource-Constrained Adaptation")
    print("=" * 80)

    results = await run_rq4_experiment(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", accuracy_threshold=0.85)
    analysis = results["adaptation_analysis"]

    print(
        f"\n✅ RQ4 Complete: Tested {len(analysis.get('environment_results', {}))} environments "
        f"with conclusion: {results['conclusion']}"
    )
    print("\n💡 Key Findings:")
    for finding in analysis.get("key_findings", []):
        print(f"   • {finding}")

    return results


async def run_all_experiments():
    """Run complete experiment suite"""
    print("\n" + "=" * 80)
    print("🚀 Agentic Carbon-Efficient LLM Compression Framework")
    print("   Complete Experiment Suite")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rq1_results = await run_rq1()
    rq2_results = await run_rq2()
    rq3_results = await run_rq3()
    rq4_results = await run_rq4()

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"experiment_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "rq1": rq1_results,
                "rq2": rq2_results,
                "rq3": rq3_results,
                "rq4": rq4_results,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n💾 Results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("✨ Experiment Suite Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
