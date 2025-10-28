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


async def run_rq2():
    """Run RQ2: Agent-Driven Optimization"""
    print("\n" + "=" * 80)
    print("📊 RQ2: Agent-Driven Pruning and Quantization")
    print("=" * 80)

    results = await run_rq2_experiment(
        model="google/gemma-12b", accuracy_threshold=0.93, carbon_budget=5.0
    )

    print(f"\n✅ RQ2 Complete: {results['pareto_frontier_size']} Pareto-optimal solutions found")
    print("\n💡 Key Findings:")
    for finding in results["key_findings"]:
        print(f"   • {finding}")

    return results


async def run_all_experiments():
    """Run complete experiment suite"""
    print("\n" + "=" * 80)
    print("🚀 Agentic Carbon-Efficient LLM Compression Framework")
    print("   Complete Experiment Suite")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # RQ2 is fully implemented
    rq2_results = await run_rq2()

    # TODO: Implement RQ1, RQ3, RQ4 in future iterations
    print("\n⚠️  RQ1, RQ3, RQ4 implementations are placeholders - to be completed")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"experiment_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "rq2": rq2_results,
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
