#!/usr/bin/env python3
"""
Simple Optimization Example

Demonstrates basic usage of the agentic compression framework.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_compression.graph.workflow import run_compression_optimization
from agentic_compression.optimization.agent_driven import run_rq2_experiment


async def example_1_workflow():
    """Example 1: Using the LangGraph workflow"""
    print("\n" + "=" * 60)
    print("Example 1: LangGraph Workflow Optimization")
    print("=" * 60)

    results = await run_compression_optimization(
        objective="Compress for edge deployment with minimal carbon footprint",
        carbon_budget=5.0,
        max_iterations=5,
        accuracy_threshold=0.90,
    )

    print("\n✅ Optimization Complete!")
    print(f"   Total Solutions: {results['total_solutions']}")
    print(f"   Pareto Optimal: {results['pareto_optimal_count']}")
    print(f"   Carbon Used: {results['carbon_used']:.4f} kg / {results['carbon_budget']} kg")

    if results["best_solution"]:
        best = results["best_solution"]
        print("\n🏆 Best Solution:")
        print(f"   Accuracy: {best['average_accuracy']:.3f}")
        print(f"   CO₂: {best['co2_kg']:.4f} kg")
        print(f"   Memory: {best['memory_gb']:.1f} GB")


async def example_2_rq2():
    """Example 2: Running RQ2 experiment directly"""
    print("\n" + "=" * 60)
    print("Example 2: RQ2 Agent-Driven Optimization")
    print("=" * 60)

    results = await run_rq2_experiment(
        model="google/gemma-12b", accuracy_threshold=0.93, carbon_budget=5.0
    )

    print("\n✅ RQ2 Experiment Complete!")
    print(f"   Pareto Frontier Size: {results['pareto_frontier_size']}")

    print("\n📊 Carbon Impact Analysis:")
    impact = results["carbon_impact_analysis"]
    print(
        f"   Accuracy Range: {impact['accuracy_range']['min']:.3f} - {impact['accuracy_range']['max']:.3f}"
    )
    print(
        f"   Carbon Range: {impact['carbon_range']['min']:.4f} - {impact['carbon_range']['max']:.4f} kg"
    )
    print(f"   Reduction from Baseline: {impact['carbon_range']['reduction_from_baseline']}")

    print("\n💡 Key Findings:")
    for finding in results["key_findings"]:
        print(f"   • {finding}")


async def main():
    """Run all examples"""
    print("\n🚀 Agentic Compression Framework - Simple Examples")

    # Example 1: LangGraph workflow
    await example_1_workflow()

    # Example 2: RQ2 experiment
    await example_2_rq2()

    print("\n" + "=" * 60)
    print("✨ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
