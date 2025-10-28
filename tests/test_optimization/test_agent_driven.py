"""
Tests for RQ2: Agent-Driven Optimization.
"""

import pytest

from agentic_compression.optimization.agent_driven import (
    AgentDrivenOptimization,
    run_rq2_experiment,
)


class TestAgentDrivenOptimization:
    """Test Agent-Driven Optimization (RQ2)"""

    @pytest.mark.asyncio
    async def test_optimization_initialization(self):
        """Test initialization of optimization"""
        optimizer = AgentDrivenOptimization()

        assert optimizer.solutions == []
        assert optimizer.pareto_frontier == []
        assert optimizer.carbon_used == 0.0

    @pytest.mark.asyncio
    async def test_run_optimization_basic(self):
        """Test basic optimization run"""
        optimizer = AgentDrivenOptimization()

        results = await optimizer.run_optimization(
            model="google/gemma3-270m",
            accuracy_threshold=0.90,
            carbon_budget=2.0,  # Small budget for quick test
            max_iterations=3,  # Few iterations
        )

        # Should have some solutions
        assert len(optimizer.solutions) > 0

        # Should have results
        assert "pareto_frontier_size" in results
        assert "carbon_impact_analysis" in results
        assert "key_findings" in results

    @pytest.mark.asyncio
    async def test_optimization_respects_carbon_budget(self):
        """Test that optimization respects carbon budget"""
        optimizer = AgentDrivenOptimization()

        carbon_budget = 1.0  # Very small budget

        results = await optimizer.run_optimization(
            model="google/gemma3-270m",
            accuracy_threshold=0.90,
            carbon_budget=carbon_budget,
            max_iterations=10,
        )

        # Should not exceed carbon budget
        assert optimizer.carbon_used <= carbon_budget * 1.1  # 10% tolerance

    @pytest.mark.asyncio
    async def test_optimization_generates_pareto_frontier(self):
        """Test that optimization generates Pareto frontier"""
        optimizer = AgentDrivenOptimization()

        results = await optimizer.run_optimization(
            model="google/gemma3-270m",
            accuracy_threshold=0.85,
            carbon_budget=3.0,
            max_iterations=5,
        )

        # Should have Pareto frontier
        assert len(optimizer.pareto_frontier) > 0
        assert results["pareto_frontier_size"] == len(optimizer.pareto_frontier)


class TestRQ2Experiment:
    """Test complete RQ2 experiment"""

    @pytest.mark.asyncio
    async def test_run_rq2_experiment_complete(self):
        """Test running complete RQ2 experiment"""
        results = await run_rq2_experiment(
            model="google/gemma3-270m", accuracy_threshold=0.90, carbon_budget=2.0
        )

        # Should have all required fields
        assert "experiment" in results
        assert "model" in results
        assert "parameters" in results
        assert "optimization_results" in results
        assert "conclusion" in results

        # Check experiment name
        assert "RQ2" in results["experiment"]

    @pytest.mark.asyncio
    async def test_rq2_experiment_different_models(self):
        """Test RQ2 with different models"""
        models = ["google/gemma3-270m", "google/gemma-12b"]

        for model in models:
            results = await run_rq2_experiment(
                model=model, accuracy_threshold=0.85, carbon_budget=1.5
            )

            assert results["model"] == model
            assert "optimization_results" in results

    @pytest.mark.asyncio
    async def test_rq2_carbon_impact_analysis(self):
        """Test carbon impact analysis in RQ2"""
        results = await run_rq2_experiment(
            model="google/gemma3-270m", accuracy_threshold=0.90, carbon_budget=2.0
        )

        impact = results["optimization_results"]["carbon_impact_analysis"]

        # Should have impact metrics
        assert "carbon_reduction_percent" in impact or "best_carbon_config" in impact


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
