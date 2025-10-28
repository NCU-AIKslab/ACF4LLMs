"""
Tests for Pareto frontier algorithms.
"""

import pytest

from agentic_compression.core.metrics import EvaluationMetrics, ParetoSolution
from agentic_compression.core.pareto import (
    calculate_crowding_distance,
    calculate_frontier_diversity,
    compute_composite_score,
    compute_pareto_frontier,
    dominates,
    select_best_solution,
    weighted_dominates,
)


class TestDominance:
    """Test dominance checking functions"""

    def test_dominates_strictly_better(self):
        """Test when one solution strictly dominates another"""
        # Solution A: better in all objectives
        metrics_a = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},  # Higher is better
            latency_ms=30.0,  # Lower is better
            memory_gb=4.0,  # Lower is better
            energy_kwh=0.02,  # Lower is better
            co2_kg=0.01,  # Lower is better
        )

        # Solution B: worse in all objectives
        metrics_b = EvaluationMetrics(
            accuracy={"gsm8k": 0.90},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        assert dominates(metrics_a, metrics_b)
        assert not dominates(metrics_b, metrics_a)

    def test_dominates_equal_solutions(self):
        """Test when solutions are equal"""
        metrics_a = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        metrics_b = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        # Equal solutions don't dominate each other
        assert not dominates(metrics_a, metrics_b)
        assert not dominates(metrics_b, metrics_a)

    def test_dominates_pareto_trade_off(self):
        """Test when solutions have trade-offs (neither dominates)"""
        # Solution A: high accuracy, high carbon
        metrics_a = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        # Solution B: lower accuracy, lower carbon
        metrics_b = EvaluationMetrics(
            accuracy={"gsm8k": 0.90},
            latency_ms=30.0,
            memory_gb=4.0,
            energy_kwh=0.02,
            co2_kg=0.01,
        )

        # Neither dominates (trade-off situation)
        assert not dominates(metrics_a, metrics_b)
        assert not dominates(metrics_b, metrics_a)


class TestWeightedDominance:
    """Test weighted dominance functions"""

    def test_weighted_dominates_accuracy_focused(self):
        """Test weighted dominance with accuracy-focused weights"""
        weights = {
            "accuracy": 2.0,
            "latency": -0.3,
            "memory": -0.3,
            "energy": -0.3,
            "co2": -0.3,
        }

        # Solution A: high accuracy
        metrics_a = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        # Solution B: lower accuracy, better resources
        metrics_b = EvaluationMetrics(
            accuracy={"gsm8k": 0.90},
            latency_ms=30.0,
            memory_gb=4.0,
            energy_kwh=0.02,
            co2_kg=0.01,
        )

        # With accuracy-focused weights, A should dominate B
        assert weighted_dominates(metrics_a, metrics_b, weights)

    def test_weighted_dominates_carbon_focused(self):
        """Test weighted dominance with carbon-focused weights"""
        weights = {
            "accuracy": 0.5,
            "latency": -0.5,
            "memory": -0.5,
            "energy": -1.0,
            "co2": -2.0,
        }

        # Solution A: high accuracy, high carbon
        metrics_a = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        # Solution B: lower accuracy, lower carbon
        metrics_b = EvaluationMetrics(
            accuracy={"gsm8k": 0.90},
            latency_ms=30.0,
            memory_gb=4.0,
            energy_kwh=0.02,
            co2_kg=0.01,
        )

        # With carbon-focused weights, B should dominate A
        assert weighted_dominates(metrics_b, metrics_a, weights)


class TestComputeParetoFrontier:
    """Test Pareto frontier computation"""

    def test_compute_pareto_frontier_basic(self):
        """Test basic Pareto frontier computation"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.90},
                    latency_ms=30.0,
                    memory_gb=4.0,
                    energy_kwh=0.02,
                    co2_kg=0.01,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.85},  # Dominated
                    latency_ms=60.0,
                    memory_gb=10.0,
                    energy_kwh=0.08,
                    co2_kg=0.03,
                )
            ),
        ]

        frontier = compute_pareto_frontier(solutions)

        # Should have 2 Pareto-optimal solutions (first two)
        assert len(frontier) == 2

        # All frontier solutions should be marked as Pareto-optimal
        for sol in frontier:
            assert sol.is_pareto_optimal

    def test_compute_pareto_frontier_all_optimal(self):
        """Test when all solutions are Pareto-optimal"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.90},
                    latency_ms=30.0,
                    memory_gb=6.0,
                    energy_kwh=0.03,
                    co2_kg=0.015,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.85},
                    latency_ms=20.0,
                    memory_gb=4.0,
                    energy_kwh=0.02,
                    co2_kg=0.01,
                )
            ),
        ]

        frontier = compute_pareto_frontier(solutions)

        # All three should be on the frontier (trade-offs)
        assert len(frontier) == 3

    def test_compute_pareto_frontier_single_solution(self):
        """Test Pareto frontier with single solution"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            )
        ]

        frontier = compute_pareto_frontier(solutions)

        assert len(frontier) == 1
        assert frontier[0].is_pareto_optimal


class TestCrowdingDistance:
    """Test crowding distance calculation"""

    def test_calculate_crowding_distance(self):
        """Test crowding distance calculation"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                ),
                is_pareto_optimal=True,
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.90},
                    latency_ms=40.0,
                    memory_gb=6.0,
                    energy_kwh=0.04,
                    co2_kg=0.015,
                ),
                is_pareto_optimal=True,
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.85},
                    latency_ms=30.0,
                    memory_gb=4.0,
                    energy_kwh=0.03,
                    co2_kg=0.01,
                ),
                is_pareto_optimal=True,
            ),
        ]

        calculate_crowding_distance(solutions)

        # Boundary solutions should have infinite crowding distance
        assert solutions[0].crowding_distance == float("inf")
        assert solutions[-1].crowding_distance == float("inf")

        # Middle solution should have finite crowding distance
        assert 0 < solutions[1].crowding_distance < float("inf")

    def test_calculate_crowding_distance_two_solutions(self):
        """Test crowding distance with only two solutions"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.90},
                    latency_ms=30.0,
                    memory_gb=4.0,
                    energy_kwh=0.02,
                    co2_kg=0.01,
                )
            ),
        ]

        calculate_crowding_distance(solutions)

        # Both should have infinite distance (boundaries)
        assert solutions[0].crowding_distance == float("inf")
        assert solutions[1].crowding_distance == float("inf")


class TestSelectBestSolution:
    """Test best solution selection"""

    def test_select_best_solution_balanced(self):
        """Test selecting best solution with balanced weights"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.90},
                    latency_ms=30.0,
                    memory_gb=4.0,
                    energy_kwh=0.02,
                    co2_kg=0.01,
                )
            ),
        ]

        best = select_best_solution(solutions, criteria="balanced")

        assert best is not None
        assert best in solutions

    def test_select_best_solution_accuracy(self):
        """Test selecting best solution by accuracy"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.90},
                    latency_ms=30.0,
                    memory_gb=4.0,
                    energy_kwh=0.02,
                    co2_kg=0.01,
                )
            ),
        ]

        best = select_best_solution(solutions, criteria="accuracy")

        # Should select solution with highest accuracy
        assert best.metrics.average_accuracy() == 0.95


class TestCompositeScore:
    """Test composite score calculation"""

    def test_compute_composite_score(self):
        """Test composite score with different weights"""
        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        weights = {
            "accuracy": 1.0,
            "latency": -0.5,
            "memory": -0.5,
            "energy": -0.5,
            "co2": -0.5,
        }

        score = compute_composite_score(metrics, weights)

        # Score should be a single number
        assert isinstance(score, (int, float))

    def test_composite_score_accuracy_focused(self):
        """Test composite score favors accuracy"""
        metrics_high_acc = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        metrics_low_acc = EvaluationMetrics(
            accuracy={"gsm8k": 0.85},
            latency_ms=30.0,
            memory_gb=4.0,
            energy_kwh=0.02,
            co2_kg=0.01,
        )

        weights = {"accuracy": 10.0, "latency": -0.1, "memory": -0.1, "energy": -0.1, "co2": -0.1}

        score_high = compute_composite_score(metrics_high_acc, weights)
        score_low = compute_composite_score(metrics_low_acc, weights)

        # High accuracy should have higher score
        assert score_high > score_low


class TestFrontierDiversity:
    """Test frontier diversity calculation"""

    def test_calculate_frontier_diversity(self):
        """Test diversity metric calculation"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.90},
                    latency_ms=40.0,
                    memory_gb=6.0,
                    energy_kwh=0.04,
                    co2_kg=0.015,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.85},
                    latency_ms=30.0,
                    memory_gb=4.0,
                    energy_kwh=0.03,
                    co2_kg=0.01,
                )
            ),
        ]

        diversity = calculate_frontier_diversity(solutions)

        # Diversity should be a positive number
        assert diversity >= 0

    def test_frontier_diversity_two_solutions(self):
        """Test diversity with two solutions"""
        solutions = [
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.95},
                    latency_ms=50.0,
                    memory_gb=8.0,
                    energy_kwh=0.05,
                    co2_kg=0.02,
                )
            ),
            ParetoSolution(
                metrics=EvaluationMetrics(
                    accuracy={"gsm8k": 0.85},
                    latency_ms=30.0,
                    memory_gb=4.0,
                    energy_kwh=0.03,
                    co2_kg=0.01,
                )
            ),
        ]

        diversity = calculate_frontier_diversity(solutions)

        assert diversity >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
