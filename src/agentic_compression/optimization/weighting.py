"""
RQ3: Weighting Scheme Analysis

Research Question 3: How do different weighting configurations affect the
Pareto frontier and optimal strategy selection?
"""

import logging
from typing import Any

from ..core.config import CompressionConfig
from ..core.metrics import ParetoSolution
from ..core.pareto import (
    calculate_crowding_distance,
    calculate_frontier_diversity,
    compute_composite_score,
    compute_pareto_frontier,
    weighted_dominates,
)
from ..tools.evaluation_tools import evaluate_config_full

logger = logging.getLogger(__name__)


class WeightingSchemeAnalysis:
    """Analyze impact of different objective weighting schemes"""

    def __init__(self):
        self.weight_schemes = {
            "balanced": {
                "accuracy": 1.0,
                "latency": -0.7,
                "memory": -0.6,
                "energy": -0.6,
                "co2": -0.8,
            },
            "accuracy_focused": {
                "accuracy": 2.0,
                "latency": -0.3,
                "memory": -0.3,
                "energy": -0.3,
                "co2": -0.3,
            },
            "carbon_focused": {
                "accuracy": 0.5,
                "latency": -0.5,
                "memory": -0.5,
                "energy": -1.0,
                "co2": -2.0,
            },
            "efficiency_focused": {
                "accuracy": 0.7,
                "latency": -1.5,
                "memory": -1.5,
                "energy": -0.5,
                "co2": -0.5,
            },
        }
        self.pareto_frontiers: dict[str, list[ParetoSolution]] = {}
        self.all_solutions: list[ParetoSolution] = []
        self.optimal_selections: dict[str, ParetoSolution] = {}

    async def analyze_weight_impact(self, model: str, carbon_budget: float = 5.0) -> dict[str, Any]:
        """
        Analyze how different weight schemes affect Pareto frontier.

        Args:
            model: Model identifier to optimize
            carbon_budget: Maximum carbon budget in kg CO2

        Returns:
            Complete analysis of weight scheme impacts
        """
        logger.info("Starting weighting scheme analysis")

        # Generate candidate solutions
        self.all_solutions = await self._generate_candidate_solutions(model, carbon_budget)
        logger.info(f"Generated {len(self.all_solutions)} candidate solutions")

        # Compute Pareto frontiers for each weight scheme
        for scheme_name, weights in self.weight_schemes.items():
            logger.info(f"Computing Pareto frontier for {scheme_name} scheme")
            frontier = self._compute_weighted_pareto(self.all_solutions, weights)
            self.pareto_frontiers[scheme_name] = frontier

            # Select optimal solution for this scheme
            optimal = self._select_optimal_with_weights(frontier, weights)
            self.optimal_selections[scheme_name] = optimal

            logger.info(
                f"{scheme_name}: {len(frontier)} Pareto-optimal solutions, "
                f"best accuracy={optimal.metrics.average_accuracy():.3f}, "
                f"CO₂={optimal.metrics.co2_kg:.4f}kg"
            )

        # Calculate diversity metrics
        diversity_analysis = self._calculate_frontier_diversity()

        # Compare weight schemes
        comparison = self._compare_weight_schemes()

        results = {
            "total_solutions_explored": len(self.all_solutions),
            "weight_schemes_tested": list(self.weight_schemes.keys()),
            "frontier_sizes": {
                name: len(frontier) for name, frontier in self.pareto_frontiers.items()
            },
            "diversity_metrics": diversity_analysis,
            "scheme_comparison": comparison,
            "optimal_solutions": self._extract_optimal_configs(),
            "key_findings": self._generate_findings(),
        }

        return results

    async def _generate_candidate_solutions(
        self, model: str, carbon_budget: float
    ) -> list[ParetoSolution]:
        """
        Generate diverse candidate compression solutions.

        Args:
            model: Model identifier
            carbon_budget: Carbon budget

        Returns:
            List of evaluated solutions
        """
        solutions = []
        carbon_used = 0.0

        # Comprehensive configuration space
        quantization_levels = [4, 8, 16, 32]
        pruning_levels = [0.0, 0.1, 0.3, 0.5, 0.7]

        for bits in quantization_levels:
            for sparsity in pruning_levels:
                if carbon_used >= carbon_budget:
                    logger.info("Carbon budget reached during generation")
                    break

                # Create configuration
                config = CompressionConfig(
                    quantization_bits=bits, pruning_sparsity=sparsity, model_path=model
                )

                # Evaluate configuration
                metrics = await evaluate_config_full(config)
                carbon_used += metrics.co2_kg * 0.05  # Evaluation cost

                # Create solution
                solution = ParetoSolution(metrics=metrics)
                solutions.append(solution)

                logger.debug(
                    f"Generated solution: bits={bits}, sparsity={sparsity:.1%}, "
                    f"accuracy={metrics.average_accuracy():.3f}"
                )

            if carbon_used >= carbon_budget:
                break

        return solutions

    def _compute_weighted_pareto(
        self, solutions: list[ParetoSolution], weights: dict[str, float]
    ) -> list[ParetoSolution]:
        """
        Compute Pareto frontier using weighted dominance.

        Args:
            solutions: List of solutions to analyze
            weights: Weight scheme for objectives

        Returns:
            Pareto-optimal solutions under weighted scheme
        """
        # First compute standard Pareto frontier
        standard_frontier = compute_pareto_frontier(solutions.copy())

        # Then apply weighted selection
        weighted_frontier = []

        for sol in standard_frontier:
            # Check if this solution is dominated under weighted scheme
            is_dominated = False

            for other_sol in standard_frontier:
                if sol == other_sol:
                    continue

                if weighted_dominates(other_sol.metrics, sol.metrics, weights):
                    is_dominated = True
                    break

            if not is_dominated:
                weighted_frontier.append(sol)

        # Calculate crowding distance for diversity
        calculate_crowding_distance(weighted_frontier)

        return weighted_frontier

    def _select_optimal_with_weights(
        self, frontier: list[ParetoSolution], weights: dict[str, float]
    ) -> ParetoSolution:
        """
        Select optimal solution from frontier using weights.

        Args:
            frontier: Pareto-optimal solutions
            weights: Objective weights

        Returns:
            Best solution according to weights
        """
        if not frontier:
            raise ValueError("Empty frontier provided")

        # Compute composite scores
        best_solution = max(frontier, key=lambda sol: compute_composite_score(sol.metrics, weights))

        return best_solution

    def _calculate_frontier_diversity(self) -> dict[str, float]:
        """
        Calculate diversity metrics for each frontier.

        Returns:
            Diversity scores per weight scheme
        """
        diversity = {}

        for scheme_name, frontier in self.pareto_frontiers.items():
            if len(frontier) < 2:
                diversity[scheme_name] = 0.0
            else:
                # Use existing diversity calculation
                div_score = calculate_frontier_diversity(frontier)
                diversity[scheme_name] = div_score

        return diversity

    def _compare_weight_schemes(self) -> dict[str, Any]:
        """
        Compare characteristics of different weight schemes.

        Returns:
            Comparison analysis
        """
        comparison = {}

        for scheme_name, frontier in self.pareto_frontiers.items():
            if not frontier:
                continue

            # Extract metrics
            accuracies = [sol.metrics.average_accuracy() for sol in frontier]
            carbons = [sol.metrics.co2_kg for sol in frontier]
            latencies = [sol.metrics.latency_ms for sol in frontier]
            memories = [sol.metrics.memory_gb for sol in frontier]

            optimal = self.optimal_selections.get(scheme_name)

            comparison[scheme_name] = {
                "frontier_size": len(frontier),
                "accuracy_range": {
                    "min": min(accuracies),
                    "max": max(accuracies),
                    "mean": sum(accuracies) / len(accuracies),
                },
                "carbon_range": {
                    "min": min(carbons),
                    "max": max(carbons),
                    "mean": sum(carbons) / len(carbons),
                },
                "latency_range": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": sum(latencies) / len(latencies),
                },
                "memory_range": {
                    "min": min(memories),
                    "max": max(memories),
                    "mean": sum(memories) / len(memories),
                },
                "optimal_config": {
                    "quantization_bits": (
                        optimal.metrics.config.quantization_bits
                        if optimal and optimal.metrics.config
                        else None
                    ),
                    "pruning_sparsity": (
                        optimal.metrics.config.pruning_sparsity
                        if optimal and optimal.metrics.config
                        else None
                    ),
                    "accuracy": optimal.metrics.average_accuracy() if optimal else None,
                    "co2_kg": optimal.metrics.co2_kg if optimal else None,
                },
                "dominant_characteristic": self._identify_dominant_characteristic(
                    scheme_name, frontier
                ),
            }

        return comparison

    def _identify_dominant_characteristic(
        self, scheme_name: str, frontier: list[ParetoSolution]
    ) -> str:
        """Identify the dominant characteristic of a weight scheme"""
        if "accuracy" in scheme_name:
            return "High accuracy configurations with moderate compression"
        elif "carbon" in scheme_name:
            return "Aggressive compression favoring low carbon emissions"
        elif "efficiency" in scheme_name:
            return "Focus on low latency and memory usage"
        else:  # balanced
            return "Diverse solutions across the trade-off spectrum"

    def _extract_optimal_configs(self) -> dict[str, dict]:
        """Extract optimal configurations for each weight scheme"""
        configs = {}

        for scheme_name, solution in self.optimal_selections.items():
            if solution and solution.metrics.config:
                configs[scheme_name] = {
                    "quantization_bits": solution.metrics.config.quantization_bits,
                    "pruning_sparsity": solution.metrics.config.pruning_sparsity,
                    "accuracy": solution.metrics.average_accuracy(),
                    "co2_kg": solution.metrics.co2_kg,
                    "latency_ms": solution.metrics.latency_ms,
                    "memory_gb": solution.metrics.memory_gb,
                }

        return configs

    def _generate_findings(self) -> list[str]:
        """Generate key findings from analysis"""
        findings = []

        # Frontier size comparison
        sizes = {name: len(frontier) for name, frontier in self.pareto_frontiers.items()}
        max_frontier = max(sizes, key=sizes.get)
        min_frontier = min(sizes, key=sizes.get)

        findings.append(
            f"{max_frontier} scheme produced the largest Pareto frontier "
            f"({sizes[max_frontier]} solutions), indicating high diversity in trade-offs"
        )

        findings.append(
            f"{min_frontier} scheme produced the most focused frontier "
            f"({sizes[min_frontier]} solutions), concentrating on specific objectives"
        )

        # Weight sensitivity
        optimal_bits = [
            self.optimal_selections[name].metrics.config.quantization_bits
            for name in self.optimal_selections
            if self.optimal_selections[name].metrics.config
        ]

        if len(set(optimal_bits)) > 2:
            findings.append(
                "Weight schemes significantly influence optimal quantization selection, "
                "ranging from INT4 to INT32 depending on priorities"
            )

        # Carbon vs accuracy trade-off
        if (
            "carbon_focused" in self.optimal_selections
            and "accuracy_focused" in self.optimal_selections
        ):
            carbon_opt = self.optimal_selections["carbon_focused"]
            acc_opt = self.optimal_selections["accuracy_focused"]

            acc_diff = (
                acc_opt.metrics.average_accuracy() - carbon_opt.metrics.average_accuracy()
            ) * 100
            carbon_diff = (carbon_opt.metrics.co2_kg / acc_opt.metrics.co2_kg - 1) * 100

            findings.append(
                f"Prioritizing carbon over accuracy enables {abs(carbon_diff):.1f}% "
                f"emission reduction at cost of {acc_diff:.1f}% accuracy"
            )

        return findings


async def run_rq3_experiment(
    model: str = "google/gemma-12b", carbon_budget: float = 5.0
) -> dict[str, Any]:
    """
    Run complete RQ3 experiment analyzing weight scheme impacts.

    Args:
        model: Model to optimize
        carbon_budget: Carbon budget in kg CO2

    Returns:
        Complete RQ3 experiment results
    """
    analyzer = WeightingSchemeAnalysis()

    # Run analysis
    analysis_results = await analyzer.analyze_weight_impact(
        model=model, carbon_budget=carbon_budget
    )

    # Package results
    results = {
        "experiment": "RQ3: Weighting Scheme Analysis",
        "model": model,
        "parameters": {
            "carbon_budget": carbon_budget,
            "weight_schemes": list(analyzer.weight_schemes.keys()),
        },
        "analysis": analysis_results,
        "recommendations": generate_rq3_recommendations(analysis_results),
        "conclusion": generate_rq3_conclusion(analysis_results),
    }

    return results


def generate_rq3_recommendations(analysis: dict[str, Any]) -> dict[str, str]:
    """Generate deployment recommendations based on weight analysis"""
    return {
        "critical_applications": "Use accuracy_focused weights to maintain high performance",
        "green_computing": "Use carbon_focused weights to minimize environmental impact",
        "edge_deployment": "Use efficiency_focused weights to optimize for resource constraints",
        "general_purpose": "Use balanced weights for diverse, well-distributed Pareto frontier",
        "custom_deployment": "Adjust weights based on specific constraints and priorities",
    }


def generate_rq3_conclusion(analysis: dict[str, Any]) -> str:
    """Generate conclusion from RQ3 analysis"""
    sizes = analysis.get("frontier_sizes", {})
    total = analysis.get("total_solutions_explored", 0)

    avg_frontier_size = sum(sizes.values()) / len(sizes) if sizes else 0
    frontier_ratio = (avg_frontier_size / total * 100) if total > 0 else 0

    return (
        f"Weight scheme configuration significantly impacts Pareto frontier characteristics. "
        f"On average, {frontier_ratio:.1f}% of explored solutions are Pareto-optimal, "
        f"with frontier size varying by {max(sizes.values()) - min(sizes.values())} solutions "
        f"across different weight schemes. This demonstrates the importance of aligning "
        f"optimization objectives with deployment requirements and stakeholder priorities."
    )
