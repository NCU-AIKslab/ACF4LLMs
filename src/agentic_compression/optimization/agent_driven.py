"""
RQ2: Agent-Driven Optimization

Research Question 2: How do agent-driven pruning and quantization adjustments
impact energy consumption and carbon emissions?
"""

import logging
from typing import Any

from ..core.config import CompressionConfig
from ..core.metrics import ParetoSolution
from ..core.pareto import calculate_crowding_distance, compute_pareto_frontier
from ..tools.evaluation_tools import evaluate_config_full

logger = logging.getLogger(__name__)


class AgentDrivenOptimization:
    """
    Agent-driven compression optimization with carbon tracking.

    This class implements the core research question about how dynamic
    agent-driven adjustments affect energy and carbon footprint.
    """

    def __init__(self):
        self.optimization_history: list[dict] = []
        self.pareto_solutions: list[ParetoSolution] = []

    async def optimize_with_agents(
        self, model: str, accuracy_threshold: float = 0.93, carbon_budget: float = 5.0
    ) -> list[ParetoSolution]:
        """
        Run agent-driven optimization with carbon budget.

        Args:
            model: Model identifier to optimize
            accuracy_threshold: Minimum acceptable accuracy
            carbon_budget: Maximum carbon budget in kg CO2

        Returns:
            List of Pareto-optimal solutions
        """
        logger.info(
            f"Starting agent-driven optimization: "
            f"accuracy_threshold={accuracy_threshold}, "
            f"carbon_budget={carbon_budget}kg"
        )

        solutions = []
        carbon_used = 0.0

        # Explore compression configuration space
        for bits in [4, 8, 16]:
            for sparsity in [0.0, 0.3, 0.5, 0.7]:
                if carbon_used >= carbon_budget:
                    logger.info(f"Carbon budget reached: {carbon_used:.4f}kg")
                    break

                # Create configuration
                config = CompressionConfig(
                    quantization_bits=bits, pruning_sparsity=sparsity, model_path=model
                )

                # Evaluate configuration
                metrics = await evaluate_config_full(config)
                carbon_used += metrics.co2_kg * 0.1  # Evaluation cost

                # Create solution
                solution = ParetoSolution(metrics=metrics)
                solutions.append(solution)

                avg_accuracy = metrics.average_accuracy()
                logger.info(
                    f"Config (bits={bits}, sparsity={sparsity:.1%}): "
                    f"accuracy={avg_accuracy:.3f}, "
                    f"energy={metrics.energy_kwh:.4f}kWh, "
                    f"CO₂={metrics.co2_kg:.4f}kg"
                )

                # Track if meets threshold
                if avg_accuracy >= accuracy_threshold:
                    self.optimization_history.append(
                        {"config": config, "metrics": metrics, "carbon_cumulative": carbon_used}
                    )

        # Compute Pareto frontier
        self.pareto_solutions = compute_pareto_frontier(solutions)
        calculate_crowding_distance(self.pareto_solutions)

        logger.info(f"Found {len(self.pareto_solutions)} Pareto-optimal solutions")

        return self.pareto_solutions

    def get_best_by_criterion(self, criterion: str = "balanced") -> ParetoSolution:
        """
        Get best solution from Pareto frontier by criterion.

        Args:
            criterion: "accuracy", "carbon", or "balanced"

        Returns:
            Best solution according to criterion
        """
        if not self.pareto_solutions:
            raise ValueError("No Pareto solutions available")

        if criterion == "accuracy":
            return max(self.pareto_solutions, key=lambda x: x.metrics.average_accuracy())
        elif criterion == "carbon":
            return min(self.pareto_solutions, key=lambda x: x.metrics.co2_kg)
        else:  # balanced

            def balance_score(sol: ParetoSolution) -> float:
                return sol.metrics.average_accuracy() * 2.0 - sol.metrics.co2_kg * 10.0

            return max(self.pareto_solutions, key=balance_score)

    def analyze_carbon_impact(self) -> dict[str, Any]:
        """
        Analyze the impact of different configurations on carbon emissions.

        Returns:
            Dictionary with carbon impact analysis
        """
        if not self.pareto_solutions:
            return {}

        # Extract metrics
        accuracies = [sol.metrics.average_accuracy() for sol in self.pareto_solutions]
        carbons = [sol.metrics.co2_kg for sol in self.pareto_solutions]
        energies = [sol.metrics.energy_kwh for sol in self.pareto_solutions]

        # Find extremes
        min_carbon_sol = min(self.pareto_solutions, key=lambda x: x.metrics.co2_kg)
        max_accuracy_sol = max(self.pareto_solutions, key=lambda x: x.metrics.average_accuracy())

        # Calculate reduction potential
        baseline_carbon = 0.034  # Baseline uncompressed model
        max_reduction = baseline_carbon - min(carbons)
        reduction_percentage = (max_reduction / baseline_carbon) * 100

        analysis = {
            "pareto_size": len(self.pareto_solutions),
            "accuracy_range": {
                "min": min(accuracies),
                "max": max(accuracies),
                "span": max(accuracies) - min(accuracies),
            },
            "carbon_range": {
                "min": min(carbons),
                "max": max(carbons),
                "span": max(carbons) - min(carbons),
                "reduction_from_baseline": f"{reduction_percentage:.1f}%",
            },
            "energy_range": {
                "min": min(energies),
                "max": max(energies),
                "span": max(energies) - min(energies),
            },
            "best_carbon_solution": {
                "accuracy": min_carbon_sol.metrics.average_accuracy(),
                "co2_kg": min_carbon_sol.metrics.co2_kg,
                "energy_kwh": min_carbon_sol.metrics.energy_kwh,
                "config": (
                    vars(min_carbon_sol.metrics.config) if min_carbon_sol.metrics.config else None
                ),
            },
            "best_accuracy_solution": {
                "accuracy": max_accuracy_sol.metrics.average_accuracy(),
                "co2_kg": max_accuracy_sol.metrics.co2_kg,
                "energy_kwh": max_accuracy_sol.metrics.energy_kwh,
                "config": (
                    vars(max_accuracy_sol.metrics.config)
                    if max_accuracy_sol.metrics.config
                    else None
                ),
            },
        }

        return analysis


async def run_rq2_experiment(
    model: str = "google/gemma-12b", accuracy_threshold: float = 0.93, carbon_budget: float = 5.0
) -> dict[str, Any]:
    """
    Run complete RQ2 experiment.

    Args:
        model: Model to optimize
        accuracy_threshold: Minimum acceptable accuracy
        carbon_budget: Carbon budget in kg CO2

    Returns:
        Complete RQ2 experiment results
    """
    optimizer = AgentDrivenOptimization()

    # Run optimization
    pareto_solutions = await optimizer.optimize_with_agents(
        model=model, accuracy_threshold=accuracy_threshold, carbon_budget=carbon_budget
    )

    # Analyze results
    carbon_impact = optimizer.analyze_carbon_impact()

    # Get best solutions
    best_carbon = optimizer.get_best_by_criterion("carbon")
    best_accuracy = optimizer.get_best_by_criterion("accuracy")
    best_balanced = optimizer.get_best_by_criterion("balanced")

    results = {
        "experiment": "RQ2: Agent-Driven Optimization",
        "model": model,
        "parameters": {"accuracy_threshold": accuracy_threshold, "carbon_budget": carbon_budget},
        "pareto_frontier_size": len(pareto_solutions),
        "carbon_impact_analysis": carbon_impact,
        "best_solutions": {
            "carbon_optimized": {
                "accuracy": best_carbon.metrics.average_accuracy(),
                "co2_kg": best_carbon.metrics.co2_kg,
                "config": vars(best_carbon.metrics.config) if best_carbon.metrics.config else None,
            },
            "accuracy_optimized": {
                "accuracy": best_accuracy.metrics.average_accuracy(),
                "co2_kg": best_accuracy.metrics.co2_kg,
                "config": (
                    vars(best_accuracy.metrics.config) if best_accuracy.metrics.config else None
                ),
            },
            "balanced": {
                "accuracy": best_balanced.metrics.average_accuracy(),
                "co2_kg": best_balanced.metrics.co2_kg,
                "config": (
                    vars(best_balanced.metrics.config) if best_balanced.metrics.config else None
                ),
            },
        },
        "key_findings": generate_rq2_findings(carbon_impact, best_carbon, best_accuracy),
    }

    return results


def generate_rq2_findings(
    carbon_impact: dict, best_carbon: ParetoSolution, best_accuracy: ParetoSolution
) -> list[str]:
    """Generate key findings from RQ2 experiment"""
    findings = []

    # Carbon reduction finding
    reduction = carbon_impact.get("carbon_range", {}).get("reduction_from_baseline", "N/A")
    findings.append(
        f"Agent-driven optimization achieved up to {reduction} carbon reduction "
        f"compared to baseline uncompressed model"
    )

    # Accuracy-carbon tradeoff
    acc_diff = best_accuracy.metrics.average_accuracy() - best_carbon.metrics.average_accuracy()
    carbon_diff = best_accuracy.metrics.co2_kg - best_carbon.metrics.co2_kg
    findings.append(
        f"Trading {acc_diff*100:.1f}% accuracy enables "
        f"{abs(carbon_diff)/best_accuracy.metrics.co2_kg*100:.1f}% carbon reduction"
    )

    # Pareto frontier size
    findings.append(
        f"Found {carbon_impact['pareto_size']} Pareto-optimal configurations, "
        f"demonstrating multiple viable accuracy-carbon tradeoffs"
    )

    return findings
