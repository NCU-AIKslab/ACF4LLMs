"""
RQ1: Dynamic vs Static Compression Comparison

Research Question 1: How does the multi-agent compression framework compare
to static one-shot compression approaches?
"""

import logging
from typing import Any

from ..core.config import CompressionConfig
from ..core.metrics import EvaluationMetrics
from ..tools.evaluation_tools import evaluate_config_full

logger = logging.getLogger(__name__)


class DynamicVsStaticComparison:
    """Compare dynamic agent-driven vs static compression approaches"""

    def __init__(self):
        self.dynamic_results: list[EvaluationMetrics] = []
        self.static_results: list[EvaluationMetrics] = []
        self.dynamic_history: list[dict] = []
        self.static_history: list[dict] = []

    async def run_dynamic_compression(
        self,
        model: str,
        accuracy_threshold: float = 0.93,
        carbon_budget: float = 5.0,
        max_iterations: int = 20,
    ) -> dict[str, Any]:
        """
        Run dynamic compression with iterative feedback adjustment.

        Args:
            model: Model identifier to optimize
            accuracy_threshold: Minimum acceptable accuracy
            carbon_budget: Maximum carbon budget in kg CO2
            max_iterations: Maximum optimization iterations

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting dynamic compression optimization")

        # Start with baseline config
        config = CompressionConfig(
            quantization_bits=16,
            pruning_sparsity=0.0,
            model_path=model,
            accuracy_threshold=accuracy_threshold,
            carbon_budget=carbon_budget,
        )

        carbon_used = 0.0
        best_config = None
        best_metrics = None
        iterations_to_converge = 0

        for iteration in range(max_iterations):
            # Evaluate current configuration
            metrics = await self._evaluate_config(config)
            carbon_used += metrics.co2_kg * 0.1  # Evaluation cost

            self.dynamic_results.append(metrics)
            self.dynamic_history.append(
                {
                    "iteration": iteration,
                    "config": vars(config),
                    "metrics": metrics.to_dict(),
                    "carbon_used": carbon_used,
                }
            )

            avg_accuracy = metrics.average_accuracy()
            logger.info(
                f"Dynamic iter {iteration}: accuracy={avg_accuracy:.3f}, "
                f"CO₂={metrics.co2_kg:.4f}kg, bits={config.quantization_bits}, "
                f"sparsity={config.pruning_sparsity:.2f}"
            )

            # Check if we found a good solution
            if avg_accuracy >= accuracy_threshold and not best_config:
                best_config = config
                best_metrics = metrics
                iterations_to_converge = iteration + 1
                logger.info(f"Converged at iteration {iterations_to_converge}")

            # Check carbon budget
            if carbon_used >= carbon_budget:
                logger.info(f"Carbon budget exhausted: {carbon_used:.4f}kg")
                break

            # Dynamically adjust configuration based on feedback
            config = await self._adjust_config_dynamically(config, metrics, accuracy_threshold)

        return {
            "approach": "dynamic",
            "iterations": len(self.dynamic_results),
            "iterations_to_converge": iterations_to_converge,
            "carbon_used": carbon_used,
            "best_accuracy": best_metrics.average_accuracy() if best_metrics else 0,
            "best_carbon": best_metrics.co2_kg if best_metrics else 0,
            "best_config": vars(best_config) if best_config else None,
            "history": self.dynamic_history,
        }

    async def run_static_compression(
        self, model: str, accuracy_threshold: float = 0.93, carbon_budget: float = 5.0
    ) -> dict[str, Any]:
        """
        Run static one-shot compression with predefined configurations.

        Args:
            model: Model identifier to optimize
            accuracy_threshold: Minimum acceptable accuracy
            carbon_budget: Maximum carbon budget in kg CO2

        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting static compression optimization")

        # Predefined static configurations
        static_configs = [
            {"quantization_bits": 32, "pruning_sparsity": 0.0},
            {"quantization_bits": 16, "pruning_sparsity": 0.0},
            {"quantization_bits": 16, "pruning_sparsity": 0.1},
            {"quantization_bits": 8, "pruning_sparsity": 0.0},
            {"quantization_bits": 8, "pruning_sparsity": 0.3},
            {"quantization_bits": 8, "pruning_sparsity": 0.5},
            {"quantization_bits": 4, "pruning_sparsity": 0.3},
            {"quantization_bits": 4, "pruning_sparsity": 0.5},
        ]

        carbon_used = 0.0
        best_config = None
        best_metrics = None

        for idx, cfg_dict in enumerate(static_configs):
            if carbon_used >= carbon_budget:
                logger.info(f"Carbon budget reached: {carbon_used:.4f}kg")
                break

            # Create configuration
            config = CompressionConfig(
                quantization_bits=cfg_dict["quantization_bits"],
                pruning_sparsity=cfg_dict["pruning_sparsity"],
                model_path=model,
                accuracy_threshold=accuracy_threshold,
            )

            # Evaluate configuration
            metrics = await self._evaluate_config(config)
            carbon_used += metrics.co2_kg * 0.1  # Evaluation cost

            self.static_results.append(metrics)
            self.static_history.append(
                {
                    "config_index": idx,
                    "config": vars(config),
                    "metrics": metrics.to_dict(),
                    "carbon_used": carbon_used,
                }
            )

            avg_accuracy = metrics.average_accuracy()
            logger.info(
                f"Static config {idx}: accuracy={avg_accuracy:.3f}, " f"CO₂={metrics.co2_kg:.4f}kg"
            )

            # Select best configuration that meets threshold
            if avg_accuracy >= accuracy_threshold:
                if best_metrics is None or metrics.co2_kg < best_metrics.co2_kg:
                    best_config = config
                    best_metrics = metrics

        return {
            "approach": "static",
            "configs_evaluated": len(self.static_results),
            "carbon_used": carbon_used,
            "best_accuracy": best_metrics.average_accuracy() if best_metrics else 0,
            "best_carbon": best_metrics.co2_kg if best_metrics else 0,
            "best_config": vars(best_config) if best_config else None,
            "history": self.static_history,
        }

    async def _adjust_config_dynamically(
        self,
        current_config: CompressionConfig,
        current_metrics: EvaluationMetrics,
        accuracy_threshold: float,
    ) -> CompressionConfig:
        """
        Adjust configuration dynamically based on feedback.

        Args:
            current_config: Current compression configuration
            current_metrics: Current evaluation metrics
            accuracy_threshold: Target accuracy threshold

        Returns:
            Adjusted configuration
        """
        avg_accuracy = current_metrics.average_accuracy()

        # Determine adjustment strategy based on current performance
        if avg_accuracy > accuracy_threshold + 0.02:
            # We have accuracy headroom - can compress more aggressively
            new_bits = max(current_config.quantization_bits // 2, 4)
            new_sparsity = min(current_config.pruning_sparsity + 0.2, 0.7)
            logger.debug("Accuracy above threshold - increasing compression")

        elif avg_accuracy < accuracy_threshold - 0.02:
            # Below threshold - need to reduce compression
            new_bits = min(current_config.quantization_bits * 2, 32)
            new_sparsity = max(current_config.pruning_sparsity - 0.1, 0.0)
            logger.debug("Accuracy below threshold - reducing compression")

        else:
            # Near threshold - fine-tune
            if current_config.quantization_bits > 8:
                new_bits = 8
            elif current_config.quantization_bits == 8:
                new_bits = 4
            else:
                new_bits = current_config.quantization_bits

            new_sparsity = min(current_config.pruning_sparsity + 0.1, 0.7)
            logger.debug("Near threshold - fine-tuning")

        return CompressionConfig(
            quantization_bits=new_bits,
            pruning_sparsity=new_sparsity,
            model_path=current_config.model_path,
            accuracy_threshold=current_config.accuracy_threshold,
            carbon_budget=current_config.carbon_budget,
        )

    async def _evaluate_config(self, config: CompressionConfig) -> EvaluationMetrics:
        """
        Evaluate a compression configuration.

        Args:
            config: Configuration to evaluate

        Returns:
            Evaluation metrics
        """
        return await evaluate_config_full(config)

    def compare_results(self) -> dict[str, Any]:
        """
        Compare dynamic vs static compression results.

        Returns:
            Comprehensive comparison analysis
        """
        if not self.dynamic_results or not self.static_results:
            return {"error": "No results to compare"}

        # Extract metrics
        dynamic_accuracies = [m.average_accuracy() for m in self.dynamic_results]
        static_accuracies = [m.average_accuracy() for m in self.static_results]

        dynamic_carbons = [m.co2_kg for m in self.dynamic_results]
        static_carbons = [m.co2_kg for m in self.static_results]

        # Find best solutions
        dynamic_best = max(self.dynamic_results, key=lambda m: m.average_accuracy())
        static_best = max(self.static_results, key=lambda m: m.average_accuracy())

        # Calculate improvements
        accuracy_improvement = (
            dynamic_best.average_accuracy() - static_best.average_accuracy()
        ) * 100

        carbon_improvement = (static_best.co2_kg - dynamic_best.co2_kg) / static_best.co2_kg * 100

        comparison = {
            "dynamic_approach": {
                "avg_accuracy": sum(dynamic_accuracies) / len(dynamic_accuracies),
                "best_accuracy": max(dynamic_accuracies),
                "avg_carbon": sum(dynamic_carbons) / len(dynamic_carbons),
                "best_carbon": min(dynamic_carbons),
                "iterations": len(self.dynamic_results),
                "convergence_speed": self.dynamic_history[-1].get("iteration", 0) + 1,
            },
            "static_approach": {
                "avg_accuracy": sum(static_accuracies) / len(static_accuracies),
                "best_accuracy": max(static_accuracies),
                "avg_carbon": sum(static_carbons) / len(static_carbons),
                "best_carbon": min(static_carbons),
                "configs_tested": len(self.static_results),
            },
            "improvements": {
                "accuracy_gain_percent": accuracy_improvement,
                "carbon_reduction_percent": carbon_improvement,
                "convergence_speed_ratio": (len(self.static_results) / len(self.dynamic_results)),
            },
            "key_findings": self._generate_findings(accuracy_improvement, carbon_improvement),
        }

        return comparison

    def _generate_findings(
        self, accuracy_improvement: float, carbon_improvement: float
    ) -> list[str]:
        """Generate key findings from comparison"""
        findings = []

        if accuracy_improvement > 2.0:
            findings.append(
                f"Dynamic approach achieved {accuracy_improvement:.1f}% higher "
                f"accuracy than static one-shot compression"
            )
        elif accuracy_improvement < -2.0:
            findings.append(
                f"Static approach achieved {abs(accuracy_improvement):.1f}% higher "
                f"accuracy, indicating well-tuned predefined configurations"
            )
        else:
            findings.append(
                f"Both approaches achieved comparable accuracy "
                f"(difference: {abs(accuracy_improvement):.1f}%)"
            )

        if carbon_improvement > 15.0:
            findings.append(
                f"Dynamic optimization reduced carbon emissions by "
                f"{carbon_improvement:.1f}% through adaptive configuration"
            )
        elif carbon_improvement < -15.0:
            findings.append(
                f"Static approach was more carbon-efficient "
                f"({abs(carbon_improvement):.1f}% lower emissions)"
            )

        findings.append(
            f"Dynamic approach explored {len(self.dynamic_results)} configurations "
            f"vs {len(self.static_results)} static configurations"
        )

        return findings


async def run_rq1_experiment(
    model: str = "google/gemma-12b",
    accuracy_threshold: float = 0.93,
    carbon_budget: float = 5.0,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """
    Run complete RQ1 experiment comparing dynamic vs static compression.

    Args:
        model: Model to optimize
        accuracy_threshold: Minimum acceptable accuracy
        carbon_budget: Carbon budget in kg CO2
        max_iterations: Maximum iterations for dynamic approach

    Returns:
        Complete RQ1 experiment results
    """
    comparator = DynamicVsStaticComparison()

    # Run both approaches
    dynamic_results = await comparator.run_dynamic_compression(
        model=model,
        accuracy_threshold=accuracy_threshold,
        carbon_budget=carbon_budget,
        max_iterations=max_iterations,
    )

    static_results = await comparator.run_static_compression(
        model=model, accuracy_threshold=accuracy_threshold, carbon_budget=carbon_budget
    )

    # Compare results
    comparison = comparator.compare_results()

    results = {
        "experiment": "RQ1: Dynamic vs Static Compression",
        "model": model,
        "parameters": {
            "accuracy_threshold": accuracy_threshold,
            "carbon_budget": carbon_budget,
            "max_iterations": max_iterations,
        },
        "dynamic_results": dynamic_results,
        "static_results": static_results,
        "comparison": comparison,
        "conclusion": generate_rq1_conclusion(comparison),
    }

    return results


def generate_rq1_conclusion(comparison: dict[str, Any]) -> str:
    """Generate conclusion from RQ1 comparison"""
    improvements = comparison.get("improvements", {})
    accuracy_gain = improvements.get("accuracy_gain_percent", 0)
    carbon_reduction = improvements.get("carbon_reduction_percent", 0)

    if accuracy_gain > 2 and carbon_reduction > 15:
        return (
            "Dynamic multi-agent compression significantly outperforms static "
            "approaches in both accuracy and carbon efficiency, demonstrating "
            "the value of adaptive optimization."
        )
    elif accuracy_gain > 2:
        return (
            "Dynamic approach achieves higher accuracy through iterative "
            "refinement, though carbon efficiency is comparable to static methods."
        )
    elif carbon_reduction > 15:
        return (
            "Dynamic optimization excels in carbon efficiency by adapting "
            "configurations to balance accuracy and emissions."
        )
    else:
        return (
            "Both approaches achieve comparable results, with dynamic methods "
            "offering greater adaptability and static methods providing "
            "predictability."
        )
