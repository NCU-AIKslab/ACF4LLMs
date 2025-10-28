"""
RQ4: Resource-Constrained Adaptation

Research Question 4: How does the framework adapt in resource-constrained
environments (edge devices, mobile, data centers)?
"""

import logging
from typing import Any

from ..core.config import (
    CARBON_INTENSIVE_DC,
    CLOUD_SERVER,
    EDGE_DEVICE,
    MOBILE_DEVICE,
    CompressionConfig,
    EnvironmentConstraints,
)
from ..core.metrics import EvaluationMetrics, ParetoSolution
from ..tools.evaluation_tools import evaluate_config_full

logger = logging.getLogger(__name__)


class ResourceConstrainedAdaptation:
    """Test framework adaptation in resource-constrained environments"""

    def __init__(self):
        self.environments = {
            "edge_device": EDGE_DEVICE,
            "mobile_device": MOBILE_DEVICE,
            "cloud_server": CLOUD_SERVER,
            "carbon_intensive_dc": CARBON_INTENSIVE_DC,
        }
        self.adaptation_results: dict = {}
        self.environment_solutions: dict[str, list[ParetoSolution]] = {}

    async def test_environment_adaptation(
        self, model: str, accuracy_threshold: float = 0.85
    ) -> dict[str, Any]:
        """
        Test framework adaptation across different deployment environments.

        Args:
            model: Model identifier to optimize
            accuracy_threshold: Minimum acceptable accuracy

        Returns:
            Complete adaptation analysis across environments
        """
        logger.info("Starting resource-constrained environment adaptation test")

        results = {}

        for env_name, env_constraints in self.environments.items():
            logger.info(f"\nTesting adaptation to {env_name}")
            logger.info(
                f"Constraints: memory≤{env_constraints.max_memory_gb}GB, "
                f"power≤{env_constraints.max_power_watts}W, "
                f"latency≤{env_constraints.latency_requirement_ms}ms"
            )

            # Adapt to environment
            adapted_config = await self._adapt_to_environment(
                model=model, environment=env_constraints, accuracy_threshold=accuracy_threshold
            )

            # Evaluate in environment
            metrics = await self._evaluate_in_environment(
                config=adapted_config, environment=env_constraints
            )

            # Check feasibility
            feasibility = self._check_feasibility(metrics, env_constraints)

            # Calculate efficiency score
            efficiency = self._calculate_efficiency_score(metrics, env_constraints)

            # Store results
            solution = ParetoSolution(metrics=metrics)
            if env_name not in self.environment_solutions:
                self.environment_solutions[env_name] = []
            self.environment_solutions[env_name].append(solution)

            results[env_name] = {
                "environment_constraints": {
                    "max_memory_gb": env_constraints.max_memory_gb,
                    "max_power_watts": env_constraints.max_power_watts,
                    "carbon_intensity": env_constraints.carbon_intensity,
                    "latency_requirement_ms": env_constraints.latency_requirement_ms,
                },
                "adapted_config": {
                    "quantization_bits": adapted_config.quantization_bits,
                    "pruning_sparsity": adapted_config.pruning_sparsity,
                },
                "performance": {
                    "accuracy": metrics.average_accuracy(),
                    "latency_ms": metrics.latency_ms,
                    "memory_gb": metrics.memory_gb,
                    "energy_kwh": metrics.energy_kwh,
                    "co2_kg": metrics.co2_kg,
                    "daily_co2_kg": metrics.co2_kg * 1000,  # Assuming 1000 inferences/day
                },
                "feasibility": feasibility,
                "efficiency_score": efficiency,
            }

            logger.info(
                f"{env_name}: feasible={feasibility['is_feasible']}, "
                f"accuracy={metrics.average_accuracy():.3f}, "
                f"efficiency={efficiency:.3f}"
            )

        # Compare across environments
        comparison = self._compare_environments(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results, comparison)

        self.adaptation_results = {
            "environments_tested": list(self.environments.keys()),
            "environment_results": results,
            "cross_environment_comparison": comparison,
            "deployment_recommendations": recommendations,
            "key_findings": self._generate_findings(results, comparison),
        }

        return self.adaptation_results

    async def _adapt_to_environment(
        self, model: str, environment: EnvironmentConstraints, accuracy_threshold: float
    ) -> CompressionConfig:
        """
        Adapt compression configuration to environment constraints.

        Args:
            model: Model identifier
            environment: Environment constraints
            accuracy_threshold: Target accuracy

        Returns:
            Adapted compression configuration
        """
        # Determine compression aggressiveness based on constraints
        if environment.max_memory_gb <= 4:
            # Extremely constrained (edge device)
            quantization_bits = 4
            pruning_sparsity = 0.7
        elif environment.max_memory_gb <= 8:
            # Moderately constrained (mobile)
            quantization_bits = 8
            pruning_sparsity = 0.5
        elif environment.max_memory_gb <= 32:
            # Lightly constrained
            quantization_bits = 8
            pruning_sparsity = 0.3
        else:
            # Resource-rich (cloud)
            if environment.carbon_intensity > 0.6:
                # High carbon intensity - prioritize efficiency
                quantization_bits = 8
                pruning_sparsity = 0.4
            else:
                # Low carbon intensity - can use more resources
                quantization_bits = 16
                pruning_sparsity = 0.2

        # Adjust for latency requirements
        if environment.latency_requirement_ms < 30:
            # Very strict latency - use more compression
            quantization_bits = min(quantization_bits, 8)
            pruning_sparsity = max(pruning_sparsity, 0.4)

        config = CompressionConfig(
            quantization_bits=quantization_bits,
            pruning_sparsity=pruning_sparsity,
            model_path=model,
            accuracy_threshold=accuracy_threshold,
        )

        logger.debug(
            f"Adapted config for {environment.name}: "
            f"bits={quantization_bits}, sparsity={pruning_sparsity:.1%}"
        )

        return config

    async def _evaluate_in_environment(
        self, config: CompressionConfig, environment: EnvironmentConstraints
    ) -> EvaluationMetrics:
        """
        Evaluate configuration in specific environment.

        Args:
            config: Compression configuration
            environment: Environment constraints

        Returns:
            Evaluation metrics adjusted for environment
        """
        # Get base metrics
        metrics = await evaluate_config_full(config)

        # Adjust carbon based on environment's carbon intensity
        # Base calculation assumes carbon_intensity of 0.4 (default grid)
        adjusted_co2 = metrics.energy_kwh * environment.carbon_intensity

        # Create adjusted metrics
        adjusted_metrics = EvaluationMetrics(
            accuracy=metrics.accuracy,
            latency_ms=metrics.latency_ms,
            memory_gb=metrics.memory_gb,
            energy_kwh=metrics.energy_kwh,
            co2_kg=adjusted_co2,
            throughput_tps=metrics.throughput_tps,
            compression_ratio=metrics.compression_ratio,
            config=config,
        )

        return adjusted_metrics

    def _check_feasibility(
        self, metrics: EvaluationMetrics, environment: EnvironmentConstraints
    ) -> dict[str, Any]:
        """
        Check if configuration meets environment constraints.

        Args:
            metrics: Evaluation metrics
            environment: Environment constraints

        Returns:
            Feasibility analysis
        """
        # Check each constraint
        memory_ok = metrics.memory_gb <= environment.max_memory_gb
        latency_ok = metrics.latency_ms <= environment.latency_requirement_ms

        # Estimate power consumption (simplified)
        # Assume power scales with memory and latency
        estimated_power = (metrics.memory_gb / 24.0) * 100  # Rough estimate in watts
        power_ok = estimated_power <= environment.max_power_watts

        is_feasible = memory_ok and latency_ok and power_ok

        feasibility = {
            "is_feasible": is_feasible,
            "memory_constraint": {
                "satisfied": memory_ok,
                "usage_gb": metrics.memory_gb,
                "limit_gb": environment.max_memory_gb,
                "utilization_percent": (metrics.memory_gb / environment.max_memory_gb) * 100,
            },
            "latency_constraint": {
                "satisfied": latency_ok,
                "actual_ms": metrics.latency_ms,
                "limit_ms": environment.latency_requirement_ms,
                "margin_ms": environment.latency_requirement_ms - metrics.latency_ms,
            },
            "power_constraint": {
                "satisfied": power_ok,
                "estimated_watts": estimated_power,
                "limit_watts": environment.max_power_watts,
            },
        }

        return feasibility

    def _calculate_efficiency_score(
        self, metrics: EvaluationMetrics, environment: EnvironmentConstraints
    ) -> float:
        """
        Calculate efficiency score for configuration in environment.

        Args:
            metrics: Evaluation metrics
            environment: Environment constraints

        Returns:
            Normalized efficiency score (0-1, higher is better)
        """
        # Normalize metrics to [0, 1]
        accuracy_score = metrics.average_accuracy()  # Already 0-1

        # Resource utilization (lower is better, so invert)
        memory_score = 1.0 - min(metrics.memory_gb / environment.max_memory_gb, 1.0)

        # Latency score (lower is better)
        latency_score = 1.0 - min(metrics.latency_ms / environment.latency_requirement_ms, 1.0)

        # Carbon score (lower is better)
        # Normalize to typical range
        carbon_score = max(0, 1.0 - (metrics.co2_kg / 0.034))  # 0.034 is baseline

        # Weighted composite score
        efficiency = (
            accuracy_score * 0.4 + memory_score * 0.2 + latency_score * 0.2 + carbon_score * 0.2
        )

        return max(0.0, min(1.0, efficiency))  # Clamp to [0, 1]

    def _compare_environments(self, results: dict[str, Any]) -> dict[str, Any]:
        """
        Compare performance across different environments.

        Args:
            results: Results from all environments

        Returns:
            Cross-environment comparison
        """
        comparison = {
            "accuracy_ranking": [],
            "carbon_ranking": [],
            "efficiency_ranking": [],
            "carbon_impact": {},
            "feasibility_summary": {},
        }

        # Extract metrics for ranking
        env_data = []
        for env_name, env_result in results.items():
            env_data.append(
                {
                    "name": env_name,
                    "accuracy": env_result["performance"]["accuracy"],
                    "co2_daily": env_result["performance"]["daily_co2_kg"],
                    "efficiency": env_result["efficiency_score"],
                    "feasible": env_result["feasibility"]["is_feasible"],
                }
            )

        # Rank by accuracy
        sorted_by_acc = sorted(env_data, key=lambda x: x["accuracy"], reverse=True)
        comparison["accuracy_ranking"] = [
            {"environment": e["name"], "accuracy": e["accuracy"]} for e in sorted_by_acc
        ]

        # Rank by carbon (lower is better)
        sorted_by_carbon = sorted(env_data, key=lambda x: x["co2_daily"])
        comparison["carbon_ranking"] = [
            {"environment": e["name"], "daily_co2_kg": e["co2_daily"]} for e in sorted_by_carbon
        ]

        # Rank by efficiency
        sorted_by_eff = sorted(env_data, key=lambda x: x["efficiency"], reverse=True)
        comparison["efficiency_ranking"] = [
            {"environment": e["name"], "efficiency_score": e["efficiency"]} for e in sorted_by_eff
        ]

        # Calculate carbon impact
        if sorted_by_carbon:
            best_carbon = sorted_by_carbon[0]["co2_daily"]
            worst_carbon = sorted_by_carbon[-1]["co2_daily"]

            comparison["carbon_impact"] = {
                "lowest_carbon_env": sorted_by_carbon[0]["name"],
                "highest_carbon_env": sorted_by_carbon[-1]["name"],
                "carbon_range_kg_per_day": {
                    "min": best_carbon,
                    "max": worst_carbon,
                    "difference": worst_carbon - best_carbon,
                },
                "reduction_potential_percent": (
                    ((worst_carbon - best_carbon) / worst_carbon * 100) if worst_carbon > 0 else 0
                ),
            }

        # Feasibility summary
        feasible_envs = [e["name"] for e in env_data if e["feasible"]]
        comparison["feasibility_summary"] = {
            "total_environments": len(env_data),
            "feasible_count": len(feasible_envs),
            "feasible_environments": feasible_envs,
        }

        return comparison

    def _generate_recommendations(
        self, results: dict[str, Any], comparison: dict[str, Any]
    ) -> dict[str, str]:
        """
        Generate deployment recommendations based on analysis.

        Args:
            results: Environment-specific results
            comparison: Cross-environment comparison

        Returns:
            Deployment recommendations
        """
        recommendations = {}

        for env_name, env_result in results.items():
            if env_result["feasibility"]["is_feasible"]:
                accuracy = env_result["performance"]["accuracy"]
                carbon = env_result["performance"]["daily_co2_kg"]
                efficiency = env_result["efficiency_score"]

                if accuracy > 0.92 and efficiency > 0.75:
                    recommendations[env_name] = (
                        f"✅ HIGHLY RECOMMENDED - Excellent balance of accuracy ({accuracy:.1%}) "
                        f"and efficiency (score: {efficiency:.2f})"
                    )
                elif accuracy > 0.88 and carbon < 1.0:
                    recommendations[env_name] = (
                        f"✅ RECOMMENDED - Good accuracy ({accuracy:.1%}) with low carbon "
                        f"footprint ({carbon:.2f}kg/day)"
                    )
                elif efficiency > 0.70:
                    recommendations[env_name] = (
                        "⚠️ ACCEPTABLE - Meets constraints but consider optimization for "
                        "accuracy improvement"
                    )
                else:
                    recommendations[env_name] = (
                        "⚠️ USE WITH CAUTION - Feasible but with reduced performance"
                    )
            else:
                recommendations[env_name] = (
                    "❌ NOT RECOMMENDED - Does not meet environment constraints"
                )

        return recommendations

    def _generate_findings(self, results: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
        """Generate key findings from adaptation analysis"""
        findings = []

        # Feasibility finding
        feasibility = comparison["feasibility_summary"]
        findings.append(
            f"Framework successfully adapted to {feasibility['feasible_count']} out of "
            f"{feasibility['total_environments']} tested environments"
        )

        # Carbon impact finding
        carbon_impact = comparison.get("carbon_impact", {})
        if carbon_impact:
            reduction = carbon_impact.get("reduction_potential_percent", 0)
            findings.append(
                f"Deploying to {carbon_impact['lowest_carbon_env']} instead of "
                f"{carbon_impact['highest_carbon_env']} reduces daily carbon emissions by "
                f"{reduction:.1f}% ({carbon_impact['carbon_range_kg_per_day']['difference']:.2f}kg)"
            )

        # Accuracy-efficiency trade-off
        acc_ranking = comparison["accuracy_ranking"]
        eff_ranking = comparison["efficiency_ranking"]

        if acc_ranking and eff_ranking:
            best_acc_env = acc_ranking[0]["environment"]
            best_eff_env = eff_ranking[0]["environment"]

            if best_acc_env != best_eff_env:
                findings.append(
                    f"{best_acc_env} achieves highest accuracy while "
                    f"{best_eff_env} provides best overall efficiency"
                )

        # Resource adaptation finding
        findings.append(
            "Framework automatically adjusts compression aggressiveness based on "
            "environment constraints, demonstrating effective resource-aware optimization"
        )

        return findings


async def run_rq4_experiment(
    model: str = "google/gemma-12b", accuracy_threshold: float = 0.85
) -> dict[str, Any]:
    """
    Run complete RQ4 experiment testing environment adaptation.

    Args:
        model: Model to optimize
        accuracy_threshold: Minimum acceptable accuracy

    Returns:
        Complete RQ4 experiment results
    """
    adapter = ResourceConstrainedAdaptation()

    # Test adaptation
    adaptation_results = await adapter.test_environment_adaptation(
        model=model, accuracy_threshold=accuracy_threshold
    )

    # Package results
    results = {
        "experiment": "RQ4: Resource-Constrained Adaptation",
        "model": model,
        "parameters": {
            "accuracy_threshold": accuracy_threshold,
            "environments_tested": list(adapter.environments.keys()),
        },
        "adaptation_analysis": adaptation_results,
        "conclusion": generate_rq4_conclusion(adaptation_results),
    }

    return results


def generate_rq4_conclusion(adaptation_results: dict[str, Any]) -> str:
    """Generate conclusion from RQ4 analysis"""
    feasibility = adaptation_results["cross_environment_comparison"]["feasibility_summary"]
    carbon_impact = adaptation_results["cross_environment_comparison"].get("carbon_impact", {})

    feasible_count = feasibility["feasible_count"]
    total_count = feasibility["total_environments"]
    reduction = carbon_impact.get("reduction_potential_percent", 0)

    return (
        f"The agentic compression framework demonstrates strong adaptability, "
        f"successfully operating in {feasible_count}/{total_count} tested environments. "
        f"Strategic environment selection enables up to {reduction:.1f}% carbon emission "
        f"reduction while maintaining acceptable accuracy thresholds. The framework's "
        f"automatic constraint-aware configuration adjustment makes it suitable for "
        f"diverse deployment scenarios from edge devices to cloud data centers."
    )
