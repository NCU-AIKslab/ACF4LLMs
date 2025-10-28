"""
Visualization utilities for Pareto frontiers and optimization results.
"""

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

from ..core.metrics import ParetoSolution


def plot_2d_pareto(
    solutions: list[ParetoSolution],
    x_metric: str = "co2",
    y_metric: str = "accuracy",
    save_path: str | None = None,
    show_labels: bool = True,
) -> plt.Figure:
    """
    Plot 2D Pareto frontier.

    Args:
        solutions: List of solutions (both Pareto and non-Pareto)
        x_metric: Metric for x-axis (co2, energy, latency, memory)
        y_metric: Metric for y-axis (accuracy, co2, energy)
        save_path: Path to save figure (optional)
        show_labels: Whether to show configuration labels

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 8))

    # Separate Pareto and non-Pareto solutions
    pareto_x, pareto_y = [], []
    non_pareto_x, non_pareto_y = [], []

    for sol in solutions:
        x_val = _get_metric_value(sol, x_metric)
        y_val = _get_metric_value(sol, y_metric)

        if sol.is_pareto_optimal:
            pareto_x.append(x_val)
            pareto_y.append(y_val)
        else:
            non_pareto_x.append(x_val)
            non_pareto_y.append(y_val)

    # Plot points
    if non_pareto_x:
        plt.scatter(
            non_pareto_x, non_pareto_y, alpha=0.4, s=50, label=f"Non-Pareto ({len(non_pareto_x)})"
        )

    if pareto_x:
        plt.scatter(
            pareto_x,
            pareto_y,
            color="red",
            s=100,
            marker="*",
            label=f"Pareto Optimal ({len(pareto_x)})",
            zorder=5,
        )

        # Connect Pareto frontier
        sorted_indices = np.argsort(pareto_x)
        sorted_x = [pareto_x[i] for i in sorted_indices]
        sorted_y = [pareto_y[i] for i in sorted_indices]
        plt.plot(sorted_x, sorted_y, "r--", alpha=0.5, linewidth=2)

        # Add labels if requested
        if show_labels:
            for sol in solutions:
                if sol.is_pareto_optimal and sol.metrics.config:
                    bits = sol.metrics.config.quantization_bits
                    sparsity = sol.metrics.config.pruning_sparsity
                    if bits in [4, 8, 16] and sparsity in [0.0, 0.3, 0.5, 0.7]:
                        x_val = _get_metric_value(sol, x_metric)
                        y_val = _get_metric_value(sol, y_metric)
                        plt.annotate(
                            f"INT{bits}\n{sparsity:.0%}",
                            (x_val, y_val),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                            alpha=0.7,
                        )

    # Labels and formatting
    plt.xlabel(_get_metric_label(x_metric), fontsize=12)
    plt.ylabel(_get_metric_label(y_metric), fontsize=12)
    plt.title(
        f"Pareto Frontier: {_get_metric_label(y_metric)} vs {_get_metric_label(x_metric)}",
        fontsize=14,
    )
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved 2D Pareto plot to: {save_path}")

    return plt.gcf()


def plot_3d_pareto(solutions: list[ParetoSolution], save_path: str | None = None) -> plt.Figure:
    """
    Plot 3D Pareto frontier (Accuracy vs CO2 vs Energy).

    Args:
        solutions: List of solutions
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Separate data
    pareto_acc, pareto_co2, pareto_energy = [], [], []
    non_pareto_acc, non_pareto_co2, non_pareto_energy = [], [], []

    for sol in solutions:
        acc = sol.metrics.average_accuracy()
        co2 = sol.metrics.co2_kg
        energy = sol.metrics.energy_kwh

        if sol.is_pareto_optimal:
            pareto_acc.append(acc)
            pareto_co2.append(co2)
            pareto_energy.append(energy)
        else:
            non_pareto_acc.append(acc)
            non_pareto_co2.append(co2)
            non_pareto_energy.append(energy)

    # Plot
    if non_pareto_acc:
        ax.scatter(
            non_pareto_co2,
            non_pareto_energy,
            non_pareto_acc,
            alpha=0.3,
            s=30,
            c="blue",
            label="Non-Pareto",
        )

    if pareto_acc:
        ax.scatter(
            pareto_co2, pareto_energy, pareto_acc, c="red", s=60, marker="*", label="Pareto Optimal"
        )

    ax.set_xlabel("CO₂ (kg)", fontsize=10)
    ax.set_ylabel("Energy (kWh)", fontsize=10)
    ax.set_zlabel("Accuracy", fontsize=10)
    ax.set_title("3D Pareto Frontier: Multi-Objective Trade-offs", fontsize=12)
    ax.legend()
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved 3D Pareto plot to: {save_path}")

    return fig


def generate_analysis_report(
    solutions: list[ParetoSolution], pareto_frontier: list[ParetoSolution]
) -> dict:
    """
    Generate comprehensive analysis report.

    Args:
        solutions: All solutions explored
        pareto_frontier: Pareto-optimal solutions

    Returns:
        Dictionary with analysis metrics
    """
    report = {
        "total_solutions": len(solutions),
        "pareto_optimal_count": len(pareto_frontier),
        "pareto_ratio": len(pareto_frontier) / len(solutions) * 100 if solutions else 0,
        "best_accuracy": None,
        "best_carbon": None,
        "best_balanced": None,
        "configuration_distribution": {},
    }

    if not pareto_frontier:
        return report

    # Find best solutions
    best_acc = max(pareto_frontier, key=lambda x: x.metrics.average_accuracy())
    best_co2 = min(pareto_frontier, key=lambda x: x.metrics.co2_kg)

    def balance_score(sol):
        return sol.metrics.average_accuracy() * 2.0 - sol.metrics.co2_kg * 10.0

    best_balanced = max(pareto_frontier, key=balance_score)

    report["best_accuracy"] = {
        "config": vars(best_acc.metrics.config) if best_acc.metrics.config else None,
        "metrics": best_acc.metrics.to_dict(),
    }

    report["best_carbon"] = {
        "config": vars(best_co2.metrics.config) if best_co2.metrics.config else None,
        "metrics": best_co2.metrics.to_dict(),
    }

    report["best_balanced"] = {
        "config": vars(best_balanced.metrics.config) if best_balanced.metrics.config else None,
        "metrics": best_balanced.metrics.to_dict(),
    }

    # Configuration distribution
    for sol in pareto_frontier:
        if sol.metrics.config:
            bits = sol.metrics.config.quantization_bits
            if bits not in report["configuration_distribution"]:
                report["configuration_distribution"][bits] = 0
            report["configuration_distribution"][bits] += 1

    return report


def _get_metric_value(solution: ParetoSolution, metric: str) -> float:
    """Extract metric value from solution"""
    if metric == "accuracy":
        return solution.metrics.average_accuracy()
    elif metric == "co2":
        return solution.metrics.co2_kg
    elif metric == "energy":
        return solution.metrics.energy_kwh
    elif metric == "latency":
        return solution.metrics.latency_ms
    elif metric == "memory":
        return solution.metrics.memory_gb
    else:
        return 0.0


def _get_metric_label(metric: str) -> str:
    """Get display label for metric"""
    labels = {
        "accuracy": "Average Accuracy",
        "co2": "CO₂ Emissions (kg)",
        "energy": "Energy (kWh)",
        "latency": "Latency (ms)",
        "memory": "Memory (GB)",
    }
    return labels.get(metric, metric.capitalize())
