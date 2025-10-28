"""
Plotly visualizations for Streamlit UI.

Provides interactive 2D/3D Pareto frontier plots and analysis charts.
"""

import plotly.express as px
import plotly.graph_objects as go

from ..core.metrics import ParetoSolution


def create_2d_pareto_plot(
    solutions: list[ParetoSolution],
    x_metric: str = "co2_kg",
    y_metric: str = "accuracy",
    title: str = "Pareto Frontier: Accuracy vs Carbon",
) -> go.Figure:
    """
    Create interactive 2D Pareto frontier plot.

    Args:
        solutions: List of solutions
        x_metric: X-axis metric name
        y_metric: Y-axis metric name
        title: Plot title

    Returns:
        Plotly figure
    """
    # Extract data
    x_data = []
    y_data = []
    labels = []
    colors = []

    for i, sol in enumerate(solutions):
        if x_metric == "accuracy":
            x_val = sol.metrics.average_accuracy()
        else:
            x_val = getattr(sol.metrics, x_metric, 0)

        if y_metric == "accuracy":
            y_val = sol.metrics.average_accuracy()
        else:
            y_val = getattr(sol.metrics, y_metric, 0)

        x_data.append(x_val)
        y_data.append(y_val)

        config = sol.metrics.config
        if config:
            label = f"INT{config.quantization_bits}, {config.pruning_sparsity:.0%} pruned"
        else:
            label = f"Solution {i+1}"

        labels.append(label)
        colors.append("Pareto-optimal" if sol.is_pareto_optimal else "Dominated")

    # Create figure
    fig = px.scatter(
        x=x_data,
        y=y_data,
        color=colors,
        hover_data={"Configuration": labels},
        title=title,
        labels={
            "x": x_metric.replace("_", " ").title(),
            "y": y_metric.replace("_", " ").title(),
        },
        color_discrete_map={"Pareto-optimal": "#e74c3c", "Dominated": "#95a5a6"},
    )

    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(height=600, hovermode="closest")

    return fig


def create_3d_pareto_plot(
    solutions: list[ParetoSolution], title: str = "3D Pareto Frontier"
) -> go.Figure:
    """
    Create interactive 3D Pareto frontier plot.

    Args:
        solutions: List of solutions
        title: Plot title

    Returns:
        Plotly figure
    """
    # Extract data
    x_data = []  # CO2
    y_data = []  # Energy
    z_data = []  # Accuracy
    labels = []
    colors = []

    for i, sol in enumerate(solutions):
        x_data.append(sol.metrics.co2_kg)
        y_data.append(sol.metrics.energy_kwh)
        z_data.append(sol.metrics.average_accuracy())

        config = sol.metrics.config
        if config:
            label = f"INT{config.quantization_bits}<br>{config.pruning_sparsity:.0%} pruned"
        else:
            label = f"Solution {i+1}"

        labels.append(label)
        colors.append(1 if sol.is_pareto_optimal else 0)

    # Create 3D scatter
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_data,
                y=y_data,
                z=z_data,
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors,
                    colorscale=[[0, "#95a5a6"], [1, "#e74c3c"]],
                    showscale=False,
                    line=dict(width=1, color="white"),
                ),
                text=labels,
                hovertemplate="<b>%{text}</b><br>"
                + "CO₂: %{x:.4f}kg<br>"
                + "Energy: %{y:.4f}kWh<br>"
                + "Accuracy: %{z:.1%}<br>"
                + "<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="CO₂ Emissions (kg)",
            yaxis_title="Energy (kWh)",
            zaxis_title="Accuracy",
        ),
        height=700,
    )

    return fig


def create_weight_comparison_chart(comparison_data: dict) -> go.Figure:
    """
    Create weight scheme comparison chart.

    Args:
        comparison_data: Weight scheme comparison data

    Returns:
        Plotly figure
    """

    # Extract data
    schemes = []
    frontier_sizes = []
    avg_accuracies = []
    avg_carbons = []

    for scheme, data in comparison_data.items():
        schemes.append(scheme.replace("_", " ").title())
        frontier_sizes.append(data["frontier_size"])
        avg_accuracies.append(data["accuracy_range"]["mean"])
        avg_carbons.append(data["carbon_range"]["mean"])

    # Create subplot with multiple traces
    fig = go.Figure()

    fig.add_trace(go.Bar(name="Frontier Size", x=schemes, y=frontier_sizes, marker_color="#3498db"))

    fig.update_layout(
        title="Weight Scheme Comparison: Frontier Sizes",
        xaxis_title="Weight Scheme",
        yaxis_title="Pareto Frontier Size",
        height=500,
    )

    return fig


def create_environment_comparison_chart(env_results: dict) -> go.Figure:
    """
    Create environment comparison chart.

    Args:
        env_results: Environment-specific results

    Returns:
        Plotly figure
    """
    envs = []
    accuracies = []
    carbons = []
    efficiencies = []

    for env_name, result in env_results.items():
        envs.append(env_name.replace("_", " ").title())
        accuracies.append(result["performance"]["accuracy"])
        carbons.append(result["performance"]["daily_co2_kg"])
        efficiencies.append(result["efficiency_score"])

    # Create grouped bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(name="Accuracy", x=envs, y=accuracies, marker_color="#2ecc71", yaxis="y"))

    fig.add_trace(
        go.Bar(name="Efficiency Score", x=envs, y=efficiencies, marker_color="#3498db", yaxis="y")
    )

    fig.add_trace(
        go.Bar(
            name="Daily CO₂ (kg)",
            x=envs,
            y=carbons,
            marker_color="#e74c3c",
            yaxis="y2",
            opacity=0.7,
        )
    )

    fig.update_layout(
        title="Environment Comparison",
        xaxis_title="Environment",
        yaxis=dict(title="Accuracy / Efficiency", side="left"),
        yaxis2=dict(title="Daily CO₂ (kg)", side="right", overlaying="y"),
        barmode="group",
        height=600,
    )

    return fig


def create_radar_chart(metrics_dict: dict, title: str = "Multi-Objective Comparison") -> go.Figure:
    """
    Create radar chart for multi-objective comparison.

    Args:
        metrics_dict: Dictionary of metrics
        title: Plot title

    Returns:
        Plotly figure
    """
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Normalize values to 0-1 scale
    normalized_values = []
    for v in values:
        if isinstance(v, (int, float)):
            normalized_values.append(min(v, 1.0))
        else:
            normalized_values.append(0.5)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(r=normalized_values, theta=categories, fill="toself", name="Performance")
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title=title
    )

    return fig


def create_parallel_coordinates(solutions: list[ParetoSolution]) -> go.Figure:
    """
    Create parallel coordinates plot for high-dimensional exploration.

    Args:
        solutions: List of solutions

    Returns:
        Plotly figure
    """
    import pandas as pd

    # Extract data
    data = []
    for sol in solutions:
        config = sol.metrics.config
        data.append(
            {
                "Quantization": config.quantization_bits if config else 8,
                "Sparsity": config.pruning_sparsity if config else 0.0,
                "Accuracy": sol.metrics.average_accuracy(),
                "CO₂": sol.metrics.co2_kg,
                "Latency": sol.metrics.latency_ms,
                "Memory": sol.metrics.memory_gb,
                "Pareto": 1 if sol.is_pareto_optimal else 0,
            }
        )

    df = pd.DataFrame(data)

    fig = px.parallel_coordinates(
        df,
        dimensions=["Quantization", "Sparsity", "Accuracy", "CO₂", "Latency", "Memory"],
        color="Pareto",
        color_continuous_scale=[[0, "#95a5a6"], [1, "#e74c3c"]],
        title="Parallel Coordinates: Multi-Dimensional Exploration",
    )

    fig.update_layout(height=600)

    return fig
