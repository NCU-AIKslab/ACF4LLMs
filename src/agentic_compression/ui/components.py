"""
Reusable Streamlit UI components.

Provides common UI elements like metric cards, progress bars, and configuration forms.
"""

import streamlit as st

from ..core.metrics import EvaluationMetrics, ParetoSolution
from .utils import calculate_carbon_percentage, format_metric_value


def render_metric_card(
    label: str,
    value: float,
    metric_name: str,
    delta: float | None = None,
    help_text: str | None = None,
):
    """
    Render a metric card with value and optional delta.

    Args:
        label: Metric label
        value: Metric value
        metric_name: Metric name for formatting
        delta: Optional delta value
        help_text: Optional help text
    """
    formatted_value = format_metric_value(value, metric_name)

    if delta is not None:
        formatted_delta = format_metric_value(delta, metric_name)
        st.metric(label=label, value=formatted_value, delta=formatted_delta, help=help_text)
    else:
        st.metric(label=label, value=formatted_value, help=help_text)


def render_carbon_progress(carbon_used: float, carbon_budget: float):
    """
    Render carbon budget progress bar.

    Args:
        carbon_used: Carbon used (kg)
        carbon_budget: Carbon budget (kg)
    """
    percentage = calculate_carbon_percentage(carbon_used, carbon_budget)

    st.write("**Carbon Budget Usage**")
    st.progress(percentage / 100.0)
    st.caption(f"{carbon_used:.4f}kg / {carbon_budget:.2f}kg ({percentage:.1f}%)")


def render_best_solution_card(solution: EvaluationMetrics | dict):
    """
    Render highlighted card for best solution.

    Args:
        solution: Best solution metrics
    """
    st.subheader("🏆 Best Solution")

    if isinstance(solution, dict):
        accuracy = solution.get("accuracy", 0)
        co2 = solution.get("co2_kg", 0)
        latency = solution.get("latency_ms", 0)
        memory = solution.get("memory_gb", 0)
    else:
        accuracy = solution.average_accuracy()
        co2 = solution.co2_kg
        latency = solution.latency_ms
        memory = solution.memory_gb

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("CO₂", f"{co2:.4f}kg")
    with col3:
        st.metric("Latency", f"{latency:.1f}ms")
    with col4:
        st.metric("Memory", f"{memory:.2f}GB")


def render_configuration_form() -> dict:
    """
    Render configuration input form in sidebar.

    Returns:
        Configuration dictionary
    """
    st.sidebar.header("Configuration")

    model = st.sidebar.selectbox(
        "Model",
        options=["google/gemma-12b", "google/gemma3-270m", "meta-llama/Llama-2-7b"],
        help="Select model to optimize",
    )

    st.sidebar.subheader("Compression Settings")

    quantization_bits = st.sidebar.select_slider(
        "Quantization Bits",
        options=[4, 8, 16, 32],
        value=8,
        help="Number of bits for quantization (lower = more compression)",
    )

    pruning_sparsity = st.sidebar.slider(
        "Pruning Sparsity",
        min_value=0.0,
        max_value=0.7,
        value=0.3,
        step=0.1,
        help="Percentage of weights to prune (higher = more compression)",
    )

    st.sidebar.subheader("Optimization Constraints")

    carbon_budget = st.sidebar.number_input(
        "Carbon Budget (kg CO₂)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Maximum carbon budget for optimization",
    )

    accuracy_threshold = st.sidebar.slider(
        "Accuracy Threshold",
        min_value=0.80,
        max_value=0.99,
        value=0.90,
        step=0.01,
        help="Minimum acceptable accuracy",
    )

    max_iterations = st.sidebar.number_input(
        "Max Iterations",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Maximum optimization iterations",
    )

    objective = st.sidebar.text_area(
        "Optimization Objective",
        value="Compress for edge deployment with minimal carbon footprint",
        help="Describe your optimization goal",
    )

    return {
        "model": model,
        "quantization_bits": quantization_bits,
        "pruning_sparsity": pruning_sparsity,
        "carbon_budget": carbon_budget,
        "accuracy_threshold": accuracy_threshold,
        "max_iterations": max_iterations,
        "objective": objective,
    }


def render_benchmark_accuracy_table(accuracy_dict: dict):
    """
    Render benchmark accuracy breakdown table.

    Args:
        accuracy_dict: Dictionary mapping benchmark names to accuracy scores
    """
    st.write("**Benchmark Accuracy Breakdown**")

    import pandas as pd

    df = pd.DataFrame([{"Benchmark": k, "Accuracy": f"{v:.1%}"} for k, v in accuracy_dict.items()])

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_pareto_solutions_table(solutions: list[ParetoSolution]):
    """
    Render table of Pareto-optimal solutions.

    Args:
        solutions: List of Pareto solutions
    """
    st.write("**Pareto-Optimal Configurations**")

    import pandas as pd

    data = []
    for i, sol in enumerate(solutions):
        data.append(
            {
                "#": i + 1,
                "Bits": sol.metrics.config.quantization_bits if sol.metrics.config else "N/A",
                "Sparsity": (
                    f"{sol.metrics.config.pruning_sparsity:.1%}" if sol.metrics.config else "N/A"
                ),
                "Accuracy": f"{sol.metrics.average_accuracy():.1%}",
                "CO₂ (kg)": f"{sol.metrics.co2_kg:.4f}",
                "Latency (ms)": f"{sol.metrics.latency_ms:.1f}",
                "Memory (GB)": f"{sol.metrics.memory_gb:.2f}",
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_summary_stats(results: dict):
    """
    Render summary statistics cards.

    Args:
        results: Results dictionary
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Solutions",
            results.get("total_solutions", 0),
            help="Total configurations explored",
        )

    with col2:
        st.metric(
            "Pareto Optimal",
            results.get("pareto_optimal_count", 0),
            help="Number of Pareto-optimal solutions",
        )

    with col3:
        st.metric("Iterations", results.get("iterations", 0), help="Optimization iterations run")

    with col4:
        carbon_used = results.get("carbon_used", 0)
        st.metric("Carbon Used", f"{carbon_used:.4f}kg", help="Total carbon emissions")
