"""
Page 1: Quick Optimization

Simple interface for running a single compression optimization.
"""

import streamlit as st

from ...graph.workflow import run_compression_optimization
from ..components import (
    render_benchmark_accuracy_table,
    render_best_solution_card,
    render_carbon_progress,
    render_configuration_form,
    render_pareto_solutions_table,
    render_summary_stats,
)
from ..utils import export_results_json, init_session_state, run_async
from ..visualizations import create_2d_pareto_plot

st.set_page_config(page_title="Quick Optimization", page_icon="🚀", layout="wide")

# Initialize session state
init_session_state("optimization_results", None)
init_session_state("optimization_running", False)

# Header
st.title("🚀 Quick Optimization")
st.markdown("Run a single optimization experiment with custom configuration and constraints.")

# Configuration form in sidebar
config = render_configuration_form()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Optimization Objective")
    st.info(config["objective"])

with col2:
    if st.button(
        "▶️ Run Optimization",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.optimization_running,
    ):
        st.session_state.optimization_running = True
        st.session_state.optimization_results = None

# Run optimization
if st.session_state.optimization_running:
    with st.status("Running optimization...", expanded=True) as status:
        st.write("Planning optimization strategy...")

        try:
            # Run workflow
            results = run_async(
                run_compression_optimization(
                    objective=config["objective"],
                    carbon_budget=config["carbon_budget"],
                    max_iterations=config["max_iterations"],
                    accuracy_threshold=config["accuracy_threshold"],
                )
            )

            st.write("✅ Optimization complete!")
            status.update(label="Optimization complete!", state="complete")

            st.session_state.optimization_results = results
            st.session_state.optimization_running = False

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            status.update(label="Optimization failed!", state="error")
            st.session_state.optimization_running = False

# Display results
if st.session_state.optimization_results:
    results = st.session_state.optimization_results

    st.divider()
    st.header("Results")

    # Summary statistics
    render_summary_stats(results)

    # Carbon budget progress
    st.divider()
    render_carbon_progress(
        carbon_used=results.get("carbon_used", 0),
        carbon_budget=config["carbon_budget"],
    )

    # Best solution
    st.divider()
    if "best_solution" in results:
        render_best_solution_card(results["best_solution"])

    # Pareto frontier visualization
    st.divider()
    st.subheader("📊 Pareto Frontier Visualization")

    if "pareto_frontier" in results and results["pareto_frontier"]:
        from ...core.metrics import ParetoSolution

        # Convert dict to ParetoSolution objects
        pareto_solutions = []
        for sol_dict in results["pareto_frontier"]:
            from ...core.metrics import EvaluationMetrics

            metrics = EvaluationMetrics(
                accuracy=sol_dict.get("accuracy", {}),
                latency_ms=sol_dict.get("latency_ms", 0),
                memory_gb=sol_dict.get("memory_gb", 0),
                energy_kwh=sol_dict.get("energy_kwh", 0),
                co2_kg=sol_dict.get("co2_kg", 0),
                throughput_tps=sol_dict.get("throughput_tps", 0),
                compression_ratio=sol_dict.get("compression_ratio", 1.0),
            )
            pareto_solutions.append(ParetoSolution(metrics=metrics))

        # Create 2D plot
        fig = create_2d_pareto_plot(pareto_solutions)
        st.plotly_chart(fig, use_container_width=True)

        # Solutions table
        st.divider()
        render_pareto_solutions_table(pareto_solutions)

    # Benchmark accuracy
    if "best_solution" in results and "accuracy" in results["best_solution"]:
        st.divider()
        render_benchmark_accuracy_table(results["best_solution"]["accuracy"])

    # Export results
    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        json_str = export_results_json(results)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json_str,
            file_name="optimization_results.json",
            mime="application/json",
        )

else:
    # Placeholder
    st.info("👆 Configure your optimization settings and click **Run Optimization** to start.")
