"""
Page 4: Interactive 3D Explorer

Interactive 3D visualization and high-dimensional exploration.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st

from agentic_compression.optimization.agent_driven import run_rq2_experiment
from agentic_compression.ui.components import render_pareto_solutions_table
from agentic_compression.ui.utils import export_results_json, init_session_state, run_async
from agentic_compression.ui.visualizations import (
    create_3d_pareto_plot,
    create_parallel_coordinates,
    create_radar_chart,
)

st.set_page_config(page_title="3D Explorer", page_icon="🎯", layout="wide")

# Initialize session state
init_session_state("exploration_results", None)

# Header
st.title("🎯 Interactive 3D Explorer")
st.markdown("Explore the multi-dimensional solution space with interactive visualizations.")

# Configuration
st.sidebar.header("Exploration Settings")

model = st.sidebar.selectbox(
    "Model", ["google/gemma-12b", "google/gemma3-270m"], help="Model to explore"
)

carbon_budget = st.sidebar.number_input(
    "Carbon Budget (kg)", min_value=1.0, max_value=20.0, value=5.0, step=0.5
)

accuracy_threshold = st.sidebar.slider(
    "Accuracy Threshold", min_value=0.80, max_value=0.99, value=0.93, step=0.01
)

# Run exploration
if st.sidebar.button("🔍 Explore Solution Space", type="primary"):
    with st.spinner("Exploring solution space..."):
        try:
            # Run RQ2 experiment to generate diverse solutions
            results = run_async(
                run_rq2_experiment(
                    model=model,
                    accuracy_threshold=accuracy_threshold,
                    carbon_budget=carbon_budget,
                )
            )

            st.session_state.exploration_results = results
            st.sidebar.success("✅ Exploration complete!")

        except Exception as e:
            st.sidebar.error(f"Exploration failed: {str(e)}")

# Display results
if st.session_state.exploration_results:
    results = st.session_state.exploration_results

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pareto Frontier Size", results.get("pareto_frontier_size", 0))

    with col2:
        st.metric("Total Solutions", len(results.get("solutions", [])))

    with col3:
        pareto_ratio = (
            results.get("pareto_frontier_size", 0) / len(results.get("solutions", []))
            if results.get("solutions")
            else 0
        )
        st.metric("Pareto Ratio", f"{pareto_ratio:.1%}")

    # Visualization tabs
    st.divider()
    tab1, tab2, tab3 = st.tabs(["3D Pareto Frontier", "Parallel Coordinates", "Radar Chart"])

    pareto_solutions = results.get("pareto_frontier", [])

    with tab1:
        st.subheader("3D Pareto Frontier: Accuracy vs CO₂ vs Energy")

        if pareto_solutions:
            fig = create_3d_pareto_plot(pareto_solutions)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "🖱️ **Interactive controls**: "
                "Click and drag to rotate, scroll to zoom, double-click to reset"
            )

        else:
            st.warning("No Pareto solutions to display")

    with tab2:
        st.subheader("Parallel Coordinates: Multi-Dimensional Exploration")

        all_solutions = results.get("solutions", [])

        if all_solutions:
            fig = create_parallel_coordinates(all_solutions)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "💡 **Tip**: Drag axis ranges to filter solutions. "
                "Red lines indicate Pareto-optimal solutions."
            )

        else:
            st.warning("No solutions to display")

    with tab3:
        st.subheader("Radar Chart: Multi-Objective Performance")

        if pareto_solutions:
            # Get best solution
            best_solution = max(pareto_solutions, key=lambda s: s.metrics.average_accuracy())

            # Normalize metrics for radar chart
            metrics_dict = {
                "Accuracy": best_solution.metrics.average_accuracy(),
                "Low Carbon": 1.0 - min(best_solution.metrics.co2_kg / 0.05, 1.0),
                "Low Latency": 1.0 - min(best_solution.metrics.latency_ms / 100, 1.0),
                "Low Memory": 1.0 - min(best_solution.metrics.memory_gb / 24, 1.0),
                "Low Energy": 1.0 - min(best_solution.metrics.energy_kwh / 0.05, 1.0),
                "High Throughput": min(best_solution.metrics.throughput_tps / 2000, 1.0),
            }

            fig = create_radar_chart(metrics_dict, title="Best Solution Performance Profile")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No solutions to display")

    # Pareto solutions table
    st.divider()
    st.subheader("Pareto-Optimal Solutions")

    if pareto_solutions:
        render_pareto_solutions_table(pareto_solutions)

    # Key findings
    if "key_findings" in results:
        st.divider()
        st.subheader("Key Findings")
        for finding in results["key_findings"]:
            st.write(f"- {finding}")

    # Carbon impact analysis
    if "carbon_impact_analysis" in results:
        st.divider()
        st.subheader("Carbon Impact Analysis")

        impact = results["carbon_impact_analysis"]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Carbon Reduction Potential",
                f"{impact.get('carbon_reduction_percent', 0):.1f}%",
            )

        with col2:
            st.metric(
                "Best Carbon Config",
                f"{impact.get('best_carbon_config', {}).get('quantization_bits', 'N/A')} bits",
            )

        with col3:
            st.metric(
                "CO₂ Savings",
                f"{impact.get('absolute_reduction_kg', 0):.4f}kg",
            )

    # Export
    st.divider()
    json_str = export_results_json(results)
    st.download_button(
        "📥 Download Exploration Results",
        data=json_str,
        file_name="exploration_results.json",
        mime="application/json",
    )

else:
    st.info("👈 Configure exploration settings and click **Explore Solution Space** to start.")

    st.divider()
    st.subheader("What to Expect")
    st.write(
        "After the exploration runs, this page renders live Plotly figures for the Pareto frontier, "
        "parallel coordinates, and radar charts. All visualizations are derived from the real "
        "solutions generated by `run_rq2_experiment`."
    )
