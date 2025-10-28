"""
Page 3: Experiment Comparison

Run and compare multiple configurations side-by-side.
"""

import streamlit as st

from ...graph.workflow import run_compression_optimization
from ..utils import export_results_json, init_session_state, run_async

st.set_page_config(page_title="Experiment Comparison", page_icon="🔬", layout="wide")

# Initialize session state
init_session_state("experiments", [])
init_session_state("experiment_counter", 0)

# Header
st.title("🔬 Experiment Comparison")
st.markdown("Run multiple configurations and compare results side-by-side.")

# Sidebar: Add new experiment
st.sidebar.header("Add New Experiment")

exp_name = st.sidebar.text_input(
    "Experiment Name", value=f"Experiment {st.session_state.experiment_counter + 1}"
)

exp_objective = st.sidebar.selectbox(
    "Optimization Goal",
    [
        "Edge deployment with minimal carbon",
        "Carbon-efficient compression",
        "Accuracy-focused optimization",
        "Balanced approach",
        "Custom",
    ],
)

if exp_objective == "Custom":
    exp_objective = st.sidebar.text_area("Custom Objective", value="")

col1, col2 = st.sidebar.columns(2)
with col1:
    exp_bits = st.selectbox("Quant Bits", [4, 8, 16, 32], index=1)
with col2:
    exp_sparsity = st.selectbox("Pruning %", [0.0, 0.1, 0.3, 0.5, 0.7], index=2)

col3, col4 = st.sidebar.columns(2)
with col3:
    exp_carbon = st.number_input("Carbon Budget", 1.0, 10.0, 5.0, 0.5)
with col4:
    exp_threshold = st.number_input("Accuracy Threshold", 0.80, 0.99, 0.90, 0.01)

if st.sidebar.button("➕ Add & Run Experiment", type="primary"):
    with st.spinner(f"Running {exp_name}..."):
        try:
            results = run_async(
                run_compression_optimization(
                    objective=exp_objective,
                    carbon_budget=exp_carbon,
                    max_iterations=10,
                    accuracy_threshold=exp_threshold,
                )
            )

            st.session_state.experiments.append(
                {
                    "name": exp_name,
                    "objective": exp_objective,
                    "carbon_budget": exp_carbon,
                    "accuracy_threshold": exp_threshold,
                    "results": results,
                }
            )
            st.session_state.experiment_counter += 1
            st.sidebar.success(f"✅ {exp_name} completed!")

        except Exception as e:
            st.sidebar.error(f"Failed: {str(e)}")

# Main content: Comparison view
if st.session_state.experiments:
    st.subheader(f"Comparing {len(st.session_state.experiments)} Experiments")

    # Comparison table
    import pandas as pd

    comparison_data = []
    for exp in st.session_state.experiments:
        res = exp["results"]
        best = res.get("best_solution", {})

        comparison_data.append(
            {
                "Experiment": exp["name"],
                "Objective": (
                    exp["objective"][:40] + "..."
                    if len(exp["objective"]) > 40
                    else exp["objective"]
                ),
                "Accuracy": (
                    f"{best.get('accuracy', 0):.1%}"
                    if isinstance(best.get("accuracy"), (int, float))
                    else "N/A"
                ),
                "CO₂ (kg)": f"{best.get('co2_kg', 0):.4f}",
                "Latency (ms)": f"{best.get('latency_ms', 0):.1f}",
                "Memory (GB)": f"{best.get('memory_gb', 0):.2f}",
                "Pareto Size": res.get("pareto_optimal_count", 0),
                "Carbon Used": f"{res.get('carbon_used', 0):.4f}kg",
            }
        )

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Visualizations
    st.divider()
    st.subheader("Visualization Comparison")

    # Collect all solutions for overlay
    from ...core.metrics import EvaluationMetrics, ParetoSolution

    all_solutions = []
    experiment_labels = []

    for exp in st.session_state.experiments:
        if "pareto_frontier" in exp["results"]:
            for sol_dict in exp["results"]["pareto_frontier"]:
                metrics = EvaluationMetrics(
                    accuracy=sol_dict.get("accuracy", {}),
                    latency_ms=sol_dict.get("latency_ms", 0),
                    memory_gb=sol_dict.get("memory_gb", 0),
                    energy_kwh=sol_dict.get("energy_kwh", 0),
                    co2_kg=sol_dict.get("co2_kg", 0),
                    throughput_tps=sol_dict.get("throughput_tps", 0),
                    compression_ratio=sol_dict.get("compression_ratio", 1.0),
                )
                solution = ParetoSolution(metrics=metrics)
                all_solutions.append(solution)
                experiment_labels.append(exp["name"])

    if all_solutions:
        # Create overlay plot
        import plotly.graph_objects as go

        fig = go.Figure()

        # Plot each experiment separately
        for exp in st.session_state.experiments:
            if "pareto_frontier" in exp["results"]:
                x_data = []
                y_data = []

                for sol_dict in exp["results"]["pareto_frontier"]:
                    x_data.append(sol_dict.get("co2_kg", 0))
                    if isinstance(sol_dict.get("accuracy"), dict):
                        # Average accuracy from dict
                        acc_values = list(sol_dict.get("accuracy", {}).values())
                        y_val = sum(acc_values) / len(acc_values) if acc_values else 0
                    else:
                        y_val = sol_dict.get("accuracy", 0)
                    y_data.append(y_val)

                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="markers",
                        name=exp["name"],
                        marker=dict(size=10),
                    )
                )

        fig.update_layout(
            title="Pareto Frontier Overlay: All Experiments",
            xaxis_title="CO₂ Emissions (kg)",
            yaxis_title="Accuracy",
            height=600,
            hovermode="closest",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Export all experiments
    st.divider()
    col1, col2 = st.columns([1, 3])

    with col1:
        export_data = {
            "experiments": st.session_state.experiments,
            "total_count": len(st.session_state.experiments),
        }
        json_str = export_results_json(export_data)
        st.download_button(
            "📥 Download All Results",
            data=json_str,
            file_name="all_experiments.json",
            mime="application/json",
        )

    with col2:
        if st.button("🗑️ Clear All Experiments", type="secondary"):
            st.session_state.experiments = []
            st.session_state.experiment_counter = 0
            st.rerun()

else:
    st.info("👈 Add your first experiment using the sidebar to start comparing.")
