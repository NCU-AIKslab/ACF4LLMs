"""
Page 2: Advanced Visualization

Advanced visualization features for RQ1, RQ3, RQ4 analyses.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import streamlit as st

from agentic_compression.optimization.dynamic_vs_static import run_rq1_experiment
from agentic_compression.optimization.resource_adaptation import run_rq4_experiment
from agentic_compression.optimization.weighting import run_rq3_experiment
from agentic_compression.ui.utils import export_results_json, init_session_state, run_async
from agentic_compression.ui.visualizations import (
    create_environment_comparison_chart,
    create_weight_comparison_chart,
)

st.set_page_config(page_title="Advanced Visualization", page_icon="📊", layout="wide")

# Initialize session state
init_session_state("rq_results", {})

# Header
st.title("📊 Advanced Visualization")
st.markdown("Explore research questions with advanced visualizations and analysis.")

# Tabs for different RQs
tab1, tab2, tab3 = st.tabs(
    ["RQ1: Dynamic vs Static", "RQ3: Weight Analysis", "RQ4: Environment Adaptation"]
)

# ============================================================================
# RQ1: Dynamic vs Static Comparison
# ============================================================================
with tab1:
    st.subheader("RQ1: Dynamic vs Static Compression Comparison")
    st.markdown("Compare the multi-agent dynamic approach against static one-shot compression.")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        rq1_model = st.selectbox(
            "Model", ["google/gemma-12b", "google/gemma3-270m"], key="rq1_model"
        )

    with col2:
        rq1_threshold = st.slider("Accuracy Threshold", 0.85, 0.95, 0.93, 0.01, key="rq1_threshold")

    with col3:
        rq1_budget = st.number_input("Carbon Budget (kg)", 1.0, 10.0, 5.0, 0.5, key="rq1_budget")

    if st.button("▶️ Run RQ1 Experiment", type="primary", key="run_rq1"):
        with st.spinner("Running RQ1 experiment..."):
            try:
                results = run_async(
                    run_rq1_experiment(
                        model=rq1_model,
                        accuracy_threshold=rq1_threshold,
                        carbon_budget=rq1_budget,
                    )
                )
                st.session_state.rq_results["rq1"] = results
                st.success("✅ RQ1 experiment complete!")

            except Exception as e:
                st.error(f"Experiment failed: {str(e)}")

    # Display RQ1 results
    if "rq1" in st.session_state.rq_results:
        results = st.session_state.rq_results["rq1"]

        st.divider()
        st.subheader("Results Comparison")

        # Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Dynamic Approach")
            dyn = results["comparison"]["dynamic_approach"]
            st.metric("Best Accuracy", f"{dyn['best_accuracy']:.1%}")
            st.metric("Best Carbon", f"{dyn['best_carbon']:.4f}kg")
            st.metric("Iterations", dyn["iterations"])

        with col2:
            st.markdown("### Static Approach")
            sta = results["comparison"]["static_approach"]
            st.metric("Best Accuracy", f"{sta['best_accuracy']:.1%}")
            st.metric("Best Carbon", f"{sta['best_carbon']:.4f}kg")
            st.metric("Configs Tested", sta["configs_tested"])

        # Key findings
        st.divider()
        st.subheader("Key Findings")
        for finding in results["comparison"]["key_findings"]:
            st.write(f"- {finding}")

        # Conclusion
        st.info(f"**Conclusion**: {results['conclusion']}")

        # Export
        json_str = export_results_json(results)
        st.download_button(
            "📥 Download RQ1 Results",
            data=json_str,
            file_name="rq1_results.json",
            mime="application/json",
        )

# ============================================================================
# RQ3: Weighting Scheme Analysis
# ============================================================================
with tab2:
    st.subheader("RQ3: Weighting Scheme Impact Analysis")
    st.markdown("Analyze how different objective weights affect the Pareto frontier.")

    col1, col2 = st.columns([1, 1])

    with col1:
        rq3_model = st.selectbox(
            "Model", ["google/gemma-12b", "google/gemma3-270m"], key="rq3_model"
        )

    with col2:
        rq3_budget = st.number_input("Carbon Budget (kg)", 1.0, 10.0, 5.0, 0.5, key="rq3_budget")

    if st.button("▶️ Run RQ3 Experiment", type="primary", key="run_rq3"):
        with st.spinner("Running RQ3 experiment..."):
            try:
                results = run_async(run_rq3_experiment(model=rq3_model, carbon_budget=rq3_budget))
                st.session_state.rq_results["rq3"] = results
                st.success("✅ RQ3 experiment complete!")

            except Exception as e:
                st.error(f"Experiment failed: {str(e)}")

    # Display RQ3 results
    if "rq3" in st.session_state.rq_results:
        results = st.session_state.rq_results["rq3"]

        st.divider()
        st.subheader("Weight Scheme Comparison")

        # Frontier sizes chart
        if "scheme_comparison" in results["analysis"]:
            fig = create_weight_comparison_chart(results["analysis"]["scheme_comparison"])
            st.plotly_chart(fig, use_container_width=True)

        # Optimal configurations
        st.divider()
        st.subheader("Optimal Configurations by Weight Scheme")

        if "optimal_solutions" in results["analysis"]:
            import pandas as pd

            df = pd.DataFrame(results["analysis"]["optimal_solutions"]).T
            st.dataframe(df, use_container_width=True)

        # Key findings
        st.divider()
        st.subheader("Key Findings")
        for finding in results["analysis"]["key_findings"]:
            st.write(f"- {finding}")

        # Recommendations
        st.divider()
        st.subheader("Recommendations")
        for use_case, rec in results["recommendations"].items():
            st.write(f"**{use_case.replace('_', ' ').title()}**: {rec}")

        # Export
        json_str = export_results_json(results)
        st.download_button(
            "📥 Download RQ3 Results",
            data=json_str,
            file_name="rq3_results.json",
            mime="application/json",
        )

# ============================================================================
# RQ4: Environment Adaptation
# ============================================================================
with tab3:
    st.subheader("RQ4: Resource-Constrained Environment Adaptation")
    st.markdown("Test framework adaptation across different deployment environments.")

    col1, col2 = st.columns([1, 1])

    with col1:
        rq4_model = st.selectbox(
            "Model", ["google/gemma-12b", "google/gemma3-270m"], key="rq4_model"
        )

    with col2:
        rq4_threshold = st.slider("Accuracy Threshold", 0.80, 0.95, 0.85, 0.01, key="rq4_threshold")

    if st.button("▶️ Run RQ4 Experiment", type="primary", key="run_rq4"):
        with st.spinner("Running RQ4 experiment..."):
            try:
                results = run_async(
                    run_rq4_experiment(model=rq4_model, accuracy_threshold=rq4_threshold)
                )
                st.session_state.rq_results["rq4"] = results
                st.success("✅ RQ4 experiment complete!")

            except Exception as e:
                st.error(f"Experiment failed: {str(e)}")

    # Display RQ4 results
    if "rq4" in st.session_state.rq_results:
        results = st.session_state.rq_results["rq4"]

        st.divider()
        st.subheader("Environment Comparison")

        # Environment comparison chart
        if "environment_results" in results["adaptation_analysis"]:
            fig = create_environment_comparison_chart(
                results["adaptation_analysis"]["environment_results"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Feasibility summary
        st.divider()
        st.subheader("Feasibility Summary")
        feasibility = results["adaptation_analysis"]["cross_environment_comparison"][
            "feasibility_summary"
        ]
        st.metric(
            "Feasible Environments",
            f"{feasibility['feasible_count']} / {feasibility['total_environments']}",
        )
        st.write("**Feasible:** " + ", ".join(feasibility["feasible_environments"]))

        # Deployment recommendations
        st.divider()
        st.subheader("Deployment Recommendations")
        for env, rec in results["adaptation_analysis"]["deployment_recommendations"].items():
            st.write(f"**{env.replace('_', ' ').title()}**: {rec}")

        # Key findings
        st.divider()
        st.subheader("Key Findings")
        for finding in results["adaptation_analysis"]["key_findings"]:
            st.write(f"- {finding}")

        # Conclusion
        st.info(f"**Conclusion**: {results['conclusion']}")

        # Export
        json_str = export_results_json(results)
        st.download_button(
            "📥 Download RQ4 Results",
            data=json_str,
            file_name="rq4_results.json",
            mime="application/json",
        )
