"""
Agentic Carbon-Efficient LLM Compression Framework - Streamlit UI

Main application entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Agentic Compression Framework",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #ecf0f1;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<p class="main-header">🌱 Agentic Compression Framework</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Carbon-Efficient LLM Compression with Multi-Agent Optimization</p>',
    unsafe_allow_html=True,
)

# Introduction
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
        Welcome to the **Agentic Carbon-Efficient LLM Compression Framework**!
        This tool implements research from Liu et al. (2025) using **LangGraph**
        multi-agent systems to optimize Large Language Models while balancing
        **accuracy**, **efficiency**, and **carbon footprint**.
        """
    )

# Features
st.markdown("---")
st.header("✨ Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="feature-card">
        <h3>🚀 Quick Optimization</h3>
        <p>Run single optimization experiments with custom configurations and constraints.</p>
        <p><strong>Navigate to:</strong> Quick Optimization →</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-card">
        <h3>🔬 Experiment Comparison</h3>
        <p>Run multiple configurations and compare results side-by-side.</p>
        <p><strong>Navigate to:</strong> Experiment Comparison →</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
        <h3>📊 Advanced Visualization</h3>
        <p>Explore research questions (RQ1-4) with advanced visualizations.</p>
        <p><strong>Navigate to:</strong> Advanced Visualization →</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-card">
        <h3>🎯 Interactive 3D Explorer</h3>
        <p>Explore multi-dimensional solution space with interactive 3D plots.</p>
        <p><strong>Navigate to:</strong> Interactive 3D Explorer →</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Research Questions
st.markdown("---")
st.header("🔬 Research Questions")

with st.expander("**RQ1: Dynamic vs Static Compression**"):
    st.markdown(
        """
        **Question:** How does the multi-agent compression framework compare to static
        one-shot compression approaches?

        **Methodology:** Compare agent-driven dynamic optimization against predefined
        static configurations.

        **Metrics:** Accuracy, carbon efficiency, convergence speed
        """
    )

with st.expander("**RQ2: Agent-Driven Optimization Impact**"):
    st.markdown(
        """
        **Question:** What is the impact of dynamic pruning/quantization adjustment on
        energy consumption and carbon footprint?

        **Methodology:** Analyze carbon reduction through adaptive configuration.

        **Metrics:** Energy consumption, CO₂ emissions, Pareto frontier characteristics
        """
    )

with st.expander("**RQ3: Weighting Scheme Configuration**"):
    st.markdown(
        """
        **Question:** How do different weighting configurations affect the Pareto frontier
        and optimal strategy selection?

        **Methodology:** Test balanced, accuracy-focused, carbon-focused, and efficiency-focused
        weight schemes.

        **Metrics:** Frontier size, diversity, optimal configurations per scheme
        """
    )

with st.expander("**RQ4: Resource-Constrained Adaptation**"):
    st.markdown(
        """
        **Question:** How does the framework adapt in resource-constrained environments?

        **Methodology:** Test deployment in edge devices, mobile, cloud, and carbon-intensive
        data centers.

        **Metrics:** Feasibility, efficiency scores, deployment recommendations
        """
    )

# Architecture Overview
st.markdown("---")
st.header("🏗️ Architecture")

st.markdown(
    """
    The framework uses **LangGraph** for agent orchestration with the following components:

    - **Core Modules:** Configuration, metrics, Pareto frontier algorithms
    - **Optimization:** Research question implementations (RQ1-4)
    - **Tools:** LangChain tools for compression, evaluation, and carbon monitoring
    - **Workflow:** LangGraph state-based workflow with checkpointing
    - **Visualization:** Interactive Plotly plots and analysis
    """
)

# Quick Start
st.markdown("---")
st.header("🚀 Quick Start Guide")

st.markdown(
    """
    1. **👈 Navigate** to **Quick Optimization** in the sidebar
    2. **⚙️ Configure** your optimization settings:
       - Select model (Gemma, LLaMA, etc.)
       - Set quantization bits and pruning sparsity
       - Define carbon budget and accuracy threshold
    3. **▶️ Run** the optimization
    4. **📊 Analyze** results with interactive visualizations
    5. **📥 Download** results as JSON for further analysis

    **Tip:** Start with the default settings to understand the framework, then experiment
    with different configurations!
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #7f8c8d;">
    <p><strong>Agentic Carbon-Efficient LLM Compression Framework v2.0</strong></p>
    <p>Research by Liu et al. (2025) | Implementation with LangGraph</p>
    <p>🌱 Optimizing AI for a Sustainable Future 🌱</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation help
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        ### 📖 Navigation

        Use the page selector above to navigate between:

        - **🚀 Quick Optimization**
        - **📊 Advanced Visualization**
        - **🔬 Experiment Comparison**
        - **🎯 Interactive 3D Explorer**

        ### 💡 Need Help?

        - Check the main page for overview
        - Each page has built-in instructions
        - Export results for offline analysis

        ### 🔗 Resources

        - [Documentation](https://github.com)
        - [Research Paper](https://arxiv.org)
        - [Report Issues](https://github.com/issues)
        """
    )
