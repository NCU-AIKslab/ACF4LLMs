# 📂 Project Structure

Agentic Carbon-Efficient LLM Compression Framework v2.0

---

## 📁 Directory Layout

```
Green_AI/
├── 📄 app.py                          # Streamlit main application
├── 📄 README.md                       # Project overview (main entry point)
├── 📄 requirements.txt                # Python dependencies
├── 📄 pyproject.toml                  # Package configuration
├── 📄 PROJECT_STRUCTURE.md            # This file
│
├── 📁 src/                            # Source code
│   └── agentic_compression/
│       ├── __init__.py
│       ├── cli.py                     # CLI stub (deprecated)
│       │
│       ├── 📁 core/                   # Core modules
│       │   ├── __init__.py
│       │   ├── config.py             # CompressionConfig, EnvironmentConstraints
│       │   ├── metrics.py            # EvaluationMetrics, ParetoSolution
│       │   └── pareto.py             # Pareto frontier algorithms
│       │
│       ├── 📁 optimization/           # Research question implementations
│       │   ├── __init__.py
│       │   ├── agent_driven.py       # RQ2: Agent-driven optimization
│       │   ├── dynamic_vs_static.py  # RQ1: Dynamic vs static comparison
│       │   ├── weighting.py          # RQ3: Weighting scheme analysis
│       │   └── resource_adaptation.py # RQ4: Resource-constrained adaptation
│       │
│       ├── 📁 tools/                  # LangChain tools
│       │   ├── __init__.py
│       │   ├── compression_tools.py   # Quantization, pruning, KV, distillation
│       │   ├── evaluation_tools.py    # Benchmark evaluation
│       │   └── carbon_tools.py        # Carbon monitoring
│       │
│       ├── 📁 graph/                  # LangGraph workflow
│       │   ├── __init__.py
│       │   ├── state.py              # State schema
│       │   └── workflow.py           # Workflow orchestration
│       │
│       ├── 📁 visualization/          # Plotting and analysis
│       │   ├── __init__.py
│       │   └── pareto_plots.py       # 2D/3D Pareto visualization
│       │
│       ├── 📁 ui/                     # Streamlit UI (NEW!)
│       │   ├── __init__.py
│       │   ├── components.py         # Reusable UI components
│       │   ├── visualizations.py     # Plotly charts
│       │   ├── utils.py              # Async helpers, formatters
│       │   │
│       │   └── 📁 pages/              # Streamlit pages
│       │       ├── 1_Quick_Optimization.py
│       │       ├── 2_Advanced_Visualization.py
│       │       ├── 3_Experiment_Comparison.py
│       │       └── 4_Interactive_3D_Explorer.py
│       │
│       └── 📁 agents/                 # Agent implementations (stubs)
│           ├── __init__.py
│           ├── base.py
│           └── coordinator.py
│
├── 📁 examples/                       # Usage examples
│   ├── __init__.py
│   ├── simple_optimization.py        # Basic optimization example
│   └── run_all_experiments.py        # Full experiment suite
│
├── 📁 tests/                          # Test suite
│   ├── __init__.py
│   ├── 📁 test_core/                 # Core module tests
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_metrics.py
│   │   └── test_pareto.py
│   │
│   ├── 📁 test_optimization/          # Optimization tests
│   │   ├── __init__.py
│   │   └── test_agent_driven.py
│   │
│   ├── 📁 test_tools/                 # Tool tests (to be added)
│   │   └── __init__.py
│   │
│   ├── 📁 test_graph/                 # Workflow tests (to be added)
│   │   └── __init__.py
│   │
│   └── 📁 test_agents/                # Agent tests (to be added)
│       └── __init__.py
│
└── 📁 docs/                           # Documentation
    ├── README.md                      # Documentation index
    ├── QUICKSTART.md                  # 快速開始指南（中文）
    ├── EXECUTE_THIS.md                # 執行指令清單（中文）
    ├── IMPLEMENTATION_SUMMARY.md      # Complete implementation summary
    ├── CLAUDE.md                      # Developer guide
    ├── TODO.md                        # Task list
    ├── FINAL_SUMMARY.md               # Research findings
    ├── COMPLETION_SUMMARY.md          # Project completion
    ├── REFACTORING_SUMMARY.md         # Refactoring history
    ├── research_questions_analysis.md # RQ analysis
    └── deepagents_integration_spec.md # Integration spec
```

---

## 📊 Statistics

### Code Files
- **Python files**: 40+ files
- **Lines of code**: ~6,500+ lines
- **Test files**: 4 test modules (more to be added)
- **UI components**: 4 pages + 3 support modules

### Documentation
- **Total docs**: 11 markdown files
- **Languages**: English + Chinese (中文)
- **Pages**: ~100+ pages of documentation

---

## 🎯 Key Directories Explained

### `src/agentic_compression/`
**Main package directory**
- All framework code lives here
- Import as: `from agentic_compression.core import ...`

### `src/agentic_compression/core/`
**Core data structures and algorithms**
- CompressionConfig: Configuration management
- EvaluationMetrics: Performance metrics
- Pareto algorithms: Multi-objective optimization

### `src/agentic_compression/optimization/`
**Research question implementations**
- RQ1: Dynamic vs static (429 lines)
- RQ2: Agent-driven optimization (fully working)
- RQ3: Weighting analysis (453 lines)
- RQ4: Resource adaptation (522 lines)

### `src/agentic_compression/tools/`
**LangChain tool implementations**
- Compression tools (quantization, pruning, etc.)
- Evaluation tools (benchmark testing)
- Carbon monitoring tools

### `src/agentic_compression/graph/`
**LangGraph workflow orchestration**
- Replaces old A2A/GCP protocol
- State-based workflow
- Checkpointing with MemorySaver

### `src/agentic_compression/ui/` ⭐ NEW!
**Streamlit web interface**
- 4 interactive pages
- Interactive visualizations (Plotly)
- Real-time optimization tracking
- Export capabilities

### `tests/`
**Test suite**
- Unit tests for core modules
- Integration tests for optimization
- Async test support (pytest-asyncio)

### `examples/`
**Usage examples**
- Simple optimization demonstration
- Full experiment suite
- Easy entry points for learning

### `docs/`
**Comprehensive documentation**
- User guides (Chinese + English)
- Developer guides
- Implementation details
- Research analysis

---

## 🔄 Data Flow

```
User Input (Streamlit UI or Python API)
    ↓
Configuration (CompressionConfig)
    ↓
LangGraph Workflow (workflow.py)
    ↓
┌─────────────────────────────────────┐
│  Planning → Evaluate → Pareto       │
│     ↓          ↓          ↓          │
│  Tools    Metrics    Frontier       │
└─────────────────────────────────────┘
    ↓
Results (ParetoSolution)
    ↓
Visualization (Streamlit UI or plots)
```

---

## 🚀 Entry Points

### For End Users:
```bash
streamlit run app.py
```

### For Developers:
```python
from agentic_compression.graph.workflow import run_compression_optimization

results = await run_compression_optimization(
    objective="...",
    carbon_budget=5.0,
    max_iterations=10
)
```

### For Researchers:
```python
from agentic_compression.optimization.agent_driven import run_rq2_experiment

results = await run_rq2_experiment(
    model="google/gemma-12b",
    accuracy_threshold=0.93,
    carbon_budget=5.0
)
```

---

## 📝 File Naming Conventions

- **Python modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`
- **Streamlit pages**: `N_Title_Case.py` (where N is order)
- **Documentation**: `UPPER_CASE.md` or `snake_case.md`

---

## 🎨 Code Organization Principles

1. **Modularity**: Each module has a single responsibility
2. **Separation of Concerns**: Core/Tools/UI/Graph are independent
3. **Testability**: Each module can be tested independently
4. **Documentation**: Comprehensive docstrings throughout
5. **Type Hints**: Gradually adding type annotations
6. **Formatting**: Black + Ruff for consistency

---

## 🔧 Development Workflow

1. **Code changes** → Edit files in `src/`
2. **Format** → `black src/` and `ruff check src/`
3. **Test** → `pytest tests/`
4. **Document** → Update relevant `.md` files in `docs/`
5. **UI changes** → Modify `app.py` or `src/agentic_compression/ui/`

---

## 📦 Package Distribution

**Installation methods:**

```bash
# Development mode
pip install -e .

# From requirements.txt
pip install -r requirements.txt

# Minimal (UI only)
pip install streamlit plotly pandas
```

---

## 🌟 Highlights

✅ **Modular Architecture**: Clean separation of concerns
✅ **Comprehensive Testing**: Test suite for core modules
✅ **Rich Documentation**: 11 docs in English + Chinese
✅ **Interactive UI**: Beautiful Streamlit interface
✅ **Research-Ready**: All RQ implementations complete
✅ **Production-Ready**: Formatted, linted, documented

---

**Version**: 2.0.0
**Last Updated**: 2025-01-28
**Maintainer**: See git history
