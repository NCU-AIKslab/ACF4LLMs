# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic Carbon-Efficient LLM Compression Framework** implementing research from Liu et al. (2025). The framework uses multi-agent systems with **LangGraph** to optimize Large Language Models while balancing accuracy, efficiency, and carbon footprint.

**Key Change**: Refactored from single-file scripts to a **standard Python package structure** with modular components. **All GCP/A2A Protocol dependencies removed** - now uses LangGraph-native agent orchestration.

## Package Structure

```
src/agentic_compression/
├── agents/          # Agent implementations (placeholder for future)
├── core/            # Core data structures and algorithms
│   ├── config.py    # CompressionConfig, EnvironmentConstraints
│   ├── metrics.py   # EvaluationMetrics, ParetoSolution
│   └── pareto.py    # Pareto frontier algorithms
├── optimization/    # Research question implementations
│   ├── agent_driven.py       # RQ2 (fully implemented)
│   ├── dynamic_vs_static.py  # RQ1 (placeholder)
│   ├── weighting.py          # RQ3 (placeholder)
│   └── resource_adaptation.py # RQ4 (placeholder)
├── tools/           # LangChain tools for agents
│   ├── compression_tools.py  # Quantization, pruning, etc.
│   ├── evaluation_tools.py   # Benchmark evaluation
│   └── carbon_tools.py       # Carbon monitoring
├── visualization/   # Plotting and analysis
│   └── pareto_plots.py       # 2D/3D Pareto visualization
└── graph/           # LangGraph workflow (replaces A2A/GCP)
    ├── state.py     # Workflow state schema
    └── workflow.py  # LangGraph workflow orchestration

examples/            # Usage examples
├── simple_optimization.py     # Quick start
└── run_all_experiments.py     # Full experiment suite
```

## Core Architecture

### LangGraph Workflow (Replaces A2A/GCP)

The framework now uses **LangGraph** for agent orchestration instead of custom A2A Protocol:
- **No GCP dependencies** - completely local
- State-based workflow with typed schemas (`CompressionState`)
- Nodes: plan → evaluate → pareto → refine
- Checkpointing with `MemorySaver`

Key file: `src/agentic_compression/graph/workflow.py`

### Pareto Frontier Analysis

The system computes Pareto-optimal solutions across multiple objectives:
- **Maximize**: Model accuracy across benchmarks (GSM8K, TruthfulQA, CommonsenseQA, HumanEval, BIG-Bench)
- **Minimize**: Latency, memory usage, energy consumption, CO₂ emissions

Key implementation:
- `src/agentic_compression/core/pareto.py:compute_pareto_frontier()` - Pareto dominance computation
- `src/agentic_compression/core/pareto.py:calculate_crowding_distance()` - Diversity measurement
- `src/agentic_compression/visualization/pareto_plots.py` - Visualization utilities

## Development Commands

### Running Experiments

```bash
# Quick start example
python examples/simple_optimization.py

# Run all experiments (RQ2 fully implemented)
python examples/run_all_experiments.py

# Install package in development mode
pip install -e .
```

### Testing

```bash
# Run tests (to be implemented)
pytest tests/

# Run with coverage
pytest --cov=agentic_compression tests/
```

### Package Installation

```bash
# Install from source
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,quantization,carbon]"
```

### Dependencies

Install core requirements:
```bash
pip install -r requirements.txt
```

Key dependencies (NO GCP):
- **LangChain/LangGraph** (>= 0.1.0): Multi-agent orchestration (replaces A2A/GCP)
- **PyTorch** (>= 2.1.0): Model operations
- **Transformers** (>= 4.36.0): HuggingFace model support
- **NumPy/SciPy**: Numerical optimization
- **Matplotlib**: Pareto frontier visualization

**Removed dependencies**:
- ❌ google-cloud-aiplatform (GCP not needed)
- ❌ google-auth (GCP not needed)
- ❌ grpcio (A2A protocol removed)
- ❌ DeepAgents (using standard LangGraph)

## Key Configuration Objects

### CompressionConfig
Defined in `src/agentic_compression/core/config.py`:
```python
from agentic_compression.core import CompressionConfig

config = CompressionConfig(
    quantization_bits=8,          # 4, 8, 16, or 32
    pruning_sparsity=0.3,         # 0.0 to 0.7
    context_length=4096,          # 1k to 32k
    carbon_budget=5.0,            # kg CO2
    accuracy_threshold=0.93       # minimum accuracy
)
```

### EvaluationMetrics
Defined in `src/agentic_compression/core/metrics.py`:
```python
from agentic_compression.core import EvaluationMetrics

metrics = EvaluationMetrics(
    accuracy={"gsm8k": 0.93, ...},  # Per-benchmark accuracy
    latency_ms=65.0,
    memory_gb=8.2,
    energy_kwh=0.042,
    co2_kg=0.017,
    throughput_tps=1500,
    compression_ratio=2.5
)
```

### Using the LangGraph Workflow
```python
from agentic_compression.graph.workflow import run_compression_optimization

results = await run_compression_optimization(
    objective="edge deployment with low carbon",
    carbon_budget=5.0,
    max_iterations=10
)
```

## Research Questions (RQ1-RQ4)

The enhanced implementation tests four core research questions:

1. **RQ1 (Dynamic vs Static)**: Compare agentic dynamic compression to static one-shot approaches
   - Implementation: `DynamicVsStaticComparison` class

2. **RQ2 (Agent-Driven Optimization)**: Impact of dynamic pruning/quantization adjustment on energy/carbon
   - Implementation: `AgentDrivenOptimization` class

3. **RQ3 (Weight Configuration)**: How weight schemes affect Pareto frontier and strategy selection
   - Implementation: `WeightingSchemeAnalysis` class
   - Weight schemes: balanced, accuracy_focused, carbon_focused, efficiency_focused

4. **RQ4 (Resource Constraints)**: Framework adaptation in resource-constrained environments
   - Implementation: `ResourceConstrainedAdaptation` class
   - Test environments: edge_device, mobile_device, cloud_server, carbon_intensive_dc

## Important Implementation Notes

### Pareto Dominance Logic

The framework uses multi-objective dominance checking in `_dominates()` methods:
- Solution A dominates B if: A is no worse in all objectives AND better in at least one
- Objectives: accuracy (maximize), latency/memory/energy/co2 (minimize)
- Non-dominated solutions form the Pareto frontier

### Carbon-Aware Scheduling

Carbon intensity varies by time and location:
- Real-time monitoring via `CarbonMonitorTool` (simulated)
- Budget tracking across optimization iterations
- Optimal scheduling windows (typically 2-6 AM when grid is cleaner)

### Local-Only Mode

For environments without cloud access:
- Use `simple_local_example.py` or `a2a_protocol_local.py`
- Implements simplified A2A Protocol without GCP dependencies
- All agent communication via local asyncio queues
- Results stored in local filesystem

## File Organization (After Refactoring)

### Package Modules
- `src/agentic_compression/core/` - Core data structures (config, metrics, pareto algorithms)
- `src/agentic_compression/tools/` - LangChain tools (compression, evaluation, carbon)
- `src/agentic_compression/graph/` - LangGraph workflow (replaces A2A/GCP)
- `src/agentic_compression/optimization/` - Research question implementations (RQ1-4)
- `src/agentic_compression/visualization/` - Pareto plotting and analysis

### Examples
- `examples/simple_optimization.py` - Quick start demonstration
- `examples/run_all_experiments.py` - Full experiment suite

### Documentation
- `README.md` - Updated with new API
- `CLAUDE.md` - This file (updated for new structure)
- `FINAL_SUMMARY.md` - Research findings (unchanged)
- `pyproject.toml` - Package metadata and dependencies

### Obsolete Files (Deleted)
- ❌ `agentic_compression_enhanced.py` - Refactored into modules
- ❌ `agentic_compression_deepagents.py` - Refactored into modules
- ❌ `pareto_demo.py` - Replaced by visualization module
- ❌ `simple_local_example.py` - Replaced by examples/
- ❌ `a2a_protocol_local.py` - Removed (using LangGraph now)

## Typical Workflows

### Adding a New Compression Strategy

1. Create tool class in `agentic_compression_deepagents.py`:
   ```python
   class NewStrategyTool(BaseTool):
       name = "strategy_name"
       description = "..."
       def _run(self, **kwargs) -> str:
           # Implementation
   ```

2. Create sub-agent definition:
   ```python
   NewStrategySubAgent = {
       "name": "strategy_specialist",
       "description": "Expert in ...",
       "tools": ["strategy_name", "evaluate_model"]
   }
   ```

3. Update coordinator to delegate to new agent

### Modifying Weight Schemes

Edit `WeightingSchemeAnalysis.weight_schemes` in `agentic_compression_enhanced.py`:
```python
self.weight_schemes = {
    "custom_scheme": {
        "accuracy": 1.5,      # Maximize
        "latency": -1.0,      # Minimize (negative weight)
        "co2": -2.0           # Minimize strongly
    }
}
```

### Adding New Benchmarks

Update evaluation logic in `_evaluate_config()` methods:
1. Add benchmark name to accuracy dictionary
2. Adjust sensitivity factors for compression impact
3. Update benchmark descriptions in documentation

## Model Support

Currently configured for:
- **Primary**: Google Gemma (gemma-12b, gemma3-270m)
- **Architecture**: Supports any HuggingFace Transformers model
- **Planned**: LLaMA, Mistral, vision-language models

## Environment Variables

Required for full functionality:
```bash
OPENAI_API_KEY=...              # For LLM agent reasoning
GCP_PROJECT=...                 # For A2A Protocol (optional)
GOOGLE_APPLICATION_CREDENTIALS=... # For GCP auth (optional)
```

For local-only mode, no environment variables required.

## Performance Expectations

Based on simulated experiments in `FINAL_SUMMARY.md`:

| Configuration | Accuracy | Memory | CO₂ Reduction |
|--------------|----------|---------|---------------|
| INT8 + 30% pruning | 93-94% | 8-12 GB | 40-45% |
| INT4 + 50% pruning | 85-90% | 4-6 GB | 65-70% |
| INT16 + 10% pruning | 95-96% | 16-20 GB | 15-20% |

## Known Limitations

- Framework uses simulated evaluations (see `await asyncio.sleep()` calls)
- No actual model compression is performed - focuses on optimization framework
- Carbon tracking is simulated; integrate real APIs for production
- No formal test suite - experiments serve as integration tests
- DeepAgents implementation requires specific LangChain versions

## Common Pitfalls

1. **Compression Factor Calculation**: Be careful with division by zero when `pruning_sparsity` approaches 1.0 (add small epsilon)
2. **Pareto Dominance**: Ensure consistent objective directions (maximize vs minimize) in `_dominates()` methods
3. **Async Operations**: All agent communications are async; use `await` properly
4. **Weight Normalization**: Composite scores in `_compute_composite_score()` may need normalization across different scales

## Citation

If extending this work, cite:
```bibtex
@inproceedings{liu2025agentic,
  title={Agentic Carbon-Efficient Compression for Large Language Models},
  author={Liu, Yan-Ru and Lin, Chien-Chang and Wang, Ting-An and Chang, Kai-En and Yang, Stephen J.H.},
  booktitle={International Conference on AI for a Sustainable Society},
  year={2025}
}
```
