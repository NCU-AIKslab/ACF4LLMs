# Agentic Carbon-Efficient LLM Compression Framework

> **🎉 Version 2.0 - Refactored!**
> Refactored from single-file scripts into a **standard Python package** with modular components.
> **All GCP/A2A Protocol dependencies removed** - now uses **LangGraph-native** agent orchestration.

## Overview

This repository implements the **Agentic Carbon-Efficient Compression Framework** for Large Language Models (LLMs) based on the research paper by Liu et al. (2025). The framework uses a multi-agent system architecture with **LangGraph** to dynamically optimize model compression while balancing accuracy, efficiency, and carbon footprint.

### What's New in v2.0

- ✅ **Standard Python package structure** (`src/agentic_compression/`)
- ✅ **No GCP dependencies** - completely local, no cloud account needed
- ✅ **LangGraph-native workflow** - replaces custom A2A Protocol
- ✅ **Modular design** - separated concerns (tools, optimization, visualization)
- ✅ **Proper package configuration** (`pyproject.toml`)
- ✅ **Clean API** with examples

## Key Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for quantization, pruning, KV context optimization, and distillation
- 🔄 **Dynamic Adaptation**: Real-time adjustment of compression strategies based on performance feedback
- 📊 **Multi-Objective Optimization**: Pareto-optimal trade-off identification across multiple metrics
- 🌱 **Carbon-Aware**: Integrated carbon monitoring and budget constraints
- 🔗 **Google A2A Protocol**: Secure agent-to-agent communication
- 📈 **Comprehensive Evaluation**: Benchmarking across GSM8K, TruthfulQA, CommonsenseQA, HumanEval, and BIG-Bench Hard

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestrator                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Coordinator Agent (LangChain)              │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                  │
│     ┌──────────────────────┴────────────────────────┐       │
│     │                                                │       │
│  ┌──▼─────────┐ ┌────────────┐ ┌──────────┐ ┌──────▼────┐  │
│  │Quantization│ │  Pruning   │ │KV Context│ │Distillation│  │
│  │   Agent    │ │   Agent    │ │  Agent   │ │   Agent    │  │
│  └────────────┘ └────────────┘ └──────────┘ └────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │     Multi-Objective Evaluation Engine (Python)       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, for actual model compression)
- **No GCP account required!** ✨

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/your-org/agentic-compression-framework.git
cd agentic-compression-framework
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package**:
```bash
# Install from source (development mode)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

4. **Optional environment variables**:
```bash
# Only needed if using OpenAI/Anthropic LLMs for agent reasoning
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_key
```

**Note**: No GCP credentials needed - framework runs completely locally!

## Usage

### Web UI (Streamlit) - **NEW! ✨**

Launch the interactive web interface:

```bash
streamlit run app.py
```

The Streamlit UI provides **4 interactive pages**:

1. **🚀 Quick Optimization** - Run single optimization experiments with custom configurations
2. **📊 Advanced Visualization** - Explore RQ1, RQ3, RQ4 analyses with interactive charts
3. **🔬 Experiment Comparison** - Run and compare multiple configurations side-by-side
4. **🎯 Interactive 3D Explorer** - Explore multi-dimensional Pareto frontiers

**Features**:
- Real-time progress tracking
- Interactive Plotly visualizations (2D/3D Pareto frontiers, parallel coordinates, radar charts)
- Side-by-side experiment comparison
- Export results as JSON
- No coding required!

### Quick Start (Programmatic)

```python
import asyncio
from agentic_compression.graph.workflow import run_compression_optimization

async def main():
    # Run optimization using LangGraph workflow
    results = await run_compression_optimization(
        objective="Compress for edge deployment with minimal carbon",
        carbon_budget=5.0,  # kg CO2
        max_iterations=10,
        accuracy_threshold=0.93
    )

    print(f"Found {results['pareto_optimal_count']} Pareto-optimal solutions")
    print(f"Best solution: {results['best_solution']}")

asyncio.run(main())
```

### Using Research Question Implementations

```python
from agentic_compression.optimization.agent_driven import run_rq2_experiment

# Run RQ2: Agent-Driven Optimization
results = await run_rq2_experiment(
    model="google/gemma-12b",
    accuracy_threshold=0.93,
    carbon_budget=5.0
)

print(f"Carbon impact: {results['carbon_impact_analysis']}")
print(f"Key findings: {results['key_findings']}")
```

### Command Line Interface

```bash
# Run simple example
python examples/simple_optimization.py

# Run all experiments
python examples/run_all_experiments.py
```

### Docker Deployment

```bash
# Build Docker image
docker build -t agentic-compression:latest .

# Run container with GPU support
docker run --gpus all \
    -v $(pwd)/results:/app/results \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    agentic-compression:latest
```

### Kubernetes Deployment

```bash
# Apply Kubernetes configuration
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -l app=compression-framework

# View logs
kubectl logs -f deployment/agentic-compression-framework
```

## Configuration

### Compression Strategies

The framework supports multiple compression techniques:

1. **Quantization**: INT4, INT8, INT16 precision reduction
2. **Pruning**: Structured and unstructured sparsity (up to 70%)
3. **KV Context**: Context window optimization (1k-32k tokens)
4. **Distillation**: Layer reduction (6-24 layers)

### Search Strategies

- `bayesian`: Bayesian optimization with Gaussian processes
- `evolutionary`: Genetic algorithm-based search
- `grid`: Exhaustive grid search
- `random`: Random sampling baseline

### Evaluation Benchmarks

- **GSM8K**: Mathematical reasoning
- **TruthfulQA**: Truthfulness and factual consistency
- **CommonsenseQA**: Commonsense reasoning
- **HumanEval**: Code generation
- **BIG-Bench Hard**: Multi-domain challenging tasks

## Monitoring

### Prometheus Metrics

The framework exposes metrics on port 8000:

- `compression_requests_total`: Total compression requests by strategy
- `compression_duration_seconds`: Time spent in compression
- `model_accuracy`: Current model accuracy per benchmark
- `carbon_emissions_kg`: Real-time carbon emissions

### Grafana Dashboard

Import the provided dashboard (`monitoring/dashboard.json`) to visualize:
- Pareto frontier evolution
- Carbon budget utilization
- Accuracy vs. efficiency trade-offs
- Agent activity patterns

## API Reference

### Framework API

```python
class AgenticCompressionFramework:
    def __init__(self, base_model: str, carbon_budget: float, accuracy_threshold: float)
    async def optimize(self, objectives: Dict, search_strategy: str, num_iterations: int) -> Dict
    def apply_compression(self, config: Dict) -> Model
```

### Agent APIs

```python
class QuantizationAgent:
    async def quantize_model(self, model_path: str, config: CompressionConfig) -> Dict

class PruningAgent:
    async def prune_model(self, model_path: str, config: CompressionConfig) -> Dict

class KVContextAgent:
    async def optimize_context(self, model_path: str, config: CompressionConfig) -> Dict

class DistillationAgent:
    async def distill_model(self, model_path: str, config: CompressionConfig) -> Dict
```

## Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run with coverage
pytest --cov=agentic_compression tests/

# Run specific test
pytest tests/unit/test_quantization_agent.py::TestQuantizationAgent::test_int8_quantization
```

## Performance Benchmarks

Expected compression results on Google Gemma-12B:

| Configuration | Accuracy | Latency | Memory | Energy | CO₂ |
|--------------|----------|---------|---------|---------|------|
| Baseline | 100% | 100ms | 24GB | 0.084 kWh | 0.034 kg |
| INT8 | 98% | 60ms | 12GB | 0.050 kWh | 0.020 kg |
| 50% Pruning | 96% | 55ms | 12GB | 0.045 kWh | 0.018 kg |
| INT4 + 30% Pruning | 94% | 35ms | 6GB | 0.030 kWh | 0.012 kg |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in evaluation
   - Use gradient checkpointing
   - Enable tensor parallelism

2. **Carbon Budget Exceeded**:
   - Increase budget or reduce iterations
   - Use more aggressive early stopping
   - Schedule during low carbon intensity periods

3. **Poor Compression Results**:
   - Adjust weight parameters in objectives
   - Try different search strategies
   - Increase number of iterations

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[QUICKSTART.md](docs/QUICKSTART.md)** - 快速開始指南（中文）
- **[EXECUTE_THIS.md](docs/EXECUTE_THIS.md)** - 執行指令清單（中文）
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - Complete implementation summary
- **[CLAUDE.md](docs/CLAUDE.md)** - Developer guide for Claude Code
- **[TODO.md](docs/TODO.md)** - Task list and implementation status
- **[FINAL_SUMMARY.md](docs/FINAL_SUMMARY.md)** - Research findings summary
- **[COMPLETION_SUMMARY.md](docs/COMPLETION_SUMMARY.md)** - Project completion summary
- **[REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md)** - Refactoring details
- **[research_questions_analysis.md](docs/research_questions_analysis.md)** - RQ analysis
- **[deepagents_integration_spec.md](docs/deepagents_integration_spec.md)** - DeepAgents integration

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{liu2025agentic,
  title={Agentic Carbon-Efficient Compression for Large Language Models: Balancing Accuracy and Energy Efficiency},
  author={Liu, Yan-Ru and Lin, Chien-Chang and Wang, Ting-An and Chang, Kai-En and Yang, Stephen J.H.},
  booktitle={Proceedings of International Conference on AI for a Sustainable Society},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain team for the agent framework
- Google for the A2A Protocol
- NVIDIA for GPU optimization tools
- The Green AI community

## Contact

- **Lead Author**: Yan-Ru Liu (liu76214@gmail.com)
- **GitHub Issues**: [Submit an issue](https://github.com/your-org/agentic-compression/issues)
- **Discussion Forum**: [Join our Discord](https://discord.gg/your-invite)

## Roadmap

- [x] ~~Web UI for monitoring and control~~ ✅ **Streamlit UI implemented!**
- [ ] Support for more model architectures (LLaMA, Mistral, etc.)
- [ ] Integration with HuggingFace Model Hub
- [ ] Real-time carbon intensity API integration
- [ ] Distributed multi-GPU optimization
- [ ] AutoML integration for hyperparameter tuning
- [ ] Support for vision-language models
- [ ] Edge deployment optimization

---

**Version**: 2.0.0
**Last Updated**: January 2025
**Status**: Active Development

---

## Screenshots

### Streamlit Web UI

**Home Page**:
- Framework overview
- Research questions summary
- Quick start guide

**Quick Optimization Page**:
- Interactive parameter configuration
- Real-time optimization progress
- Pareto frontier visualization
- Results export

**Advanced Visualization Page**:
- RQ1: Dynamic vs Static comparison
- RQ3: Weight scheme analysis
- RQ4: Environment adaptation testing

**3D Explorer Page**:
- Interactive 3D Pareto frontiers
- Parallel coordinates visualization
- Multi-objective radar charts
