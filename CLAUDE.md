# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Compressor is an intelligent, LLM-driven multi-agent system for optimizing Large Language Models across multiple objectives: accuracy, latency, VRAM usage, energy consumption, and CO₂ emissions. The system uses LangChain and LangGraph to create intelligent agents that make optimization decisions, with Pareto frontier analysis to find optimal trade-offs between objectives.

## Core Architecture

### Multi-Agent System Structure
- **Orchestrator** (`llm_compressor/core/orchestrator.py`): LangGraph-based coordinator for LLM-driven agents
- **Registry** (`llm_compressor/core/registry.py`): Experiment tracking and artifact management
- **Pareto Analyzer** (`llm_compressor/core/pareto.py`): Multi-objective optimization analysis
- **Metrics Collector** (`llm_compressor/core/metrics.py`): Performance and resource monitoring

### LLM-Driven Agent Framework
- **Base LLM Agent** (`llm_compressor/agents/llm_base.py`): LangChain-powered base class for intelligent agents
- **LangGraph Workflow**: StateGraph-based multi-agent coordination with conditional routing
- **Decision Framework**: Structured LLM decision-making with confidence scoring and reasoning

### LLM-Driven Agents
Located in `llm_compressor/agents/`:
- `llm_quantization.py`: Intelligent quantization strategy selection using LLM reasoning
- `llm_pruning.py`: Smart pruning decisions with sparsity pattern optimization
- `llm_distillation.py`: Adaptive knowledge distillation with student model sizing
- `llm_kv_optimization.py`: Context-aware attention optimization strategies
- `llm_performance.py`: Intelligent performance monitoring and carbon footprint analysis
- `llm_evaluation.py`: Comprehensive model evaluation with benchmark orchestration
- `llm_recipe_planner.py`: LLM-driven optimization recipe generation and portfolio planning

### Model Runners
- Abstract interface in `llm_compressor/core/runners.py` supports both vLLM and TensorRT-LLM backends
- Default backend is vLLM with GPU acceleration

## Common Development Commands

### Installation and Setup
```bash
# Quick setup and run
make quickstart

# Manual installation
make install
make setup-data

# Development setup
make setup-dev
```

### Running Optimization
```bash
# LLM-driven baseline measurement
python scripts/run_search.py --config configs/default.yaml --recipes baseline --output reports/baseline

# LLM-driven conservative optimization
python scripts/run_search.py --config configs/default.yaml --recipes conservative --output reports/conservative

# LLM-driven aggressive optimization
python scripts/run_search.py --config configs/default.yaml --recipes aggressive --output reports/aggressive

# LLM-planned optimization portfolio
python scripts/run_search.py --config configs/default.yaml --recipes llm_planned --output reports/llm_planned

# Using make commands
make run                    # Conservative optimization
make run-baseline          # Baseline measurement
make run-aggressive        # Aggressive optimization
```

#### Results Export
```bash
# Export results
make export
python scripts/export_report.py --db experiments.db --output analysis_report

# Export from JSON summary
make export-json
python scripts/export_report.py --input reports/execution_summary.json --output analysis_report
```

### Testing and Quality Assurance
```bash
# Full test suite
make test
python -m pytest tests/ -v --cov=llm_compressor --cov-report=html --cov-report=term

# Quick tests
make test-quick
python -m pytest tests/ -v -x --tb=short

# Linting
make lint
flake8 llm_compressor/ scripts/ --max-line-length=100 --ignore=E203,W503
mypy llm_compressor/ --ignore-missing-imports

# Code formatting
make format
black llm_compressor/ scripts/ --line-length=100
isort llm_compressor/ scripts/ --profile black

# Format checking
make format-check
black llm_compressor/ scripts/ --line-length=100 --check
isort llm_compressor/ scripts/ --profile black --check-only

# All quality checks
make check

# CI/CD pipeline tests
make ci-test
make ci-build
```

### Docker Operations
```bash
# Build Docker image
make build

# Run in Docker
make run-docker

# Interactive Docker session
make run-interactive

# Quickstart with Docker
make quickstart-docker

# Clean Docker resources
make clean-docker
```

## Configuration System

### Main Configuration
- Primary config: `configs/default.yaml`
- Baseline recipes: `configs/recipes_baseline.yaml`
- Model-specific configs available (e.g., `configs/gemma3_270m.yaml`, `configs/gemma3_baseline.yaml`)

### Key Configuration Sections
- `model`: Base model settings, sequence length
- `hardware`: GPU type, VRAM limits, parallelization
- `constraints`: Accuracy drop limits, latency thresholds, carbon budget
- `agents`: Individual agent configurations and parameters
- `search`: Optimization method (bayesian, evolutionary, grid, random)
- `evaluation`: MMLU, GSM8K, MT-Bench, Safety benchmarks

## Agent Development

### Adding New Agents
1. Inherit from `BaseAgent` in `agents/base.py`
2. Implement the `execute()` method returning `AgentResult`
3. Register in orchestrator's `_initialize_agents()` method
4. Add configuration parameters in YAML config files

### Agent Result Structure
```python
@dataclass
class AgentResult:
    success: bool
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
```

## Hardware Requirements
- **GPU**: NVIDIA GPU with 40GB+ VRAM (A100/H100 recommended)
- **CUDA**: 11.8+
- **Python**: 3.10+
- **Docker**: 20.10+ (optional)

## Evaluation Datasets
Built-in support for:
- **GSM8K**: Mathematical reasoning
- **TruthfulQA**: Factual knowledge and truthfulness
- **CommonsenseQA**: Commonsense reasoning
- **HumanEval**: Code generation
- **BIG-Bench Hard**: Complex reasoning tasks
- **Safety**: Red-teaming and toxicity evaluation

## Monitoring and Debugging
```bash
# Debug mode with verbose logging
make debug
python scripts/run_search.py --config configs/default.yaml --log-level DEBUG --output reports/debug

# System monitoring
make monitor
watch -n 5 'nvidia-smi; echo ""; ps aux | grep python | head -10'

# Validate configuration
make validate-config
python scripts/run_search.py --config configs/default.yaml --dry-run

# Environment information
make env-info

# Performance benchmarking
make benchmark

# Multi-GPU execution
make run-multi-gpu
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_search.py --config configs/default.yaml --output reports/multi_gpu

# Custom recipes
make run-custom RECIPE=path/to/custom_recipe.yaml
```

## Optimization Techniques
- **Quantization**: AWQ (4-bit), GPTQ, BitsAndBytes (INT8/FP16)
- **Attention**: FlashAttention, PagedAttention for memory efficiency  
- **Pruning**: Structured/unstructured, N:M sparsity patterns
- **Distillation**: LoRA, QLoRA with temperature scaling
- **KV Optimization**: Sliding window, compression techniques

## Cleanup and Maintenance
```bash
# Clean generated files
make clean
rm -rf reports/* artifacts/* logs/*

# Clean Docker resources
make clean-docker

# Full cleanup (files + Docker)
make clean-all

# Remove Python cache files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete
```

## Key Dependencies

### Core Dependencies
- PyTorch ≥ 2.1.0 with CUDA support
- Transformers ≥ 4.36.0
- Quantization: auto-gptq, autoawq, bitsandbytes
- Serving: vLLM ≥ 0.3.0 (default) or TensorRT-LLM
- Optimization: scipy, optuna, DEAP
- Visualization: plotly, matplotlib, seaborn
- Monitoring: pynvml, GPUtil, psutil
- Development: pytest, black, isort, flake8, mypy

### LLM Agent Framework Dependencies
- **LangChain**: langchain ≥ 0.1.0, langchain-community, langchain-core
- **LangGraph**: langgraph ≥ 0.0.20 for workflow orchestration
- **LangSmith**: langsmith ≥ 0.0.80 for tracing and monitoring
- **LLM Providers**:
  - OpenAI: openai ≥ 1.12.0
  - Anthropic: anthropic ≥ 0.8.0
  - Google: google-generativeai ≥ 0.3.0

## LLM-Driven Agent System

### Environment Setup for LLM Agents
```bash
# Set API keys for LLM providers
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"

# Optional: Set default LLM provider
export DEFAULT_LLM_PROVIDER="openai"  # or "anthropic", "google"
```

### LLM Agent Configuration
- **Primary Config**: `configs/default.yaml`
- **Provider Selection**: Configure LLM provider per agent type
- **Temperature Control**: Adjust reasoning vs consistency (0.1 for consistency, 0.7 for creativity)
- **Token Limits**: Control response length and context usage

### LLM Agent Features
- **Intelligent Decision Making**: LLM-powered optimization strategy selection
- **Adaptive Planning**: Dynamic adjustment based on intermediate results
- **Risk Assessment**: Confidence scoring and uncertainty quantification
- **Contextual Reasoning**: Hardware-aware and task-specific optimization
- **Portfolio Generation**: Diverse optimization strategy exploration
- **Multi-Objective Balancing**: Intelligent trade-off reasoning

### LangGraph Workflow Architecture
- **StateGraph**: Conditional node routing based on agent outputs
- **Message Passing**: Structured communication between agents
- **Error Handling**: Graceful failure recovery and retry mechanisms
- **Workflow Coordination**: Intelligent agent sequencing and parallelization
- **Result Aggregation**: Multi-agent output synthesis

### Monitoring LLM Agent Performance
```bash
# Enable LLM request logging
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="your-langsmith-api-key"

# Debug LLM decisions
python scripts/run_search.py --config configs/default.yaml --log-level DEBUG

# Monitor token usage
grep "token_usage" llm_optimization.log
```

### LLM Agent Troubleshooting
- **API Rate Limits**: Agents gracefully handle rate limiting with exponential backoff
- **Mock Mode**: System runs with simulated responses when API keys unavailable
- **Fallback Mechanisms**: Traditional algorithmic agents as backup
- **Validation**: Structured output parsing with error recovery
- **Cost Control**: Token usage monitoring and budget management
- to