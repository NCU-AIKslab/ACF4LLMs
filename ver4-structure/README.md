# GSM8K Optimization Workflow System

A comprehensive ML workflow system for optimizing Qwen3 and Qwen2.5-Math models on the GSM8K mathematical reasoning dataset. The system provides multi-objective optimization balancing accuracy, latency, VRAM usage, and carbon emissions.

## Features

- **Multi-Objective Optimization**: Simultaneously optimize for accuracy, latency, memory usage, and environmental impact
- **Advanced Quantization**: Support for BitsAndBytes (4-bit/8-bit), GPTQ, and AWQ quantization methods
- **Structured Sparsity**: 2:4 semi-structured sparsity for NVIDIA Tensor Core acceleration
- **LoRA/QLoRA Fine-tuning**: Parameter-efficient fine-tuning with quantization support
- **Runtime Optimization**: FlashAttention-2 and vLLM PagedAttention integration
- **RAG Integration**: FAISS-based retrieval for mathematical knowledge
- **Carbon Tracking**: Environmental impact monitoring with CodeCarbon
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **Pareto Optimization**: Multi-objective hyperparameter optimization with Optuna

## Architecture

The system consists of 14 main components working in a coordinated pipeline:

```
ORCHESTRATOR → RECIPE_PLANNER → SEARCH (HPO) →
QUANTIZATION → PRUNING → DISTILLATION → KV_OPTIMIZATION →
RAG → EVALUATION → METRICS_COLLECTION → REGISTRY → PARETO
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gsm8k-optimization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install additional GPU-optimized packages:
```bash
# For FAISS GPU support
pip install faiss-gpu

# For vLLM (if using server throughput recipe)
pip install vllm

# For FlashAttention (if supported)
pip install flash-attn --no-build-isolation
```

## Quick Start

### 1. List Available Recipes

```bash
python scripts/run_optimization.py list
```

Available recipes:
- `accuracy`: Maximize accuracy with QLoRA fine-tuning
- `latency`: Optimize for speed and memory efficiency
- `server`: High-throughput serving optimization
- `small_gpu`: Resource-constrained environment optimization

### 2. Run Single Evaluation

```bash
python scripts/run_optimization.py eval --recipe accuracy --samples 100
```

### 3. Run Hyperparameter Optimization

```bash
python scripts/run_optimization.py hpo --recipe accuracy --trials 50 --timeout 4
```

## Configuration

### Recipe Configuration

Recipes are defined in YAML files in the `configs/` directory. Each recipe specifies:

- **Pipeline stages**: Which optimization techniques to apply
- **Hyperparameters**: Default values and search spaces
- **Resource limits**: Memory, time, and compute constraints
- **Evaluation settings**: Batch sizes, metrics, and tracking options

Example recipe structure:
```yaml
name: "accuracy_first"
base_model: "Qwen/Qwen2.5-Math-7B"
pipeline:
  - stage: quantize_bnb
    args:
      quant_type: "nf4"
      compute_dtype: "bfloat16"
  - stage: distill_lora
    args:
      lora_r: 32
      lora_alpha: 64
decode:
  temperature: 0.2
  max_new_tokens: 256
```

### Hyperparameter Optimization

The `configs/hpo_spaces.yaml` file defines search spaces for multi-objective optimization:

```yaml
decode:
  temperature:
    type: "float"
    low: 0.1
    high: 0.9
lora:
  lora_r:
    type: "categorical"
    choices: [8, 16, 32, 64]
objectives:
  primary:
    - name: "accuracy"
      direction: "maximize"
  secondary:
    - name: "latency_ms"
      direction: "minimize"
```

## Core Components

### Quantization Modules

1. **BitsAndBytes** (`src/stages/quantize_bnb.py`)
   - 4-bit NF4/FP4 quantization
   - QLoRA training support
   - Memory-efficient inference

2. **GPTQ** (`src/stages/quantize_gptq.py`)
   - Post-training weight quantization
   - Math-aware calibration
   - ExLlama kernel support

3. **AWQ** (`src/stages/quantize_awq.py`)
   - Activation-aware quantization
   - Preserves important weights
   - Optimized for inference

### Sparsity and Pruning

**2:4 Structured Sparsity** (`src/stages/prune_sparsity.py`)
- NVIDIA Tensor Core acceleration
- Hardware-aware pruning patterns
- Recovery fine-tuning support

### Evaluation System

**GSM8K Evaluator** (`src/eval/gsm8k_eval.py`)
- Canonical `####` answer parsing
- Comprehensive error analysis
- Performance and resource monitoring

### Multi-Objective Optimization

**Optuna Integration** (`src/search/hpo.py`)
- Pareto-optimal solution finding
- Multiple sampling strategies
- Visualization and analysis tools

## Usage Examples

### Custom Recipe Creation

Create a new recipe by copying and modifying an existing configuration:

```bash
cp configs/recipe_accuracy.yaml configs/recipe_custom.yaml
# Edit the new file with your settings
python scripts/run_optimization.py eval --recipe custom
```

### Programmatic Usage

```python
from src.config import load_recipe
from src.search.hpo import MultiObjectiveSearch
from src.eval.gsm8k_eval import GSM8KEvaluator

# Load configuration
recipe = load_recipe("accuracy")

# Create evaluator
evaluator = GSM8KEvaluator()

# Run optimization
search = MultiObjectiveSearch("my_study")
# ... (see scripts/run_optimization.py for full example)
```

### Model Export and Deployment

The system supports exporting optimized models for deployment:

```python
from src.stages.quantize_awq import AWQQuantizer

quantizer = AWQQuantizer()
model_artifact = quantizer.apply_awq_quantization(
    "Qwen/Qwen2.5-Math-7B",
    save_path="./optimized_models/qwen_awq"
)
```

## Monitoring and Analysis

### Carbon Footprint Tracking

The system automatically tracks energy consumption and CO2 emissions:

```python
from src.monitor.metrics_collector import CarbonTracker

with CarbonTracker("my_experiment").track_emissions():
    # Your optimization code here
    pass
```

### MLflow Integration

Experiments are automatically logged to MLflow:

```bash
mlflow ui --backend-store-uri ./mlruns
```

### Pareto Front Analysis

View multi-objective optimization results:

```python
pareto_front = search.get_pareto_front()
search.plot_pareto_front("results/pareto_front.html")
```

## Results and Evaluation

The system provides comprehensive evaluation metrics:

- **Accuracy**: Exact match on extracted numeric answers
- **Latency**: P50/P90/P99 inference time percentiles
- **Memory**: Peak VRAM usage during inference
- **Throughput**: Tokens per second generation rate
- **Energy**: Power consumption in kWh
- **CO2**: Carbon emissions in grams

### Error Analysis

Automatic error categorization includes:
- Arithmetic slips
- Wrong calculations
- Format errors (missing `####`)
- Answer extraction failures

## Hardware Requirements

### Minimum Requirements
- GPU: 8GB VRAM (small_gpu recipe)
- CPU: 8 cores
- RAM: 16GB
- Storage: 50GB

### Recommended
- GPU: 24GB VRAM (A6000, RTX 4090)
- CPU: 16 cores
- RAM: 64GB
- Storage: 200GB SSD

### Tensor Core Support
For 2:4 sparsity acceleration:
- NVIDIA Ampere (RTX 30xx, A100) or newer
- CUDA 11.0+
- PyTorch with sparse tensor support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e .
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black src/ scripts/ tests/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{gsm8k_optimization_system,
  title={GSM8K Optimization Workflow System},
  author={},
  year={2024},
  url={}
}
```

## Comprehensive GSM8K Testing Suite

### 200-Sample Test Architecture

The system includes a comprehensive testing suite for large-scale evaluation of mathematical reasoning models on the GSM8K dataset. The architecture supports real model inference testing with complete data preservation.

#### Test Scripts Overview

1. **gsm8k_200_efficient_complete.py** - Primary 200-question test with streaming saves
2. **gsm8k_200_complete_test.py** - Original comprehensive test (slower but detailed)
3. **gsm8k_100_fast_results.json** - 100-question baseline results
4. **gsm8k_200_final_complete_results.json** - Complete 200-question evaluation results

#### Architecture Components

```
GSM8K Test Suite Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Test Orchestrator                           │
├─────────────────────────────────────────────────────────────────┤
│  Dataset Loading     │  Model Loading    │  Inference Engine   │
│  ────────────────    │  ──────────────   │  ───────────────────  │
│  • GSM8K Dataset    │  • Qwen2.5-Math   │  • Real GPU Inference│
│  • 200 Questions    │  • 4-bit Quant    │  • Answer Extraction │
│  • Complete Text    │  • RTX 4070 Opt   │  • Performance Track │
├─────────────────────────────────────────────────────────────────┤
│  Data Storage        │  Progress Track   │  Results Analysis    │
│  ──────────────      │  ──────────────   │  ──────────────────   │
│  • NO Truncation    │  • Streaming Save │  • Accuracy Metrics  │
│  • Complete Q&A     │  • Every 10 Items │  • Timing Analysis   │
│  • Full Responses   │  • Error Recovery │  • Memory Monitoring │
└─────────────────────────────────────────────────────────────────┘
```

#### Key Features

**1. Complete Data Preservation**
- **No Truncation**: Full questions, answers, and responses saved
- **Comprehensive Storage**: All 200 questions with complete metadata
- **JSON Format**: Structured data with 2,237 lines (0.26 MB)

**2. Real Model Inference**
- **Actual Neural Network**: Qwen/Qwen2.5-Math-1.5B-Instruct
- **GPU Acceleration**: RTX 4070 with 4-bit NF4 quantization (1.1GB VRAM)
- **Genuine Responses**: No simulation - real transformer inference

**3. Robust Architecture**
- **Streaming Saves**: Results saved every 10 questions to prevent data loss
- **Error Recovery**: 100% processing success rate (0 errors in 200 questions)
- **Performance Monitoring**: Real-time accuracy and timing tracking

#### Test Configuration

```python
# Core Test Parameters
{
    "total_questions": 200,
    "dataset": "openai/gsm8k",
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "quantization": "4-bit NF4",
    "max_new_tokens": 150,
    "temperature": 0.01,
    "sampling": "greedy",
    "optimization": "speed_and_completeness_balanced"
}
```

#### Usage Examples

**Run 200-Question Comprehensive Test:**
```bash
python gsm8k_200_efficient_complete.py
```

**Key Results Structure:**
```json
{
    "question_id": 0,
    "question_full": "Complete original question text...",
    "expected_answer": "18",
    "gsm8k_full_answer": "Complete GSM8K solution...",
    "generated_response_full": "Complete model response...",
    "extracted_answer": "18",
    "correct": true,
    "inference_time_ms": 15573.28,
    "processed_at": "2025-09-16 16:05:34"
}
```

#### Performance Benchmarks

**Achieved Results (RTX 4070 Laptop):**
- **Completion Rate**: 200/200 questions (100%)
- **Accuracy**: 24.0% (48/200 correct)
- **Processing Speed**: 15.48 seconds per question average
- **Total Time**: 51.61 minutes for complete evaluation
- **Memory Efficiency**: 1,101.5 MB with 4-bit quantization
- **Stability**: Zero processing errors

**Hardware Utilization:**
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8GB)
- **Memory Usage**: ~14% of available VRAM
- **Quantization**: 4-bit NF4 with double quantization
- **PyTorch**: v2.5.1+cu121

#### Data Analysis Features

**1. Complete Question Tracking**
```python
# Every question includes full data
{
    "question_full": "Complete question text without truncation",
    "gsm8k_full_answer": "Original GSM8K solution with all steps",
    "generated_response_full": "Complete model-generated response"
}
```

**2. Performance Metrics**
```python
{
    "accuracy_percentage": 24.0,
    "average_inference_time_ms": 15480.09,
    "questions_per_minute": 3.88,
    "total_test_time_minutes": 51.61,
    "gpu_memory_mb": 1101.49
}
```

**3. System Information**
```python
{
    "device": "cuda",
    "gpu_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
    "pytorch_version": "2.5.1+cu121",
    "data_storage": "COMPLETE - no truncation"
}
```

#### File Structure

```
gsm8k-optimization/
├── gsm8k_200_efficient_complete.py     # Main 200-question test
├── gsm8k_200_complete_test.py          # Detailed version
├── gsm8k_200_final_complete_results.json  # Complete results
├── gsm8k_200_efficient_complete.json   # Incremental saves
├── qwen_math_inference_test.py         # 3-question demo
├── gsm8k_100_fast_results.json         # 100-question baseline
└── README.md                           # This documentation
```

#### Integration with Main System

The 200-sample test integrates with the main optimization system:

1. **Dataset Compatibility**: Uses same GSM8K loader as main system
2. **Model Loading**: Compatible with quantization modules
3. **Results Format**: Standardized JSON structure
4. **Performance Tracking**: Same metrics as optimization pipeline

#### Research Applications

**Academic Use Cases:**
- Mathematical reasoning evaluation benchmarks
- Model comparison studies
- Quantization effectiveness analysis
- GPU optimization research

**Production Applications:**
- Model quality assessment before deployment
- Performance regression testing
- Hardware capability evaluation
- Large-scale inference validation

## Acknowledgments

- OpenAI for the GSM8K dataset
- Hugging Face for Transformers library
- The Qwen team for the mathematical reasoning models
- Optuna team for multi-objective optimization
- NVIDIA for Tensor Core sparse support