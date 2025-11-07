# Real Evaluation System - User Guide

## 🎯 Overview

The Agentic Compression Framework now uses **REAL model evaluation** instead of simulation!

### What's Real Now?

✅ **Model Loading**: Uses HuggingFace Transformers
✅ **Quantization**: Uses bitsandbytes (4-bit/8-bit NF4)
✅ **Pruning**: Uses PyTorch pruning (unstructured & 2:4 structured)
✅ **Evaluation**: Uses lm-evaluation-harness on 5 benchmarks
✅ **GPU Monitoring**: Uses pynvml for real metrics
✅ **Carbon**: Calculated from actual GPU usage

### Five Real Benchmarks

1. **GSM8K** - Mathematical reasoning (8-shot)
2. **TruthfulQA** - Truthfulness and factual consistency
3. **CommonsenseQA** - Commonsense reasoning (5-shot)
4. **HumanEval** - Code generation (pass@1)
5. **BigBench** - Multi-domain challenging tasks

---

## 🚀 Quick Start

### 1. Quick Test (Development Mode)

Run a quick test with 50 samples per benchmark:

```bash
conda activate greenAI
python test_real_evaluation.py
```

This will:
- Test gemma3-270m model (small, fast)
- Test 4-bit and 8-bit quantization
- Test combined quantization + pruning
- Complete in ~10-15 minutes on RTX 4090

### 2. Using Streamlit UI

```bash
streamlit run src/agentic_compression/ui/app.py
```

Then:
1. Go to "Quick Optimization" page
2. Select model (gemma3-270m recommended for testing)
3. Choose quantization bits (4, 8, 16, or 32)
4. Set pruning sparsity (0.0-0.7)
5. Click "Run Optimization"
6. **Wait for REAL evaluation** (this takes time!)

---

## ⏱️ Expected Evaluation Times

### gemma3-270m (Small Model)
- **Full evaluation**: 5-10 minutes
- **Quick test (50 samples)**: 2-3 minutes
- **Memory**: 0.5-2GB VRAM (depending on quantization)

### gemma-12b (Large Model)
- **Full evaluation**: 20-30 minutes
- **Quick test**: 5-7 minutes
- **Memory**: 4-12GB VRAM (depending on quantization)

---

## 📊 Understanding Real Results

### Accuracy Interpretation

Real benchmark results will vary by:
- Model size and quality
- Quantization level (4-bit < 8-bit < 16-bit < 32-bit)
- Pruning sparsity (higher sparsity = lower accuracy)
- Benchmark difficulty

**Typical Ranges** (gemma3-270m, 8-bit):
- GSM8K: 30-50% (math is hard!)
- TruthfulQA: 40-60%
- CommonsenseQA: 50-70%
- HumanEval: 10-30% (code generation is challenging)
- BigBench: 40-60%

**Don't expect 90%+ accuracy on all benchmarks!**
Real evaluation shows actual model performance.

### Memory & Carbon

- **Memory**: Shows actual GPU VRAM usage
- **Latency**: Measured during inference
- **Energy**: Calculated from GPU power draw
- **CO2**: Based on grid carbon intensity (default: 0.4 kg/kWh)

---

## 🔧 Configuration

### Quick Test Mode

Edit `src/agentic_compression/core/config.py`:

```python
# Quick evaluation (50 samples)
EVAL_CONFIG_QUICK = EvaluationConfig(
    limit=50,
    batch_size=4,
    quick_test=True,
)

# Full evaluation (all samples)
EVAL_CONFIG_FULL = EvaluationConfig(
    limit=None,
    batch_size=8,
    quick_test=False,
)
```

### Model Selection

Recommended models for testing:
- `google/gemma3-270m` - Fast, small (270M params)
- `google/gemma-12b` - Slower, large (12B params)
- `meta-llama/Llama-2-7b` - Medium (7B params)

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solution 1**: Use smaller model
```python
config.model_path = "google/gemma3-270m"
```

**Solution 2**: Use more quantization
```python
config.quantization_bits = 4  # Instead of 8
```

**Solution 3**: Reduce batch size
```python
runner = BenchmarkRunner(batch_size=2)  # Instead of 8
```

### Evaluation Too Slow

**Use quick test mode**:
```python
runner = BenchmarkRunner(limit=50)  # Only 50 samples
```

### Import Errors

```bash
conda activate greenAI
pip install lm-eval transformers bitsandbytes accelerate pynvml
```

### Model Download Issues

Models are cached in `./model_cache/`. If download fails:
```bash
rm -rf ./model_cache/
# Then retry - HuggingFace will re-download
```

---

## 📈 Performance Tips

### 1. Use GPU Efficiently

- **Quantization**: Load with `load_in_4bit=True` directly
- **Batch size**: Use largest batch that fits in VRAM
- **Device map**: Use `device_map="auto"` for multi-GPU

### 2. Cache Models

Models are cached after first download:
- **Location**: `./model_cache/`
- **Keep it**: Don't delete unless troubleshooting
- **Size**: gemma3-270m ~1GB, gemma-12b ~24GB

### 3. Parallel Evaluation

The system evaluates all 5 benchmarks, but:
- Benchmarks run sequentially (memory constraint)
- Within each benchmark, batches run in parallel
- Use larger `batch_size` for faster evaluation

---

## 🎓 Advanced Usage

### Custom Benchmarks

Edit `BENCHMARK_CONFIGS` in `config.py` to add benchmarks:

```python
BENCHMARK_CONFIGS["my_task"] = {
    "description": "My custom task",
    "task_name": "my_task",  # lm-eval task name
    "num_fewshot": 5,
    "primary_metric": "acc",
}
```

### Custom Evaluation Config

```python
from agentic_compression.evaluation import BenchmarkRunner

runner = BenchmarkRunner(
    batch_size=16,        # Larger batches
    num_fewshot=8,        # More examples
    limit=100,            # Quick test
    device="cuda:1",      # Specific GPU
)
```

### Export Results

Results are saved in `EvaluationMetrics` format:

```python
metrics = await evaluate_config_full(config)
results_dict = metrics.to_dict()

# Save to JSON
import json
with open("results.json", "w") as f:
    json.dump(results_dict, f, indent=2)
```

---

## 📚 References

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [bitsandbytes quantization](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)
- [PyTorch pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

---

**Questions?** Check logs in the terminal or create an issue on GitHub.
