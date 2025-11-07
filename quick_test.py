#!/usr/bin/env python3
"""
Ultra-quick smoke test for real evaluation system.

Tests basic imports and infrastructure without running full evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("QUICK SMOKE TEST - Real Evaluation System")
print("=" * 60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from agentic_compression.inference.model_loader import ModelLoader
    from agentic_compression.inference.quantizer import RealQuantizer
    from agentic_compression.inference.pruner import RealPruner
    from agentic_compression.evaluation.lm_harness_adapter import LMHarnessAdapter
    from agentic_compression.evaluation.benchmark_runner import BenchmarkRunner
    from agentic_compression.tools.evaluation_tools import evaluate_config_full
    from agentic_compression.core.config import CompressionConfig, BENCHMARK_CONFIGS
    print("   ✅ All imports successful")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Check lm-eval
print("\n2. Checking lm-eval installation...")
try:
    import lm_eval
    print(f"   ✅ lm-eval version: {lm_eval.__version__}")
except ImportError:
    print("   ❌ lm-eval not installed")
    sys.exit(1)

# Test 3: Check PyTorch and CUDA
print("\n3. Checking PyTorch and CUDA...")
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        print(f"   ✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   ✅ GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"   ❌ PyTorch check failed: {e}")
    sys.exit(1)

# Test 4: Check transformers
print("\n4. Checking transformers...")
try:
    import transformers
    print(f"   ✅ Transformers version: {transformers.__version__}")
except ImportError:
    print("   ❌ transformers not installed")
    sys.exit(1)

# Test 5: Check bitsandbytes
print("\n5. Checking bitsandbytes...")
try:
    import bitsandbytes
    print(f"   ✅ bitsandbytes version: {bitsandbytes.__version__}")
except ImportError:
    print("   ⚠️  bitsandbytes not installed (quantization will not work)")

# Test 6: Check configuration
print("\n6. Checking configuration...")
try:
    print(f"   ✅ {len(BENCHMARK_CONFIGS)} benchmarks configured:")
    for name, config in BENCHMARK_CONFIGS.items():
        print(f"      - {name}: {config['description']}")
except Exception as e:
    print(f"   ❌ Config check failed: {e}")
    sys.exit(1)

# Test 7: Test compression config creation
print("\n7. Testing compression config creation...")
try:
    config = CompressionConfig(
        model_path="google/gemma3-270m",
        quantization_bits=8,
        pruning_sparsity=0.3,
    )
    print(f"   ✅ Config created: {config.quantization_bits}-bit, {config.pruning_sparsity:.1%} sparsity")
except Exception as e:
    print(f"   ❌ Config creation failed: {e}")
    sys.exit(1)

# Test 8: Test quantizer metrics calculation
print("\n8. Testing quantizer...")
try:
    metrics = RealQuantizer.calculate_compression_metrics(32, 4)
    print(f"   ✅ 4-bit compression ratio: {metrics['compression_ratio']:.1f}x")
    print(f"   ✅ Memory reduction: {metrics['memory_reduction_percent']:.1f}%")
except Exception as e:
    print(f"   ❌ Quantizer test failed: {e}")
    sys.exit(1)

# Test 9: Test pruner metrics calculation
print("\n9. Testing pruner...")
try:
    metrics = RealPruner.calculate_pruning_metrics(0.5)
    print(f"   ✅ 50% sparsity metrics calculated")
    print(f"   ✅ Theoretical speedup: {metrics['theoretical_speedup']:.2f}x")
except Exception as e:
    print(f"   ❌ Pruner test failed: {e}")
    sys.exit(1)

# Test 10: Check lm-eval tasks
print("\n10. Checking lm-eval tasks...")
try:
    from lm_eval import tasks
    available_tasks = tasks.get_task_list() if hasattr(tasks, 'get_task_list') else []
    if available_tasks:
        print(f"   ✅ {len(available_tasks)} tasks available in lm-eval")

        # Check if our benchmarks are available
        required_tasks = set(config['task_name'] for config in BENCHMARK_CONFIGS.values())
        available_set = set(available_tasks)

        for task in required_tasks:
            if task in available_set:
                print(f"   ✅ Task '{task}' found")
            else:
                print(f"   ⚠️  Task '{task}' not found (may need dataset download)")
    else:
        print("   ⚠️  Could not list tasks (may need to run evaluation first)")
except Exception as e:
    print(f"   ⚠️  Task check warning: {e}")

# Summary
print("\n" + "=" * 60)
print("SMOKE TEST SUMMARY")
print("=" * 60)
print("✅ All critical components are working!")
print("\nNext steps:")
print("1. Run quick test: python test_real_evaluation.py")
print("2. Or start Streamlit: streamlit run src/agentic_compression/ui/app.py")
print("\nNote: First run will download models (~1GB for gemma3-270m)")
print("=" * 60)
