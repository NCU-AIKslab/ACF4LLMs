#!/usr/bin/env python3
"""
Quick test script for real evaluation system.

Tests the entire pipeline: model loading -> compression -> evaluation
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_compression.core.config import CompressionConfig, EVAL_CONFIG_QUICK
from agentic_compression.tools.evaluation_tools import evaluate_config_full

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_small_model_quick():
    """Test with small model (270M) and quick evaluation."""
    logger.info("=" * 60)
    logger.info("Test 1: Small Model (gemma3-270m) with Quick Evaluation")
    logger.info("=" * 60)

    # Use small model for quick testing
    config = CompressionConfig(
        model_path="google/gemma3-270m",
        quantization_bits=8,  # 8-bit quantization
        pruning_sparsity=0.0,  # No pruning for first test
        accuracy_threshold=0.70,
    )

    logger.info(f"Config: {config.quantization_bits}-bit, sparsity={config.pruning_sparsity}")
    logger.info("Starting evaluation...")

    try:
        metrics = await evaluate_config_full(config)

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS:")
        logger.info("=" * 60)
        logger.info(f"Average Accuracy: {metrics.average_accuracy():.3f}")
        logger.info(f"Memory: {metrics.memory_gb:.2f} GB")
        logger.info(f"Latency: {metrics.latency_ms:.1f} ms")
        logger.info(f"CO2: {metrics.co2_kg:.4f} kg")
        logger.info(f"Compression Ratio: {metrics.compression_ratio:.2f}x")

        logger.info("\nPer-Benchmark Accuracy:")
        for benchmark, acc in metrics.accuracy.items():
            logger.info(f"  {benchmark}: {acc:.3f}")

        logger.info("\n✅ Test 1 PASSED!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Test 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_quantization_comparison():
    """Test different quantization levels."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Quantization Comparison (4-bit vs 8-bit)")
    logger.info("=" * 60)

    results = []

    for bits in [8, 4]:
        logger.info(f"\nTesting {bits}-bit quantization...")

        config = CompressionConfig(
            model_path="google/gemma3-270m",
            quantization_bits=bits,
            pruning_sparsity=0.0,
        )

        try:
            metrics = await evaluate_config_full(config)
            results.append({
                "bits": bits,
                "accuracy": metrics.average_accuracy(),
                "memory_gb": metrics.memory_gb,
                "co2_kg": metrics.co2_kg,
            })
            logger.info(f"  ✅ {bits}-bit: accuracy={metrics.average_accuracy():.3f}")

        except Exception as e:
            logger.error(f"  ❌ {bits}-bit failed: {str(e)}")
            return False

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON:")
    logger.info("=" * 60)
    for r in results:
        logger.info(
            f"{r['bits']}-bit: Acc={r['accuracy']:.3f}, "
            f"Mem={r['memory_gb']:.2f}GB, CO2={r['co2_kg']:.4f}kg"
        )

    logger.info("\n✅ Test 2 PASSED!")
    return True


async def test_compression_and_pruning():
    """Test combined quantization and pruning."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Combined Quantization + Pruning")
    logger.info("=" * 60)

    config = CompressionConfig(
        model_path="google/gemma3-270m",
        quantization_bits=8,
        pruning_sparsity=0.3,  # 30% pruning
    )

    logger.info(f"Config: {config.quantization_bits}-bit + {config.pruning_sparsity:.0%} pruning")

    try:
        metrics = await evaluate_config_full(config)

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS:")
        logger.info("=" * 60)
        logger.info(f"Average Accuracy: {metrics.average_accuracy():.3f}")
        logger.info(f"Memory: {metrics.memory_gb:.2f} GB")
        logger.info(f"CO2: {metrics.co2_kg:.4f} kg")
        logger.info(f"Compression Ratio: {metrics.compression_ratio:.2f}x")

        logger.info("\n✅ Test 3 PASSED!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Test 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    logger.info("\n" + "🚀" * 30)
    logger.info("REAL EVALUATION SYSTEM TEST SUITE")
    logger.info("🚀" * 30 + "\n")

    logger.info("NOTE: Using QUICK evaluation mode (50 samples per benchmark)")
    logger.info("For full evaluation, modify EVAL_CONFIG_QUICK in config.py\n")

    # Monkey-patch BenchmarkRunner to use quick config
    from agentic_compression.evaluation import benchmark_runner
    original_init = benchmark_runner.BenchmarkRunner.__init__

    def quick_init(self, **kwargs):
        kwargs["limit"] = 50  # Force quick mode
        original_init(self, **kwargs)

    benchmark_runner.BenchmarkRunner.__init__ = quick_init

    tests = [
        ("Small Model Quick Test", test_small_model_quick()),
        ("Quantization Comparison", test_quantization_comparison()),
        ("Compression + Pruning", test_compression_and_pruning()),
    ]

    results = []
    for name, test_coro in tests:
        try:
            result = await test_coro
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {str(e)}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! System is ready.")
        return 0
    else:
        logger.error("\n⚠️  SOME TESTS FAILED. Please check the logs.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
