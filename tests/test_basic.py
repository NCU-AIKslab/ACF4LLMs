#!/usr/bin/env python3
"""Basic tests for the Agentic Compression Framework."""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.common.schemas import ModelSpec, CompressionStrategy, EvaluationResult
        print("✅ Common schemas imported")
    except ImportError as e:
        print(f"❌ Failed to import common schemas: {e}")
        return False

    try:
        from src.coordinator.spec_inference import infer_spec
        print("✅ Spec inference imported")
    except ImportError as e:
        print(f"❌ Failed to import spec inference: {e}")
        return False

    try:
        from src.coordinator.pareto import ParetoFrontier
        print("✅ Pareto frontier imported")
    except ImportError as e:
        print(f"❌ Failed to import Pareto frontier: {e}")
        return False

    try:
        from src.agents.quantization_agent import get_quantization_subagent
        print("✅ Quantization agent imported")
    except ImportError as e:
        print(f"❌ Failed to import quantization agent: {e}")
        return False

    try:
        from src.agents.evaluation_agent import get_evaluation_subagent
        print("✅ Evaluation agent imported")
    except ImportError as e:
        print(f"❌ Failed to import evaluation agent: {e}")
        return False

    try:
        from src.coordinator.coordinator import CompressionCoordinator
        print("✅ Coordinator imported")
    except ImportError as e:
        print(f"❌ Failed to import coordinator: {e}")
        return False

    return True


def test_spec_inference():
    """Test specification inference."""
    print("\nTesting spec inference...")

    from src.coordinator.spec_inference import infer_spec, format_spec_summary

    # Test with known models
    test_cases = [
        ("gpt2", "gsm8k"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "commonsenseqa"),
        ("mistralai/Mistral-7B-v0.1", "humaneval"),
    ]

    for model, dataset in test_cases:
        try:
            spec = infer_spec(model, dataset)
            assert spec.model_name == model
            assert spec.model_size_gb > 0
            assert len(spec.preferred_methods) > 0
            print(f"✅ Spec inference works for {model}")
        except Exception as e:
            print(f"❌ Spec inference failed for {model}: {e}")
            return False

    return True


def test_pareto_frontier():
    """Test Pareto frontier tracking."""
    print("\nTesting Pareto frontier...")

    from src.coordinator.pareto import ParetoFrontier
    from src.common.schemas import CompressionStrategy, EvaluationResult, CompressionMethod
    from uuid import uuid4

    try:
        # Create frontier
        frontier = ParetoFrontier()

        # Add some test solutions
        for i in range(3):
            strategy = CompressionStrategy(
                episode_id=i,
                strategy_id=str(uuid4())[:8],
                methods=[CompressionMethod.AUTOROUND],
                quantization_bits=4 if i == 0 else 8,
            )

            result = EvaluationResult(
                strategy_id=strategy.strategy_id,
                checkpoint_path=f"/test/checkpoint_{i}",
                model_size_gb=10.0 - i,
                compression_ratio=1.5 + i * 0.5,
                accuracy=0.95 - i * 0.05,
                latency_ms=100 + i * 10,
                throughput_tokens_per_sec=100,
                memory_gb=8.0,
                evaluation_time_sec=60.0,
            )

            is_pareto, dominated = frontier.add_solution(strategy, result)
            print(f"  Solution {i}: Pareto={is_pareto}, Dominates {len(dominated)} solutions")

        # Check frontier
        assert len(frontier.solutions) > 0
        print(f"✅ Pareto frontier works, {len(frontier.solutions)} solutions on frontier")

    except Exception as e:
        print(f"❌ Pareto frontier test failed: {e}")
        return False

    return True


def test_mock_quantization():
    """Test mock quantization tool."""
    print("\nTesting mock quantization...")

    from src.agents.quantization_agent import quantize_model

    try:
        result = quantize_model.invoke({
            "model_path": "gpt2",
            "method": "autoround",
            "bit_width": 4,
        })

        assert "checkpoint_path" in result
        assert "model_size_gb" in result
        assert "compression_ratio" in result
        assert result["compression_ratio"] > 1

        print(f"✅ Mock quantization works, compression ratio: {result['compression_ratio']:.1f}x")

    except Exception as e:
        print(f"❌ Mock quantization failed: {e}")
        return False

    return True


def test_mock_evaluation():
    """Test mock evaluation tool."""
    print("\nTesting mock evaluation...")

    from src.agents.evaluation_agent import evaluate_model

    try:
        result = evaluate_model.invoke({
            "checkpoint_path": "/test/checkpoint",
            "benchmarks": ["gsm8k", "commonsenseqa"],
            "use_proxy": True,
        })

        assert "benchmark_scores" in result
        assert "average_accuracy" in result
        assert "latency_ms" in result
        assert len(result["benchmark_scores"]) == 2

        print(f"✅ Mock evaluation works, avg accuracy: {result['average_accuracy']:.3f}")

    except Exception as e:
        print(f"❌ Mock evaluation failed: {e}")
        return False

    return True


def test_coordinator_creation():
    """Test coordinator creation."""
    print("\nTesting coordinator creation...")

    from src.coordinator.coordinator import CompressionCoordinator

    try:
        coordinator = CompressionCoordinator(
            model_name="gpt2",
            dataset="gsm8k",
            max_episodes=1,
            use_mock=True,  # Use mock to avoid DeepAgents dependency
        )

        assert coordinator.spec.model_name == "gpt2"
        assert coordinator.max_episodes == 1
        print("✅ Coordinator created successfully")

    except Exception as e:
        print(f"❌ Coordinator creation failed: {e}")
        return False

    return True


def run_all_tests():
    """Run all basic tests."""
    print("=" * 60)
    print("🧪 Running Basic Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Spec Inference", test_spec_inference),
        ("Pareto Frontier", test_pareto_frontier),
        ("Mock Quantization", test_mock_quantization),
        ("Mock Evaluation", test_mock_evaluation),
        ("Coordinator Creation", test_coordinator_creation),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} tests failed")

    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)