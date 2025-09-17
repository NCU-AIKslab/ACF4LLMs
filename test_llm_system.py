#!/usr/bin/env python3
"""Test script for the LLM-driven agent system."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all LLM agents can be imported."""
    print("Testing imports...")

    try:
        from llm_compressor.core.orchestrator import Orchestrator
        print("✓ Orchestrator import successful")

        from llm_compressor.agents.llm_quantization import LLMQuantizationAgent
        print("✓ LLM Quantization Agent import successful")

        from llm_compressor.agents.llm_pruning import LLMPruningAgent
        print("✓ LLM Pruning Agent import successful")

        from llm_compressor.agents.llm_distillation import LLMDistillationAgent
        print("✓ LLM Distillation Agent import successful")

        from llm_compressor.agents.llm_kv_optimization import LLMKVOptimizationAgent
        print("✓ LLM KV Optimization Agent import successful")

        from llm_compressor.agents.llm_performance import LLMPerformanceAgent
        print("✓ LLM Performance Agent import successful")

        from llm_compressor.agents.llm_evaluation import LLMEvaluationAgent
        print("✓ LLM Evaluation Agent import successful")

        from llm_compressor.agents.llm_recipe_planner import LLMRecipePlannerAgent
        print("✓ LLM Recipe Planner Agent import successful")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    config_path = "llm_compressor/configs/default.yaml"

    if not Path(config_path).exists():
        print(f"✗ Configuration file not found: {config_path}")
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✓ Configuration loaded successfully")
        print(f"  - LLM provider: {config.get('llm', {}).get('provider', 'not set')}")
        print(f"  - Model: {config.get('model', {}).get('base_model', 'not set')}")
        print(f"  - Agents configured: {len(config.get('agents', {}))}")

        return True

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_orchestrator_initialization():
    """Test orchestrator initialization."""
    print("\nTesting orchestrator initialization...")

    # Setup mock API keys
    os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"
    os.environ["ANTHROPIC_API_KEY"] = "mock-key-for-testing"
    os.environ["GOOGLE_API_KEY"] = "mock-key-for-testing"

    try:
        from llm_compressor.core.orchestrator import Orchestrator

        config_path = "llm_compressor/configs/default.yaml"
        orchestrator = Orchestrator(config_path)

        print(f"✓ Orchestrator initialized successfully")
        print(f"  - Number of agents: {len(orchestrator.agents)}")
        print(f"  - Agent types: {list(orchestrator.agents.keys())}")

        return True

    except Exception as e:
        print(f"✗ Orchestrator initialization failed: {e}")
        return False

def test_agent_validation():
    """Test agent recipe validation."""
    print("\nTesting agent validation...")

    os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"

    try:
        from llm_compressor.agents.llm_quantization import LLMQuantizationAgent

        config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.1
            }
        }

        agent = LLMQuantizationAgent("test_quantization", config)

        # Test recipe validation
        valid_recipe = {"quantization": {"method": "awq", "bits": 4}}
        invalid_recipe = {"pruning": {"method": "structured"}}

        assert agent.validate_recipe(valid_recipe), "Valid recipe should pass validation"
        assert not agent.validate_recipe(invalid_recipe), "Invalid recipe should fail validation"

        print("✓ Agent validation working correctly")
        return True

    except Exception as e:
        print(f"✗ Agent validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM-DRIVEN AGENT SYSTEM TEST")
    print("=" * 60)

    tests = [
        test_imports,
        test_config_loading,
        test_orchestrator_initialization,
        test_agent_validation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! LLM-driven system is ready.")
        print("\nTo run the system:")
        print("1. Set your API keys:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export ANTHROPIC_API_KEY='your-key'")
        print("   export GOOGLE_API_KEY='your-key'")
        print("2. Run optimization:")
        print("   python scripts/run_search.py --config llm_compressor/configs/default.yaml --recipes conservative")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)