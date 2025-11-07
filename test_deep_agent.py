"""
Test script for Compression Deep Agent

This script demonstrates basic usage of the Deep Agent for compression optimization.
"""

import asyncio
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def test_agent_initialization():
    """Test 1: Initialize the Deep Agent"""
    print("\n" + "=" * 80)
    print("TEST 1: Initializing Compression Deep Agent")
    print("=" * 80)

    try:
        from src.agentic_compression.agents.compression_deep_agent import (
            create_compression_deep_agent,
        )

        # Create agent (requires ANTHROPIC_API_KEY environment variable)
        agent = create_compression_deep_agent(
            model_name="claude-3-5-sonnet-20241022",
            workspace_dir="./workspace",
        )

        print("✓ Agent initialized successfully")
        print(f"  - Workspace: ./workspace")
        print(f"  - Tools available: {len(agent.tools)}")
        print(f"  - Tool names: {[tool.name for tool in agent.tools][:5]}...")  # Show first 5

        return agent

    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return None


def test_lora_tools():
    """Test 2: Test LoRA tools directly"""
    print("\n" + "=" * 80)
    print("TEST 2: Testing LoRA Tools")
    print("=" * 80)

    try:
        from src.agentic_compression.agents.sub_agents.lora_sub_agent import (
            create_lora_tools,
        )

        tools = create_lora_tools()
        print(f"✓ Created {len(tools)} LoRA tools")

        # Test configure_lora tool
        configure_tool = tools[0]  # ConfigureLoRATool
        result = configure_tool._run(
            base_model="meta-llama/Llama-2-7b-hf",
            rank=8,
            alpha=16,
        )

        print(f"✓ ConfigureLoRA test result:")
        print(result)

        # Test estimate tool
        estimate_tool = tools[2]  # EstimateLoRAImpactTool
        result = estimate_tool._run(
            base_model="meta-llama/Llama-2-7b-hf",
            rank=8,
        )

        print(f"✓ EstimateLoRAImpact test result:")
        print(result)

        return True

    except Exception as e:
        print(f"✗ LoRA tools test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_distillation_tools():
    """Test 3: Test distillation tools directly"""
    print("\n" + "=" * 80)
    print("TEST 3: Testing Distillation Tools")
    print("=" * 80)

    try:
        from src.agentic_compression.agents.sub_agents.distillation_sub_agent import (
            create_distillation_tools,
        )

        tools = create_distillation_tools()
        print(f"✓ Created {len(tools)} distillation tools")

        # Test setup_distillation tool
        setup_tool = tools[0]  # SetupDistillationTool
        result = setup_tool._run(
            teacher_model="meta-llama/Llama-2-7b-hf",
            student_scale=0.5,
            temperature=2.0,
            alpha=0.7,
        )

        print(f"✓ SetupDistillation test result:")
        print(result)

        # Test estimate tool
        estimate_tool = tools[1]  # EstimateDistillationTool
        result = estimate_tool._run(
            teacher_model="meta-llama/Llama-2-7b-hf",
            compression_ratio=2.0,
        )

        print(f"✓ EstimateDistillation test result:")
        print(result)

        return True

    except Exception as e:
        print(f"✗ Distillation tools test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mlflow_tools():
    """Test 4: Test MLflow tracking tools"""
    print("\n" + "=" * 80)
    print("TEST 4: Testing MLflow Tracking Tools")
    print("=" * 80)

    try:
        from src.agentic_compression.agents.tracking_tool import create_tracking_tools

        tools = create_tracking_tools(tracking_uri="./mlruns")
        print(f"✓ Created {len(tools)} tracking tools")

        # Test log_experiment tool
        log_tool = tools[0]  # LogExperimentTool
        result = log_tool._run(
            config={"technique": "quantization", "bits": 8},
            metrics={"accuracy": 0.654, "latency_ms": 45.3, "memory_mb": 3421},
            model_name="test-model",
            tags={"test": "true"},
        )

        print(f"✓ LogExperiment test result:")
        print(result)

        return True

    except Exception as e:
        print(f"✗ MLflow tools test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_workspace():
    """Test 5: Test workspace functionality"""
    print("\n" + "=" * 80)
    print("TEST 5: Testing Workspace Manager")
    print("=" * 80)

    try:
        from src.agentic_compression.agents.compression_deep_agent import (
            WorkspaceManager,
        )

        workspace = WorkspaceManager(workspace_dir="./workspace")
        print(f"✓ Workspace initialized at: {workspace.workspace_dir}")

        # Test save experiment
        filepath = workspace.save_experiment(
            experiment_id="test_exp_001",
            config={"technique": "quantization", "bits": 8},
            metrics={"accuracy": 0.654},
        )
        print(f"✓ Saved experiment to: {filepath}")

        # Test load experiments
        experiments = workspace.load_experiments()
        print(f"✓ Loaded {len(experiments)} experiments")

        # Test save knowledge
        filepath = workspace.save_knowledge(
            topic="quantization_best_practices",
            content="# Quantization Best Practices\n\n- Use 8-bit for best accuracy/size trade-off\n- 4-bit requires calibration",
        )
        print(f"✓ Saved knowledge to: {filepath}")

        # Test load knowledge
        content = workspace.load_knowledge("quantization_best_practices")
        print(f"✓ Loaded knowledge: {content[:50]}...")

        return True

    except Exception as e:
        print(f"✗ Workspace test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent_simple_task():
    """Test 6: Run agent on a simple task (requires API key)"""
    print("\n" + "=" * 80)
    print("TEST 6: Testing Agent on Simple Task")
    print("=" * 80)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠ Skipping agent task test (ANTHROPIC_API_KEY not set)")
        print("  Set the API key to test the full agent:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        return None

    try:
        from src.agentic_compression.agents.compression_deep_agent import (
            create_compression_deep_agent,
        )

        agent = create_compression_deep_agent()

        # Simple task: estimate LoRA impact
        print("Running agent on task: Estimate LoRA compression for Llama-2-7B...")
        result = agent.run(
            "Estimate the impact of applying LoRA (rank=8) compression to meta-llama/Llama-2-7b-hf. "
            "Use the estimate_lora_impact tool."
        )

        print(f"✓ Agent task completed:")
        print(f"  Output: {result.get('output', '')[:200]}...")

        return True

    except Exception as e:
        print(f"✗ Agent task test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("COMPRESSION DEEP AGENT TEST SUITE")
    print("=" * 80)

    results = {}

    # Test 1: Agent initialization (skip if no API key for now)
    if os.getenv("ANTHROPIC_API_KEY"):
        results["initialization"] = test_agent_initialization() is not None
    else:
        print("\n⚠ Skipping agent initialization test (no ANTHROPIC_API_KEY)")
        results["initialization"] = None

    # Test 2: LoRA tools
    results["lora_tools"] = test_lora_tools()

    # Test 3: Distillation tools
    results["distillation_tools"] = test_distillation_tools()

    # Test 4: MLflow tools
    results["mlflow_tools"] = test_mlflow_tools()

    # Test 5: Workspace
    results["workspace"] = test_workspace()

    # Test 6: Agent simple task (requires API key)
    results["agent_task"] = test_agent_simple_task()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"

        print(f"  {test_name:<25} {status}")

    passed = sum(1 for r in results.values() if r is True)
    total = len([r for r in results.values() if r is not None])
    print(f"\nPassed: {passed}/{total}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
