#!/usr/bin/env python3
"""Run LLM-driven optimization search using LangChain and LangGraph."""

import argparse
import sys
import os
import logging
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm_compressor.core.orchestrator import Orchestrator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('llm_optimization.log')
        ]
    )


def check_api_keys():
    """Check if required API keys are set."""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = []

    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)

    if missing_keys:
        print("Warning: Missing API keys for LLM providers:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nThe system will attempt to run with mock responses.")
        print("For full functionality, set the API keys as environment variables.")
        return False

    return True


def validate_config_file(config_path: str) -> bool:
    """Validate configuration file exists and is readable."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return False

    if not config_file.is_file():
        print(f"Error: Configuration path is not a file: {config_path}")
        return False

    try:
        with open(config_file, 'r') as f:
            import yaml
            yaml.safe_load(f)
        return True
    except Exception as e:
        print(f"Error: Invalid YAML configuration file: {e}")
        return False


def run_llm_optimization(config_path: str, recipes: str, output_dir: str) -> dict:
    """Run LLM-driven optimization."""

    print(f"Starting LLM-driven optimization...")
    print(f"Config: {config_path}")
    print(f"Recipes: {recipes}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    try:
        # Initialize LLM orchestrator
        print("Initializing LLM orchestrator...")
        orchestrator = Orchestrator(config_path)

        print(f"Initialized {len(orchestrator.agents)} LLM agents:")
        for agent_name in orchestrator.agents.keys():
            print(f"  - {agent_name}")
        print()

        # Generate recipes based on strategy
        if recipes == "baseline":
            print("Generating baseline recipes...")
            recipe_list = [
                {
                    "id": "baseline_recipe",
                    "name": "baseline",
                    "description": "Baseline performance measurement",
                    "pipeline": ["perf_carbon", "eval_safety"],
                    "optimization_level": "minimal"
                }
            ]

        elif recipes == "conservative":
            print("Generating conservative optimization recipes...")
            recipe_list = [
                {
                    "id": "conservative_quantization",
                    "name": "conservative_quantization",
                    "description": "Conservative 8-bit quantization",
                    "pipeline": ["quantization", "perf_carbon", "eval_safety"],
                    "quantization": {
                        "method": "bitsandbytes",
                        "load_in_8bit": True
                    },
                    "optimization_level": "conservative"
                },
                {
                    "id": "conservative_attention",
                    "name": "conservative_attention",
                    "description": "FlashAttention optimization",
                    "pipeline": ["kv_longcontext", "perf_carbon", "eval_safety"],
                    "kv_optimization": {
                        "method": "flash_attention",
                        "version": "flash_attention_2"
                    },
                    "optimization_level": "conservative"
                }
            ]

        elif recipes == "aggressive":
            print("Generating aggressive optimization recipes...")
            recipe_list = [
                {
                    "id": "aggressive_quantization",
                    "name": "aggressive_quantization",
                    "description": "Aggressive 4-bit quantization",
                    "pipeline": ["quantization", "perf_carbon", "eval_safety"],
                    "quantization": {
                        "method": "awq",
                        "bits": 4,
                        "group_size": 128
                    },
                    "optimization_level": "aggressive"
                },
                {
                    "id": "aggressive_combined",
                    "name": "aggressive_combined",
                    "description": "Combined quantization and pruning",
                    "pipeline": ["quantization", "pruning_sparsity", "perf_carbon", "eval_safety"],
                    "quantization": {
                        "method": "gptq",
                        "bits": 4
                    },
                    "pruning": {
                        "method": "unstructured",
                        "sparsity_ratio": 0.5
                    },
                    "optimization_level": "aggressive"
                }
            ]

        elif recipes == "llm_planned":
            print("Using LLM-planned recipes...")
            # Use recipe planner agent to generate recipes
            planner_recipe = {
                "id": "llm_planner",
                "recipe_planning": {
                    "strategy": "optimization_portfolio",
                    "num_recipes": 5,
                    "diversity": "high"
                }
            }

            context = {
                "config": orchestrator.config,
                "model_path": orchestrator.config.get("model", {}).get("base_model", "google/gemma-3-4b-it"),
                "hardware_config": orchestrator.config.get("hardware", {})
            }

            if "recipe_planner" in orchestrator.agents:
                planner_result = orchestrator.agents["recipe_planner"].execute(planner_recipe, context)
                if planner_result.success:
                    recipe_list = planner_result.artifacts.get("recipe_portfolio", [])
                    print(f"LLM planner generated {len(recipe_list)} recipes")
                else:
                    print(f"LLM planner failed: {planner_result.error}")
                    recipe_list = []
            else:
                print("Recipe planner agent not available, using default recipes")
                recipe_list = []

        else:
            print(f"Unknown recipe strategy: {recipes}")
            return {"success": False, "error": f"Unknown recipe strategy: {recipes}"}

        if not recipe_list:
            print("No recipes generated, using baseline recipe")
            recipe_list = [
                {
                    "id": "fallback_baseline",
                    "name": "baseline",
                    "pipeline": ["perf_carbon", "eval_safety"],
                    "optimization_level": "minimal"
                }
            ]

        # Execute recipes using LangGraph workflow
        print(f"\nExecuting {len(recipe_list)} recipes using LangGraph workflow...")
        results = []

        for i, recipe in enumerate(recipe_list, 1):
            print(f"\n[{i}/{len(recipe_list)}] Executing recipe: {recipe.get('name', recipe['id'])}")
            print(f"Description: {recipe.get('description', 'No description')}")
            print(f"Pipeline: {recipe.get('pipeline', [])}")

            start_time = time.time()

            try:
                # Execute recipe through LangGraph workflow
                result = orchestrator.execute_recipe(recipe)

                execution_time = time.time() - start_time
                result["execution_time"] = execution_time

                print(f"✓ Recipe completed in {execution_time:.1f}s")
                print(f"  Success: {result.get('success', False)}")

                if result.get('success'):
                    metrics = result.get('metrics', {})
                    print(f"  Metrics: {len(metrics)} collected")

                    # Show key metrics
                    if 'composite_score' in metrics:
                        print(f"  Composite Score: {metrics['composite_score']:.3f}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")

                results.append(result)

            except Exception as e:
                execution_time = time.time() - start_time
                error_result = {
                    "recipe_id": recipe["id"],
                    "success": False,
                    "error": str(e),
                    "execution_time": execution_time
                }
                results.append(error_result)
                print(f"✗ Recipe failed in {execution_time:.1f}s: {e}")

        # Analyze results
        print(f"\n{'='*60}")
        print("OPTIMIZATION RESULTS SUMMARY")
        print(f"{'='*60}")

        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]

        print(f"Total recipes executed: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")

        if successful_results:
            print(f"\nBest results:")
            # Sort by composite score if available
            sorted_results = sorted(
                successful_results,
                key=lambda x: x.get("metrics", {}).get("composite_score", 0),
                reverse=True
            )

            for i, result in enumerate(sorted_results[:3], 1):
                metrics = result.get("metrics", {})
                recipe_id = result.get("recipe_id", "unknown")
                composite_score = metrics.get("composite_score", 0)
                print(f"  {i}. {recipe_id}: Score {composite_score:.3f}")

        if failed_results:
            print(f"\nFailed recipes:")
            for result in failed_results:
                recipe_id = result.get("recipe_id", "unknown")
                error = result.get("error", "Unknown error")
                print(f"  - {recipe_id}: {error}")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "llm_optimization_results.json"

        import json
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "config_path": config_path,
                "recipe_strategy": recipes,
                "total_recipes": len(results),
                "successful_recipes": len(successful_results),
                "results": results
            }, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

        return {
            "success": True,
            "total_recipes": len(results),
            "successful_recipes": len(successful_results),
            "results": results,
            "output_file": str(results_file)
        }

    except Exception as e:
        print(f"Error during LLM optimization: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM-driven model optimization using LangChain and LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline performance measurement
  python scripts/run_search.py --config llm_compressor/configs/default.yaml --recipes baseline

  # Run conservative optimization
  python scripts/run_search.py --config llm_compressor/configs/default.yaml --recipes conservative

  # Run aggressive optimization
  python scripts/run_search.py --config llm_compressor/configs/default.yaml --recipes aggressive

  # Use LLM-planned recipes
  python scripts/run_search.py --config llm_compressor/configs/default.yaml --recipes llm_planned

Environment Variables:
  OPENAI_API_KEY      OpenAI API key for GPT models
  ANTHROPIC_API_KEY   Anthropic API key for Claude models
  GOOGLE_API_KEY      Google API key for Gemini models
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="llm_compressor/configs/default.yaml",
        help="Path to configuration file (default: llm_compressor/configs/default.yaml)"
    )

    parser.add_argument(
        "--recipes",
        type=str,
        choices=["baseline", "conservative", "aggressive", "llm_planned"],
        default="conservative",
        help="Recipe generation strategy (default: conservative)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for results (default: reports)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Print banner
    print("=" * 60)
    print("LLM-DRIVEN MODEL OPTIMIZATION")
    print("Using LangChain and LangGraph")
    print("=" * 60)

    # Check API keys
    has_api_keys = check_api_keys()
    if not has_api_keys:
        print("\nContinuing with mock mode...\n")

    # Validate configuration
    if not validate_config_file(args.config):
        sys.exit(1)

    # Run optimization
    result = run_llm_optimization(args.config, args.recipes, args.output)

    if result["success"]:
        print(f"\n✓ LLM optimization completed successfully!")
        print(f"✓ {result['successful_recipes']}/{result['total_recipes']} recipes succeeded")
        print(f"✓ Results saved to: {result['output_file']}")
        sys.exit(0)
    else:
        print(f"\n✗ LLM optimization failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()