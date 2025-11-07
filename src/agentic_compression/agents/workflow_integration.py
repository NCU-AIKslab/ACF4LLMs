"""
Integration module to connect Deep Agent with LangGraph workflow.

This module provides a bridge between the autonomous Deep Agent and the
existing LangGraph compression optimization workflow.
"""

import json
import logging
from typing import Any, Dict, List

from ..core.config import CompressionConfig
from ..graph.state import CompressionState
from .compression_deep_agent import CompressionDeepAgent, create_compression_deep_agent

logger = logging.getLogger(__name__)

# Global agent instance (lazy initialization)
_agent_instance: CompressionDeepAgent = None


def get_agent_instance() -> CompressionDeepAgent:
    """
    Get or create the global Deep Agent instance.

    Returns:
        Singleton CompressionDeepAgent instance
    """
    global _agent_instance
    if _agent_instance is None:
        logger.info("Initializing Compression Deep Agent...")
        _agent_instance = create_compression_deep_agent(
            model_name="claude-3-5-sonnet-20241022",
            workspace_dir="./workspace",
            mlflow_tracking_uri="./mlruns",
            temperature=0.7,
        )
        logger.info("Compression Deep Agent initialized successfully")
    return _agent_instance


def parse_agent_plan(agent_output: Dict[str, Any]) -> List[CompressionConfig]:
    """
    Parse Deep Agent output into CompressionConfig objects.

    The agent returns a structured plan with recommended configurations.
    This function extracts those configs and converts them to CompressionConfig objects.

    Args:
        agent_output: Output from Deep Agent's plan_compression method

    Returns:
        List of CompressionConfig objects to evaluate
    """
    configs = []

    # Try to parse agent output
    try:
        # Agent output is in result['output']
        output_text = agent_output.get("output", "")

        # Try to find JSON configurations in the output
        # The agent should structure recommendations as JSON
        import re

        # Look for JSON blocks or structured config data
        json_pattern = r"\{[^{}]*\"(?:quantization_bits|bits)\"[^{}]*\}"
        json_matches = re.findall(json_pattern, output_text, re.DOTALL)

        for match in json_matches:
            try:
                config_dict = json.loads(match)

                # Normalize field names (agent might use different names)
                normalized = {}
                if "bits" in config_dict:
                    normalized["quantization_bits"] = config_dict["bits"]
                elif "quantization_bits" in config_dict:
                    normalized["quantization_bits"] = config_dict["quantization_bits"]
                else:
                    normalized["quantization_bits"] = 8  # Default

                if "sparsity" in config_dict:
                    normalized["pruning_sparsity"] = config_dict["sparsity"]
                elif "pruning_sparsity" in config_dict:
                    normalized["pruning_sparsity"] = config_dict["pruning_sparsity"]
                else:
                    normalized["pruning_sparsity"] = 0.0  # Default

                # Add other fields if present
                if "lora_rank" in config_dict:
                    normalized["lora_rank"] = config_dict["lora_rank"]
                if "distillation_ratio" in config_dict:
                    normalized["distillation_ratio"] = config_dict["distillation_ratio"]

                configs.append(CompressionConfig(**normalized))

            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse config: {match}, error: {e}")
                continue

        # If no configs found, use fallback defaults
        if not configs:
            logger.warning("No configurations found in agent output, using fallback")
            configs = _get_fallback_configs()

    except Exception as e:
        logger.error(f"Error parsing agent plan: {e}")
        configs = _get_fallback_configs()

    return configs


def _get_fallback_configs() -> List[CompressionConfig]:
    """
    Get fallback configurations if agent fails to provide any.

    Returns:
        List of sensible default CompressionConfig objects
    """
    return [
        CompressionConfig(quantization_bits=8, pruning_sparsity=0.3),
        CompressionConfig(quantization_bits=4, pruning_sparsity=0.5),
        CompressionConfig(quantization_bits=16, pruning_sparsity=0.1),
    ]


async def plan_optimization_with_agent(state: CompressionState) -> CompressionState:
    """
    Planning node using Deep Agent (replaces hardcoded planning logic).

    This function:
    1. Gets the Deep Agent instance
    2. Calls agent.plan_compression with current state
    3. Parses agent output into CompressionConfig objects
    4. Updates state with planned configurations

    Args:
        state: Current workflow state

    Returns:
        Updated state with agent-planned configurations
    """
    logger.info(f"Planning optimization with Deep Agent for: {state['objective']}")

    try:
        # Get agent
        agent = get_agent_instance()

        # Prepare task for agent
        model_name = state.get("model_name", "unknown-model")
        objective = state["objective"]
        carbon_budget = state.get("carbon_budget", None)
        constraints = state.get("constraints", {})

        # Query agent for existing experiments (agent has memory)
        previous_solutions = state.get("solutions", [])
        if previous_solutions:
            constraints["previous_attempts"] = len(previous_solutions)

        # Call agent to plan compression
        logger.info("Calling Deep Agent to plan compression strategy...")
        agent_result = agent.plan_compression(
            model_name=model_name,
            objective=objective,
            carbon_budget=carbon_budget,
            constraints=constraints,
        )

        logger.info(f"Agent planning completed: {agent_result.get('output', '')[:200]}...")

        # Parse agent output into configs
        configs_to_explore = parse_agent_plan(agent_result)

        logger.info(f"Parsed {len(configs_to_explore)} configurations from agent plan")

        # Store configurations to explore
        state["strategy_results"]["planned_configs"] = [
            vars(config) for config in configs_to_explore
        ]

        # Store agent reasoning for transparency
        state["strategy_results"]["agent_reasoning"] = agent_result.get("output", "")

        # Add planning message
        from ..graph.workflow import create_message

        state["messages"].append(
            create_message(
                f"Deep Agent created optimization plan with {len(configs_to_explore)} configurations. "
                f"Agent reasoning: {agent_result.get('output', '')[:150]}..."
            )
        )

        logger.info("Planning with Deep Agent completed successfully")

    except Exception as e:
        logger.error(f"Error in Deep Agent planning: {e}")
        # Fallback to default configs
        configs_to_explore = _get_fallback_configs()
        state["strategy_results"]["planned_configs"] = [
            vars(config) for config in configs_to_explore
        ]
        state["strategy_results"]["agent_error"] = str(e)

        from ..graph.workflow import create_message

        state["messages"].append(
            create_message(
                f"Deep Agent encountered an error: {e}. Using fallback configurations."
            )
        )

    return state


async def refine_search_with_agent(state: CompressionState) -> CompressionState:
    """
    Refinement node using Deep Agent (replaces hardcoded refinement).

    This function:
    1. Passes current solutions to agent
    2. Agent reflects and suggests improvements
    3. Parses new configurations to try
    4. Updates state

    Args:
        state: Current workflow state

    Returns:
        Updated state with refined configurations
    """
    logger.info("Refining search with Deep Agent")

    try:
        # Get agent
        agent = get_agent_instance()

        # Prepare current solutions for agent
        current_solutions = [
            {
                "config": {
                    "quantization_bits": sol.config.quantization_bits,
                    "pruning_sparsity": sol.config.pruning_sparsity,
                },
                "metrics": {
                    "accuracy": sol.metrics.accuracy,
                    "latency_ms": sol.metrics.latency_ms,
                    "memory_mb": sol.metrics.memory_mb,
                    "carbon_kg": sol.metrics.carbon_kg,
                },
            }
            for sol in state["solutions"]
        ]

        # Call agent to reflect and improve
        logger.info("Calling Deep Agent to reflect and suggest improvements...")
        agent_result = agent.reflect_and_improve(
            current_solutions=current_solutions,
            objective=state["objective"],
        )

        logger.info(f"Agent refinement completed: {agent_result.get('output', '')[:200]}...")

        # Parse new configs
        new_configs = parse_agent_plan(agent_result)

        if new_configs:
            # Add new configs to planned configs
            state["strategy_results"]["planned_configs"] = [
                vars(config) for config in new_configs
            ]

            from ..graph.workflow import create_message

            state["messages"].append(
                create_message(
                    f"Deep Agent suggested {len(new_configs)} refined configurations. "
                    f"Reasoning: {agent_result.get('output', '')[:150]}..."
                )
            )

            logger.info(f"Agent suggested {len(new_configs)} refined configurations")
        else:
            logger.info("Agent did not suggest new configurations, search complete")
            from ..graph.workflow import create_message

            state["messages"].append(
                create_message("Deep Agent determined no further refinement needed")
            )

    except Exception as e:
        logger.error(f"Error in Deep Agent refinement: {e}")
        from ..graph.workflow import create_message

        state["messages"].append(
            create_message(f"Deep Agent refinement error: {e}")
        )

    return state


def enable_deep_agent_workflow():
    """
    Enable Deep Agent in the workflow by replacing planning functions.

    This function should be called during initialization to use Deep Agent
    instead of the hardcoded planning logic.

    Usage:
        from agentic_compression.agents.workflow_integration import enable_deep_agent_workflow
        enable_deep_agent_workflow()
    """
    # Import workflow module
    from ..graph import workflow

    # Replace planning functions
    logger.info("Enabling Deep Agent workflow integration...")

    # Store original functions for fallback
    workflow._original_plan_optimization = workflow.plan_optimization
    workflow._original_refine_search = workflow.refine_search

    # Replace with agent versions
    workflow.plan_optimization = plan_optimization_with_agent
    workflow.refine_search = refine_search_with_agent

    logger.info("Deep Agent workflow integration enabled successfully")


def disable_deep_agent_workflow():
    """
    Disable Deep Agent and restore original workflow functions.
    """
    from ..graph import workflow

    if hasattr(workflow, "_original_plan_optimization"):
        workflow.plan_optimization = workflow._original_plan_optimization
        workflow.refine_search = workflow._original_refine_search
        logger.info("Deep Agent workflow integration disabled, restored originals")
    else:
        logger.warning("No original functions found to restore")
