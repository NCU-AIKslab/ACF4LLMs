"""
LangGraph workflow for compression optimization.

This replaces the A2A/GCP protocol with LangGraph-native agent orchestration.
"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, StateGraph

from ..core.config import CompressionConfig
from ..core.metrics import ParetoSolution
from ..core.pareto import compute_pareto_frontier, select_best_solution
from ..tools.evaluation_tools import evaluate_config_full
from .state import CompressionState, create_initial_state

logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Nodes
# ============================================================================


async def plan_optimization(state: CompressionState) -> CompressionState:
    """
    Planning node: Create optimization strategy based on objective.

    Args:
        state: Current workflow state

    Returns:
        Updated state with optimization plan
    """
    logger.info(f"Planning optimization for: {state['objective']}")

    # Parse objective and constraints to create initial configurations to explore
    objective = state["objective"].lower()
    # constraints = state["constraints"]  # Reserved for future use

    # Determine configurations to explore based on objective
    configs_to_explore = []

    if "edge" in objective or "mobile" in objective:
        # Aggressive compression for edge/mobile
        configs_to_explore = [
            CompressionConfig(quantization_bits=4, pruning_sparsity=0.7),
            CompressionConfig(quantization_bits=4, pruning_sparsity=0.5),
            CompressionConfig(quantization_bits=8, pruning_sparsity=0.5),
        ]
    elif "carbon" in objective or "green" in objective:
        # Carbon-focused
        configs_to_explore = [
            CompressionConfig(quantization_bits=4, pruning_sparsity=0.6),
            CompressionConfig(quantization_bits=8, pruning_sparsity=0.4),
            CompressionConfig(quantization_bits=8, pruning_sparsity=0.3),
        ]
    elif "accuracy" in objective:
        # Accuracy-focused
        configs_to_explore = [
            CompressionConfig(quantization_bits=16, pruning_sparsity=0.1),
            CompressionConfig(quantization_bits=8, pruning_sparsity=0.2),
            CompressionConfig(quantization_bits=8, pruning_sparsity=0.3),
        ]
    else:
        # Balanced approach
        configs_to_explore = [
            CompressionConfig(quantization_bits=8, pruning_sparsity=0.3),
            CompressionConfig(quantization_bits=4, pruning_sparsity=0.5),
            CompressionConfig(quantization_bits=16, pruning_sparsity=0.1),
        ]

    # Store configurations to explore
    state["strategy_results"]["planned_configs"] = [vars(config) for config in configs_to_explore]

    # Add planning message
    state["messages"].append(
        AIMessage(
            content=f"Created optimization plan with {len(configs_to_explore)} configurations to explore"
        )
    )

    return state


async def evaluate_configurations(state: CompressionState) -> CompressionState:
    """
    Evaluation node: Evaluate planned configurations.

    Args:
        state: Current workflow state

    Returns:
        Updated state with evaluation results
    """
    logger.info("Evaluating configurations")

    planned_configs_dict = state["strategy_results"].get("planned_configs", [])

    for config_dict in planned_configs_dict:
        # Recreate config object
        config = CompressionConfig(**config_dict)

        # Evaluate configuration
        metrics = await evaluate_config_full(config)

        # Create solution
        solution = ParetoSolution(metrics=metrics)
        state["solutions"].append(solution)

        # Update carbon usage (estimate)
        state["carbon_used"] += 0.01  # Small carbon cost per evaluation

        logger.info(
            f"Evaluated config: bits={config.quantization_bits}, "
            f"sparsity={config.pruning_sparsity:.1%}, "
            f"accuracy={metrics.average_accuracy():.3f}, "
            f"co2={metrics.co2_kg:.4f}kg"
        )

    # Increment iterations
    state["iterations"] += 1

    # Add evaluation message
    state["messages"].append(
        AIMessage(content=f"Evaluated {len(planned_configs_dict)} configurations")
    )

    return state


async def compute_pareto(state: CompressionState) -> CompressionState:
    """
    Pareto computation node: Calculate Pareto frontier.

    Args:
        state: Current workflow state

    Returns:
        Updated state with Pareto frontier
    """
    logger.info("Computing Pareto frontier")

    if not state["solutions"]:
        logger.warning("No solutions to compute Pareto frontier")
        return state

    # Compute Pareto frontier
    pareto_frontier = compute_pareto_frontier(state["solutions"])

    state["pareto_frontier"] = pareto_frontier

    # Select best solution based on objective
    if "accuracy" in state["objective"].lower():
        criterion = "accuracy"
    elif "carbon" in state["objective"].lower() or "green" in state["objective"].lower():
        criterion = "carbon"
    else:
        criterion = "balanced"

    best = select_best_solution(pareto_frontier, criterion=criterion)
    state["best_solution"] = best

    # Add message
    state["messages"].append(
        AIMessage(
            content=f"Found {len(pareto_frontier)} Pareto-optimal solutions. "
            f"Best solution: accuracy={best.metrics.average_accuracy():.3f}, "
            f"CO2={best.metrics.co2_kg:.4f}kg"
        )
    )

    logger.info(f"Pareto frontier: {len(pareto_frontier)} solutions")

    return state


def should_continue(state: CompressionState) -> Literal["continue", "finish"]:
    """
    Conditional edge: Decide whether to continue optimization.

    Args:
        state: Current workflow state

    Returns:
        "continue" or "finish"
    """
    if state["error"]:
        logger.warning(f"Stopping due to error: {state['error']}")
        return "finish"

    if state["carbon_used"] >= state["carbon_budget"]:
        logger.info("Carbon budget reached")
        return "finish"

    if state["iterations"] >= state["max_iterations"]:
        logger.info("Max iterations reached")
        return "finish"

    if not state.get("should_continue", True):
        logger.info("Optimization completed")
        return "finish"

    return "continue"


async def refine_search(state: CompressionState) -> CompressionState:
    """
    Refinement node: Refine search around promising solutions.

    Args:
        state: Current workflow state

    Returns:
        Updated state with refined configurations
    """
    logger.info("Refining search around best solutions")

    if not state["pareto_frontier"]:
        state["should_continue"] = False
        return state

    # Take top 2 solutions from Pareto frontier
    top_solutions = state["pareto_frontier"][:2]

    refined_configs = []
    for sol in top_solutions:
        base_config = sol.metrics.config
        if base_config:
            # Create variations
            if base_config.quantization_bits > 4:
                refined_configs.append(
                    CompressionConfig(
                        quantization_bits=base_config.quantization_bits // 2,
                        pruning_sparsity=base_config.pruning_sparsity,
                    )
                )
            if base_config.pruning_sparsity < 0.6:
                refined_configs.append(
                    CompressionConfig(
                        quantization_bits=base_config.quantization_bits,
                        pruning_sparsity=min(base_config.pruning_sparsity + 0.1, 0.7),
                    )
                )

    # Update planned configs for next iteration
    state["strategy_results"]["planned_configs"] = [vars(config) for config in refined_configs]

    # Check if we have enough iterations left
    if state["iterations"] >= state["max_iterations"] - 1:
        state["should_continue"] = False

    return state


# ============================================================================
# Workflow Creation
# ============================================================================


def create_compression_workflow(llm=None, checkpointer=None):
    """
    Create the compression optimization workflow using LangGraph.

    This replaces the A2A/GCP protocol with native LangGraph orchestration.

    Args:
        llm: Optional LLM for agent reasoning (not used in current implementation)
        checkpointer: Optional checkpointer for persistence

    Returns:
        Compiled LangGraph workflow
    """
    # Create workflow graph
    workflow = StateGraph(CompressionState)

    # Add nodes
    workflow.add_node("plan", plan_optimization)
    workflow.add_node("evaluate", evaluate_configurations)
    workflow.add_node("pareto", compute_pareto)
    workflow.add_node("refine", refine_search)

    # Define edges
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "evaluate")
    workflow.add_edge("evaluate", "pareto")

    # Conditional edge: continue or finish
    workflow.add_conditional_edges(
        "pareto",
        should_continue,
        {
            "continue": "refine",
            "finish": END,
        },
    )

    workflow.add_edge("refine", "evaluate")

    # Compile workflow
    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled_workflow = workflow.compile(checkpointer=checkpointer)

    return compiled_workflow


# ============================================================================
# Helper Functions
# ============================================================================


async def run_compression_optimization(
    objective: str, carbon_budget: float = 5.0, max_iterations: int = 10, **constraints
) -> dict:
    """
    Run complete compression optimization workflow.

    Args:
        objective: Optimization objective (e.g., "edge deployment", "carbon-efficient")
        carbon_budget: Carbon budget in kg CO2
        max_iterations: Maximum optimization iterations
        **constraints: Additional constraints

    Returns:
        Dictionary with optimization results
    """
    # Create initial state
    initial_state = create_initial_state(
        objective=objective,
        carbon_budget=carbon_budget,
        max_iterations=max_iterations,
        **constraints,
    )

    # Create workflow
    workflow = create_compression_workflow()

    # Run workflow
    final_state = await workflow.ainvoke(initial_state)

    # Extract results
    results = {
        "objective": objective,
        "total_solutions": len(final_state["solutions"]),
        "pareto_optimal_count": len(final_state["pareto_frontier"]),
        "best_solution": (
            final_state["best_solution"].metrics.to_dict() if final_state["best_solution"] else None
        ),
        "carbon_used": final_state["carbon_used"],
        "carbon_budget": final_state["carbon_budget"],
        "iterations": final_state["iterations"],
        "pareto_frontier": [sol.metrics.to_dict() for sol in final_state["pareto_frontier"]],
    }

    return results
