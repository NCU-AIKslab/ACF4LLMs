"""
LangGraph state definitions for compression workflow.
"""

from typing import Annotated, Any

try:
    from langchain_core.messages import BaseMessage
except ImportError:
    # Fallback when langchain_core is not installed
    BaseMessage = dict

try:
    from langgraph.graph.message import add_messages
except ImportError:
    # Fallback function when langgraph is not installed
    def add_messages(left, right):
        """Fallback message concatenation."""
        return left + right if isinstance(left, list) and isinstance(right, list) else right

from typing_extensions import TypedDict

from ..core.config import CompressionConfig
from ..core.metrics import EvaluationMetrics, ParetoSolution


class CompressionState(TypedDict):
    """
    State for the compression optimization workflow.

    This state is passed between nodes in the LangGraph workflow,
    replacing the A2A protocol message passing.

    Attributes:
        messages: Agent messages (using LangGraph message passing)
        config: Current compression configuration
        objective: User's optimization objective
        constraints: Resource constraints and requirements
        solutions: List of explored solutions
        pareto_frontier: Current Pareto-optimal solutions
        best_solution: Current best solution
        carbon_used: Total carbon budget consumed
        iterations: Number of optimization iterations
        strategy_results: Results from each compression strategy
        evaluation_results: Benchmark evaluation results
        should_continue: Whether to continue optimization
        error: Any error that occurred
    """

    # Agent communication
    messages: Annotated[list[BaseMessage], add_messages]

    # Configuration and objectives
    config: CompressionConfig | None
    objective: str
    constraints: dict[str, Any]

    # Optimization state
    solutions: list[ParetoSolution]
    pareto_frontier: list[ParetoSolution]
    best_solution: ParetoSolution | None

    # Carbon tracking
    carbon_used: float
    carbon_budget: float

    # Iteration tracking
    iterations: int
    max_iterations: int

    # Strategy results
    strategy_results: dict[str, dict[str, Any]]
    evaluation_results: dict[str, EvaluationMetrics]

    # Control flow
    should_continue: bool
    error: str | None


def create_initial_state(
    objective: str, carbon_budget: float = 10.0, max_iterations: int = 50, **constraints
) -> CompressionState:
    """
    Create initial state for compression workflow.

    Args:
        objective: User's optimization objective
        carbon_budget: Maximum carbon budget in kg CO2
        max_iterations: Maximum optimization iterations
        **constraints: Additional constraints (accuracy_threshold, max_memory_gb, etc.)

    Returns:
        Initial compression state
    """
    return CompressionState(
        messages=[],
        config=None,
        objective=objective,
        constraints=constraints,
        solutions=[],
        pareto_frontier=[],
        best_solution=None,
        carbon_used=0.0,
        carbon_budget=carbon_budget,
        iterations=0,
        max_iterations=max_iterations,
        strategy_results={},
        evaluation_results={},
        should_continue=True,
        error=None,
    )


def update_state_with_solution(
    state: CompressionState, solution: ParetoSolution, carbon_cost: float
) -> CompressionState:
    """
    Update state with a new solution.

    Args:
        state: Current state
        solution: New solution to add
        carbon_cost: Carbon cost of evaluating this solution

    Returns:
        Updated state
    """
    # Add solution
    state["solutions"].append(solution)

    # Update carbon tracking
    state["carbon_used"] += carbon_cost

    # Increment iterations
    state["iterations"] += 1

    # Check if should continue
    if state["carbon_used"] >= state["carbon_budget"]:
        state["should_continue"] = False
        state["error"] = "Carbon budget exceeded"

    if state["iterations"] >= state["max_iterations"]:
        state["should_continue"] = False

    return state


def update_pareto_frontier(
    state: CompressionState, solutions: list[ParetoSolution]
) -> CompressionState:
    """
    Update Pareto frontier in state.

    Args:
        state: Current state
        solutions: New Pareto-optimal solutions

    Returns:
        Updated state
    """
    state["pareto_frontier"] = solutions
    return state
