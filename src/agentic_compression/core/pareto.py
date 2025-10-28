"""
Pareto frontier computation and multi-objective optimization algorithms.
"""


import numpy as np

from .metrics import EvaluationMetrics, ParetoSolution


def dominates(metrics_a: EvaluationMetrics, metrics_b: EvaluationMetrics) -> bool:
    """
    Check if solution A dominates solution B in multi-objective optimization.

    A solution dominates another if it is no worse in all objectives and
    strictly better in at least one objective.

    Args:
        metrics_a: Metrics for solution A
        metrics_b: Metrics for solution B

    Returns:
        True if A dominates B, False otherwise
    """
    objectives = {
        "accuracy": (metrics_a.average_accuracy(), metrics_b.average_accuracy(), True),  # maximize
        "latency": (metrics_a.latency_ms, metrics_b.latency_ms, False),  # minimize
        "memory": (metrics_a.memory_gb, metrics_b.memory_gb, False),  # minimize
        "energy": (metrics_a.energy_kwh, metrics_b.energy_kwh, False),  # minimize
        "co2": (metrics_a.co2_kg, metrics_b.co2_kg, False),  # minimize
    }

    better_in_at_least_one = False

    for obj, (val_a, val_b, maximize) in objectives.items():
        if maximize:
            # For maximization: A must be >= B in all, and > B in at least one
            if val_a < val_b:
                return False
            if val_a > val_b:
                better_in_at_least_one = True
        else:
            # For minimization: A must be <= B in all, and < B in at least one
            if val_a > val_b:
                return False
            if val_a < val_b:
                better_in_at_least_one = True

    return better_in_at_least_one


def weighted_dominates(
    metrics_a: EvaluationMetrics,
    metrics_b: EvaluationMetrics,
    weights: dict[str, float],
    threshold: float = 1.05,
) -> bool:
    """
    Check if solution A dominates B using weighted composite scores.

    Args:
        metrics_a: Metrics for solution A
        metrics_b: Metrics for solution B
        weights: Dictionary of objective weights
        threshold: Minimum ratio for domination (default 1.05 = 5% better)

    Returns:
        True if A dominates B with weighted scoring
    """
    score_a = compute_composite_score(metrics_a, weights)
    score_b = compute_composite_score(metrics_b, weights)

    return score_a > score_b * threshold


def compute_composite_score(metrics: EvaluationMetrics, weights: dict[str, float]) -> float:
    """
    Compute weighted composite score for multi-objective optimization.

    Args:
        metrics: Evaluation metrics
        weights: Dictionary of objective weights (positive for maximize, negative for minimize)

    Returns:
        Weighted composite score
    """
    # Convert all metrics to "higher is better" form
    values = {
        "accuracy": metrics.average_accuracy(),
        "latency": 1.0 / (metrics.latency_ms + 1e-6),
        "memory": 1.0 / (metrics.memory_gb + 1e-6),
        "energy": 1.0 / (metrics.energy_kwh + 1e-6),
        "co2": 1.0 / (metrics.co2_kg + 1e-6),
    }

    score = 0.0
    for key, weight in weights.items():
        if key in values:
            score += abs(weight) * values[key]

    return score


def compute_pareto_frontier(solutions: list[ParetoSolution]) -> list[ParetoSolution]:
    """
    Compute the Pareto frontier from a list of solutions.

    Uses dominance relationships to identify non-dominated solutions.

    Args:
        solutions: List of solutions to analyze

    Returns:
        List of Pareto-optimal solutions
    """
    n = len(solutions)

    # Reset all domination relationships
    for sol in solutions:
        sol.reset_domination()

    # Compute domination relationships
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(solutions[i].metrics, solutions[j].metrics):
                    solutions[i].dominates.append(j)
                    solutions[j].dominated_by.append(i)

    # Identify Pareto-optimal solutions (not dominated by any other)
    pareto_frontier = []
    for sol in solutions:
        if len(sol.dominated_by) == 0:
            sol.is_pareto_optimal = True
            sol.rank = 0
            pareto_frontier.append(sol)

    return pareto_frontier


def compute_pareto_fronts(solutions: list[ParetoSolution]) -> list[list[ParetoSolution]]:
    """
    Compute multiple Pareto fronts (non-dominated sorting).

    Args:
        solutions: List of solutions to analyze

    Returns:
        List of fronts, where front[0] is the Pareto frontier
    """
    fronts = []
    remaining = solutions.copy()

    rank = 0
    while remaining:
        # Find non-dominated solutions in remaining set
        current_front = []

        for sol in remaining:
            sol.reset_domination()

        # Compute domination within remaining solutions
        for i, sol_i in enumerate(remaining):
            for j, sol_j in enumerate(remaining):
                if i != j:
                    if dominates(sol_i.metrics, sol_j.metrics):
                        sol_i.dominates.append(j)
                        sol_j.dominated_by.append(i)

        # Extract non-dominated solutions
        for sol in remaining:
            if len(sol.dominated_by) == 0:
                sol.is_pareto_optimal = rank == 0
                sol.rank = rank
                current_front.append(sol)

        if not current_front:
            break

        fronts.append(current_front)
        remaining = [sol for sol in remaining if sol not in current_front]
        rank += 1

    return fronts


def calculate_crowding_distance(solutions: list[ParetoSolution]) -> None:
    """
    Calculate crowding distance for solutions (in-place modification).

    Crowding distance measures how close a solution is to its neighbors,
    used to maintain diversity in the Pareto frontier.

    Args:
        solutions: List of solutions (modified in-place)
    """
    n = len(solutions)
    if n == 0:
        return

    # Initialize distances
    for sol in solutions:
        sol.crowding_distance = 0.0

    # Calculate distance for each objective
    objectives = ["accuracy", "energy", "co2"]

    for obj in objectives:
        # Sort by objective
        if obj == "accuracy":
            sorted_sols = sorted(solutions, key=lambda x: x.metrics.average_accuracy())
        elif obj == "energy":
            sorted_sols = sorted(solutions, key=lambda x: x.metrics.energy_kwh)
        else:  # co2
            sorted_sols = sorted(solutions, key=lambda x: x.metrics.co2_kg)

        # Boundary solutions get infinite distance
        sorted_sols[0].crowding_distance = float("inf")
        sorted_sols[-1].crowding_distance = float("inf")

        # Calculate distance for interior solutions
        if n > 2:
            for i in range(1, n - 1):
                if obj == "accuracy":
                    range_val = (
                        sorted_sols[-1].metrics.average_accuracy()
                        - sorted_sols[0].metrics.average_accuracy()
                    )
                    if range_val > 0:
                        distance = (
                            sorted_sols[i + 1].metrics.average_accuracy()
                            - sorted_sols[i - 1].metrics.average_accuracy()
                        ) / range_val
                    else:
                        distance = 0
                elif obj == "energy":
                    range_val = (
                        sorted_sols[-1].metrics.energy_kwh - sorted_sols[0].metrics.energy_kwh
                    )
                    if range_val > 0:
                        distance = (
                            sorted_sols[i + 1].metrics.energy_kwh
                            - sorted_sols[i - 1].metrics.energy_kwh
                        ) / range_val
                    else:
                        distance = 0
                else:  # co2
                    range_val = sorted_sols[-1].metrics.co2_kg - sorted_sols[0].metrics.co2_kg
                    if range_val > 0:
                        distance = (
                            sorted_sols[i + 1].metrics.co2_kg - sorted_sols[i - 1].metrics.co2_kg
                        ) / range_val
                    else:
                        distance = 0

                sorted_sols[i].crowding_distance += distance


def select_best_solution(
    pareto_frontier: list[ParetoSolution],
    weights: dict[str, float] | None = None,
    criterion: str = "balanced",
) -> ParetoSolution | None:
    """
    Select the best solution from Pareto frontier based on criterion.

    Args:
        pareto_frontier: List of Pareto-optimal solutions
        weights: Optional weights for composite scoring
        criterion: Selection criterion ("balanced", "accuracy", "carbon", "weighted")

    Returns:
        Selected best solution or None if frontier is empty
    """
    if not pareto_frontier:
        return None

    if criterion == "accuracy":
        return max(pareto_frontier, key=lambda x: x.metrics.average_accuracy())

    elif criterion == "carbon":
        return min(pareto_frontier, key=lambda x: x.metrics.co2_kg)

    elif criterion == "balanced":
        # Simple balanced scoring: maximize accuracy, minimize carbon
        def balance_score(sol: ParetoSolution) -> float:
            return sol.metrics.average_accuracy() * 2.0 - sol.metrics.co2_kg * 10.0

        return max(pareto_frontier, key=balance_score)

    elif criterion == "weighted" and weights:
        return max(pareto_frontier, key=lambda x: compute_composite_score(x.metrics, weights))

    else:
        # Default to balanced
        return select_best_solution(pareto_frontier, criterion="balanced")


def compute_hypervolume(
    solutions: list[ParetoSolution], reference_point: dict[str, float]
) -> float:
    """
    Compute hypervolume indicator for Pareto frontier quality.

    The hypervolume is the volume of objective space dominated by the Pareto frontier.

    Args:
        solutions: List of solutions (typically Pareto-optimal)
        reference_point: Reference point for hypervolume calculation

    Returns:
        Hypervolume value
    """
    if not solutions:
        return 0.0

    # Simple 2D hypervolume calculation (accuracy vs co2)
    # For production, use more sophisticated algorithms for higher dimensions

    points = [(sol.metrics.average_accuracy(), sol.metrics.co2_kg) for sol in solutions]

    # Sort by first objective (accuracy) descending
    points.sort(reverse=True, key=lambda p: p[0])

    ref_acc = reference_point.get("accuracy", 0.0)
    ref_co2 = reference_point.get("co2", 1.0)

    hypervolume = 0.0
    prev_acc = ref_acc

    for acc, co2 in points:
        if acc > ref_acc and co2 < ref_co2:
            hypervolume += (acc - prev_acc) * (ref_co2 - co2)
            prev_acc = acc

    return hypervolume


def calculate_frontier_diversity(frontier: list[ParetoSolution]) -> float:
    """
    Calculate diversity metric for Pareto frontier.

    Measures the spread of solutions across objective space.

    Args:
        frontier: List of Pareto-optimal solutions

    Returns:
        Average pairwise distance between solutions
    """
    if len(frontier) < 2:
        return 0.0

    distances = []
    for i in range(len(frontier)):
        for j in range(i + 1, len(frontier)):
            dist = solution_distance(frontier[i].metrics, frontier[j].metrics)
            distances.append(dist)

    return np.mean(distances) if distances else 0.0


def solution_distance(metrics_a: EvaluationMetrics, metrics_b: EvaluationMetrics) -> float:
    """
    Calculate normalized Euclidean distance between two solutions.

    Args:
        metrics_a: Metrics for solution A
        metrics_b: Metrics for solution B

    Returns:
        Normalized distance
    """
    # Normalize features to [0, 1] range for fair comparison
    features_a = np.array(
        [
            metrics_a.average_accuracy(),
            metrics_a.latency_ms / 100.0,
            metrics_a.memory_gb / 24.0,
            metrics_a.energy_kwh / 0.084,
            metrics_a.co2_kg / 0.034,
        ]
    )

    features_b = np.array(
        [
            metrics_b.average_accuracy(),
            metrics_b.latency_ms / 100.0,
            metrics_b.memory_gb / 24.0,
            metrics_b.energy_kwh / 0.084,
            metrics_b.co2_kg / 0.034,
        ]
    )

    return float(np.linalg.norm(features_a - features_b))
