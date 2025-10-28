"""Core data structures and algorithms"""

from .config import CompressionConfig
from .metrics import EvaluationMetrics, ParetoSolution
from .pareto import compute_pareto_frontier, dominates

__all__ = [
    "CompressionConfig",
    "EvaluationMetrics",
    "ParetoSolution",
    "compute_pareto_frontier",
    "dominates",
]
