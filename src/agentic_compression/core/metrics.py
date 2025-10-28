"""
Evaluation metrics and Pareto solution data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime

from .config import CompressionConfig


@dataclass
class EvaluationMetrics:
    """
    Comprehensive metrics for evaluating compressed models.

    Attributes:
        accuracy: Dictionary of accuracy scores per benchmark
        latency_ms: Inference latency in milliseconds
        memory_gb: Memory usage in gigabytes
        energy_kwh: Energy consumption in kilowatt-hours
        co2_kg: Carbon emissions in kilograms CO2
        throughput_tps: Throughput in tokens per second
        compression_ratio: Ratio of compressed to original size
        timestamp: ISO format timestamp of evaluation
        config: Configuration used for compression (optional)
    """

    accuracy: dict[str, float]
    latency_ms: float
    memory_gb: float
    energy_kwh: float
    co2_kg: float
    throughput_tps: float
    compression_ratio: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config: CompressionConfig | None = None

    def average_accuracy(self) -> float:
        """Calculate average accuracy across all benchmarks"""
        if not self.accuracy:
            return 0.0
        return sum(self.accuracy.values()) / len(self.accuracy)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for serialization"""
        return {
            "accuracy": self.accuracy,
            "average_accuracy": self.average_accuracy(),
            "latency_ms": self.latency_ms,
            "memory_gb": self.memory_gb,
            "energy_kwh": self.energy_kwh,
            "co2_kg": self.co2_kg,
            "throughput_tps": self.throughput_tps,
            "compression_ratio": self.compression_ratio,
            "timestamp": self.timestamp,
        }


@dataclass
class ParetoSolution:
    """
    A solution in the Pareto frontier optimization.

    Attributes:
        metrics: Evaluation metrics for this solution
        dominates: List of solution indices this solution dominates
        dominated_by: List of solution indices that dominate this solution
        is_pareto_optimal: Whether this solution is on the Pareto frontier
        crowding_distance: Crowding distance for diversity measurement
        rank: Pareto rank (0 = optimal, 1 = first dominated front, etc.)
    """

    metrics: EvaluationMetrics
    dominates: list[int] = field(default_factory=list)
    dominated_by: list[int] = field(default_factory=list)
    is_pareto_optimal: bool = False
    crowding_distance: float = 0.0
    rank: int = -1

    def reset_domination(self):
        """Reset domination relationships for recomputation"""
        self.dominates.clear()
        self.dominated_by.clear()
        self.is_pareto_optimal = False
        self.crowding_distance = 0.0
        self.rank = -1


@dataclass
class OptimizationResult:
    """
    Complete results from an optimization run.

    Attributes:
        pareto_solutions: List of solutions on the Pareto frontier
        all_solutions: All solutions explored during optimization
        best_accuracy: Solution with highest accuracy
        best_carbon: Solution with lowest carbon emissions
        best_balanced: Solution with best balance across objectives
        total_iterations: Number of optimization iterations
        total_carbon_used: Total carbon budget consumed (kg CO2)
        convergence_history: History of best solutions per iteration
    """

    pareto_solutions: list[ParetoSolution]
    all_solutions: list[ParetoSolution]
    best_accuracy: ParetoSolution | None = None
    best_carbon: ParetoSolution | None = None
    best_balanced: ParetoSolution | None = None
    total_iterations: int = 0
    total_carbon_used: float = 0.0
    convergence_history: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """Generate summary statistics"""
        return {
            "total_solutions": len(self.all_solutions),
            "pareto_optimal_count": len(self.pareto_solutions),
            "pareto_ratio": (
                len(self.pareto_solutions) / len(self.all_solutions) if self.all_solutions else 0
            ),
            "total_iterations": self.total_iterations,
            "total_carbon_used": self.total_carbon_used,
            "best_accuracy": (
                self.best_accuracy.metrics.average_accuracy() if self.best_accuracy else 0
            ),
            "best_carbon": self.best_carbon.metrics.co2_kg if self.best_carbon else 0,
        }


@dataclass
class ComparisonResult:
    """
    Results from comparing different optimization strategies.

    Attributes:
        strategy_name: Name of the optimization strategy
        average_accuracy: Average accuracy achieved
        average_carbon: Average carbon emissions
        convergence_speed: Iterations to reach threshold
        final_metrics: Final evaluation metrics
    """

    strategy_name: str
    average_accuracy: float
    average_carbon: float
    convergence_speed: int
    final_metrics: EvaluationMetrics
