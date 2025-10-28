"""
Tests for EvaluationMetrics and ParetoSolution.
"""

import pytest

from agentic_compression.core.config import CompressionConfig
from agentic_compression.core.metrics import EvaluationMetrics, ParetoSolution


class TestEvaluationMetrics:
    """Test EvaluationMetrics class"""

    def test_create_metrics(self):
        """Test creating evaluation metrics"""
        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.95, "truthfulqa": 0.92},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
            throughput_tps=1500,
            compression_ratio=2.5,
        )

        assert metrics.accuracy["gsm8k"] == 0.95
        assert metrics.accuracy["truthfulqa"] == 0.92
        assert metrics.latency_ms == 50.0
        assert metrics.memory_gb == 8.0
        assert metrics.energy_kwh == 0.05
        assert metrics.co2_kg == 0.02
        assert metrics.throughput_tps == 1500
        assert metrics.compression_ratio == 2.5

    def test_average_accuracy(self):
        """Test average accuracy calculation"""
        metrics = EvaluationMetrics(
            accuracy={"benchmark1": 0.9, "benchmark2": 0.8, "benchmark3": 0.7},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        avg_accuracy = metrics.average_accuracy()
        expected = (0.9 + 0.8 + 0.7) / 3
        assert abs(avg_accuracy - expected) < 1e-6

    def test_average_accuracy_empty(self):
        """Test average accuracy with empty dict"""
        metrics = EvaluationMetrics(
            accuracy={}, latency_ms=50.0, memory_gb=8.0, energy_kwh=0.05, co2_kg=0.02
        )

        assert metrics.average_accuracy() == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = CompressionConfig(quantization_bits=8, pruning_sparsity=0.3)

        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
            config=config,
        )

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert "accuracy" in metrics_dict
        assert metrics_dict["latency_ms"] == 50.0
        assert metrics_dict["memory_gb"] == 8.0
        assert "config" in metrics_dict

    def test_metrics_with_config(self):
        """Test metrics with associated config"""
        config = CompressionConfig(quantization_bits=4, pruning_sparsity=0.5)

        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.90},
            latency_ms=30.0,
            memory_gb=4.0,
            energy_kwh=0.03,
            co2_kg=0.012,
            config=config,
        )

        assert metrics.config is not None
        assert metrics.config.quantization_bits == 4
        assert metrics.config.pruning_sparsity == 0.5


class TestParetoSolution:
    """Test ParetoSolution class"""

    def test_create_pareto_solution(self):
        """Test creating a Pareto solution"""
        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        solution = ParetoSolution(metrics=metrics)

        assert solution.metrics == metrics
        assert not solution.is_pareto_optimal  # Default is False
        assert solution.crowding_distance == 0.0  # Default

    def test_pareto_optimal_flag(self):
        """Test Pareto optimal flag"""
        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        solution = ParetoSolution(metrics=metrics, is_pareto_optimal=True)

        assert solution.is_pareto_optimal

    def test_crowding_distance(self):
        """Test crowding distance attribute"""
        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        solution = ParetoSolution(metrics=metrics, crowding_distance=5.0)

        assert solution.crowding_distance == 5.0

    def test_solution_comparison(self):
        """Test comparison between solutions"""
        metrics1 = EvaluationMetrics(
            accuracy={"gsm8k": 0.95},
            latency_ms=50.0,
            memory_gb=8.0,
            energy_kwh=0.05,
            co2_kg=0.02,
        )

        metrics2 = EvaluationMetrics(
            accuracy={"gsm8k": 0.90},
            latency_ms=30.0,
            memory_gb=4.0,
            energy_kwh=0.03,
            co2_kg=0.012,
        )

        solution1 = ParetoSolution(metrics=metrics1)
        solution2 = ParetoSolution(metrics=metrics2)

        # Solution1 has better accuracy
        assert solution1.metrics.average_accuracy() > solution2.metrics.average_accuracy()

        # Solution2 has better carbon
        assert solution2.metrics.co2_kg < solution1.metrics.co2_kg


class TestMetricsEdgeCases:
    """Test edge cases for metrics"""

    def test_zero_values(self):
        """Test metrics with zero values"""
        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 0.0},
            latency_ms=0.0,
            memory_gb=0.0,
            energy_kwh=0.0,
            co2_kg=0.0,
        )

        assert metrics.average_accuracy() == 0.0
        assert metrics.latency_ms == 0.0

    def test_very_high_values(self):
        """Test metrics with very high values"""
        metrics = EvaluationMetrics(
            accuracy={"gsm8k": 1.0},  # Perfect accuracy
            latency_ms=1000.0,  # 1 second
            memory_gb=128.0,  # 128 GB
            energy_kwh=1.0,  # 1 kWh
            co2_kg=0.5,  # 500g CO2
        )

        assert metrics.average_accuracy() == 1.0
        assert metrics.memory_gb == 128.0

    def test_multiple_benchmarks(self):
        """Test metrics with many benchmarks"""
        accuracy = {
            "gsm8k": 0.95,
            "truthfulqa": 0.92,
            "commonsenseqa": 0.88,
            "humaneval": 0.85,
            "bigbench": 0.90,
        }

        metrics = EvaluationMetrics(
            accuracy=accuracy, latency_ms=50.0, memory_gb=8.0, energy_kwh=0.05, co2_kg=0.02
        )

        assert len(metrics.accuracy) == 5
        avg = sum(accuracy.values()) / len(accuracy)
        assert abs(metrics.average_accuracy() - avg) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
