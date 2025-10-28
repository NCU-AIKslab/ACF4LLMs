"""
Tools for model evaluation and benchmarking.
"""

import asyncio
import json
import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..core.config import BENCHMARK_CONFIGS, CompressionConfig
from ..core.metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class EvaluationInput(BaseModel):
    """Input schema for evaluation tool"""

    benchmark: str = Field(
        description="Benchmark to run (gsm8k, truthfulqa, commonsenseqa, humaneval, bigbench)"
    )
    config_json: str = Field(description="JSON string of compression configuration")


class EvaluationTool(BaseTool):
    """Tool for evaluating compressed models on benchmarks."""

    name: str = "evaluate_model"
    description: str = """Evaluate compressed model on specified benchmark.

    Args:
        benchmark: Benchmark name (gsm8k, truthfulqa, commonsenseqa, humaneval, bigbench)
        config_json: JSON string with compression configuration

    Returns:
        JSON with accuracy, latency_ms, memory_gb, energy_kwh, co2_kg
    """
    args_schema: type[BaseModel] = EvaluationInput

    def _run(self, benchmark: str, config_json: str) -> str:
        """Execute evaluation (simulated)"""
        logger.info(f"Evaluating on {benchmark}")

        # Parse config
        try:
            config_dict = json.loads(config_json)
        except (json.JSONDecodeError, TypeError, ValueError):
            config_dict = {"quantization_bits": 8, "pruning_sparsity": 0.0}

        # Get benchmark configuration
        bench_config = BENCHMARK_CONFIGS.get(
            benchmark, {"base_accuracy": 0.80, "sensitivity_factor": 1.0}
        )

        # Calculate accuracy impact from compression
        bits = config_dict.get("quantization_bits", 8)
        sparsity = config_dict.get("pruning_sparsity", 0.0)

        quant_impact = (32 - bits) / 32 * 0.15
        prune_impact = sparsity * 0.20
        total_impact = (quant_impact + prune_impact) * bench_config["sensitivity_factor"]

        # Calculate metrics
        accuracy = max(bench_config["base_accuracy"] - total_impact, 0.3)
        compression_factor = (32 / bits) * (1 / (1 - sparsity + 0.01))

        result = {
            "benchmark": benchmark,
            "accuracy": f"{accuracy:.3f}",
            "latency_ms": int(100 / compression_factor),
            "memory_gb": round(24 / compression_factor, 2),
            "energy_kwh": round(0.084 / compression_factor, 4),
            "co2_kg": round(0.034 / compression_factor, 4),
            "throughput_tps": int(1000 * compression_factor),
        }

        return json.dumps(result, indent=2)

    async def _arun(self, benchmark: str, config_json: str) -> str:
        """Async version"""
        await asyncio.sleep(0.1)  # Simulate evaluation time
        return self._run(benchmark, config_json)


async def evaluate_config_full(config: CompressionConfig) -> EvaluationMetrics:
    """
    Evaluate a configuration across all benchmarks.

    Args:
        config: Compression configuration to evaluate

    Returns:
        Complete evaluation metrics
    """
    await asyncio.sleep(0.1)  # Simulate evaluation time

    # Calculate compression factor
    compression_factor = (32 / config.quantization_bits) * (
        1 / (1 - config.pruning_sparsity + 0.01)
    )

    # Calculate accuracy for each benchmark
    accuracy = {}
    for benchmark, bench_config in BENCHMARK_CONFIGS.items():
        base_acc = bench_config["base_accuracy"]
        sensitivity = bench_config["sensitivity_factor"]

        quant_impact = (32 - config.quantization_bits) / 32 * 0.15
        prune_impact = config.pruning_sparsity * 0.20

        total_impact = (quant_impact + prune_impact) * sensitivity
        accuracy[benchmark] = max(base_acc - total_impact, 0.3)

    # Calculate resource metrics
    metrics = EvaluationMetrics(
        accuracy=accuracy,
        latency_ms=100 / compression_factor,
        memory_gb=24 / compression_factor,
        energy_kwh=0.084 / compression_factor,
        co2_kg=0.034 / compression_factor,
        throughput_tps=1000 * compression_factor,
        compression_ratio=compression_factor,
        config=config,
    )

    return metrics
