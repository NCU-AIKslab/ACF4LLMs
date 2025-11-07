"""
Benchmark runner for comprehensive model evaluation.

Coordinates evaluation across multiple benchmarks and collects performance metrics.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core.config import BENCHMARK_CONFIGS, CompressionConfig
from ..core.metrics import EvaluationMetrics
from .lm_harness_adapter import LMHarnessAdapter

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not available, GPU monitoring disabled")

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks on compressed models.

    Features:
    - Multi-benchmark evaluation
    - Performance metrics collection
    - GPU monitoring
    - Carbon emission estimation
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_fewshot: int = 5,
        limit: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        Initialize benchmark runner.

        Args:
            batch_size: Batch size for evaluation
            num_fewshot: Number of few-shot examples
            limit: Limit samples for quick testing (None = full eval)
            device: Device to run on
        """
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.device = device
        self.adapter = LMHarnessAdapter(batch_size=batch_size, device=device)

        # Initialize GPU monitoring
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_monitoring = True
                logger.info("GPU monitoring enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_monitoring = False
        else:
            self.gpu_monitoring = False

        logger.info(
            f"Initialized BenchmarkRunner: batch_size={batch_size}, "
            f"num_fewshot={num_fewshot}, limit={limit}"
        )

    async def run_all_benchmarks(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[CompressionConfig] = None,
    ) -> EvaluationMetrics:
        """
        Run all five benchmarks on a model.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            config: Compression configuration (for metadata)

        Returns:
            EvaluationMetrics with results from all benchmarks

        Example:
            >>> runner = BenchmarkRunner()
            >>> metrics = await runner.run_all_benchmarks(model, tokenizer)
        """
        logger.info("Starting evaluation on all benchmarks")
        start_time = time.time()

        # Get list of benchmarks
        benchmarks = list(BENCHMARK_CONFIGS.keys())
        logger.info(f"Running {len(benchmarks)} benchmarks: {benchmarks}")

        # Start GPU monitoring
        gpu_start = self._get_gpu_metrics() if self.gpu_monitoring else None

        # Run evaluation
        try:
            results = self.adapter.evaluate_model(
                model=model,
                tokenizer=tokenizer,
                tasks=benchmarks,
                num_fewshot=self.num_fewshot,
                limit=self.limit,
            )

            # Extract accuracy per benchmark
            accuracy = {
                task: results[task]["accuracy"]
                for task in benchmarks
                if task in results
            }

            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Evaluation completed in {elapsed_time:.1f}s")

            # Get GPU metrics
            gpu_end = self._get_gpu_metrics() if self.gpu_monitoring else None

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                elapsed_time=elapsed_time,
                gpu_start=gpu_start,
                gpu_end=gpu_end,
                model=model,
            )

            # Create EvaluationMetrics
            metrics = EvaluationMetrics(
                accuracy=accuracy,
                latency_ms=performance_metrics["latency_ms"],
                memory_gb=performance_metrics["memory_gb"],
                energy_kwh=performance_metrics["energy_kwh"],
                co2_kg=performance_metrics["co2_kg"],
                throughput_tps=performance_metrics["throughput_tps"],
                compression_ratio=performance_metrics.get("compression_ratio", 1.0),
                config=config,
            )

            logger.info(f"Average accuracy: {metrics.average_accuracy():.3f}")
            logger.info(f"Memory usage: {metrics.memory_gb:.2f} GB")
            logger.info(f"CO2 emissions: {metrics.co2_kg:.4f} kg")

            return metrics

        except Exception as e:
            logger.error(f"Benchmark evaluation failed: {str(e)}")
            raise

    async def run_single_benchmark(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        benchmark: str,
    ) -> Dict:
        """
        Run a single benchmark.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            benchmark: Benchmark name

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running single benchmark: {benchmark}")

        try:
            result = self.adapter.evaluate_single_task(
                model=model,
                tokenizer=tokenizer,
                task=benchmark,
                num_fewshot=self.num_fewshot,
                limit=self.limit,
            )
            return result

        except Exception as e:
            logger.error(f"Failed to run benchmark {benchmark}: {str(e)}")
            raise

    def _get_gpu_metrics(self) -> Optional[Dict]:
        """Get current GPU metrics."""
        if not self.gpu_monitoring:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W

            return {
                "memory_used_gb": info.used / 1024**3,
                "memory_total_gb": info.total / 1024**3,
                "gpu_utilization_percent": utilization.gpu,
                "power_watts": power,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return None

    def _calculate_performance_metrics(
        self,
        elapsed_time: float,
        gpu_start: Optional[Dict],
        gpu_end: Optional[Dict],
        model: AutoModelForCausalLM,
    ) -> Dict:
        """
        Calculate performance metrics from evaluation.

        Args:
            elapsed_time: Total evaluation time in seconds
            gpu_start: GPU metrics at start
            gpu_end: GPU metrics at end
            model: The evaluated model

        Returns:
            Dictionary with performance metrics
        """
        # Latency (average per sample, assuming ~1000 samples total)
        avg_samples = 1000 if self.limit is None else min(self.limit, 1000)
        latency_ms = (elapsed_time / avg_samples) * 1000

        # Memory usage
        if gpu_end:
            memory_gb = gpu_end["memory_used_gb"]
        elif torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
        else:
            # Estimate based on model size
            param_count = sum(p.numel() for p in model.parameters())
            memory_gb = param_count * 4 / 1024**3  # Assuming float32

        # Energy and carbon
        if gpu_start and gpu_end:
            avg_power_watts = (gpu_start["power_watts"] + gpu_end["power_watts"]) / 2
            energy_kwh = (avg_power_watts * elapsed_time / 3600) / 1000
        else:
            # Estimate: RTX 4090 TDP ~450W, assume 60% utilization
            avg_power_watts = 450 * 0.6
            energy_kwh = (avg_power_watts * elapsed_time / 3600) / 1000

        # Carbon (using average grid intensity)
        # US average: ~0.4 kg CO2/kWh
        carbon_intensity = 0.4
        co2_kg = energy_kwh * carbon_intensity

        # Throughput (tokens per second, rough estimate)
        # Assume average of 50 tokens per sample
        total_tokens = avg_samples * 50
        throughput_tps = total_tokens / elapsed_time

        # Compression ratio (if model has quantization config)
        compression_ratio = 1.0
        if hasattr(model, "quantization_config"):
            config = model.quantization_config
            if getattr(config, "load_in_4bit", False):
                compression_ratio = 32 / 4
            elif getattr(config, "load_in_8bit", False):
                compression_ratio = 32 / 8

        return {
            "latency_ms": latency_ms,
            "memory_gb": memory_gb,
            "energy_kwh": energy_kwh,
            "co2_kg": co2_kg,
            "throughput_tps": throughput_tps,
            "compression_ratio": compression_ratio,
            "elapsed_time_s": elapsed_time,
        }

    def __del__(self):
        """Cleanup GPU monitoring."""
        if self.gpu_monitoring:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
