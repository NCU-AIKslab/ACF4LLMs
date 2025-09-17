"""
Metrics collection system for performance and resource monitoring.

Provides comprehensive tracking of latency, throughput, memory usage,
energy consumption, and CO2 emissions during model operations.
"""

import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable, ContextManager
from contextlib import contextmanager
import json
from pathlib import Path
import threading
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

import torch
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latency_ms: List[float] = field(default_factory=list)
    throughput_tokens_per_sec: float = 0.0
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    energy_consumption_kwh: float = 0.0
    co2_emissions_g: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not self.latency_ms:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}

        return {
            "p50": np.percentile(self.latency_ms, 50),
            "p90": np.percentile(self.latency_ms, 90),
            "p95": np.percentile(self.latency_ms, 95),
            "p99": np.percentile(self.latency_ms, 99)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        percentiles = self.get_latency_percentiles()
        return {
            "latency_ms_mean": statistics.mean(self.latency_ms) if self.latency_ms else 0,
            "latency_ms_p50": percentiles["p50"],
            "latency_ms_p90": percentiles["p90"],
            "latency_ms_p95": percentiles["p95"],
            "latency_ms_p99": percentiles["p99"],
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "energy_consumption_kwh": self.energy_consumption_kwh,
            "co2_emissions_g": self.co2_emissions_g,
            "timestamp": self.timestamp,
            "num_samples": len(self.latency_ms)
        }


class MetricsCollector:
    """
    Comprehensive metrics collection system.

    Tracks performance, resource usage, and environmental impact
    during model training and inference operations.
    """

    def __init__(self,
                 sample_interval: float = 0.1,
                 enable_carbon_tracking: bool = True,
                 enable_gpu_monitoring: bool = True):
        """
        Initialize metrics collector.

        Args:
            sample_interval: Interval in seconds for resource sampling
            enable_carbon_tracking: Whether to track energy and carbon
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.sample_interval = sample_interval
        self.enable_carbon_tracking = enable_carbon_tracking
        self.enable_gpu_monitoring = enable_gpu_monitoring

        self._monitoring = False
        self._monitor_thread = None
        self._metrics_history = []
        self._current_session = None

        # Initialize carbon tracker if available
        self._carbon_tracker = None
        if enable_carbon_tracking:
            self._init_carbon_tracker()

        # Initialize GPU monitoring
        self._gpu_available = torch.cuda.is_available() if enable_gpu_monitoring else False
        if self._gpu_available:
            logger.info(f"GPU monitoring enabled. Detected {torch.cuda.device_count()} GPU(s)")

    def _init_carbon_tracker(self):
        """Initialize CodeCarbon tracker for energy monitoring."""
        try:
            from codecarbon import EmissionsTracker, OfflineEmissionsTracker

            # Try online tracker first, fallback to offline
            try:
                self._carbon_tracker = EmissionsTracker(
                    project_name="gsm8k_optimization",
                    output_dir="./carbon_logs",
                    save_to_file=True,
                    log_level="ERROR",  # Reduce noise
                    tracking_mode="process"
                )
                logger.info("Initialized online carbon tracker")
            except:
                self._carbon_tracker = OfflineEmissionsTracker(
                    country_iso_code="USA",  # Default fallback
                    project_name="gsm8k_optimization",
                    output_dir="./carbon_logs",
                    save_to_file=True,
                    log_level="ERROR"
                )
                logger.info("Initialized offline carbon tracker")

        except ImportError:
            logger.warning("CodeCarbon not available. Energy tracking disabled.")
            self._carbon_tracker = None

    @contextmanager
    def measure_operation(self, operation_name: str = "operation") -> ContextManager[PerformanceMetrics]:
        """
        Context manager for measuring a specific operation.

        Args:
            operation_name: Name of the operation being measured

        Yields:
            PerformanceMetrics object that gets populated during the operation
        """
        metrics = PerformanceMetrics()

        # Start monitoring
        self.start_monitoring()

        # Start carbon tracking if available
        if self._carbon_tracker:
            self._carbon_tracker.start()

        # Record start state
        start_time = time.time()
        if self._gpu_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        start_memory = self._get_memory_usage()

        try:
            yield metrics
        finally:
            # Record end state
            end_time = time.time()
            operation_duration = end_time - start_time

            # Stop carbon tracking
            carbon_emissions = 0.0
            energy_consumption = 0.0
            if self._carbon_tracker:
                try:
                    emissions_data = self._carbon_tracker.stop()
                    if emissions_data:
                        carbon_emissions = emissions_data * 1000  # Convert to grams
                        energy_consumption = emissions_data * 0.5  # Rough estimate
                except:
                    logger.warning("Failed to stop carbon tracker")

            # Stop monitoring and collect metrics
            final_metrics = self.stop_monitoring()

            # Update the metrics object
            if final_metrics:
                metrics.latency_ms = final_metrics.latency_ms
                metrics.throughput_tokens_per_sec = final_metrics.throughput_tokens_per_sec
                metrics.memory_peak_mb = final_metrics.memory_peak_mb
                metrics.memory_allocated_mb = final_metrics.memory_allocated_mb
                metrics.cpu_usage_percent = final_metrics.cpu_usage_percent
                metrics.gpu_utilization_percent = final_metrics.gpu_utilization_percent

            metrics.energy_consumption_kwh = energy_consumption
            metrics.co2_emissions_g = carbon_emissions

            logger.info(f"Operation '{operation_name}' completed in {operation_duration:.2f}s")
            logger.info(f"Energy: {energy_consumption:.6f} kWh, CO2: {carbon_emissions:.3f}g")

    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._current_session = PerformanceMetrics()
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()

        logger.debug("Started resource monitoring")

    def stop_monitoring(self) -> Optional[PerformanceMetrics]:
        """Stop monitoring and return collected metrics."""
        if not self._monitoring:
            return None

        self._monitoring = False

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

        metrics = self._current_session
        if metrics:
            self._metrics_history.append(metrics)

        self._current_session = None
        logger.debug("Stopped resource monitoring")

        return metrics

    def _monitor_resources(self):
        """Background thread for continuous resource monitoring."""
        while self._monitoring:
            try:
                self._collect_current_metrics()
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")

    def _collect_current_metrics(self):
        """Collect current resource usage metrics."""
        if not self._current_session:
            return

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        self._current_session.cpu_usage_percent = max(
            self._current_session.cpu_usage_percent, cpu_percent
        )

        # Memory usage
        memory_mb = self._get_memory_usage()
        self._current_session.memory_peak_mb = max(
            self._current_session.memory_peak_mb, memory_mb
        )
        self._current_session.memory_allocated_mb = memory_mb

        # GPU metrics
        if self._gpu_available:
            gpu_util = self._get_gpu_utilization()
            self._current_session.gpu_utilization_percent = max(
                self._current_session.gpu_utilization_percent, gpu_util
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self._gpu_available:
            # GPU memory
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # System memory for this process
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            # Fallback: estimate from memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated()
                memory_total = torch.cuda.get_device_properties(0).total_memory
                return (memory_used / memory_total) * 100
            return 0.0

    def record_latency(self, latency_ms: float):
        """Record a single latency measurement."""
        if self._current_session:
            self._current_session.latency_ms.append(latency_ms)

    def record_throughput(self, tokens_per_second: float):
        """Record throughput measurement."""
        if self._current_session:
            self._current_session.throughput_tokens_per_sec = tokens_per_second

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the current monitoring session."""
        if not self._current_session:
            return {}

        return self._current_session.to_dict()

    def get_historical_summary(self) -> Dict[str, Any]:
        """Get summary statistics from all monitoring sessions."""
        if not self._metrics_history:
            return {}

        # Aggregate metrics across all sessions
        all_latencies = []
        all_throughputs = []
        all_memory_peaks = []
        all_energy = []
        all_co2 = []

        for metrics in self._metrics_history:
            all_latencies.extend(metrics.latency_ms)
            if metrics.throughput_tokens_per_sec > 0:
                all_throughputs.append(metrics.throughput_tokens_per_sec)
            if metrics.memory_peak_mb > 0:
                all_memory_peaks.append(metrics.memory_peak_mb)
            if metrics.energy_consumption_kwh > 0:
                all_energy.append(metrics.energy_consumption_kwh)
            if metrics.co2_emissions_g > 0:
                all_co2.append(metrics.co2_emissions_g)

        summary = {
            "total_sessions": len(self._metrics_history),
            "total_samples": len(all_latencies)
        }

        if all_latencies:
            summary.update({
                "latency_ms_mean": statistics.mean(all_latencies),
                "latency_ms_p50": np.percentile(all_latencies, 50),
                "latency_ms_p90": np.percentile(all_latencies, 90),
                "latency_ms_p99": np.percentile(all_latencies, 99)
            })

        if all_throughputs:
            summary["throughput_avg_tokens_per_sec"] = statistics.mean(all_throughputs)

        if all_memory_peaks:
            summary["memory_peak_max_mb"] = max(all_memory_peaks)

        if all_energy:
            summary["total_energy_kwh"] = sum(all_energy)

        if all_co2:
            summary["total_co2_g"] = sum(all_co2)

        return summary

    def save_metrics(self, output_path: Path):
        """Save collected metrics to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_data = {
            "current_session": self.get_session_summary(),
            "historical_summary": self.get_historical_summary(),
            "session_history": [m.to_dict() for m in self._metrics_history],
            "collection_config": {
                "sample_interval": self.sample_interval,
                "carbon_tracking_enabled": self.enable_carbon_tracking,
                "gpu_monitoring_enabled": self.enable_gpu_monitoring
            }
        }

        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        logger.info(f"Metrics saved to {output_path}")

    def reset_history(self):
        """Clear all historical metrics."""
        self._metrics_history.clear()
        logger.info("Metrics history cleared")


class CarbonTracker:
    """
    Specialized carbon footprint tracker using CodeCarbon.

    Provides context manager interface for easy CO2 emission tracking.
    """

    def __init__(self, project_name: str = "gsm8k_optimization"):
        """
        Initialize carbon tracker.

        Args:
            project_name: Name of the project for tracking
        """
        self.project_name = project_name
        self._tracker = None
        self._init_tracker()

    def _init_tracker(self):
        """Initialize the CodeCarbon tracker."""
        try:
            from codecarbon import EmissionsTracker

            self._tracker = EmissionsTracker(
                project_name=self.project_name,
                output_dir="./carbon_logs",
                save_to_file=True,
                log_level="WARNING",
                tracking_mode="process",
                on_csv_write="append"
            )

        except ImportError:
            logger.warning("CodeCarbon not available. Install with: pip install codecarbon")
            self._tracker = None

    @contextmanager
    def track_emissions(self):
        """Context manager for tracking carbon emissions."""
        if not self._tracker:
            yield {"energy_kwh": 0.0, "co2_g": 0.0}
            return

        try:
            self._tracker.start()
            yield
        finally:
            try:
                emissions_kg = self._tracker.stop()
                result = {
                    "energy_kwh": emissions_kg * 2.0 if emissions_kg else 0.0,  # Rough estimate
                    "co2_g": emissions_kg * 1000 if emissions_kg else 0.0
                }
                logger.info(f"Emissions tracked: {result['co2_g']:.3f}g CO2")
            except Exception as e:
                logger.warning(f"Failed to stop carbon tracker: {e}")
                result = {"energy_kwh": 0.0, "co2_g": 0.0}


# Global metrics collector instance
_global_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def measure_inference_batch(model_func: Callable,
                          inputs: Any,
                          batch_size: int = 1) -> Dict[str, float]:
    """
    Measure performance metrics for a batch inference operation.

    Args:
        model_func: Function that performs model inference
        inputs: Input data for the model
        batch_size: Batch size for throughput calculation

    Returns:
        Dictionary with performance metrics
    """
    collector = get_metrics_collector()

    with collector.measure_operation("batch_inference") as metrics:
        start_time = time.time()

        # Run inference
        outputs = model_func(inputs)

        end_time = time.time()

        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        collector.record_latency(latency_ms)

        # Estimate throughput (rough)
        if hasattr(outputs, 'shape') and len(outputs.shape) > 1:
            output_tokens = outputs.shape[-1] * batch_size
            throughput = output_tokens / (end_time - start_time)
            collector.record_throughput(throughput)

    return metrics.to_dict()