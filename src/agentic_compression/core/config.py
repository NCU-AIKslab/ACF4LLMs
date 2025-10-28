"""
Configuration classes and constants for the compression framework.
"""

from dataclasses import dataclass
from enum import Enum


class QuantizationBits(Enum):
    """Supported quantization bit levels"""

    INT4 = 4
    INT8 = 8
    INT16 = 16
    INT32 = 32


class PruningPattern(Enum):
    """Pruning pattern types"""

    UNSTRUCTURED = "unstructured"
    STRUCTURED_2_4 = "2:4"
    STRUCTURED_4_8 = "4:8"


class BenchmarkType(Enum):
    """Supported benchmark types"""

    GSM8K = "gsm8k"
    TRUTHFULQA = "truthfulqa"
    COMMONSENSEQA = "commonsenseqa"
    HUMANEVAL = "humaneval"
    BIGBENCH = "bigbench"


@dataclass
class CompressionConfig:
    """
    Configuration for model compression strategies.

    Attributes:
        quantization_bits: Target precision in bits (4, 8, 16, or 32)
        pruning_sparsity: Sparsity level for pruning (0.0 to 0.7)
        context_length: Context window size in tokens (1k to 32k)
        distillation_layers: Number of layers in student model (optional)
        dynamic_adjustment: Enable dynamic compression adjustment
        carbon_aware: Enable carbon-aware scheduling
        model_path: Path or identifier of the model to compress
        carbon_budget: Maximum carbon budget in kg CO2
        accuracy_threshold: Minimum acceptable accuracy
    """

    quantization_bits: int = 8
    pruning_sparsity: float = 0.0
    context_length: int = 4096
    distillation_layers: int | None = None
    dynamic_adjustment: bool = True
    carbon_aware: bool = True
    model_path: str = "google/gemma-12b"
    carbon_budget: float = 10.0
    accuracy_threshold: float = 0.95

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.quantization_bits not in [4, 8, 16, 32]:
            raise ValueError(
                f"quantization_bits must be 4, 8, 16, or 32, got {self.quantization_bits}"
            )

        if not 0.0 <= self.pruning_sparsity <= 0.7:
            raise ValueError(
                f"pruning_sparsity must be between 0.0 and 0.7, got {self.pruning_sparsity}"
            )

        if self.context_length < 1024 or self.context_length > 32768:
            raise ValueError(
                f"context_length must be between 1024 and 32768, got {self.context_length}"
            )

        if self.carbon_budget <= 0:
            raise ValueError(f"carbon_budget must be positive, got {self.carbon_budget}")

        if not 0.0 <= self.accuracy_threshold <= 1.0:
            raise ValueError(
                f"accuracy_threshold must be between 0.0 and 1.0, got {self.accuracy_threshold}"
            )


@dataclass
class EnvironmentConstraints:
    """
    Resource constraints for different deployment environments.

    Attributes:
        max_memory_gb: Maximum available memory in GB
        max_power_watts: Maximum power consumption in watts
        carbon_intensity: Carbon intensity of power grid (kgCO2/kWh)
        latency_requirement_ms: Maximum acceptable latency in milliseconds
        name: Name of the environment
    """

    max_memory_gb: float
    max_power_watts: float
    carbon_intensity: float
    latency_requirement_ms: float
    name: str = "custom"


# Predefined environment configurations
EDGE_DEVICE = EnvironmentConstraints(
    name="edge_device",
    max_memory_gb=4,
    max_power_watts=10,
    carbon_intensity=0.2,
    latency_requirement_ms=50,
)

MOBILE_DEVICE = EnvironmentConstraints(
    name="mobile_device",
    max_memory_gb=8,
    max_power_watts=5,
    carbon_intensity=0.3,
    latency_requirement_ms=100,
)

CLOUD_SERVER = EnvironmentConstraints(
    name="cloud_server",
    max_memory_gb=128,
    max_power_watts=400,
    carbon_intensity=0.5,
    latency_requirement_ms=20,
)

CARBON_INTENSIVE_DC = EnvironmentConstraints(
    name="carbon_intensive_dc",
    max_memory_gb=256,
    max_power_watts=800,
    carbon_intensity=0.8,
    latency_requirement_ms=30,
)


# Benchmark-specific configuration
BENCHMARK_CONFIGS = {
    BenchmarkType.GSM8K.value: {
        "base_accuracy": 0.95,
        "sensitivity_factor": 1.2,
        "description": "Mathematical reasoning",
    },
    BenchmarkType.TRUTHFULQA.value: {
        "base_accuracy": 0.85,
        "sensitivity_factor": 1.0,
        "description": "Truthfulness and factual consistency",
    },
    BenchmarkType.COMMONSENSEQA.value: {
        "base_accuracy": 0.90,
        "sensitivity_factor": 1.1,
        "description": "Commonsense reasoning",
    },
    BenchmarkType.HUMANEVAL.value: {
        "base_accuracy": 0.80,
        "sensitivity_factor": 1.5,
        "description": "Code generation",
    },
    BenchmarkType.BIGBENCH.value: {
        "base_accuracy": 0.88,
        "sensitivity_factor": 1.15,
        "description": "Multi-domain challenging tasks",
    },
}


# Default baseline metrics for a 12B parameter model
BASELINE_METRICS = {
    "latency_ms": 100,
    "memory_gb": 24,
    "energy_kwh": 0.084,
    "co2_kg": 0.034,
    "throughput_tps": 1000,
}
