"""
Tests for CompressionConfig and EnvironmentConstraints.
"""

import dataclasses

import pytest

from agentic_compression.core.config import (
    CARBON_INTENSIVE_DC,
    CLOUD_SERVER,
    EDGE_DEVICE,
    MOBILE_DEVICE,
    CompressionConfig,
    EnvironmentConstraints,
)


class TestCompressionConfig:
    """Tests covering validation and defaults for CompressionConfig."""

    def test_default_config(self):
        config = CompressionConfig()

        assert config.quantization_bits == 8
        assert config.pruning_sparsity == 0.0
        assert config.context_length == 4096
        assert config.distillation_layers is None
        assert config.dynamic_adjustment is True
        assert config.carbon_aware is True
        assert config.model_path == "google/gemma-12b"
        assert config.carbon_budget == 10.0
        assert config.accuracy_threshold == 0.95

    def test_custom_config(self):
        config = CompressionConfig(
            quantization_bits=4,
            pruning_sparsity=0.5,
            context_length=16384,
            distillation_layers=12,
            dynamic_adjustment=False,
            carbon_aware=False,
            model_path="google/gemma3-270m",
            carbon_budget=2.5,
            accuracy_threshold=0.8,
        )

        assert config.quantization_bits == 4
        assert config.pruning_sparsity == 0.5
        assert config.context_length == 16384
        assert config.distillation_layers == 12
        assert config.dynamic_adjustment is False
        assert config.carbon_aware is False
        assert config.model_path == "google/gemma3-270m"
        assert config.carbon_budget == 2.5
        assert config.accuracy_threshold == 0.8

    def test_invalid_quantization_bits(self):
        with pytest.raises(ValueError):
            CompressionConfig(quantization_bits=6)

    def test_invalid_pruning_range(self):
        with pytest.raises(ValueError):
            CompressionConfig(pruning_sparsity=0.8)

        with pytest.raises(ValueError):
            CompressionConfig(pruning_sparsity=-0.1)

    def test_context_length_bounds(self):
        with pytest.raises(ValueError):
            CompressionConfig(context_length=512)

        with pytest.raises(ValueError):
            CompressionConfig(context_length=65536)

    def test_accuracy_threshold_validation(self):
        with pytest.raises(ValueError):
            CompressionConfig(accuracy_threshold=1.5)

        with pytest.raises(ValueError):
            CompressionConfig(accuracy_threshold=-0.1)

    def test_carbon_budget_must_be_positive(self):
        with pytest.raises(ValueError):
            CompressionConfig(carbon_budget=0)

    def test_dataclass_serialization(self):
        config = CompressionConfig(quantization_bits=16, pruning_sparsity=0.2)
        data = dataclasses.asdict(config)

        assert data["quantization_bits"] == 16
        assert data["pruning_sparsity"] == 0.2
        assert "context_length" in data


class TestEnvironmentConstraints:
    """Tests for the built-in environment profiles."""

    def test_edge_device_profile(self):
        assert EDGE_DEVICE.name == "edge_device"
        assert EDGE_DEVICE.max_memory_gb == 4
        assert EDGE_DEVICE.max_power_watts == 10
        assert EDGE_DEVICE.latency_requirement_ms == 50
        assert EDGE_DEVICE.carbon_intensity == 0.2

    def test_mobile_device_profile(self):
        assert MOBILE_DEVICE.name == "mobile_device"
        assert MOBILE_DEVICE.max_memory_gb == 8
        assert MOBILE_DEVICE.max_power_watts == 5
        assert MOBILE_DEVICE.latency_requirement_ms == 100
        assert MOBILE_DEVICE.carbon_intensity == 0.3

    def test_cloud_server_profile(self):
        assert CLOUD_SERVER.name == "cloud_server"
        assert CLOUD_SERVER.max_memory_gb == 128
        assert CLOUD_SERVER.max_power_watts == 400
        assert CLOUD_SERVER.latency_requirement_ms == 20
        assert CLOUD_SERVER.carbon_intensity == 0.5

    def test_carbon_intensive_dc_profile(self):
        assert CARBON_INTENSIVE_DC.name == "carbon_intensive_dc"
        assert CARBON_INTENSIVE_DC.max_memory_gb == 256
        assert CARBON_INTENSIVE_DC.max_power_watts == 800
        assert CARBON_INTENSIVE_DC.latency_requirement_ms == 30
        assert CARBON_INTENSIVE_DC.carbon_intensity == 0.8

    def test_custom_environment_constraints(self):
        custom = EnvironmentConstraints(
            name="custom",
            max_memory_gb=12,
            max_power_watts=20,
            carbon_intensity=0.4,
            latency_requirement_ms=75,
        )

        assert custom.name == "custom"
        assert custom.max_memory_gb == 12
        assert custom.max_power_watts == 20
        assert custom.carbon_intensity == 0.4
        assert custom.latency_requirement_ms == 75
