"""
Tests for CompressionConfig and EnvironmentConstraints.
"""

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
    """Test CompressionConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CompressionConfig()

        assert config.quantization_bits == 8
        assert config.pruning_sparsity == 0.0
        assert config.context_length == 4096
        assert config.num_layers == 24
        assert config.accuracy_threshold == 0.90
        assert config.carbon_budget == 10.0

    def test_custom_config(self):
        """Test custom configuration values"""
        config = CompressionConfig(
            quantization_bits=4,
            pruning_sparsity=0.5,
            context_length=2048,
            num_layers=12,
            model_path="google/gemma-12b",
            accuracy_threshold=0.95,
            carbon_budget=5.0,
        )

        assert config.quantization_bits == 4
        assert config.pruning_sparsity == 0.5
        assert config.context_length == 2048
        assert config.num_layers == 12
        assert config.model_path == "google/gemma-12b"
        assert config.accuracy_threshold == 0.95
        assert config.carbon_budget == 5.0

    def test_quantization_bits_validation(self):
        """Test valid quantization bit values"""
        valid_bits = [4, 8, 16, 32]

        for bits in valid_bits:
            config = CompressionConfig(quantization_bits=bits)
            assert config.quantization_bits == bits

    def test_pruning_sparsity_range(self):
        """Test pruning sparsity within valid range"""
        # Valid sparsity values
        for sparsity in [0.0, 0.3, 0.5, 0.7]:
            config = CompressionConfig(pruning_sparsity=sparsity)
            assert config.pruning_sparsity == sparsity

    def test_context_length_values(self):
        """Test various context length values"""
        valid_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

        for length in valid_lengths:
            config = CompressionConfig(context_length=length)
            assert config.context_length == length

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = CompressionConfig(quantization_bits=4, pruning_sparsity=0.3)

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["quantization_bits"] == 4
        assert config_dict["pruning_sparsity"] == 0.3
        assert "context_length" in config_dict
        assert "num_layers" in config_dict


class TestEnvironmentConstraints:
    """Test EnvironmentConstraints class"""

    def test_edge_device_constraints(self):
        """Test edge device predefined constraints"""
        assert EDGE_DEVICE.name == "edge_device"
        assert EDGE_DEVICE.max_memory_gb == 4
        assert EDGE_DEVICE.max_power_watts == 15
        assert EDGE_DEVICE.latency_requirement_ms == 50
        assert EDGE_DEVICE.carbon_intensity > 0

    def test_mobile_device_constraints(self):
        """Test mobile device predefined constraints"""
        assert MOBILE_DEVICE.name == "mobile_device"
        assert MOBILE_DEVICE.max_memory_gb == 8
        assert MOBILE_DEVICE.max_power_watts == 10
        assert MOBILE_DEVICE.latency_requirement_ms == 30

    def test_cloud_server_constraints(self):
        """Test cloud server predefined constraints"""
        assert CLOUD_SERVER.name == "cloud_server"
        assert CLOUD_SERVER.max_memory_gb == 64
        assert CLOUD_SERVER.max_power_watts == 250
        assert CLOUD_SERVER.latency_requirement_ms == 200

    def test_carbon_intensive_dc_constraints(self):
        """Test carbon-intensive DC constraints"""
        assert CARBON_INTENSIVE_DC.name == "carbon_intensive_dc"
        assert CARBON_INTENSIVE_DC.max_memory_gb == 128
        assert CARBON_INTENSIVE_DC.carbon_intensity > CLOUD_SERVER.carbon_intensity

    def test_custom_environment(self):
        """Test custom environment constraints"""
        custom_env = EnvironmentConstraints(
            name="custom_edge",
            max_memory_gb=2,
            max_power_watts=5,
            carbon_intensity=0.3,
            latency_requirement_ms=20,
        )

        assert custom_env.name == "custom_edge"
        assert custom_env.max_memory_gb == 2
        assert custom_env.max_power_watts == 5
        assert custom_env.carbon_intensity == 0.3
        assert custom_env.latency_requirement_ms == 20

    def test_environment_comparison(self):
        """Test comparison between different environments"""
        # Edge should be more constrained than cloud
        assert EDGE_DEVICE.max_memory_gb < CLOUD_SERVER.max_memory_gb
        assert EDGE_DEVICE.max_power_watts < CLOUD_SERVER.max_power_watts
        assert EDGE_DEVICE.latency_requirement_ms < CLOUD_SERVER.latency_requirement_ms

        # Mobile should be most constrained on power
        assert MOBILE_DEVICE.max_power_watts <= EDGE_DEVICE.max_power_watts


class TestConfigIntegration:
    """Integration tests for config and environment together"""

    def test_config_with_environment_adaptation(self):
        """Test config adaptation based on environment"""
        # Edge device - should use aggressive compression
        edge_config = CompressionConfig(quantization_bits=4, pruning_sparsity=0.7)

        assert edge_config.quantization_bits == 4  # Aggressive quantization
        assert edge_config.pruning_sparsity == 0.7  # High pruning

        # Cloud server - can use less compression
        cloud_config = CompressionConfig(quantization_bits=16, pruning_sparsity=0.2)

        assert cloud_config.quantization_bits == 16  # Less aggressive
        assert cloud_config.pruning_sparsity == 0.2  # Lower pruning

    def test_carbon_budget_constraints(self):
        """Test carbon budget affects configuration choices"""
        low_carbon = CompressionConfig(carbon_budget=1.0)
        high_carbon = CompressionConfig(carbon_budget=20.0)

        assert low_carbon.carbon_budget < high_carbon.carbon_budget
        assert low_carbon.carbon_budget == 1.0
        assert high_carbon.carbon_budget == 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
