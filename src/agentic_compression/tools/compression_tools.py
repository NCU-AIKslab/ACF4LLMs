"""
LangChain tools for model compression strategies.
"""

import json
import logging

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydantic models for tool inputs
class QuantizationInput(BaseModel):
    """Input schema for quantization tool"""

    bits: int = Field(description="Target precision in bits (4, 8, 16, 32)")
    method: str = Field(default="gptq", description="Quantization method (gptq, awq, smoothquant)")


class PruningInput(BaseModel):
    """Input schema for pruning tool"""

    sparsity: float = Field(description="Target sparsity level (0.0-0.7)")
    pattern: str = Field(default="2:4", description="Pruning pattern (2:4, 4:8, unstructured)")


class KVCacheInput(BaseModel):
    """Input schema for KV cache optimization tool"""

    context_length: int = Field(description="Target context length in tokens")
    cache_strategy: str = Field(default="sliding_window", description="Cache management strategy")


class DistillationInput(BaseModel):
    """Input schema for distillation tool"""

    student_layers: int = Field(description="Number of layers in student model")
    temperature: float = Field(default=5.0, description="Distillation temperature")


# ============================================================================
# Compression Tools
# ============================================================================


class QuantizationTool(BaseTool):
    """Tool for model quantization to reduce precision."""

    name: str = "quantize_model"
    description: str = """Quantize a model to reduce precision and memory footprint.

    Use this when you need to reduce model size through precision reduction.

    Args:
        bits: Target precision in bits (4, 8, 16, 32). Lower bits = more compression.
        method: Quantization method to use:
            - gptq: General-purpose quantization (best quality)
            - awq: Activation-aware quantization (faster)
            - smoothquant: For balanced accuracy/speed

    Returns:
        JSON with compression_ratio, memory_reduction, estimated_accuracy_loss, energy_reduction
    """
    args_schema: type[BaseModel] = QuantizationInput

    def _run(self, bits: int, method: str = "gptq") -> str:
        """Execute quantization simulation"""
        logger.info(f"Quantizing model to {bits} bits using {method}")

        # Validate inputs
        if bits not in [4, 8, 16, 32]:
            raise ValueError(f"Invalid quantization bits: {bits}. Must be 4, 8, 16, or 32.")

        # Simulate quantization results
        compression_ratio = 32 / bits
        memory_reduction = 1 - (1 / compression_ratio)

        # Accuracy impact varies by method and bit width
        accuracy_impact_factors = {
            "gptq": 0.015,  # Best quality preservation
            "awq": 0.018,  # Slightly more impact
            "smoothquant": 0.020,  # Most efficient but more impact
        }
        factor = accuracy_impact_factors.get(method, 0.02)
        accuracy_impact = factor * (8 / bits)

        result = {
            "strategy": "quantization",
            "bits": bits,
            "method": method,
            "compression_ratio": f"{compression_ratio:.2f}x",
            "memory_reduction": f"{memory_reduction:.1%}",
            "estimated_accuracy_loss": f"{accuracy_impact:.1%}",
            "energy_reduction": f"{memory_reduction * 0.8:.1%}",
            "latency_improvement": f"{(compression_ratio - 1) * 0.3:.1%}",
        }

        return json.dumps(result, indent=2)

    async def _arun(self, bits: int, method: str = "gptq") -> str:
        """Async version"""
        return self._run(bits, method)


class PruningTool(BaseTool):
    """Tool for model pruning to remove redundant weights."""

    name: str = "prune_model"
    description: str = """Prune model weights to reduce parameters and computation.

    Use this to reduce model size by removing less important weights.

    Args:
        sparsity: Target sparsity level (0.0-0.7). Higher sparsity = more pruning.
            - 0.3: Conservative, minimal accuracy impact
            - 0.5: Balanced compression/accuracy
            - 0.7: Aggressive, maximum compression
        pattern: Pruning pattern:
            - 2:4: Keep 2 out of every 4 weights (structured, hardware-friendly)
            - 4:8: Keep 4 out of every 8 weights
            - unstructured: Remove any weights (highest compression)

    Returns:
        JSON with speedup, memory_reduction, estimated_accuracy_loss, energy_reduction
    """
    args_schema: type[BaseModel] = PruningInput

    def _run(self, sparsity: float, pattern: str = "2:4") -> str:
        """Execute pruning simulation"""
        logger.info(f"Pruning model with {sparsity:.1%} sparsity using {pattern} pattern")

        # Validate inputs
        if not 0.0 <= sparsity <= 0.7:
            raise ValueError(f"Invalid sparsity: {sparsity}. Must be between 0.0 and 0.7.")

        # Calculate performance impacts
        speedup = 1 / (1 - sparsity) if sparsity < 1.0 else 1.0
        memory_reduction = sparsity

        # Pattern affects hardware efficiency
        hardware_efficiency = {
            "2:4": 0.9,  # High hardware support
            "4:8": 0.85,  # Good hardware support
            "unstructured": 0.6,  # Limited hardware support
        }
        efficiency = hardware_efficiency.get(pattern, 0.7)

        # Accuracy impact depends on sparsity and pattern
        accuracy_impact = sparsity * 0.1 * (1.0 if pattern == "unstructured" else 0.8)

        result = {
            "strategy": "pruning",
            "sparsity": f"{sparsity:.1%}",
            "pattern": pattern,
            "speedup": f"{speedup:.2f}x",
            "memory_reduction": f"{memory_reduction:.1%}",
            "estimated_accuracy_loss": f"{accuracy_impact:.1%}",
            "energy_reduction": f"{sparsity * 0.7:.1%}",
            "hardware_efficiency": f"{efficiency:.1%}",
        }

        return json.dumps(result, indent=2)

    async def _arun(self, sparsity: float, pattern: str = "2:4") -> str:
        """Async version"""
        return self._run(sparsity, pattern)


class KVCacheTool(BaseTool):
    """Tool for KV cache and context window optimization."""

    name: str = "optimize_kv_cache"
    description: str = """Optimize KV cache and context window for memory/latency efficiency.

    Use this to reduce memory usage for long-context scenarios.

    Args:
        context_length: Target context length in tokens (1024-32768)
            - 2048: Short context, minimal memory
            - 4096: Standard context
            - 8192: Long context
            - 16384+: Very long context, high memory
        cache_strategy: Cache management strategy:
            - sliding_window: Fixed window, discard old tokens
            - streaming: Incremental processing
            - attention_sink: Keep important tokens

    Returns:
        JSON with memory_ratio, latency_reduction, context_coverage
    """
    args_schema: type[BaseModel] = KVCacheInput

    def _run(self, context_length: int, cache_strategy: str = "sliding_window") -> str:
        """Execute KV cache optimization simulation"""
        logger.info(f"Optimizing KV cache with {context_length} context using {cache_strategy}")

        # Validate inputs
        if not 1024 <= context_length <= 32768:
            raise ValueError(f"Invalid context_length: {context_length}. Must be 1024-32768.")

        # Calculate impacts relative to baseline (32768)
        memory_ratio = context_length / 32768

        # Strategy affects efficiency
        strategy_efficiency = {
            "sliding_window": 1.0,
            "streaming": 0.9,
            "attention_sink": 0.85,
        }
        efficiency = strategy_efficiency.get(cache_strategy, 1.0)

        latency_reduction = (1 - memory_ratio * efficiency) * 0.3

        result = {
            "strategy": "kv_cache",
            "context_length": context_length,
            "cache_strategy": cache_strategy,
            "memory_ratio": f"{memory_ratio:.1%}",
            "latency_reduction": f"{latency_reduction:.1%}",
            "memory_saved": f"{(1 - memory_ratio):.1%}",
            "efficiency_factor": f"{efficiency:.2f}",
        }

        return json.dumps(result, indent=2)

    async def _arun(self, context_length: int, cache_strategy: str = "sliding_window") -> str:
        """Async version"""
        return self._run(context_length, cache_strategy)


class DistillationTool(BaseTool):
    """Tool for knowledge distillation to create smaller models."""

    name: str = "distill_model"
    description: str = """Apply knowledge distillation to create a smaller student model.

    Use this for significant model size reduction by training a smaller model.

    Args:
        student_layers: Number of layers in student model (6-24)
            - 6: Very small, 4x compression
            - 12: Medium, 2x compression
            - 18: Moderate, 1.3x compression
        temperature: Distillation temperature (1.0-10.0)
            - Low (1-3): Harder targets, less knowledge transfer
            - Medium (4-6): Balanced
            - High (7-10): Softer targets, more knowledge transfer

    Returns:
        JSON with compression_ratio, training_required, estimated_accuracy_loss
    """
    args_schema: type[BaseModel] = DistillationInput

    def _run(self, student_layers: int, temperature: float = 5.0) -> str:
        """Execute distillation simulation"""
        logger.info(f"Distilling to {student_layers} layers with temperature {temperature}")

        # Validate inputs
        if not 6 <= student_layers <= 24:
            raise ValueError(f"Invalid student_layers: {student_layers}. Must be 6-24.")

        if not 1.0 <= temperature <= 10.0:
            raise ValueError(f"Invalid temperature: {temperature}. Must be 1.0-10.0.")

        # Calculate compression
        teacher_layers = 24  # Assume 24-layer teacher
        compression_ratio = teacher_layers / student_layers

        # Temperature affects knowledge transfer quality
        temp_factor = min(temperature / 5.0, 1.5)  # Optimal around 5.0
        accuracy_impact = 0.15 * (1 - student_layers / 24) / temp_factor

        result = {
            "strategy": "distillation",
            "student_layers": student_layers,
            "teacher_layers": teacher_layers,
            "temperature": temperature,
            "compression_ratio": f"{compression_ratio:.2f}x",
            "training_required": True,
            "estimated_accuracy_loss": f"{accuracy_impact:.1%}",
            "training_time_estimate": f"{student_layers * 2}h",
            "memory_reduction": f"{(1 - 1/compression_ratio):.1%}",
        }

        return json.dumps(result, indent=2)

    async def _arun(self, student_layers: int, temperature: float = 5.0) -> str:
        """Async version"""
        return self._run(student_layers, temperature)


# Helper function to get all compression tools
def get_all_compression_tools() -> list[BaseTool]:
    """Get list of all compression tools for LangChain agents"""
    return [
        QuantizationTool(),
        PruningTool(),
        KVCacheTool(),
        DistillationTool(),
    ]
