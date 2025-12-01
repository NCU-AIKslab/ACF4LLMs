"""Quantization Agent for model compression using LangGraph tools."""

import json
import os
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool

from src.common.schemas import CompressionMethod
from src.tools.quantization_wrapper import get_quantizer

logger = logging.getLogger(__name__)


@tool
def quantize_model(
    model_path: str,
    method: str,
    bit_width: int = 4,
    calibration_samples: int = 512,
    calibration_dataset: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply quantization to a model checkpoint.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        method: Quantization method ('autoround', 'gptq', 'int8', 'awq')
        bit_width: Number of bits for quantization (typically 4 or 8)
        calibration_samples: Number of calibration samples to use
        calibration_dataset: Dataset to use for calibration
        output_dir: Directory to save quantized model (auto-generated if None)

    Returns:
        Dictionary with quantization results including:
        - checkpoint_path: Path to quantized model
        - model_size_gb: Size of quantized model
        - compression_ratio: Compression ratio achieved
        - quantization_time_sec: Time taken for quantization
        - metadata: Additional quantization metadata
    """
    print(f"[Quantization] Starting {method.upper()} quantization with {bit_width} bits...")

    try:
        # Get the appropriate quantizer
        quantizer = get_quantizer(method)

        # Run quantization
        if method.lower() == "int8":
            # INT8 doesn't need bit_width parameter
            result = quantizer.quantize(
                model_name=model_path,
                output_dir=output_dir,
            )
        else:
            result = quantizer.quantize(
                model_name=model_path,
                bit_width=bit_width,
                calibration_samples=calibration_samples,
                calibration_dataset=calibration_dataset,
                output_dir=output_dir,
            )

        print(f"[Quantization] Completed! Saved to {result['checkpoint_path']}")
        print(f"[Quantization] Compression ratio: {result['compression_ratio']:.2f}x")
        print(f"[Quantization] Model size: {result['model_size_gb']:.1f} GB")

        return result

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise


@tool
def estimate_quantization_vram(
    model_path: str,
    method: str,
    bit_width: int = 4,
) -> Dict[str, float]:
    """Estimate VRAM requirements for quantization.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        method: Quantization method
        bit_width: Number of bits for quantization

    Returns:
        Dictionary with VRAM estimates:
        - required_vram_gb: Minimum VRAM needed
        - recommended_vram_gb: Recommended VRAM for smooth operation
    """
    # Mock estimation based on model size
    # In production, this would analyze the actual model
    base_model_size_gb = 16.0  # Assuming 8B model

    # Different methods have different memory requirements
    method_multipliers = {
        "autoround": 2.5,  # Needs model + gradients
        "gptq": 2.0,       # Needs model + calibration data
        "int8": 1.5,       # Simpler method
        "awq": 2.0,        # Similar to GPTQ
    }

    multiplier = method_multipliers.get(method.lower(), 2.0)

    return {
        "required_vram_gb": base_model_size_gb * multiplier,
        "recommended_vram_gb": base_model_size_gb * (multiplier + 0.5),
    }


@tool
def validate_quantization_compatibility(
    model_path: str,
    method: str,
    bit_width: int = 4,
) -> Dict[str, Any]:
    """Check if a model is compatible with the specified quantization method.

    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        method: Quantization method to validate
        bit_width: Number of bits for quantization

    Returns:
        Dictionary with validation results:
        - is_compatible: Whether the method is compatible
        - warnings: List of warnings
        - recommendations: List of recommendations
    """
    warnings = []
    recommendations = []
    is_compatible = True

    # Mock validation logic
    # In production, this would check actual model architecture
    model_name_lower = model_path.lower()

    # Check method compatibility
    if method.lower() == "autoround":
        if "mistral" in model_name_lower or "mixtral" in model_name_lower:
            recommendations.append("AutoRound works particularly well with Mistral models")
    elif method.lower() == "gptq":
        if bit_width not in [4, 8]:
            warnings.append(f"GPTQ typically works best with 4 or 8 bits, got {bit_width}")
    elif method.lower() == "awq":
        if "llama" not in model_name_lower:
            warnings.append("AWQ is optimized for Llama models, results may vary")

    # Check bit width
    if bit_width < 4:
        warnings.append(f"Bit width {bit_width} is very aggressive, may cause significant accuracy loss")
        recommendations.append("Consider using 4 or 8 bits for better accuracy")
    elif bit_width > 8:
        warnings.append(f"Bit width {bit_width} provides limited compression benefit")
        recommendations.append("Consider using 4 or 8 bits for better compression")

    return {
        "is_compatible": is_compatible,
        "warnings": warnings,
        "recommendations": recommendations,
    }


@tool
def list_available_quantization_methods() -> List[Dict[str, Any]]:
    """List all available quantization methods with their characteristics.

    Returns:
        List of dictionaries describing each method
    """
    methods = [
        {
            "name": "autoround",
            "description": "Adaptive rounding for weight quantization",
            "supported_bits": [4, 8],
            "pros": ["Good accuracy retention", "Fast inference"],
            "cons": ["Longer quantization time"],
            "recommended_for": ["Large language models", "Production deployments"],
        },
        {
            "name": "gptq",
            "description": "Post-training quantization with layer-wise optimization",
            "supported_bits": [2, 3, 4, 8],
            "pros": ["Excellent compression", "Wide hardware support"],
            "cons": ["Requires calibration data"],
            "recommended_for": ["Memory-constrained environments", "Edge deployment"],
        },
        {
            "name": "int8",
            "description": "Simple 8-bit integer quantization",
            "supported_bits": [8],
            "pros": ["Fast quantization", "Good hardware support"],
            "cons": ["Limited compression ratio"],
            "recommended_for": ["Quick prototyping", "Balanced compression"],
        },
        {
            "name": "awq",
            "description": "Activation-aware weight quantization",
            "supported_bits": [4],
            "pros": ["Preserves important weights", "Good for LLMs"],
            "cons": ["Limited bit width options"],
            "recommended_for": ["Llama models", "Chat applications"],
        },
    ]

    return methods


def get_quantization_subagent(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create the quantization subagent configuration for DeepAgents.

    Args:
        spec: Model specification with requirements and constraints

    Returns:
        Subagent configuration dictionary
    """
    # Extract relevant spec information
    model_name = spec.get("model_name", "unknown")
    preferred_methods = spec.get("preferred_methods", ["autoround", "gptq"])
    calibration_samples = spec.get("calibration_samples", 512)

    # Format preferred methods for prompt
    methods_str = ", ".join([m.value if hasattr(m, 'value') else str(m) for m in preferred_methods])

    prompt = f"""You are a Quantization Specialist Agent responsible for compressing neural networks.

Model: {model_name}
Preferred Methods: {methods_str}
Calibration Samples: {calibration_samples}

Your responsibilities:
1. Apply quantization methods to compress models
2. Validate compatibility before quantization
3. Estimate VRAM requirements
4. Track compression ratios and model sizes
5. Save quantized checkpoints with metadata

Available tools:
- quantize_model: Apply quantization to a model
- estimate_quantization_vram: Check VRAM requirements
- validate_quantization_compatibility: Validate method compatibility
- list_available_quantization_methods: Get information about methods

Guidelines:
- Always validate compatibility before quantizing
- Check VRAM requirements to prevent out-of-memory errors
- Start with conservative settings (8-bit) if unsure
- Use 4-bit quantization for aggressive compression
- Save detailed metadata with each quantized model
- Report compression ratio and final model size

When you receive a quantization request:
1. First, validate the method compatibility
2. Check VRAM requirements
3. Execute quantization with appropriate parameters
4. Return the checkpoint path and compression metrics
"""

    return {
        "name": "quantization",
        "description": "Applies various quantization methods to compress neural networks",
        "prompt": prompt,
        "tools": [
            quantize_model,
            estimate_quantization_vram,
            validate_quantization_compatibility,
            list_available_quantization_methods,
        ],
        "model": "anthropic:claude-3-haiku-20240307",  # Use cheaper model for specialized task
    }


# Export tools
__all__ = [
    "quantize_model",
    "estimate_quantization_vram",
    "validate_quantization_compatibility",
    "list_available_quantization_methods",
    "get_quantization_subagent",
]
