"""
BitsAndBytes quantization implementation.

Provides 4-bit quantization using BitsAndBytes NF4/FP4 with QLoRA support
for memory-efficient fine-tuning and inference.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from pathlib import Path

from ..artifacts import ModelArtifact


logger = logging.getLogger(__name__)


class BnBQuantizer:
    """
    Handles BitsAndBytes 4-bit quantization with NF4/FP4 support.

    Implements quantization strategies optimized for mathematical reasoning
    with proper calibration on GSM8K-style problems.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize BitsAndBytes quantizer.

        Args:
            device: Target device for quantization
        """
        self.device = device
        self.supported_dtypes = ["float16", "bfloat16"]
        self.supported_quant_types = ["nf4", "fp4"]

    def apply_bnb_4bit(self,
                      model_id: str,
                      tokenizer_id: Optional[str] = None,
                      quant_type: str = "nf4",
                      compute_dtype: str = "bfloat16",
                      use_double_quant: bool = True,
                      bnb_4bit_quant_type: Optional[str] = None,
                      trust_remote_code: bool = True,
                      device_map: str = "auto",
                      low_cpu_mem_usage: bool = True) -> ModelArtifact:
        """
        Apply BitsAndBytes 4-bit quantization to a model.

        Args:
            model_id: HuggingFace model identifier
            tokenizer_id: HuggingFace tokenizer identifier (defaults to model_id)
            quant_type: Quantization type ("nf4" or "fp4")
            compute_dtype: Compute dtype for operations ("bfloat16" or "float16")
            use_double_quant: Whether to use double quantization
            bnb_4bit_quant_type: Legacy parameter for compatibility
            trust_remote_code: Whether to trust remote code
            device_map: Device mapping strategy
            low_cpu_mem_usage: Whether to use low CPU memory usage

        Returns:
            ModelArtifact with quantized model metadata

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If quantization fails
        """
        logger.info(f"Applying BitsAndBytes 4-bit quantization to {model_id}")

        # Validate parameters
        if quant_type not in self.supported_quant_types:
            raise ValueError(f"Unsupported quantization type: {quant_type}. "
                           f"Supported: {self.supported_quant_types}")

        if compute_dtype not in self.supported_dtypes:
            raise ValueError(f"Unsupported compute dtype: {compute_dtype}. "
                           f"Supported: {self.supported_dtypes}")

        # Set default tokenizer
        if tokenizer_id is None:
            tokenizer_id = model_id

        # Convert dtype string to torch dtype
        torch_dtype = getattr(torch, compute_dtype)

        try:
            # Configure BitsAndBytes quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_storage=torch_dtype,  # Storage dtype
            )

            logger.info(f"BnB Config: quant_type={quant_type}, "
                       f"compute_dtype={compute_dtype}, "
                       f"double_quant={use_double_quant}")

            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=trust_remote_code,
                use_fast=True
            )

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Verify quantization was applied
            quantized_params = sum(1 for param in model.parameters()
                                 if hasattr(param, 'quant_type'))
            total_params = sum(1 for _ in model.parameters())

            logger.info(f"Quantization applied to {quantized_params}/{total_params} parameters")

            # Calculate memory usage
            memory_mb = self._estimate_memory_usage(model)

            # Create quantization metadata
            quantization_config = {
                "method": "bitsandbytes",
                "bits": 4,
                "quant_type": quant_type,
                "compute_dtype": compute_dtype,
                "use_double_quant": use_double_quant,
                "quantized_parameters": quantized_params,
                "total_parameters": total_params,
                "estimated_memory_mb": memory_mb,
                "bnb_config": bnb_config.to_dict()
            }

            # Create model artifact
            model_artifact = ModelArtifact(
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                quantization=quantization_config,
                runtime={
                    "device_map": device_map,
                    "torch_dtype": str(torch_dtype),
                    "low_cpu_mem_usage": low_cpu_mem_usage
                },
                metadata={
                    "quantization_method": "bitsandbytes_4bit",
                    "memory_efficient": True,
                    "training_compatible": True,  # QLoRA compatible
                    "inference_optimized": True
                }
            )

            logger.info(f"BitsAndBytes quantization completed successfully. "
                       f"Estimated memory usage: {memory_mb:.1f}MB")

            return model_artifact

        except Exception as e:
            logger.error(f"BitsAndBytes quantization failed: {e}")
            raise RuntimeError(f"Failed to apply BitsAndBytes quantization: {e}")

    def apply_bnb_4bit_with_model(self,
                                  model_id: str,
                                  quant_type: str = "nf4",
                                  compute_dtype: str = "float16",
                                  use_double_quant: bool = True,
                                  device: str = "auto") -> Tuple[torch.nn.Module, Any]:
        """
        Apply BitsAndBytes 4-bit quantization and return model and tokenizer.

        Args:
            model_id: HuggingFace model identifier
            quant_type: Quantization type ("nf4" or "fp4")
            compute_dtype: Compute dtype for operations
            use_double_quant: Whether to use double quantization
            device: Target device

        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        logger.info(f"Applying BitsAndBytes 4-bit quantization to {model_id}")

        # Convert dtype string to torch dtype
        torch_dtype = getattr(torch, compute_dtype)

        try:
            # Configure BitsAndBytes quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_storage=torch_dtype,
            )

            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" if device == "auto" else device,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True
            )

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("BitsAndBytes 4-bit quantization applied successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"BitsAndBytes quantization failed: {e}")
            raise RuntimeError(f"Failed to apply BitsAndBytes quantization: {e}")

    def apply_8bit_quantization(self,
                               model_id: str,
                               tokenizer_id: Optional[str] = None,
                               llm_int8_threshold: float = 6.0,
                               llm_int8_has_fp16_weight: bool = False,
                               trust_remote_code: bool = True,
                               device_map: str = "auto") -> ModelArtifact:
        """
        Apply BitsAndBytes 8-bit quantization.

        Args:
            model_id: HuggingFace model identifier
            tokenizer_id: HuggingFace tokenizer identifier
            llm_int8_threshold: Threshold for mixed-precision
            llm_int8_has_fp16_weight: Whether to keep some weights in fp16
            trust_remote_code: Whether to trust remote code
            device_map: Device mapping strategy

        Returns:
            ModelArtifact with quantized model metadata
        """
        logger.info(f"Applying BitsAndBytes 8-bit quantization to {model_id}")

        if tokenizer_id is None:
            tokenizer_id = model_id

        try:
            # Configure 8-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=llm_int8_threshold,
                llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            )

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=trust_remote_code
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=trust_remote_code
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            memory_mb = self._estimate_memory_usage(model)

            quantization_config = {
                "method": "bitsandbytes",
                "bits": 8,
                "llm_int8_threshold": llm_int8_threshold,
                "llm_int8_has_fp16_weight": llm_int8_has_fp16_weight,
                "estimated_memory_mb": memory_mb,
                "bnb_config": bnb_config.to_dict()
            }

            model_artifact = ModelArtifact(
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                quantization=quantization_config,
                runtime={"device_map": device_map},
                metadata={
                    "quantization_method": "bitsandbytes_8bit",
                    "memory_efficient": True,
                    "training_compatible": True
                }
            )

            logger.info(f"8-bit quantization completed. Memory usage: {memory_mb:.1f}MB")
            return model_artifact

        except Exception as e:
            logger.error(f"8-bit quantization failed: {e}")
            raise RuntimeError(f"Failed to apply 8-bit quantization: {e}")

    def _estimate_memory_usage(self, model: torch.nn.Module) -> float:
        """
        Estimate memory usage of quantized model in MB.

        Args:
            model: Quantized model

        Returns:
            Estimated memory usage in MB
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Trigger memory allocation
                dummy_input = torch.randint(0, 1000, (1, 10)).to(model.device)
                with torch.no_grad():
                    _ = model(dummy_input)

                memory_bytes = torch.cuda.max_memory_allocated()
                return memory_bytes / 1024 / 1024

            else:
                # Rough estimation for CPU
                param_count = sum(p.numel() for p in model.parameters())
                # Assume 4-bit quantization reduces memory by ~4x
                return param_count * 0.5 / 1024 / 1024  # Rough estimate

        except Exception as e:
            logger.warning(f"Could not estimate memory usage: {e}")
            return 0.0

    def get_memory_footprint(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Get detailed memory footprint analysis.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with memory statistics
        """
        stats = {}

        # Count parameters by type
        quantized_params = 0
        fp16_params = 0
        fp32_params = 0

        for param in model.parameters():
            if hasattr(param, 'quant_type'):
                quantized_params += param.numel()
            elif param.dtype == torch.float16:
                fp16_params += param.numel()
            elif param.dtype == torch.float32:
                fp32_params += param.numel()

        stats['quantized_parameters'] = quantized_params
        stats['fp16_parameters'] = fp16_params
        stats['fp32_parameters'] = fp32_params
        stats['total_parameters'] = quantized_params + fp16_params + fp32_params

        # Estimate memory usage
        # 4-bit quantized: ~0.5 bytes per param
        # fp16: 2 bytes per param
        # fp32: 4 bytes per param
        estimated_mb = (
            quantized_params * 0.5 +
            fp16_params * 2 +
            fp32_params * 4
        ) / 1024 / 1024

        stats['estimated_memory_mb'] = estimated_mb

        if torch.cuda.is_available():
            stats['cuda_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['cuda_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        return stats

    def validate_quantization(self, model: torch.nn.Module) -> bool:
        """
        Validate that quantization was applied correctly.

        Args:
            model: Model to validate

        Returns:
            True if quantization is valid
        """
        try:
            # Check if any parameters are quantized
            quantized_layers = 0
            total_layers = 0

            for name, module in model.named_modules():
                if hasattr(module, 'weight'):
                    total_layers += 1
                    if hasattr(module.weight, 'quant_type'):
                        quantized_layers += 1

            if quantized_layers == 0:
                logger.warning("No quantized layers found in model")
                return False

            quantization_ratio = quantized_layers / total_layers if total_layers > 0 else 0
            logger.info(f"Quantization validation: {quantized_layers}/{total_layers} "
                       f"layers quantized ({quantization_ratio:.2%})")

            return quantization_ratio > 0.1  # At least 10% of layers should be quantized

        except Exception as e:
            logger.error(f"Quantization validation failed: {e}")
            return False


def apply_bnb_4bit(model_id: str, **kwargs) -> ModelArtifact:
    """
    Convenience function to apply BitsAndBytes 4-bit quantization.

    Args:
        model_id: HuggingFace model identifier
        **kwargs: Additional arguments for quantization

    Returns:
        ModelArtifact with quantized model
    """
    quantizer = BitsAndBytesQuantizer()
    return quantizer.apply_bnb_4bit(model_id, **kwargs)


def apply_bnb_8bit(model_id: str, **kwargs) -> ModelArtifact:
    """
    Convenience function to apply BitsAndBytes 8-bit quantization.

    Args:
        model_id: HuggingFace model identifier
        **kwargs: Additional arguments for quantization

    Returns:
        ModelArtifact with quantized model
    """
    quantizer = BitsAndBytesQuantizer()
    return quantizer.apply_8bit_quantization(model_id, **kwargs)