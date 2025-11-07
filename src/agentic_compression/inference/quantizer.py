"""
Real model quantization using bitsandbytes.

Implements 4-bit and 8-bit quantization for model compression.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class RealQuantizer:
    """
    Real quantization using bitsandbytes library.

    Supports:
    - 4-bit quantization with NF4
    - 8-bit quantization
    - Double quantization for better compression
    """

    @staticmethod
    def create_quantization_config(
        bits: int,
        compute_dtype: torch.dtype = torch.bfloat16,
        use_double_quant: bool = True,
        quant_type: str = "nf4",
    ) -> BitsAndBytesConfig:
        """
        Create quantization configuration.

        Args:
            bits: Quantization bits (4 or 8)
            compute_dtype: Computation data type
            use_double_quant: Enable double quantization (4-bit only)
            quant_type: Quantization type ("nf4" or "fp4")

        Returns:
            BitsAndBytesConfig object

        Raises:
            ValueError: If bits is not 4 or 8
        """
        if bits not in [4, 8]:
            raise ValueError(f"Invalid quantization bits: {bits}. Must be 4 or 8.")

        if bits == 4:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_double_quant,
                bnb_4bit_quant_type=quant_type,
            )
            logger.info(
                f"Created 4-bit quantization config: dtype={compute_dtype}, "
                f"double_quant={use_double_quant}, type={quant_type}"
            )
        else:  # bits == 8
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            logger.info("Created 8-bit quantization config")

        return config

    @staticmethod
    def load_quantized_model(
        model_name: str,
        bits: int,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model with quantization applied.

        Args:
            model_name: Model identifier
            bits: Quantization bits (4 or 8)
            device_map: Device allocation strategy
            cache_dir: Cache directory for models
            trust_remote_code: Whether to trust remote code

        Returns:
            Tuple of (quantized_model, tokenizer)

        Example:
            >>> quantizer = RealQuantizer()
            >>> model, tokenizer = quantizer.load_quantized_model("google/gemma3-270m", bits=4)
        """
        logger.info(f"Loading {bits}-bit quantized model: {model_name}")

        # Create quantization config
        quant_config = RealQuantizer.create_quantization_config(bits)

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load quantized model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

            # Get model info
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Quantized model loaded: {param_count:,} parameters")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load quantized model: {str(e)}")
            raise

    @staticmethod
    def quantize_existing_model(
        model: AutoModelForCausalLM, bits: int
    ) -> AutoModelForCausalLM:
        """
        Quantize an already loaded model (post-training quantization).

        Note: This is limited compared to loading with quantization.
        For best results, use load_quantized_model() instead.

        Args:
            model: The model to quantize
            bits: Target quantization bits

        Returns:
            Quantized model

        Raises:
            NotImplementedError: Post-training quantization is complex
        """
        logger.warning(
            "Post-training quantization is limited. "
            "Consider using load_quantized_model() for better results."
        )

        raise NotImplementedError(
            "Post-training quantization is not fully supported. "
            "Please use load_quantized_model() to load a pre-quantized model."
        )

    @staticmethod
    def calculate_compression_metrics(
        original_bits: int = 32, quantized_bits: int = 4
    ) -> dict:
        """
        Calculate theoretical compression metrics.

        Args:
            original_bits: Original precision (default: 32)
            quantized_bits: Quantized precision (4 or 8)

        Returns:
            Dictionary with compression metrics
        """
        compression_ratio = original_bits / quantized_bits
        memory_reduction = 1 - (1 / compression_ratio)

        metrics = {
            "original_bits": original_bits,
            "quantized_bits": quantized_bits,
            "compression_ratio": compression_ratio,
            "memory_reduction_percent": memory_reduction * 100,
            "theoretical_speedup": compression_ratio * 0.7,  # Conservative estimate
        }

        return metrics

    @staticmethod
    def get_quantization_info(model: AutoModelForCausalLM) -> dict:
        """
        Get information about a quantized model.

        Args:
            model: The model to inspect

        Returns:
            Dictionary with quantization information
        """
        info = {"is_quantized": hasattr(model, "quantization_config")}

        if info["is_quantized"]:
            config = model.quantization_config
            info["quantization_method"] = config.quant_method if hasattr(config, "quant_method") else "bitsandbytes"
            info["load_in_4bit"] = getattr(config, "load_in_4bit", False)
            info["load_in_8bit"] = getattr(config, "load_in_8bit", False)

            if info["load_in_4bit"]:
                info["bits"] = 4
                info["compute_dtype"] = str(getattr(config, "bnb_4bit_compute_dtype", "unknown"))
                info["quant_type"] = getattr(config, "bnb_4bit_quant_type", "unknown")
            elif info["load_in_8bit"]:
                info["bits"] = 8
            else:
                info["bits"] = "unknown"
        else:
            info["bits"] = 32  # Assume full precision

        # Memory info
        if torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3

        return info
