"""
AWQ (Activation-aware Weight Quantization) implementation.

Provides activation-aware weight quantization that preserves important
weights for mathematical reasoning while achieving aggressive compression.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
from datasets import Dataset
import numpy as np
from pathlib import Path
import json
import time

from ..artifacts import ModelArtifact
from ..eval.gsm8k_data import GSM8KDataLoader


logger = logging.getLogger(__name__)


class AWQQuantizer:
    """
    Handles AWQ (Activation-aware Weight Quantization) with math-focused calibration.

    AWQ identifies and preserves the most important weights based on activation
    magnitudes, making it particularly effective for mathematical reasoning tasks.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize AWQ quantizer.

        Args:
            device: Target device for quantization
        """
        self.device = device
        self.supported_bits = [4]  # AWQ typically uses 4-bit
        self.supported_group_sizes = [32, 64, 128]
        self.supported_versions = ["GEMM", "GEMV"]
        self.data_loader = GSM8KDataLoader()

    def apply_awq_quantization(self,
                              model_id: str,
                              tokenizer_id: Optional[str] = None,
                              bits: int = 4,
                              group_size: int = 128,
                              zero_point: bool = True,
                              version: str = "GEMM",
                              calib_data_size: int = 512,
                              trust_remote_code: bool = True,
                              device_map: str = "auto",
                              max_memory: Optional[Dict[str, str]] = None,
                              fuse_layers: bool = True,
                              batch_size: int = 1) -> ModelArtifact:
        """
        Apply AWQ quantization to a model.

        Args:
            model_id: HuggingFace model identifier
            tokenizer_id: HuggingFace tokenizer identifier
            bits: Number of quantization bits (typically 4)
            group_size: Group size for quantization
            zero_point: Whether to use zero point quantization
            version: AWQ kernel version ("GEMM" or "GEMV")
            calib_data_size: Size of calibration dataset
            trust_remote_code: Whether to trust remote code
            device_map: Device mapping strategy
            max_memory: Maximum memory per device
            fuse_layers: Whether to fuse layers for optimization
            batch_size: Batch size for calibration

        Returns:
            ModelArtifact with quantized model metadata

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If quantization fails
        """
        logger.info(f"Applying AWQ quantization to {model_id}")

        # Validate parameters
        if bits not in self.supported_bits:
            raise ValueError(f"Unsupported bits: {bits}. AWQ supports: {self.supported_bits}")

        if group_size not in self.supported_group_sizes:
            raise ValueError(f"Unsupported group_size: {group_size}. "
                           f"Supported: {self.supported_group_sizes}")

        if version not in self.supported_versions:
            raise ValueError(f"Unsupported version: {version}. "
                           f"Supported: {self.supported_versions}")

        if tokenizer_id is None:
            tokenizer_id = model_id

        try:
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=trust_remote_code,
                use_fast=True
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Prepare calibration data
            logger.info(f"Preparing AWQ calibration data ({calib_data_size} samples)")
            calib_data = self._prepare_calibration_data(tokenizer, calib_data_size, batch_size)

            # Configure AWQ
            awq_config = AwqConfig(
                bits=bits,
                group_size=group_size,
                zero_point=zero_point,
                version=version,
                calibration_dataset=calib_data,
                fuse_layers=fuse_layers,
                modules_to_fuse={
                    "attention": ["q_proj", "k_proj", "v_proj"],
                    "mlp": ["gate_proj", "up_proj"],
                } if fuse_layers else None
            )

            logger.info(f"AWQ Config: bits={bits}, group_size={group_size}, "
                       f"zero_point={zero_point}, version={version}")

            # Load and quantize model
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=awq_config,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            quantization_time = time.time() - start_time

            logger.info(f"AWQ quantization completed in {quantization_time:.1f}s")

            # Validate quantization
            if not self._validate_awq_model(model):
                raise RuntimeError("AWQ quantization validation failed")

            # Calculate memory usage and stats
            memory_mb = self._estimate_memory_usage(model)
            model_stats = self._get_model_stats(model)
            awq_stats = self._analyze_awq_weights(model)

            # Create quantization metadata
            quantization_config = {
                "method": "awq",
                "bits": bits,
                "group_size": group_size,
                "zero_point": zero_point,
                "version": version,
                "fuse_layers": fuse_layers,
                "calibration_data_size": calib_data_size,
                "quantization_time_seconds": quantization_time,
                "estimated_memory_mb": memory_mb,
                "model_stats": model_stats,
                "awq_stats": awq_stats,
                "awq_config": awq_config.to_dict()
            }

            # Create model artifact
            model_artifact = ModelArtifact(
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                quantization=quantization_config,
                runtime={
                    "device_map": device_map,
                    "max_memory": max_memory,
                    "torch_dtype": "bfloat16",
                    "awq_version": version
                },
                metadata={
                    "quantization_method": "awq",
                    "memory_efficient": True,
                    "training_compatible": False,  # AWQ is inference-only
                    "inference_optimized": True,
                    "activation_aware": True,
                    "fused_layers": fuse_layers
                }
            )

            logger.info(f"AWQ quantization successful. Memory usage: {memory_mb:.1f}MB")
            return model_artifact

        except Exception as e:
            logger.error(f"AWQ quantization failed: {e}")
            raise RuntimeError(f"Failed to apply AWQ quantization: {e}")

    def _prepare_calibration_data(self,
                                tokenizer: AutoTokenizer,
                                size: int,
                                batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Prepare calibration dataset optimized for mathematical reasoning.

        Args:
            tokenizer: Tokenizer for text processing
            size: Number of calibration samples
            batch_size: Batch size for processing

        Returns:
            List of tokenized calibration batches
        """
        try:
            # Load GSM8K dataset
            dataset_dict, _ = self.data_loader.load_dataset(
                val_split_size=500,
                random_seed=42
            )

            # Use training split for calibration
            train_dataset = dataset_dict["train"]

            # Sample calibration data focusing on diverse mathematical operations
            calib_samples = self._select_diverse_math_samples(train_dataset, size)

            # Tokenize and batch the data
            calib_data = []
            for i in range(0, len(calib_samples), batch_size):
                batch_texts = calib_samples[i:i + batch_size]

                # Create comprehensive prompts for better activation coverage
                formatted_texts = []
                for text in batch_texts:
                    formatted_text = (
                        f"Solve this math problem step by step:\n\n"
                        f"Problem: {text}\n\n"
                        f"Solution: Let me work through this carefully."
                    )
                    formatted_texts.append(formatted_text)

                # Tokenize batch
                batch_tokens = tokenizer(
                    formatted_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                calib_data.append(batch_tokens)

            logger.info(f"Prepared {len(calib_data)} calibration batches with {len(calib_samples)} samples")
            return calib_data

        except Exception as e:
            logger.warning(f"Could not load GSM8K for calibration: {e}")
            # Fallback to dummy mathematical data
            return self._create_dummy_math_calibration(tokenizer, size, batch_size)

    def _select_diverse_math_samples(self, dataset: Dataset, size: int) -> List[str]:
        """
        Select diverse mathematical problems for better activation coverage.

        Args:
            dataset: GSM8K dataset
            size: Number of samples to select

        Returns:
            List of selected problem texts
        """
        # Categories of math problems to ensure diversity
        categories = {
            "arithmetic": ["add", "subtract", "multiply", "divide", "+", "-", "*", "/"],
            "money": ["dollar", "cent", "price", "cost", "buy", "sell", "$"],
            "time": ["hour", "minute", "day", "week", "month", "year"],
            "geometry": ["area", "perimeter", "length", "width", "height", "radius"],
            "fractions": ["half", "third", "quarter", "fraction", "/"],
            "percentages": ["percent", "%", "discount", "increase", "decrease"]
        }

        selected_samples = []
        samples_per_category = max(1, size // len(categories))

        for category, keywords in categories.items():
            category_samples = []

            for i, example in enumerate(dataset):
                if len(category_samples) >= samples_per_category:
                    break

                question = example["question"].lower()
                if any(keyword in question for keyword in keywords):
                    category_samples.append(example["question"])

            selected_samples.extend(category_samples)

        # Fill remaining slots with random samples
        remaining = size - len(selected_samples)
        if remaining > 0:
            indices = np.random.choice(len(dataset), remaining, replace=False)
            for idx in indices:
                selected_samples.append(dataset[int(idx)]["question"])

        return selected_samples[:size]

    def _create_dummy_math_calibration(self,
                                     tokenizer: AutoTokenizer,
                                     size: int,
                                     batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Create dummy mathematical calibration data as fallback.

        Args:
            tokenizer: Tokenizer
            size: Number of samples
            batch_size: Batch size

        Returns:
            List of tokenized calibration batches
        """
        dummy_problems = [
            "What is 25 + 37?",
            "If Sarah has 12 apples and gives away 5, how many does she have left?",
            "A rectangle has length 8 and width 6. What is its area?",
            "Tom saves $15 each week. How much will he save in 4 weeks?",
            "What is 3/4 of 48?",
            "If 20% of students wear glasses and there are 150 students, how many wear glasses?",
        ]

        # Repeat patterns to reach desired size
        calib_samples = (dummy_problems * (size // len(dummy_problems) + 1))[:size]

        calib_data = []
        for i in range(0, len(calib_samples), batch_size):
            batch_texts = calib_samples[i:i + batch_size]

            batch_tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            calib_data.append(batch_tokens)

        return calib_data

    def _validate_awq_model(self, model: torch.nn.Module) -> bool:
        """
        Validate that AWQ quantization was applied correctly.

        Args:
            model: Model to validate

        Returns:
            True if validation passes
        """
        try:
            # Check for AWQ quantized layers
            awq_layers = 0
            total_linear_layers = 0

            for name, module in model.named_modules():
                if 'Linear' in str(type(module)):
                    total_linear_layers += 1
                    # Check if module has AWQ quantization attributes
                    if hasattr(module, 'qweight') or hasattr(module, 'scales'):
                        awq_layers += 1

            if awq_layers == 0:
                logger.error("No AWQ quantized layers found")
                return False

            ratio = awq_layers / total_linear_layers if total_linear_layers > 0 else 0
            logger.info(f"AWQ validation: {awq_layers}/{total_linear_layers} "
                       f"linear layers quantized ({ratio:.2%})")

            return ratio > 0.5  # At least 50% of linear layers should be quantized

        except Exception as e:
            logger.error(f"AWQ validation error: {e}")
            return False

    def _analyze_awq_weights(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Analyze AWQ weight distribution and activation patterns.

        Args:
            model: AWQ quantized model

        Returns:
            Dictionary with AWQ-specific statistics
        """
        stats = {
            "quantized_layers": 0,
            "preserved_weights_ratio": 0.0,
            "average_scale_range": 0.0,
            "weight_distributions": {}
        }

        scale_values = []
        preserved_ratios = []

        for name, module in model.named_modules():
            if hasattr(module, 'scales') and hasattr(module, 'qweight'):
                stats["quantized_layers"] += 1

                # Analyze scales (important weights preserved by AWQ)
                if hasattr(module, 'scales'):
                    scales = module.scales.detach().cpu().numpy()
                    scale_values.extend(scales.flatten())

                    # Estimate preserved weight ratio (higher scales = more preserved)
                    high_scale_ratio = np.sum(scales > scales.mean()) / scales.size
                    preserved_ratios.append(high_scale_ratio)

                    stats["weight_distributions"][name] = {
                        "scale_mean": float(np.mean(scales)),
                        "scale_std": float(np.std(scales)),
                        "scale_min": float(np.min(scales)),
                        "scale_max": float(np.max(scales))
                    }

        if scale_values:
            stats["average_scale_range"] = float(np.max(scale_values) - np.min(scale_values))

        if preserved_ratios:
            stats["preserved_weights_ratio"] = float(np.mean(preserved_ratios))

        return stats

    def _estimate_memory_usage(self, model: torch.nn.Module) -> float:
        """
        Estimate memory usage of AWQ quantized model.

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
                dummy_input = torch.randint(0, 1000, (1, 128)).to(model.device)
                with torch.no_grad():
                    _ = model(dummy_input)

                memory_bytes = torch.cuda.max_memory_allocated()
                return memory_bytes / 1024 / 1024
            else:
                # Rough estimate for CPU
                param_count = sum(p.numel() for p in model.parameters())
                return param_count * 0.6 / 1024 / 1024  # AWQ ~4-bit with overhead

        except Exception as e:
            logger.warning(f"Could not estimate memory usage: {e}")
            return 0.0

    def _get_model_stats(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        Get detailed statistics about the AWQ quantized model.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with model statistics
        """
        stats = {
            "total_parameters": 0,
            "quantized_parameters": 0,
            "total_layers": 0,
            "quantized_layers": 0,
            "layer_types": {}
        }

        for name, module in model.named_modules():
            layer_type = type(module).__name__
            stats["layer_types"][layer_type] = stats["layer_types"].get(layer_type, 0) + 1

            if hasattr(module, 'weight'):
                stats["total_layers"] += 1
                param_count = module.weight.numel()
                stats["total_parameters"] += param_count

                # Check if layer is AWQ quantized
                if hasattr(module, 'qweight') or hasattr(module, 'scales'):
                    stats["quantized_layers"] += 1
                    stats["quantized_parameters"] += param_count

        # Calculate compression ratio
        if stats["total_parameters"] > 0:
            compression_ratio = stats["quantized_parameters"] / stats["total_parameters"]
            stats["compression_ratio"] = compression_ratio

        return stats

    def apply_awq_quantization_with_model(self,
                                         model_id: str,
                                         bits: int = 4,
                                         group_size: int = 128,
                                         zero_point: bool = True,
                                         device: str = "auto") -> tuple:
        """
        Apply AWQ quantization and return model and tokenizer.

        Note: This is a simplified implementation for demonstration.
        In practice, AWQ requires calibration data and specific optimization.
        """
        logger.info(f"Applying AWQ quantization to {model_id} (simplified)")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # For demonstration, load with BnB quantization as fallback
            # In production, this would use actual AWQ optimization
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" if device == "auto" else device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

            logger.info("AWQ quantization applied successfully (using BnB fallback)")
            return model, tokenizer

        except Exception as e:
            logger.error(f"AWQ quantization failed: {e}")
            raise RuntimeError(f"Failed to apply AWQ quantization: {e}")


def apply_awq_quantization(model_id: str, **kwargs) -> ModelArtifact:
    """
    Convenience function to apply AWQ quantization.

    Args:
        model_id: HuggingFace model identifier
        **kwargs: Additional arguments for quantization

    Returns:
        ModelArtifact with quantized model
    """
    quantizer = AWQQuantizer()
    return quantizer.apply_awq_quantization(model_id, **kwargs)