"""
Real model loading and management.

Handles HuggingFace model loading with caching and device management.
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages model loading with caching and memory optimization.

    Features:
    - Automatic device allocation
    - Model caching
    - Memory management
    - Support for both small and large models
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model loader.

        Args:
            cache_dir: Directory for caching models (default: ./model_cache)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./model_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._current_model = None
        self._current_tokenizer = None
        self._current_model_name = None

    def load_model(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        use_cache: bool = True,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a HuggingFace model and tokenizer.

        Args:
            model_name: Model identifier (e.g., "google/gemma3-270m")
            device_map: Device allocation strategy ("auto", "cuda:0", etc.)
            torch_dtype: Data type for model weights
            trust_remote_code: Whether to trust remote code
            use_cache: Whether to use cached model if available

        Returns:
            Tuple of (model, tokenizer)

        Example:
            >>> loader = ModelLoader()
            >>> model, tokenizer = loader.load_model("google/gemma3-270m")
        """
        logger.info(f"Loading model: {model_name}")

        # Check if already loaded
        if use_cache and self._current_model_name == model_name:
            logger.info(f"Using cached model: {model_name}")
            return self._current_model, self._current_tokenizer

        # Clear previous model if exists
        if self._current_model is not None:
            self.release_model()

        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=trust_remote_code,
            )

            # Ensure tokenizer has pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            logger.info(f"Loading model with device_map={device_map}, dtype={torch_dtype}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                cache_dir=str(self.cache_dir),
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )

            # Cache the model
            self._current_model = model
            self._current_tokenizer = tokenizer
            self._current_model_name = model_name

            # Get model size info
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model loaded successfully: {param_count:,} parameters")

            # Check GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def release_model(self):
        """
        Release current model from memory.

        This is crucial for avoiding OOM errors when loading multiple models.
        """
        if self._current_model is not None:
            logger.info(f"Releasing model: {self._current_model_name}")

            # Delete model and tokenizer
            del self._current_model
            del self._current_tokenizer
            self._current_model = None
            self._current_tokenizer = None
            self._current_model_name = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            logger.info("Model released successfully")

    def get_model_info(self, model: AutoModelForCausalLM) -> dict:
        """
        Get information about a loaded model.

        Args:
            model: The model to inspect

        Returns:
            Dictionary with model information
        """
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info = {
            "total_parameters": param_count,
            "trainable_parameters": trainable_params,
            "model_size_mb": param_count * 4 / (1024**2),  # Assuming float32
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
        }

        # Add GPU memory if available
        if torch.cuda.is_available():
            info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

        return info

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.release_model()


# Convenience function for quick model loading
def load_model_simple(
    model_name: str, device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Quick model loading without creating ModelLoader instance.

    Args:
        model_name: Model identifier
        device_map: Device allocation strategy

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_model_simple("google/gemma3-270m")
    """
    loader = ModelLoader()
    return loader.load_model(model_name, device_map=device_map)
