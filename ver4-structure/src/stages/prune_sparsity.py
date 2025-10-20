"""
Pruning and sparsity implementation with 2:4 structured sparsity support.

Provides structured and unstructured pruning techniques optimized for
mathematical reasoning models and NVIDIA Tensor Core acceleration.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from pathlib import Path
import json
import time

from ..artifacts import ModelArtifact


logger = logging.getLogger(__name__)


class SemiStructuredSparsityPruner:
    """
    Implements 2:4 semi-structured sparsity for NVIDIA Tensor Core acceleration.

    2:4 sparsity maintains 2 non-zero values for every 4 consecutive values,
    enabling hardware acceleration on Ampere and Hopper architectures.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize semi-structured sparsity pruner.

        Args:
            device: Target device for pruning operations
        """
        self.device = device
        self.supported_ratios = [0.5]  # 2:4 sparsity = 50% sparsity
        self.tensor_core_compatible = self._check_tensor_core_support()

    def _check_tensor_core_support(self) -> bool:
        """Check if current GPU supports sparse Tensor Cores."""
        if not torch.cuda.is_available():
            return False

        try:
            # Check for Ampere (8.x) or Hopper (9.x) architecture
            capability = torch.cuda.get_device_capability()
            major, minor = capability
            return major >= 8  # Ampere (8.0+) and newer support 2:4 sparsity

        except Exception as e:
            logger.warning(f"Could not detect GPU capability: {e}")
            return False

    def apply_2to4_sparsity(self,
                           model: nn.Module,
                           layers_to_prune: Optional[List[str]] = None,
                           preserve_modules: Optional[List[str]] = None,
                           recovery_finetune: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply 2:4 semi-structured sparsity to model layers.

        Args:
            model: PyTorch model to prune
            layers_to_prune: Specific layer names to prune (None = auto-detect)
            preserve_modules: Module names to skip pruning
            recovery_finetune: Whether model will undergo recovery fine-tuning

        Returns:
            Tuple of (pruned_model, pruning_stats)

        Raises:
            RuntimeError: If pruning fails
        """
        logger.info("Applying 2:4 semi-structured sparsity")

        if not self.tensor_core_compatible:
            logger.warning("GPU does not support sparse Tensor Cores. "
                          "2:4 sparsity will not provide hardware acceleration.")

        try:
            pruning_stats = {
                "total_parameters": 0,
                "pruned_parameters": 0,
                "layers_processed": 0,
                "skipped_layers": 0,
                "sparsity_ratio": 0.5,
                "structured": True,
                "pattern": "2:4"
            }

            preserve_modules = preserve_modules or []
            processed_layers = []

            # Auto-detect layers to prune if not specified
            if layers_to_prune is None:
                layers_to_prune = self._get_prunable_layers(model)

            logger.info(f"Identified {len(layers_to_prune)} layers for 2:4 pruning")

            for name, module in model.named_modules():
                if not self._should_prune_layer(name, module, layers_to_prune, preserve_modules):
                    continue

                if hasattr(module, 'weight') and module.weight is not None:
                    original_shape = module.weight.shape
                    original_params = module.weight.numel()

                    # Apply 2:4 sparsity pattern
                    sparse_weight = self._apply_2to4_pattern(module.weight)

                    # Update module weight
                    with torch.no_grad():
                        module.weight.copy_(sparse_weight)

                    # Track statistics
                    pruning_stats["total_parameters"] += original_params
                    pruning_stats["pruned_parameters"] += original_params // 2  # 50% pruned
                    pruning_stats["layers_processed"] += 1
                    processed_layers.append(name)

                    logger.debug(f"Applied 2:4 sparsity to {name}: {original_shape}")

            # Final statistics
            if pruning_stats["total_parameters"] > 0:
                actual_sparsity = pruning_stats["pruned_parameters"] / pruning_stats["total_parameters"]
                pruning_stats["actual_sparsity_ratio"] = actual_sparsity

            pruning_stats["processed_layers"] = processed_layers
            pruning_stats["tensor_core_compatible"] = self.tensor_core_compatible

            logger.info(f"2:4 sparsity applied to {pruning_stats['layers_processed']} layers. "
                       f"Pruned {pruning_stats['pruned_parameters']:,} / "
                       f"{pruning_stats['total_parameters']:,} parameters")

            return model, pruning_stats

        except Exception as e:
            logger.error(f"2:4 sparsity application failed: {e}")
            raise RuntimeError(f"Failed to apply 2:4 sparsity: {e}")

    def _apply_2to4_pattern(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply 2:4 sparsity pattern to a weight tensor.

        Args:
            weight: Weight tensor to prune

        Returns:
            Pruned weight tensor with 2:4 pattern
        """
        # Ensure weight is 2D for processing
        original_shape = weight.shape
        if weight.dim() > 2:
            weight_2d = weight.view(weight.size(0), -1)
        else:
            weight_2d = weight

        # Pad if necessary to make divisible by 4
        rows, cols = weight_2d.shape
        pad_cols = (4 - cols % 4) % 4
        if pad_cols > 0:
            weight_2d = torch.cat([weight_2d, torch.zeros(rows, pad_cols, device=weight.device)], dim=1)

        # Reshape to blocks of 4
        weight_blocks = weight_2d.view(rows, -1, 4)

        # Apply 2:4 pattern: keep top 2 values in magnitude, zero the rest
        sparse_blocks = torch.zeros_like(weight_blocks)

        for i in range(rows):
            for j in range(weight_blocks.size(1)):
                block = weight_blocks[i, j]
                # Find indices of top 2 values by magnitude
                _, top2_indices = torch.topk(torch.abs(block), k=2)
                # Keep only top 2 values
                sparse_blocks[i, j, top2_indices] = block[top2_indices]

        # Reshape back to original 2D shape
        sparse_weight_2d = sparse_blocks.view(rows, -1)

        # Remove padding if added
        if pad_cols > 0:
            sparse_weight_2d = sparse_weight_2d[:, :-pad_cols]

        # Reshape to original shape
        return sparse_weight_2d.view(original_shape)

    def _get_prunable_layers(self, model: nn.Module) -> List[str]:
        """
        Automatically identify layers suitable for 2:4 pruning.

        Args:
            model: Model to analyze

        Returns:
            List of layer names suitable for pruning
        """
        prunable_layers = []
        prunable_types = (nn.Linear, nn.Conv2d, nn.Conv1d)

        for name, module in model.named_modules():
            if isinstance(module, prunable_types):
                # Skip very small layers (< 1K parameters)
                if hasattr(module, 'weight') and module.weight.numel() >= 1024:
                    # Focus on attention and MLP layers for transformers
                    if any(keyword in name.lower() for keyword in
                          ['attn', 'attention', 'mlp', 'feed_forward', 'ffn',
                           'q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'gate_proj', 'up_proj', 'down_proj']):
                        prunable_layers.append(name)

        return prunable_layers

    def _should_prune_layer(self,
                          name: str,
                          module: nn.Module,
                          layers_to_prune: List[str],
                          preserve_modules: List[str]) -> bool:
        """
        Determine if a layer should be pruned.

        Args:
            name: Layer name
            module: Layer module
            layers_to_prune: List of layers to prune
            preserve_modules: List of modules to preserve

        Returns:
            True if layer should be pruned
        """
        # Skip if in preserve list
        if any(preserve in name for preserve in preserve_modules):
            return False

        # Skip embedding and output layers
        if any(keyword in name.lower() for keyword in ['embed', 'lm_head', 'classifier']):
            return False

        # Check if layer is in pruning list
        return any(layer_pattern in name for layer_pattern in layers_to_prune)

    def export_sparse_model(self,
                           model: nn.Module,
                           output_path: Path,
                           enable_tensor_core_optimization: bool = True) -> None:
        """
        Export sparse model with Tensor Core optimization flags.

        Args:
            model: Sparse model to export
            output_path: Path to save the model
            enable_tensor_core_optimization: Whether to enable Tensor Core optimization
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(model.state_dict(), output_path / "sparse_model.pt")

        # Create optimization metadata
        optimization_info = {
            "sparsity_type": "2:4_semi_structured",
            "tensor_core_compatible": self.tensor_core_compatible,
            "enable_tensor_core_optimization": enable_tensor_core_optimization,
            "sparse_tensor_format": "csr" if enable_tensor_core_optimization else "dense",
            "export_timestamp": time.time()
        }

        with open(output_path / "sparsity_info.json", "w") as f:
            json.dump(optimization_info, f, indent=2)

        logger.info(f"Sparse model exported to {output_path}")


class UnstructuredPruner:
    """
    Implements magnitude-based unstructured pruning.

    Removes individual weights based on magnitude thresholding,
    suitable for general sparsity without hardware constraints.
    """

    def __init__(self):
        """Initialize unstructured pruner."""
        pass

    def apply_magnitude_pruning(self,
                              model: nn.Module,
                              sparsity_ratio: float = 0.3,
                              layers_to_prune: Optional[List[str]] = None,
                              global_pruning: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply magnitude-based unstructured pruning.

        Args:
            model: Model to prune
            sparsity_ratio: Fraction of weights to prune (0.0 to 1.0)
            layers_to_prune: Specific layers to prune (None = all suitable layers)
            global_pruning: Whether to use global magnitude threshold

        Returns:
            Tuple of (pruned_model, pruning_stats)
        """
        logger.info(f"Applying magnitude pruning with {sparsity_ratio:.1%} sparsity")

        try:
            pruning_stats = {
                "total_parameters": 0,
                "pruned_parameters": 0,
                "layers_processed": 0,
                "sparsity_ratio": sparsity_ratio,
                "structured": False,
                "global_pruning": global_pruning
            }

            # Collect all weights for global threshold calculation
            all_weights = []
            layer_weights = {}

            for name, module in model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if layers_to_prune is None or any(layer in name for layer in layers_to_prune):
                        weight_flat = module.weight.data.abs().flatten()
                        layer_weights[name] = module.weight
                        all_weights.append(weight_flat)
                        pruning_stats["total_parameters"] += weight_flat.numel()

            if not all_weights:
                logger.warning("No suitable layers found for pruning")
                return model, pruning_stats

            # Calculate pruning threshold
            if global_pruning:
                all_weights_tensor = torch.cat(all_weights)
                threshold = torch.quantile(all_weights_tensor, sparsity_ratio)
                logger.info(f"Global pruning threshold: {threshold:.6f}")
            else:
                threshold = None

            # Apply pruning to each layer
            for name, weight in layer_weights.items():
                if not global_pruning:
                    # Layer-wise threshold
                    layer_threshold = torch.quantile(weight.abs().flatten(), sparsity_ratio)
                else:
                    layer_threshold = threshold

                # Create mask
                mask = weight.abs() >= layer_threshold
                pruned_weight = weight * mask

                # Update weight
                with torch.no_grad():
                    weight.copy_(pruned_weight)

                # Count pruned parameters
                pruned_count = (mask == 0).sum().item()
                pruning_stats["pruned_parameters"] += pruned_count
                pruning_stats["layers_processed"] += 1

                logger.debug(f"Pruned {pruned_count}/{weight.numel()} weights in {name}")

            # Final statistics
            actual_sparsity = pruning_stats["pruned_parameters"] / pruning_stats["total_parameters"]
            pruning_stats["actual_sparsity_ratio"] = actual_sparsity

            logger.info(f"Magnitude pruning completed. "
                       f"Actual sparsity: {actual_sparsity:.1%} "
                       f"({pruning_stats['pruned_parameters']:,} / "
                       f"{pruning_stats['total_parameters']:,} parameters)")

            return model, pruning_stats

        except Exception as e:
            logger.error(f"Magnitude pruning failed: {e}")
            raise RuntimeError(f"Failed to apply magnitude pruning: {e}")


class SparsityManager:
    """
    High-level interface for various sparsity techniques.

    Provides unified interface for different pruning methods with
    automatic method selection based on hardware and requirements.
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize sparsity manager.

        Args:
            device: Target device for sparsity operations
        """
        self.device = device
        self.semi_structured_pruner = SemiStructuredSparsityPruner(device)
        self.unstructured_pruner = UnstructuredPruner()

    def apply_sparsity(self,
                      model_id: str,
                      model: Optional[nn.Module] = None,
                      sparsity_type: str = "2:4",
                      sparsity_ratio: float = 0.5,
                      layers_to_prune: Optional[List[str]] = None,
                      recovery_finetune: bool = True,
                      save_path: Optional[Path] = None) -> ModelArtifact:
        """
        Apply sparsity to a model with automatic method selection.

        Args:
            model_id: Model identifier
            model: Pre-loaded model (optional)
            sparsity_type: Type of sparsity ("2:4", "unstructured", "auto")
            sparsity_ratio: Target sparsity ratio
            layers_to_prune: Layers to prune (None = auto-detect)
            recovery_finetune: Whether model will undergo recovery fine-tuning
            save_path: Path to save sparse model

        Returns:
            ModelArtifact with sparsity metadata

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If sparsity application fails
        """
        logger.info(f"Applying {sparsity_type} sparsity to {model_id}")

        # Validate parameters
        if sparsity_type not in ["2:4", "unstructured", "auto"]:
            raise ValueError(f"Unsupported sparsity type: {sparsity_type}")

        if not (0.0 < sparsity_ratio < 1.0):
            raise ValueError(f"Sparsity ratio must be between 0 and 1, got {sparsity_ratio}")

        try:
            # Auto-select sparsity method
            if sparsity_type == "auto":
                if self.semi_structured_pruner.tensor_core_compatible and sparsity_ratio <= 0.5:
                    sparsity_type = "2:4"
                else:
                    sparsity_type = "unstructured"
                logger.info(f"Auto-selected sparsity type: {sparsity_type}")

            # Apply appropriate sparsity method
            if sparsity_type == "2:4":
                pruned_model, pruning_stats = self.semi_structured_pruner.apply_2to4_sparsity(
                    model, layers_to_prune, recovery_finetune=recovery_finetune
                )
            else:
                pruned_model, pruning_stats = self.unstructured_pruner.apply_magnitude_pruning(
                    model, sparsity_ratio, layers_to_prune
                )

            # Save sparse model if requested
            if save_path:
                if sparsity_type == "2:4":
                    self.semi_structured_pruner.export_sparse_model(pruned_model, save_path)
                else:
                    torch.save(pruned_model.state_dict(), save_path / "sparse_model.pt")

            # Create sparsity metadata
            sparsity_config = {
                "method": sparsity_type,
                "target_sparsity_ratio": sparsity_ratio,
                "layers_pruned": pruning_stats.get("layers_processed", 0),
                "total_parameters_pruned": pruning_stats.get("pruned_parameters", 0),
                "actual_sparsity_ratio": pruning_stats.get("actual_sparsity_ratio", 0),
                "tensor_core_compatible": pruning_stats.get("tensor_core_compatible", False),
                "recovery_finetune_planned": recovery_finetune,
                "pruning_stats": pruning_stats
            }

            # Create model artifact
            model_artifact = ModelArtifact(
                model_id=model_id,
                tokenizer_id=model_id,  # Assume same as model_id
                sparsity=sparsity_config,
                metadata={
                    "sparsity_method": sparsity_type,
                    "sparse": True,
                    "hardware_accelerated": sparsity_type == "2:4" and pruning_stats.get("tensor_core_compatible", False),
                    "training_compatible": True,
                    "inference_optimized": True
                }
            )

            logger.info(f"Sparsity application completed successfully. "
                       f"Method: {sparsity_type}, "
                       f"Sparsity: {sparsity_config['actual_sparsity_ratio']:.1%}")

            return model_artifact

        except Exception as e:
            logger.error(f"Sparsity application failed: {e}")
            raise RuntimeError(f"Failed to apply sparsity: {e}")


def apply_2to4_sparsity(model: nn.Module, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to apply 2:4 semi-structured sparsity.

    Args:
        model: Model to prune
        **kwargs: Additional arguments

    Returns:
        Tuple of (pruned_model, pruning_stats)
    """
    pruner = SemiStructuredSparsityPruner()
    return pruner.apply_2to4_sparsity(model, **kwargs)


def apply_magnitude_pruning(model: nn.Module, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to apply magnitude-based pruning.

    Args:
        model: Model to prune
        **kwargs: Additional arguments

    Returns:
        Tuple of (pruned_model, pruning_stats)
    """
    pruner = UnstructuredPruner()
    return pruner.apply_magnitude_pruning(model, **kwargs)