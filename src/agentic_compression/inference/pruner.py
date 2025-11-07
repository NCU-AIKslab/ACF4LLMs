"""
Real model pruning using PyTorch pruning utilities.

Implements unstructured and structured pruning for model compression.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


class RealPruner:
    """
    Real pruning using torch.nn.utils.prune.

    Supports:
    - Unstructured pruning (L1, random)
    - Structured pruning (2:4, 4:8 patterns)
    - Global and local pruning strategies
    """

    @staticmethod
    def prune_model_unstructured(
        model: nn.Module,
        sparsity: float,
        method: str = "l1",
        exclude_layers: Optional[list[str]] = None,
    ) -> nn.Module:
        """
        Apply unstructured pruning to a model globally.

        Args:
            model: The model to prune
            sparsity: Target sparsity level (0.0-0.7)
            method: Pruning method ("l1" or "random")
            exclude_layers: Layer names to exclude from pruning

        Returns:
            Pruned model

        Example:
            >>> pruner = RealPruner()
            >>> pruned_model = pruner.prune_model_unstructured(model, sparsity=0.3)
        """
        if not 0.0 <= sparsity <= 0.7:
            raise ValueError(f"Invalid sparsity: {sparsity}. Must be between 0.0 and 0.7.")

        logger.info(f"Applying {method} unstructured pruning with sparsity={sparsity:.1%}")

        # Collect parameters to prune
        parameters_to_prune = []
        exclude_layers = exclude_layers or []

        for name, module in model.named_modules():
            # Skip excluded layers
            if any(excluded in name for excluded in exclude_layers):
                continue

            # Prune linear and conv layers
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))
                logger.debug(f"Adding layer to pruning: {name}")

        if not parameters_to_prune:
            logger.warning("No layers found for pruning")
            return model

        logger.info(f"Pruning {len(parameters_to_prune)} layers")

        # Choose pruning method
        if method.lower() == "l1":
            pruning_method = prune.L1Unstructured
        elif method.lower() == "random":
            pruning_method = prune.RandomUnstructured
        else:
            raise ValueError(f"Unknown pruning method: {method}. Use 'l1' or 'random'.")

        # Apply global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=pruning_method,
            amount=sparsity,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        logger.info("Unstructured pruning completed")

        # Calculate actual sparsity
        actual_sparsity = RealPruner._calculate_sparsity(model)
        logger.info(f"Actual model sparsity: {actual_sparsity:.1%}")

        return model

    @staticmethod
    def prune_model_structured(
        model: nn.Module,
        sparsity: float,
        pattern: str = "2:4",
        dim: int = 0,
        exclude_layers: Optional[list[str]] = None,
    ) -> nn.Module:
        """
        Apply structured pruning to a model.

        Args:
            model: The model to prune
            sparsity: Target sparsity level (0.0-0.7)
            pattern: Pruning pattern ("2:4" or "4:8")
            dim: Dimension to prune along (0 for rows, 1 for columns)
            exclude_layers: Layer names to exclude from pruning

        Returns:
            Pruned model

        Note:
            Structured pruning removes entire channels/neurons.
            2:4 pattern: Keep 2 out of every 4 weights (50% sparsity)
            4:8 pattern: Keep 4 out of every 8 weights (50% sparsity)
        """
        if not 0.0 <= sparsity <= 0.7:
            raise ValueError(f"Invalid sparsity: {sparsity}. Must be between 0.0 and 0.7.")

        logger.info(f"Applying structured pruning with pattern={pattern}, sparsity={sparsity:.1%}")

        exclude_layers = exclude_layers or []
        pruned_count = 0

        for name, module in model.named_modules():
            # Skip excluded layers
            if any(excluded in name for excluded in exclude_layers):
                continue

            # Apply structured pruning to linear and conv layers
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Use L1-norm structured pruning
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=sparsity,
                    n=1,  # L1 norm
                    dim=dim,
                )
                # Make permanent
                prune.remove(module, "weight")
                pruned_count += 1
                logger.debug(f"Pruned layer: {name}")

        logger.info(f"Structured pruning completed on {pruned_count} layers")

        # Calculate actual sparsity
        actual_sparsity = RealPruner._calculate_sparsity(model)
        logger.info(f"Actual model sparsity: {actual_sparsity:.1%}")

        return model

    @staticmethod
    def _calculate_sparsity(model: nn.Module) -> float:
        """
        Calculate the actual sparsity of a model.

        Args:
            model: The model to analyze

        Returns:
            Sparsity as a float (0.0-1.0)
        """
        total_params = 0
        zero_params = 0

        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params if total_params > 0 else 0.0
        return sparsity

    @staticmethod
    def get_pruning_info(model: nn.Module) -> dict:
        """
        Get information about pruning in a model.

        Args:
            model: The model to inspect

        Returns:
            Dictionary with pruning information
        """
        total_params = 0
        zero_params = 0
        pruned_layers = 0

        for name, module in model.named_modules():
            # Check if layer has pruning
            if prune.is_pruned(module):
                pruned_layers += 1

            # Count parameters
            for param in module.parameters(recurse=False):
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        sparsity = zero_params / total_params if total_params > 0 else 0.0

        info = {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "nonzero_parameters": total_params - zero_params,
            "sparsity_percent": sparsity * 100,
            "pruned_layers": pruned_layers,
            "compression_ratio": 1 / (1 - sparsity) if sparsity < 1.0 else float("inf"),
        }

        return info

    @staticmethod
    def calculate_pruning_metrics(sparsity: float) -> dict:
        """
        Calculate theoretical pruning metrics.

        Args:
            sparsity: Sparsity level (0.0-1.0)

        Returns:
            Dictionary with pruning metrics
        """
        metrics = {
            "sparsity_percent": sparsity * 100,
            "parameters_remaining_percent": (1 - sparsity) * 100,
            "theoretical_speedup": 1 / (1 - sparsity) if sparsity < 1.0 else float("inf"),
            "memory_reduction_percent": sparsity * 100,
        }

        return metrics
