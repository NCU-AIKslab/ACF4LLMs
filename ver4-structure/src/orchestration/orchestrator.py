"""
Core orchestrator for GSM8K optimization workflow system.

Coordinates the execution of optimization pipelines, manages model artifacts,
and provides the main entry point for running complete optimization workflows.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import RecipeConfig, load_recipe
from ..artifacts import ModelArtifact, DatasetArtifact, EvalArtifact
from ..eval.gsm8k_data import GSM8KDataLoader
from ..eval.gsm8k_eval import GSM8KEvaluator
from ..stages.quantize_bnb import BnBQuantizer
from ..stages.quantize_gptq import GPTQQuantizer
from ..stages.quantize_awq import AWQQuantizer
from ..stages.prune_sparsity import SparsityManager
from ..monitor.metrics_collector import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


class PipelineExecutionError(Exception):
    """Raised when pipeline execution fails."""
    pass


class ModelOrchestrator:
    """
    Core orchestrator for the GSM8K optimization workflow system.

    Manages the execution of optimization pipelines, coordinates between
    different optimization stages, and handles model artifacts.
    """

    def __init__(self,
                 device: Optional[str] = None,
                 cache_dir: Optional[Path] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize the model orchestrator.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            cache_dir: Directory for caching models and artifacts
            metrics_collector: Metrics collector instance
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir or Path("./model_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_collector = metrics_collector or get_metrics_collector()

        # Initialize stage processors
        self._quantizers = {
            'quantize_bnb': BnBQuantizer(),
            'quantize_gptq': GPTQQuantizer(),
            'quantize_awq': AWQQuantizer()
        }

        self._optimizers = {
            'prune_sparsity': SparsityManager()
        }

        # Pipeline state
        self._current_model = None
        self._current_tokenizer = None
        self._pipeline_artifacts = []

        logger.info(f"ModelOrchestrator initialized - Device: {self.device}")

    def execute_recipe(self,
                      recipe_name: str,
                      base_model_id: str,
                      num_eval_samples: int = 100,
                      override_params: Optional[Dict[str, Any]] = None) -> EvalArtifact:
        """
        Execute a complete optimization recipe.

        Args:
            recipe_name: Name of recipe to execute
            base_model_id: Base model identifier
            num_eval_samples: Number of samples for evaluation
            override_params: Parameters to override in recipe

        Returns:
            EvalArtifact with complete evaluation results
        """
        logger.info(f"Starting recipe execution: {recipe_name}")
        logger.info(f"Base model: {base_model_id}")
        logger.info(f"Evaluation samples: {num_eval_samples}")

        start_time = time.time()

        try:
            # Load recipe configuration
            recipe = load_recipe(recipe_name)
            if override_params:
                recipe = self._apply_overrides(recipe, override_params)

            logger.info(f"Recipe loaded: {recipe.name}")
            logger.info(f"Pipeline stages: {recipe.get_enabled_stages()}")

            # Initialize metrics tracking
            try:
                with self.metrics_collector.track_emissions():
                    # Step 1: Load base model and tokenizer
                    model_artifact = self._load_base_model(base_model_id, recipe)

                    # Step 2: Execute optimization pipeline
                    optimized_model_artifact = self._execute_pipeline(model_artifact, recipe)

                    # Step 3: Load and prepare evaluation dataset
                    dataset_artifact = self._prepare_dataset(recipe, num_eval_samples)

                    # Step 4: Run evaluation
                    eval_artifact = self._run_evaluation(
                        optimized_model_artifact,
                        dataset_artifact,
                        recipe,
                        num_eval_samples
                    )
            except AttributeError:
                # Fallback if metrics collector doesn't have track_emissions properly configured
                logger.warning("Metrics collector track_emissions not available, proceeding without carbon tracking")
                # Step 1: Load base model and tokenizer
                model_artifact = self._load_base_model(base_model_id, recipe)

                # Step 2: Execute optimization pipeline
                optimized_model_artifact = self._execute_pipeline(model_artifact, recipe)

                # Step 3: Load and prepare evaluation dataset
                dataset_artifact = self._prepare_dataset(recipe, num_eval_samples)

                # Step 4: Run evaluation
                eval_artifact = self._run_evaluation(
                    optimized_model_artifact,
                    dataset_artifact,
                    recipe,
                    num_eval_samples
                )

        except Exception as e:
            logger.error(f"Recipe execution failed: {e}")
            raise PipelineExecutionError(f"Recipe execution failed: {e}")

        execution_time = time.time() - start_time
        logger.info(f"Recipe execution completed in {execution_time:.2f} seconds")

        # Add execution metadata
        eval_artifact.metadata.update({
            'recipe_name': recipe_name,
            'base_model_id': base_model_id,
            'execution_time_seconds': execution_time,
            'device_used': self.device
        })

        return eval_artifact

    def _load_base_model(self, model_id: str, recipe: RecipeConfig) -> ModelArtifact:
        """Load base model and tokenizer."""
        logger.info(f"Loading base model: {model_id}")

        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Check if any quantization stages are present
            quantization_stages = [s for s in recipe.get_enabled_stages()
                                 if s.startswith('quantize_')]

            if not quantization_stages:
                # Load full precision model
                logger.info("Loading full precision model")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    trust_remote_code=True
                )
            else:
                # Model will be loaded by quantization stage
                logger.info("Model will be loaded by quantization stage")
                model = None

            self._current_model = model
            self._current_tokenizer = tokenizer

            # Create model artifact
            model_artifact = ModelArtifact(
                model_id=model_id,
                tokenizer_id=model_id,
                metadata={
                    'base_model': True,
                    'quantization_pending': len(quantization_stages) > 0,
                    'precision': 'float16' if self.device == 'cuda' else 'float32',
                    'cache_path': str(self.cache_dir / f"{model_id.replace('/', '_')}_base")
                }
            )

            if self.device == 'cuda' and model is not None:
                memory_used = torch.cuda.memory_allocated() / 1024**2
                model_artifact.metadata['gpu_memory_mb'] = memory_used
                logger.info(f"GPU Memory usage: {memory_used:.1f} MB")

            return model_artifact

        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise

    def _execute_pipeline(self, model_artifact: ModelArtifact, recipe: RecipeConfig) -> ModelArtifact:
        """Execute the optimization pipeline stages."""
        logger.info("Executing optimization pipeline")

        current_artifact = model_artifact
        enabled_stages = recipe.get_enabled_stages()

        for stage_name in enabled_stages:
            stage_config = recipe.get_stage_config(stage_name)
            if not stage_config:
                logger.warning(f"No configuration found for stage: {stage_name}")
                continue

            logger.info(f"Executing stage: {stage_name}")
            start_time = time.time()

            try:
                if stage_name in self._quantizers:
                    # Handle quantization stages
                    current_artifact = self._apply_quantization(
                        current_artifact, stage_name, stage_config
                    )
                elif stage_name in self._optimizers:
                    # Handle optimization stages
                    current_artifact = self._apply_optimization(
                        current_artifact, stage_name, stage_config
                    )
                else:
                    logger.warning(f"Unknown pipeline stage: {stage_name}")
                    continue

                stage_time = time.time() - start_time
                logger.info(f"Stage {stage_name} completed in {stage_time:.2f}s")

                # Update artifact metadata
                current_artifact.metadata[f'{stage_name}_applied'] = True
                current_artifact.metadata[f'{stage_name}_time_seconds'] = stage_time

            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                # Depending on configuration, either fail or continue
                if recipe.resource_limits.get('fail_on_stage_error', True):
                    raise PipelineExecutionError(f"Pipeline stage {stage_name} failed: {e}")
                else:
                    logger.warning(f"Continuing despite stage failure: {stage_name}")

        return current_artifact

    def _apply_quantization(self,
                          model_artifact: ModelArtifact,
                          stage_name: str,
                          stage_config: Dict[str, Any]) -> ModelArtifact:
        """Apply quantization stage."""
        quantizer = self._quantizers[stage_name]

        if stage_name == 'quantize_bnb':
            # Apply BitsAndBytes quantization
            model, tokenizer = quantizer.apply_bnb_4bit_with_model(
                model_id=model_artifact.model_id,
                quant_type=stage_config.get('quant_type', 'nf4'),
                compute_dtype=stage_config.get('compute_dtype', 'float16'),
                use_double_quant=stage_config.get('bnb_4bit_use_double_quant', True),
                device=self.device
            )

        elif stage_name == 'quantize_gptq':
            # Apply GPTQ quantization
            model, tokenizer = quantizer.apply_gptq_quantization_with_model(
                model_id=model_artifact.model_id,
                bits=stage_config.get('bits', 4),
                group_size=stage_config.get('group_size', 128),
                desc_act=stage_config.get('desc_act', True),
                device=self.device
            )

        elif stage_name == 'quantize_awq':
            # Apply AWQ quantization
            model, tokenizer = quantizer.apply_awq_quantization_with_model(
                model_id=model_artifact.model_id,
                bits=stage_config.get('bits', 4),
                group_size=stage_config.get('group_size', 128),
                zero_point=stage_config.get('zero_point', True),
                device=self.device
            )
        else:
            raise ValueError(f"Unknown quantization stage: {stage_name}")

        # Update current model and tokenizer
        self._current_model = model
        self._current_tokenizer = tokenizer

        # Create updated artifact
        updated_artifact = ModelArtifact(
            model_id=model_artifact.model_id,
            tokenizer_id=model_artifact.tokenizer_id,
            metadata={
                **model_artifact.metadata,
                f'{stage_name}_applied': True,
                'quantization_method': stage_name.replace('quantize_', ''),
                'quantization_config': stage_config
            }
        )

        # Update memory usage if on CUDA
        if self.device == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**2
            updated_artifact.metadata['gpu_memory_mb'] = memory_used
            logger.info(f"Post-quantization GPU Memory: {memory_used:.1f} MB")

        return updated_artifact

    def _apply_optimization(self,
                          model_artifact: ModelArtifact,
                          stage_name: str,
                          stage_config: Dict[str, Any]) -> ModelArtifact:
        """Apply optimization stage (pruning, etc.)."""
        optimizer = self._optimizers[stage_name]

        if stage_name == 'prune_sparsity':
            # Apply sparsity optimization
            model, sparsity_info = optimizer.apply_sparsity(
                model=self._current_model,
                sparsity_type=stage_config.get('sparsity_type', '2:4'),
                sparsity_ratio=stage_config.get('sparsity_ratio', 0.5),
                calibration_samples=stage_config.get('calibration_samples', 100)
            )
            self._current_model = model

        else:
            raise ValueError(f"Unknown optimization stage: {stage_name}")

        # Create updated artifact
        updated_artifact = ModelArtifact(
            model_id=model_artifact.model_id,
            tokenizer_id=model_artifact.tokenizer_id,
            metadata={
                **model_artifact.metadata,
                f'{stage_name}_applied': True,
                'optimization_method': stage_name,
                'optimization_config': stage_config
            }
        )

        return updated_artifact

    def _prepare_dataset(self, recipe: RecipeConfig, num_samples: int) -> DatasetArtifact:
        """Prepare evaluation dataset."""
        logger.info(f"Preparing GSM8K dataset ({num_samples} samples)")

        try:
            data_loader = GSM8KDataLoader()
            dataset_dict, dataset_artifact = data_loader.load_dataset(
                val_split_size=recipe.dataset.get("val_split_size", 500),
                augmentation_recipes=recipe.dataset.get("augmentation_recipes", []),
                random_seed=recipe.dataset.get("random_seed", 42)
            )

            # Select evaluation subset
            eval_dataset = dataset_dict["val"].select(range(min(num_samples, len(dataset_dict["val"]))))

            # Update artifact
            dataset_artifact.metadata.update({
                'eval_samples': num_samples,
                'actual_samples': len(eval_dataset)
            })

            return dataset_artifact

        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise

    def _run_evaluation(self,
                       model_artifact: ModelArtifact,
                       dataset_artifact: DatasetArtifact,
                       recipe: RecipeConfig,
                       num_samples: int) -> EvalArtifact:
        """Run complete model evaluation."""
        logger.info(f"Running evaluation with {num_samples} samples")

        if self._current_model is None or self._current_tokenizer is None:
            raise PipelineExecutionError("No model loaded for evaluation")

        try:
            # Create evaluator
            evaluator = GSM8KEvaluator()

            # Load actual GSM8K dataset
            from datasets import load_dataset
            gsm8k_dataset = load_dataset("openai/gsm8k", "main")
            eval_dataset = gsm8k_dataset["test"].select(range(min(num_samples, len(gsm8k_dataset["test"]))))

            # Run evaluation with current model
            eval_artifact = evaluator.evaluate(
                model=self._current_model,
                tokenizer=self._current_tokenizer,
                dataset=eval_dataset,
                model_artifact=model_artifact,
                dataset_artifact=dataset_artifact,
                batch_size=recipe.evaluation.get('batch_size', 1),
                max_new_tokens=recipe.decode.get('max_new_tokens', 200),
                temperature=recipe.decode.get('temperature', 0.1),
                top_p=recipe.decode.get('top_p', 0.9),
                num_samples=num_samples,
                run_id=f"orchestrator_{int(time.time())}"
            )

            logger.info(f"Evaluation completed - Accuracy: {eval_artifact.accuracy:.3f}")
            logger.info(f"Latency P50: {eval_artifact.latency_ms_p50:.1f}ms")
            logger.info(f"VRAM Peak: {eval_artifact.vram_peak_mb:.0f}MB")

            return eval_artifact

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise PipelineExecutionError(f"Evaluation failed: {e}")

    def _apply_overrides(self, recipe: RecipeConfig, overrides: Dict[str, Any]) -> RecipeConfig:
        """Apply parameter overrides to recipe."""
        # Implementation would merge overrides into recipe config
        # For now, return original recipe
        logger.info(f"Applying overrides: {list(overrides.keys())}")
        return recipe

    def cleanup(self):
        """Clean up resources."""
        if self._current_model is not None:
            del self._current_model
        if self._current_tokenizer is not None:
            del self._current_tokenizer
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        logger.info("Orchestrator cleanup completed")


def create_orchestrator(device: Optional[str] = None,
                       cache_dir: Optional[Path] = None) -> ModelOrchestrator:
    """
    Factory function to create a ModelOrchestrator instance.

    Args:
        device: Device to use for computation
        cache_dir: Directory for caching models

    Returns:
        ModelOrchestrator instance
    """
    return ModelOrchestrator(device=device, cache_dir=cache_dir)