"""
Configuration management system for GSM8K optimization workflow.

Provides utilities to load, validate, and manage YAML-based configuration
files for different optimization recipes and hyperparameter spaces.
"""

import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
import json

from .artifacts import ExperimentConfig


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class RecipeConfig:
    """
    Represents a complete optimization recipe configuration.
    """
    name: str
    description: str
    version: str
    base_model: str
    tokenizer_model: str
    targets: Dict[str, Any]
    pipeline: List[Dict[str, Any]]
    decode: Dict[str, Any]
    dataset: Dict[str, Any]
    evaluation: Dict[str, Any]
    resource_limits: Dict[str, Any]
    tracking: Dict[str, Any]
    hardware: Dict[str, Any] = field(default_factory=dict)
    vllm_server: Optional[Dict[str, Any]] = None
    memory_optimization: Optional[Dict[str, Any]] = None
    model_loading: Optional[Dict[str, Any]] = None
    benchmark: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecipeConfig':
        """Create RecipeConfig from dictionary."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'base_model': self.base_model,
            'tokenizer_model': self.tokenizer_model,
            'targets': self.targets,
            'pipeline': self.pipeline,
            'decode': self.decode,
            'dataset': self.dataset,
            'evaluation': self.evaluation,
            'resource_limits': self.resource_limits,
            'tracking': self.tracking,
            'hardware': self.hardware,
            'vllm_server': self.vllm_server,
            'memory_optimization': self.memory_optimization,
            'model_loading': self.model_loading,
            'benchmark': self.benchmark
        }

    def get_enabled_stages(self) -> List[str]:
        """Get list of enabled pipeline stages."""
        return [stage['stage'] for stage in self.pipeline if stage.get('enabled', True)]

    def get_stage_config(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific pipeline stage."""
        for stage in self.pipeline:
            if stage['stage'] == stage_name and stage.get('enabled', True):
                return stage.get('args', {})
        return None

    def validate(self) -> bool:
        """Validate the configuration."""
        required_fields = ['name', 'base_model', 'pipeline', 'decode', 'dataset']
        for field in required_fields:
            if not hasattr(self, field) or getattr(self, field) is None:
                raise ConfigError(f"Missing required field: {field}")

        # Validate pipeline stages
        valid_stages = {
            'quantize_bnb', 'quantize_gptq', 'quantize_awq',
            'prune_sparsity', 'distill_lora', 'kv_runtime', 'rag'
        }
        for stage in self.pipeline:
            if stage['stage'] not in valid_stages:
                raise ConfigError(f"Invalid pipeline stage: {stage['stage']}")

        # Validate objectives
        if 'objective_weights' in self.targets:
            valid_objectives = {'accuracy', 'latency', 'vram', 'co2', 'throughput'}
            for objective in self.targets['objective_weights']:
                if objective not in valid_objectives:
                    logger.warning(f"Unknown objective: {objective}")

        return True


@dataclass
class HPOConfig:
    """
    Represents hyperparameter optimization configuration.
    """
    decode: Dict[str, Any]
    lora: Dict[str, Any]
    quantization: Dict[str, Any]
    objectives: Dict[str, Any]
    study: Dict[str, Any]
    resource_constraints: Dict[str, Any]
    gptq: Optional[Dict[str, Any]] = None
    awq: Optional[Dict[str, Any]] = None
    sparsity: Optional[Dict[str, Any]] = None
    runtime: Optional[Dict[str, Any]] = None
    rag: Optional[Dict[str, Any]] = None
    data_augmentation: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    early_stopping: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HPOConfig':
        """Create HPOConfig from dictionary."""
        return cls(**data)

    def get_search_space(self, stage: str) -> Dict[str, Any]:
        """Get search space for a specific optimization stage."""
        stage_mapping = {
            'decode': self.decode,
            'lora': self.lora,
            'quantization': self.quantization,
            'gptq': self.gptq,
            'awq': self.awq,
            'sparsity': self.sparsity,
            'runtime': self.runtime,
            'rag': self.rag,
            'data_augmentation': self.data_augmentation,
            'evaluation': self.evaluation,
            'early_stopping': self.early_stopping
        }
        return stage_mapping.get(stage, {})

    def get_objectives(self) -> List[Dict[str, Any]]:
        """Get optimization objectives."""
        objectives = []
        if 'primary' in self.objectives:
            objectives.extend(self.objectives['primary'])
        if 'secondary' in self.objectives:
            objectives.extend(self.objectives['secondary'])
        return objectives


class ConfigManager:
    """
    Manages loading and validation of configuration files.
    """

    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._recipe_cache = {}
        self._hpo_cache = None

    def load_recipe(self, recipe_name: str) -> RecipeConfig:
        """
        Load a recipe configuration by name.

        Args:
            recipe_name: Name of the recipe (e.g., 'accuracy', 'latency')

        Returns:
            RecipeConfig object

        Raises:
            ConfigError: If recipe file not found or invalid
        """
        if recipe_name in self._recipe_cache:
            return self._recipe_cache[recipe_name]

        recipe_file = self.config_dir / f"recipe_{recipe_name}.yaml"
        if not recipe_file.exists():
            raise ConfigError(f"Recipe file not found: {recipe_file}")

        try:
            with open(recipe_file, 'r') as f:
                data = yaml.safe_load(f)

            recipe = RecipeConfig.from_dict(data)
            recipe.validate()

            self._recipe_cache[recipe_name] = recipe
            logger.info(f"Loaded recipe: {recipe.name}")
            return recipe

        except Exception as e:
            raise ConfigError(f"Failed to load recipe {recipe_name}: {e}")

    def load_hpo_config(self) -> HPOConfig:
        """
        Load hyperparameter optimization configuration.

        Returns:
            HPOConfig object

        Raises:
            ConfigError: If HPO config file not found or invalid
        """
        if self._hpo_cache:
            return self._hpo_cache

        hpo_file = self.config_dir / "hpo_spaces.yaml"
        if not hpo_file.exists():
            raise ConfigError(f"HPO config file not found: {hpo_file}")

        try:
            with open(hpo_file, 'r') as f:
                data = yaml.safe_load(f)

            hpo_config = HPOConfig.from_dict(data)
            self._hpo_cache = hpo_config
            logger.info("Loaded HPO configuration")
            return hpo_config

        except Exception as e:
            raise ConfigError(f"Failed to load HPO config: {e}")

    def list_recipes(self) -> List[str]:
        """
        List all available recipe configurations.

        Returns:
            List of recipe names
        """
        recipe_files = list(self.config_dir.glob("recipe_*.yaml"))
        return [f.stem.replace("recipe_", "") for f in recipe_files]

    def create_experiment_config(self, recipe_name: str, override_params: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
        """
        Create an experiment configuration from a recipe.

        Args:
            recipe_name: Name of the recipe to use
            override_params: Parameters to override in the recipe

        Returns:
            ExperimentConfig object
        """
        recipe = self.load_recipe(recipe_name)

        # Apply overrides if provided
        if override_params:
            recipe_dict = recipe.to_dict()
            recipe_dict = self._apply_overrides(recipe_dict, override_params)
            recipe = RecipeConfig.from_dict(recipe_dict)

        # Convert to experiment config
        experiment_config = ExperimentConfig(
            name=recipe.name,
            description=recipe.description,
            base_model=recipe.base_model,
            dataset_config=recipe.dataset,
            pipeline_stages=recipe.get_enabled_stages(),
            hpo_config=recipe.targets,
            objectives=list(recipe.targets.get('objective_weights', {}).keys()),
            resource_limits=recipe.resource_limits,
            metadata={
                'recipe_version': recipe.version,
                'decode_config': recipe.decode,
                'evaluation_config': recipe.evaluation,
                'hardware_config': recipe.hardware
            }
        )

        return experiment_config

    def _apply_overrides(self, config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter overrides to configuration dictionary.

        Args:
            config_dict: Original configuration
            overrides: Override parameters

        Returns:
            Updated configuration dictionary
        """
        def update_nested_dict(d: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in updates.items():
                if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                    d[key] = update_nested_dict(d[key], value)
                else:
                    d[key] = value
            return d

        return update_nested_dict(config_dict.copy(), overrides)

    def validate_config(self, config: Union[RecipeConfig, HPOConfig]) -> bool:
        """
        Validate a configuration object.

        Args:
            config: Configuration to validate

        Returns:
            True if valid

        Raises:
            ConfigError: If validation fails
        """
        if isinstance(config, RecipeConfig):
            return config.validate()
        else:
            # Basic HPO config validation
            required_sections = ['objectives', 'study']
            for section in required_sections:
                if not hasattr(config, section) or getattr(config, section) is None:
                    raise ConfigError(f"HPO config missing section: {section}")
            return True

    def export_config(self, config: Union[RecipeConfig, HPOConfig], output_path: Path) -> None:
        """
        Export configuration to YAML file.

        Args:
            config: Configuration to export
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(config, RecipeConfig):
            data = config.to_dict()
        else:
            data = config.__dict__

        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration exported to {output_path}")


# Global config manager instance
config_manager = ConfigManager()


def load_recipe(recipe_name: str) -> RecipeConfig:
    """Convenience function to load a recipe."""
    return config_manager.load_recipe(recipe_name)


def load_hpo_config() -> HPOConfig:
    """Convenience function to load HPO configuration."""
    return config_manager.load_hpo_config()


def list_recipes() -> List[str]:
    """Convenience function to list available recipes."""
    return config_manager.list_recipes()