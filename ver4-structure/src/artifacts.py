"""
Core data models and artifacts for the GSM8K optimization workflow system.

These artifacts capture the state and metadata of models, datasets, and evaluations
throughout the optimization pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import hashlib
import json
import time


@dataclass
class ModelArtifact:
    """
    Represents a model artifact with all associated metadata for optimization.

    Attributes:
        model_id: HuggingFace model identifier
        tokenizer_id: HuggingFace tokenizer identifier
        quantization: Quantization configuration (BitsAndBytes, GPTQ, AWQ)
        sparsity: Pruning and sparsity configuration
        adapters_path: Path to LoRA/QLoRA adapter weights
        runtime: Runtime optimization settings (FlashAttention, vLLM)
        config_hash: Hash of the complete configuration for reproducibility
        created_at: Timestamp of creation
        metadata: Additional custom metadata
    """
    model_id: str
    tokenizer_id: str
    quantization: Optional[Dict[str, Any]] = None
    sparsity: Optional[Dict[str, Any]] = None
    adapters_path: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    config_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate config hash after initialization."""
        if self.config_hash is None:
            self.config_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate SHA256 hash of the configuration for reproducibility."""
        config_data = {
            'model_id': self.model_id,
            'tokenizer_id': self.tokenizer_id,
            'quantization': self.quantization,
            'sparsity': self.sparsity,
            'adapters_path': self.adapters_path,
            'runtime': self.runtime
        }
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_id': self.model_id,
            'tokenizer_id': self.tokenizer_id,
            'quantization': self.quantization,
            'sparsity': self.sparsity,
            'adapters_path': self.adapters_path,
            'runtime': self.runtime,
            'config_hash': self.config_hash,
            'created_at': self.created_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelArtifact':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DatasetArtifact:
    """
    Represents a dataset artifact for GSM8K with splits and augmentations.

    Attributes:
        dataset_name: Name of the dataset (e.g., "openai/gsm8k")
        train_split: Training data split configuration
        val_split: Validation data split configuration
        test_split: Test data split configuration
        augmentation_recipes: List of data augmentation recipe IDs applied
        prompts: Prompt templates and configurations
        total_samples: Total number of samples across all splits
        created_at: Timestamp of creation
        metadata: Additional custom metadata
    """
    dataset_name: str
    train_split: Dict[str, Any]
    val_split: Dict[str, Any]
    test_split: Dict[str, Any]
    augmentation_recipes: List[str] = field(default_factory=list)
    prompts: Dict[str, str] = field(default_factory=dict)
    total_samples: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_split_info(self) -> Dict[str, int]:
        """Get summary of split sizes."""
        return {
            'train': self.train_split.get('size', 0),
            'val': self.val_split.get('size', 0),
            'test': self.test_split.get('size', 0)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dataset_name': self.dataset_name,
            'train_split': self.train_split,
            'val_split': self.val_split,
            'test_split': self.test_split,
            'augmentation_recipes': self.augmentation_recipes,
            'prompts': self.prompts,
            'total_samples': self.total_samples,
            'created_at': self.created_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetArtifact':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvalArtifact:
    """
    Represents evaluation results with comprehensive metrics.

    Captures accuracy, performance, resource usage, and environmental impact
    metrics for a model evaluation run on GSM8K.

    Attributes:
        run_id: Unique identifier for this evaluation run
        model_artifact: Reference to the model that was evaluated
        dataset_artifact: Reference to the dataset used for evaluation
        accuracy: Exact match accuracy on extracted numeric answers
        latency_ms_p50: 50th percentile latency in milliseconds
        latency_ms_p90: 90th percentile latency in milliseconds
        latency_ms_p99: 99th percentile latency in milliseconds
        tokens_per_sec: Inference throughput in tokens per second
        vram_peak_mb: Peak VRAM usage in megabytes
        energy_kwh: Total energy consumption in kilowatt-hours
        co2_g: CO2 emissions in grams
        samples_evaluated: Number of samples evaluated
        errors_by_type: Breakdown of error types and counts
        created_at: Timestamp of evaluation
        duration_seconds: Total evaluation duration
        metadata: Additional custom metadata
    """
    run_id: str
    model_artifact: ModelArtifact
    dataset_artifact: DatasetArtifact
    accuracy: float
    latency_ms_p50: float
    latency_ms_p90: float
    latency_ms_p99: float = 0.0
    tokens_per_sec: float = 0.0
    vram_peak_mb: float = 0.0
    energy_kwh: float = 0.0
    co2_g: float = 0.0
    samples_evaluated: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_efficiency_score(self) -> float:
        """
        Calculate a composite efficiency score balancing accuracy and resource usage.
        Higher is better.
        """
        if self.latency_ms_p50 == 0 or self.vram_peak_mb == 0:
            return 0.0

        # Normalize components (accuracy up, latency down, vram down, co2 down)
        acc_component = self.accuracy
        latency_component = max(0, 1 - (self.latency_ms_p50 / 10000))  # Normalize to ~10s max
        vram_component = max(0, 1 - (self.vram_peak_mb / 80000))  # Normalize to ~80GB max
        co2_component = max(0, 1 - (self.co2_g / 1000))  # Normalize to ~1kg max

        # Weighted combination
        return (acc_component * 0.4 +
                latency_component * 0.25 +
                vram_component * 0.25 +
                co2_component * 0.1)

    def get_pareto_objectives(self) -> Dict[str, float]:
        """Get the multi-objective optimization targets."""
        return {
            'accuracy': self.accuracy,  # Maximize
            'latency_ms': self.latency_ms_p50,  # Minimize
            'vram_mb': self.vram_peak_mb,  # Minimize
            'co2_g': self.co2_g  # Minimize
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'model_artifact': self.model_artifact.to_dict(),
            'dataset_artifact': self.dataset_artifact.to_dict(),
            'accuracy': self.accuracy,
            'latency_ms_p50': self.latency_ms_p50,
            'latency_ms_p90': self.latency_ms_p90,
            'latency_ms_p99': self.latency_ms_p99,
            'tokens_per_sec': self.tokens_per_sec,
            'vram_peak_mb': self.vram_peak_mb,
            'energy_kwh': self.energy_kwh,
            'co2_g': self.co2_g,
            'samples_evaluated': self.samples_evaluated,
            'errors_by_type': self.errors_by_type,
            'created_at': self.created_at,
            'duration_seconds': self.duration_seconds,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvalArtifact':
        """Create from dictionary."""
        model_artifact = ModelArtifact.from_dict(data.pop('model_artifact'))
        dataset_artifact = DatasetArtifact.from_dict(data.pop('dataset_artifact'))
        return cls(model_artifact=model_artifact, dataset_artifact=dataset_artifact, **data)


@dataclass
class ExperimentConfig:
    """
    Configuration for a complete experiment run.

    Defines the pipeline stages, hyperparameter search space, objectives,
    and resource constraints for an optimization experiment.
    """
    name: str
    description: str
    base_model: str
    dataset_config: Dict[str, Any]
    pipeline_stages: List[str]
    hpo_config: Dict[str, Any]
    objectives: List[str]
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'base_model': self.base_model,
            'dataset_config': self.dataset_config,
            'pipeline_stages': self.pipeline_stages,
            'hpo_config': self.hpo_config,
            'objectives': self.objectives,
            'resource_limits': self.resource_limits,
            'created_at': self.created_at,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)