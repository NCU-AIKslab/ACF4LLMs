"""
Knowledge Distillation Sub-Agent

Specialized sub-agent for training smaller student models from larger teachers.
"""

import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class SetupDistillationInput(BaseModel):
    """Input schema for distillation setup."""

    teacher_model: str = Field(
        description="Teacher model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    student_model: Optional[str] = Field(
        default=None,
        description="Student model name. If None, creates a smaller version of teacher."
    )
    student_scale: float = Field(
        default=0.5,
        description="Scale factor for student model (0.25 = 25% of teacher size, 0.5 = 50%)"
    )
    temperature: float = Field(
        default=2.0,
        description="Temperature for soft labels (typical range: 2-4)"
    )
    alpha: float = Field(
        default=0.7,
        description="Weight for KD loss vs hard loss (0.5-0.9). Higher = more focus on teacher."
    )


class SetupDistillationTool(BaseTool):
    """Tool to set up teacher-student distillation configuration."""

    name: str = "setup_distillation"
    description: str = """
    Configure knowledge distillation between teacher and student models.

    This tool helps you set up the distillation pipeline by:
    1. Specifying teacher and student models
    2. Configuring KD hyperparameters (temperature, alpha)
    3. Estimating compression ratio

    Example input:
    {
        "teacher_model": "meta-llama/Llama-2-7b-hf",
        "student_scale": 0.5,
        "temperature": 2.0,
        "alpha": 0.7
    }
    """
    args_schema: type[BaseModel] = SetupDistillationInput

    def _estimate_model_size(self, model_name: str) -> int:
        """Estimate model size in millions of parameters."""
        model_sizes = {
            "7b": 7000,
            "13b": 13000,
            "70b": 70000,
            "1b": 1000,
            "3b": 3000,
        }

        for key, size in model_sizes.items():
            if key in model_name.lower():
                return size

        return 1000  # Default

    def _run(
        self,
        teacher_model: str,
        student_model: Optional[str] = None,
        student_scale: float = 0.5,
        temperature: float = 2.0,
        alpha: float = 0.7,
    ) -> str:
        """Execute the tool to setup distillation."""
        try:
            # Estimate sizes
            teacher_size = self._estimate_model_size(teacher_model)
            student_size = teacher_size * student_scale if student_model is None else self._estimate_model_size(student_model)

            compression_ratio = teacher_size / student_size

            # Estimate accuracy impact based on compression ratio
            if compression_ratio < 2:
                expected_accuracy_loss = "1-2%"
            elif compression_ratio < 4:
                expected_accuracy_loss = "2-5%"
            elif compression_ratio < 10:
                expected_accuracy_loss = "5-10%"
            else:
                expected_accuracy_loss = ">10%"

            config = {
                "teacher": {
                    "model_name": teacher_model,
                    "size_m": teacher_size,
                },
                "student": {
                    "model_name": student_model or f"{teacher_model}-distilled-{int(student_scale*100)}pct",
                    "size_m": int(student_size),
                    "scale": student_scale,
                },
                "hyperparameters": {
                    "temperature": temperature,
                    "alpha": alpha,
                    "kd_loss_weight": alpha,
                    "hard_loss_weight": 1 - alpha,
                },
                "compression_ratio": round(compression_ratio, 2),
                "expected_accuracy_loss": expected_accuracy_loss,
                "estimated_speedup": f"{round(compression_ratio * 0.8, 1)}x",  # Conservative estimate
                "estimated_memory_reduction": f"{round((1 - 1/compression_ratio) * 100, 1)}%",
                "status": "success",
                "note": "Configuration ready. Use train_distillation to start training.",
            }

            return json.dumps(config, indent=2)

        except Exception as e:
            return f"✗ Failed to setup distillation: {str(e)}"


class EstimateDistillationInput(BaseModel):
    """Input schema for distillation estimation."""

    teacher_model: str = Field(
        description="Teacher model name"
    )
    compression_ratio: float = Field(
        default=2.0,
        description="Target compression ratio (e.g., 2.0 = student is 50% of teacher)"
    )


class EstimateDistillationTool(BaseTool):
    """Tool to estimate distillation impact without running training."""

    name: str = "estimate_distillation"
    description: str = """
    Estimate the impact of knowledge distillation based on compression ratio.

    This provides quick estimates without loading models or running training.

    Example input:
    {
        "teacher_model": "meta-llama/Llama-2-7b-hf",
        "compression_ratio": 2.0
    }
    """
    args_schema: type[BaseModel] = EstimateDistillationInput

    def _run(
        self,
        teacher_model: str,
        compression_ratio: float = 2.0,
    ) -> str:
        """Execute the tool to estimate distillation."""
        try:
            # Get teacher size
            model_sizes = {
                "7b": 7000,
                "13b": 13000,
                "70b": 70000,
            }

            teacher_size = 1000  # Default
            for key, size in model_sizes.items():
                if key in teacher_model.lower():
                    teacher_size = size
                    break

            student_size = teacher_size / compression_ratio

            # Empirical estimates based on distillation literature
            if compression_ratio <= 1.5:
                accuracy_loss = 0.5
                speedup = 1.3
            elif compression_ratio <= 2.0:
                accuracy_loss = 1.5
                speedup = 1.8
            elif compression_ratio <= 3.0:
                accuracy_loss = 3.0
                speedup = 2.5
            elif compression_ratio <= 5.0:
                accuracy_loss = 5.0
                speedup = 4.0
            else:
                accuracy_loss = 8.0
                speedup = 6.0

            memory_reduction = (1 - 1/compression_ratio) * 100

            result = {
                "teacher_model": teacher_model,
                "teacher_size_m": teacher_size,
                "student_size_m": int(student_size),
                "compression_ratio": compression_ratio,
                "estimated_metrics": {
                    "accuracy_loss_percent": round(accuracy_loss, 1),
                    "speedup": f"{round(speedup, 1)}x",
                    "memory_reduction_percent": round(memory_reduction, 1),
                    "parameter_reduction_percent": round((1 - 1/compression_ratio) * 100, 1),
                },
                "training_requirements": {
                    "training_data": "10K-100K samples recommended",
                    "training_time": "Hours to days depending on data and model size",
                    "gpu_memory": f"~{int(teacher_size * 0.002 + student_size * 0.002)} GB (both models in memory)",
                },
                "recommendations": {
                    "temperature": 2.0 if compression_ratio < 3 else 3.0,
                    "alpha": 0.7 if compression_ratio < 3 else 0.8,
                    "training_epochs": 3 if compression_ratio < 3 else 5,
                },
                "status": "success",
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"✗ Failed to estimate distillation: {str(e)}"


class DistillationLoss(nn.Module):
    """Custom loss for knowledge distillation."""

    def __init__(self, temperature: float = 2.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels

        Returns:
            Combined distillation loss
        """
        # Soft loss (KD loss)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        # Hard loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, labels)

        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return loss


class CompareModelsInput(BaseModel):
    """Input schema for model comparison."""

    teacher_model: str = Field(
        description="Teacher model name"
    )
    student_model: str = Field(
        description="Student model name or path"
    )


class CompareModelsTool(BaseTool):
    """Tool to compare teacher and student models."""

    name: str = "compare_models"
    description: str = """
    Compare teacher and student models side-by-side.

    This tool loads both models and reports their sizes, architectures, and differences.

    Example input:
    {
        "teacher_model": "meta-llama/Llama-2-7b-hf",
        "student_model": "path/to/student/model"
    }
    """
    args_schema: type[BaseModel] = CompareModelsInput

    def _run(
        self,
        teacher_model: str,
        student_model: str,
    ) -> str:
        """Execute the tool to compare models."""
        try:
            # Load configs (lightweight)
            teacher_config = AutoConfig.from_pretrained(teacher_model)
            try:
                student_config = AutoConfig.from_pretrained(student_model)
            except:
                return f"✗ Student model not found at '{student_model}'. Please train the student model first."

            # Get parameter counts
            def count_params(config):
                # Rough estimate based on config
                vocab_size = config.vocab_size
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                intermediate_size = config.intermediate_size

                # Embedding: vocab_size * hidden_size
                embedding_params = vocab_size * hidden_size

                # Per layer: attention + FFN
                attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O
                ffn_params = 2 * hidden_size * intermediate_size  # Up, Down
                layer_params = attention_params + ffn_params

                total = embedding_params + num_layers * layer_params
                return total / 1e6  # Convert to millions

            teacher_params = count_params(teacher_config)
            student_params = count_params(student_config)

            compression_ratio = teacher_params / student_params
            param_reduction = (1 - 1/compression_ratio) * 100

            comparison = {
                "teacher": {
                    "model_name": teacher_model,
                    "params_m": round(teacher_params, 1),
                    "num_layers": teacher_config.num_hidden_layers,
                    "hidden_size": teacher_config.hidden_size,
                    "num_attention_heads": teacher_config.num_attention_heads,
                },
                "student": {
                    "model_name": student_model,
                    "params_m": round(student_params, 1),
                    "num_layers": student_config.num_hidden_layers,
                    "hidden_size": student_config.hidden_size,
                    "num_attention_heads": student_config.num_attention_heads,
                },
                "comparison": {
                    "compression_ratio": round(compression_ratio, 2),
                    "parameter_reduction_percent": round(param_reduction, 1),
                    "layer_ratio": f"{student_config.num_hidden_layers}/{teacher_config.num_hidden_layers}",
                    "hidden_size_ratio": f"{student_config.hidden_size}/{teacher_config.hidden_size}",
                },
                "status": "success",
            }

            return json.dumps(comparison, indent=2)

        except Exception as e:
            return f"✗ Failed to compare models: {str(e)}"


def create_distillation_tools():
    """
    Factory function to create all distillation-related tools.

    Returns:
        List of distillation tools for LangChain agent
    """
    return [
        SetupDistillationTool(),
        EstimateDistillationTool(),
        CompareModelsTool(),
    ]
