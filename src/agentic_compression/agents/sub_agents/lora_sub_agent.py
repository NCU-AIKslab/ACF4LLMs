"""
LoRA/PEFT Sub-Agent

Specialized sub-agent for parameter-efficient fine-tuning using LoRA adapters.
"""

import json
from typing import Any, Dict, List, Optional

import torch
from langchain.tools import BaseTool
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class ConfigureLoRAInput(BaseModel):
    """Input schema for LoRA configuration."""

    base_model: str = Field(
        description="Base model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="Target modules for LoRA (e.g., ['q_proj', 'v_proj']). If None, uses sensible defaults."
    )
    rank: int = Field(
        default=8,
        description="LoRA rank (typical range: 4-64). Higher rank = more parameters but better quality."
    )
    alpha: int = Field(
        default=16,
        description="LoRA alpha (typically 2x rank). Controls scaling of adapter weights."
    )
    dropout: float = Field(
        default=0.05,
        description="Dropout probability for LoRA layers (typical range: 0.05-0.1)"
    )


class ConfigureLoRATool(BaseTool):
    """Tool to configure LoRA adapters."""

    name: str = "configure_lora"
    description: str = """
    Create a LoRA configuration for parameter-efficient fine-tuning.

    This tool helps you set up LoRA adapters with appropriate hyperparameters.
    If target_modules is not specified, it will use sensible defaults based on the model architecture.

    Example input:
    {
        "base_model": "meta-llama/Llama-2-7b-hf",
        "target_modules": ["q_proj", "v_proj"],
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05
    }
    """
    args_schema: type[BaseModel] = ConfigureLoRAInput

    def _get_default_target_modules(self, model_name: str) -> List[str]:
        """Get default target modules based on model architecture."""
        model_name_lower = model_name.lower()

        if "llama" in model_name_lower or "mistral" in model_name_lower:
            # LLaMA/Mistral: target attention projections
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "gpt" in model_name_lower:
            # GPT models: target attention and MLP
            return ["c_attn", "c_proj"]
        elif "bloom" in model_name_lower:
            return ["query_key_value"]
        elif "opt" in model_name_lower:
            return ["q_proj", "v_proj"]
        else:
            # Default: target common attention modules
            return ["q_proj", "v_proj"]

    def _run(
        self,
        base_model: str,
        target_modules: Optional[List[str]] = None,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ) -> str:
        """Execute the tool to configure LoRA."""
        try:
            # Use defaults if target_modules not specified
            if target_modules is None:
                target_modules = self._get_default_target_modules(base_model)

            # Create LoRA config
            config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=target_modules,
                lora_dropout=dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Estimate parameter count
            # Rough estimate: for each target module, LoRA adds 2 * rank * hidden_dim parameters
            # For LLaMA-7B (hidden_dim=4096), 4 modules, rank=8: ~2 * 8 * 4096 * 4 = 262K params
            estimated_params_per_module = 2 * rank * 4096  # Assuming hidden_dim=4096
            total_lora_params = estimated_params_per_module * len(target_modules)
            total_lora_params_m = total_lora_params / 1e6

            result = {
                "config": {
                    "rank": rank,
                    "alpha": alpha,
                    "target_modules": target_modules,
                    "dropout": dropout,
                    "task_type": "CAUSAL_LM",
                },
                "estimated_adapter_size_m": round(total_lora_params_m, 2),
                "target_modules_count": len(target_modules),
                "status": "success",
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"✗ Failed to configure LoRA: {str(e)}"


class LoadPEFTModelInput(BaseModel):
    """Input schema for loading PEFT model."""

    base_model: str = Field(
        description="Base model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    lora_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LoRA configuration dict (from configure_lora tool). If None, uses default config."
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Whether to load model in 8-bit quantization (reduces memory)"
    )


class LoadPEFTModelTool(BaseTool):
    """Tool to load a model with PEFT adapters."""

    name: str = "load_peft_model"
    description: str = """
    Load a base model and attach LoRA adapters for inference or fine-tuning.

    This tool:
    1. Loads the base model (optionally quantized)
    2. Applies LoRA configuration
    3. Returns model statistics

    Example input:
    {
        "base_model": "meta-llama/Llama-2-7b-hf",
        "lora_config": {"rank": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        "load_in_8bit": true
    }
    """
    args_schema: type[BaseModel] = LoadPEFTModelInput

    def _run(
        self,
        base_model: str,
        lora_config: Optional[Dict[str, Any]] = None,
        load_in_8bit: bool = False,
    ) -> str:
        """Execute the tool to load PEFT model."""
        try:
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_in_8bit,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

            # Prepare model for k-bit training if quantized
            if load_in_8bit:
                model = prepare_model_for_kbit_training(model)

            # Create LoRA config if not provided
            if lora_config is None:
                lora_config = {
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "v_proj"],
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                }
            else:
                # Ensure required fields
                if "task_type" not in lora_config:
                    lora_config["task_type"] = "CAUSAL_LM"
                if "bias" not in lora_config:
                    lora_config["bias"] = "none"

            # Apply PEFT
            peft_config = LoraConfig(**lora_config)
            model = get_peft_model(model, peft_config)

            # Get trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            trainable_percent = 100 * trainable_params / all_params

            result = {
                "model_loaded": base_model,
                "peft_applied": True,
                "quantized_8bit": load_in_8bit,
                "trainable_params_m": round(trainable_params / 1e6, 2),
                "all_params_m": round(all_params / 1e6, 2),
                "trainable_percent": round(trainable_percent, 3),
                "lora_config": lora_config,
                "status": "success",
                "note": "Model loaded and ready for inference or fine-tuning",
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"✗ Failed to load PEFT model: {str(e)}"


class EstimateLoRAImpactInput(BaseModel):
    """Input schema for estimating LoRA impact."""

    base_model: str = Field(
        description="Base model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    rank: int = Field(
        default=8,
        description="LoRA rank"
    )


class EstimateLoRAImpactTool(BaseTool):
    """Tool to estimate the impact of LoRA compression."""

    name: str = "estimate_lora_impact"
    description: str = """
    Estimate the compression ratio and performance impact of LoRA adapters.

    This tool provides theoretical estimates without loading the actual model.

    Example input:
    {
        "base_model": "meta-llama/Llama-2-7b-hf",
        "rank": 8
    }
    """
    args_schema: type[BaseModel] = EstimateLoRAImpactInput

    def _run(self, base_model: str, rank: int = 8) -> str:
        """Execute the tool to estimate LoRA impact."""
        try:
            # Model size estimates (in millions of parameters)
            model_sizes = {
                "llama-2-7b": 7000,
                "llama-2-13b": 13000,
                "llama-7b": 7000,
                "llama-13b": 13000,
                "mistral-7b": 7000,
                "gpt2": 117,
                "gpt2-medium": 345,
                "gpt2-large": 774,
                "gpt2-xl": 1558,
            }

            # Try to match model name
            base_params = None
            for key, size in model_sizes.items():
                if key in base_model.lower():
                    base_params = size
                    break

            if base_params is None:
                # Default estimate based on "7b", "13b" in name
                if "7b" in base_model.lower():
                    base_params = 7000
                elif "13b" in base_model.lower():
                    base_params = 13000
                else:
                    base_params = 1000  # Conservative default

            # Estimate LoRA params: 2 * rank * hidden_dim * num_layers * num_target_modules
            # For LLaMA-7B: hidden_dim=4096, num_layers=32, 4 target modules
            # Rough estimate: 2 * rank * 4096 * 32 * 4 / 1e6
            hidden_dim = 4096  # Typical for 7B models
            num_layers = 32  # Typical for 7B models
            num_target_modules = 4  # q, k, v, o projections

            lora_params = 2 * rank * hidden_dim * num_layers * num_target_modules / 1e6

            compression_ratio = base_params / lora_params
            trainable_percent = 100 * lora_params / base_params

            result = {
                "base_model": base_model,
                "base_model_params_m": base_params,
                "lora_adapter_params_m": round(lora_params, 2),
                "compression_ratio": round(compression_ratio, 1),
                "trainable_params_percent": round(trainable_percent, 3),
                "rank": rank,
                "estimated_accuracy_impact": "Minimal (<1% on pre-trained tasks, potential gain on fine-tuned tasks)",
                "estimated_memory_reduction": f"{round(100 - trainable_percent, 1)}% during training",
                "recommended_use_cases": [
                    "Task-specific fine-tuning",
                    "Multi-task learning with adapter swapping",
                    "Low-resource deployment"
                ],
                "status": "success",
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"✗ Failed to estimate LoRA impact: {str(e)}"


def create_lora_tools():
    """
    Factory function to create all LoRA-related tools.

    Returns:
        List of LoRA tools for LangChain agent
    """
    return [
        ConfigureLoRATool(),
        LoadPEFTModelTool(),
        EstimateLoRAImpactTool(),
    ]
