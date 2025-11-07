"""
MLflow Experiment Tracking Tool for Deep Agent

Provides LangChain tools for automatic experiment logging.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class ExperimentLogInput(BaseModel):
    """Input schema for logging experiments."""

    config: Dict[str, Any] = Field(
        description="Compression configuration (e.g., {'technique': 'quantization', 'bits': 8})"
    )
    metrics: Dict[str, float] = Field(
        description="Evaluation metrics (e.g., {'accuracy': 0.654, 'latency_ms': 45.3, 'memory_mb': 3421})"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Base model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional tags for the experiment"
    )


class LogExperimentTool(BaseTool):
    """Tool to log compression experiments to MLflow."""

    name: str = "log_experiment"
    description: str = """
    Log a compression experiment to MLflow for tracking and reproducibility.

    Use this tool whenever you complete an experiment to record:
    - Configuration parameters (compression technique, hyperparameters)
    - Evaluation metrics (accuracy, latency, memory, energy, carbon)
    - Model information

    Example input:
    {
        "config": {"technique": "quantization", "bits": 8, "strategy": "dynamic"},
        "metrics": {"accuracy": 0.654, "latency_ms": 45.3, "memory_mb": 3421, "carbon_kg": 0.012},
        "model_name": "meta-llama/Llama-2-7b-hf",
        "tags": {"objective": "speedup", "status": "success"}
    }
    """
    args_schema: type[BaseModel] = ExperimentLogInput

    def __init__(self, tracking_uri: str = "./mlruns", experiment_name: str = "compression_optimization"):
        """Initialize MLflow tracking."""
        super().__init__()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def _run(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        model_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Execute the tool to log experiment."""
        try:
            with mlflow.start_run() as run:
                # Log parameters
                for key, value in config.items():
                    if isinstance(value, (dict, list)):
                        mlflow.log_param(key, json.dumps(value))
                    else:
                        mlflow.log_param(key, value)

                # Log model name if provided
                if model_name:
                    mlflow.log_param("model_name", model_name)

                # Log metrics
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Add timestamp
                mlflow.set_tag("timestamp", datetime.now().isoformat())

                run_id = run.info.run_id

            return f"✓ Experiment logged to MLflow with run_id={run_id}. Config: {config}, Metrics: {metrics}"

        except Exception as e:
            return f"✗ Failed to log experiment: {str(e)}"


class QueryExperimentsInput(BaseModel):
    """Input schema for querying experiments."""

    filter_string: Optional[str] = Field(
        default=None,
        description="MLflow filter query (e.g., 'params.technique = \"quantization\"')"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return"
    )
    order_by: Optional[str] = Field(
        default="metrics.accuracy DESC",
        description="Sort order (e.g., 'metrics.accuracy DESC', 'metrics.latency_ms ASC')"
    )


class QueryExperimentsTool(BaseTool):
    """Tool to query previous experiments from MLflow."""

    name: str = "query_experiments"
    description: str = """
    Query previous compression experiments from MLflow to learn from history.

    Use this tool to:
    - Find similar experiments before running new ones
    - Identify best configurations for a specific technique
    - Analyze trends and patterns

    Example input:
    {
        "filter_string": "params.technique = 'quantization' and params.bits = '8'",
        "max_results": 5,
        "order_by": "metrics.accuracy DESC"
    }
    """
    args_schema: type[BaseModel] = QueryExperimentsInput
    _experiment_name: str = "compression_optimization"  # Private attribute with default
    _tracking_uri: str = "./mlruns"

    def __init__(self, tracking_uri: str = "./mlruns", experiment_name: str = "compression_optimization"):
        """Initialize MLflow tracking."""
        super().__init__()
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)

    def _run(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 10,
        order_by: Optional[str] = "metrics.accuracy DESC",
    ) -> str:
        """Execute the tool to query experiments."""
        try:
            experiment = mlflow.get_experiment_by_name(self._experiment_name)
            if not experiment:
                return f"No experiment found with name '{self._experiment_name}'"

            # Search runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=[order_by] if order_by else None,
            )

            if runs.empty:
                return "No experiments found matching the query."

            # Format results
            results = []
            for idx, run in runs.iterrows():
                result = {
                    "run_id": run["run_id"],
                    "params": {k.replace("params.", ""): v for k, v in run.items() if k.startswith("params.")},
                    "metrics": {k.replace("metrics.", ""): v for k, v in run.items() if k.startswith("metrics.")},
                }
                results.append(result)

            return json.dumps(results, indent=2)

        except Exception as e:
            return f"✗ Failed to query experiments: {str(e)}"


class GetBestConfigInput(BaseModel):
    """Input schema for getting best configuration."""

    metric: str = Field(
        default="accuracy",
        description="Metric to optimize (e.g., 'accuracy', 'latency_ms', 'carbon_kg')"
    )
    higher_is_better: bool = Field(
        default=True,
        description="Whether higher metric values are better (True for accuracy, False for latency/carbon)"
    )
    technique_filter: Optional[str] = Field(
        default=None,
        description="Filter by technique (e.g., 'quantization', 'pruning', 'distillation')"
    )


class GetBestConfigTool(BaseTool):
    """Tool to get the best configuration for a specific metric."""

    name: str = "get_best_config"
    description: str = """
    Retrieve the best compression configuration based on a specific metric.

    Use this tool to:
    - Find the configuration that achieved highest accuracy
    - Find the configuration with lowest latency/carbon
    - Get starting points for new experiments

    Example input:
    {
        "metric": "accuracy",
        "higher_is_better": true,
        "technique_filter": "quantization"
    }
    """
    args_schema: type[BaseModel] = GetBestConfigInput
    _experiment_name: str = "compression_optimization"
    _tracking_uri: str = "./mlruns"

    def __init__(self, tracking_uri: str = "./mlruns", experiment_name: str = "compression_optimization"):
        """Initialize MLflow tracking."""
        super().__init__()
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)

    def _run(
        self,
        metric: str = "accuracy",
        higher_is_better: bool = True,
        technique_filter: Optional[str] = None,
    ) -> str:
        """Execute the tool to get best configuration."""
        try:
            experiment = mlflow.get_experiment_by_name(self._experiment_name)
            if not experiment:
                return f"No experiment found with name '{self._experiment_name}'"

            # Build filter
            filter_string = None
            if technique_filter:
                filter_string = f"params.technique = '{technique_filter}'"

            # Build order_by
            order_by = f"metrics.{metric} {'DESC' if higher_is_better else 'ASC'}"

            # Search runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=1,
                order_by=[order_by],
            )

            if runs.empty:
                return "No experiments found matching the criteria."

            # Get best run
            best_run = runs.iloc[0]
            config = {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")}
            metrics = {k.replace("metrics.", ""): v for k, v in best_run.items() if k.startswith("metrics.")}

            result = {
                "run_id": best_run["run_id"],
                "best_config": config,
                "achieved_metrics": metrics,
                "optimized_metric": metric,
                "metric_value": metrics.get(metric),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return f"✗ Failed to get best config: {str(e)}"


def create_tracking_tools(tracking_uri: str = "./mlruns", experiment_name: str = "compression_optimization"):
    """
    Factory function to create all MLflow tracking tools.

    Args:
        tracking_uri: MLflow tracking URI (default: ./mlruns)
        experiment_name: MLflow experiment name

    Returns:
        List of tracking tools for LangChain agent
    """
    return [
        LogExperimentTool(tracking_uri=tracking_uri, experiment_name=experiment_name),
        QueryExperimentsTool(tracking_uri=tracking_uri, experiment_name=experiment_name),
        GetBestConfigTool(tracking_uri=tracking_uri, experiment_name=experiment_name),
    ]
