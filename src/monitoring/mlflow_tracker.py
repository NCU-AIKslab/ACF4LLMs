"""MLflow integration for experiment tracking and monitoring."""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import MLflow, but make it optional
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed, experiment tracking will be limited")


class MLflowTracker:
    """MLflow tracker for compression experiments."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.artifact_location = artifact_location or "./mlruns"
        self.active_run = None

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.experiment_id = self._get_or_create_experiment()
            self.client = MlflowClient()
        else:
            self.experiment_id = None
            self.client = None

    def _get_or_create_experiment(self) -> str:
        """Get or create MLflow experiment.

        Returns:
            Experiment ID
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
            else:
                return mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.artifact_location,
                )
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return "0"

    def start_episode(
        self,
        episode_id: int,
        strategy: Dict[str, Any],
        model_spec: Dict[str, Any],
    ) -> Optional[str]:
        """Start tracking a compression episode.

        Args:
            episode_id: Episode number
            strategy: Compression strategy
            model_spec: Model specification

        Returns:
            Run ID or None
        """
        if not MLFLOW_AVAILABLE:
            return None

        try:
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"episode_{episode_id}",
            )
            self.active_run = run

            # Log strategy parameters
            mlflow.log_param("episode_id", episode_id)
            mlflow.log_param("strategy_id", strategy.get("strategy_id"))
            mlflow.log_param("methods", json.dumps(strategy.get("methods", [])))

            # Log quantization parameters if present
            if "quantization_bits" in strategy:
                mlflow.log_param("quantization_bits", strategy["quantization_bits"])
            if "quantization_method" in strategy:
                mlflow.log_param("quantization_method", strategy["quantization_method"])
            if "pruning_ratio" in strategy:
                mlflow.log_param("pruning_ratio", strategy["pruning_ratio"])

            # Log model spec
            mlflow.log_param("model_name", model_spec.get("model_name"))
            mlflow.log_param("model_size_gb", model_spec.get("model_size_gb"))
            mlflow.log_param("primary_objective", model_spec.get("primary_objective"))

            # Log strategy as artifact
            strategy_file = "strategy.json"
            with open(strategy_file, "w") as f:
                json.dump(strategy, f, indent=2, default=str)
            mlflow.log_artifact(strategy_file)
            os.remove(strategy_file)

            return run.info.run_id

        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None

    def log_compression_results(
        self,
        checkpoint_path: str,
        compression_ratio: float,
        quantization_time: float,
        method: str,
    ):
        """Log compression results.

        Args:
            checkpoint_path: Path to compressed model
            compression_ratio: Compression ratio achieved
            quantization_time: Time taken for compression
            method: Compression method used
        """
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            mlflow.log_metric("compression_ratio", compression_ratio)
            mlflow.log_metric("compression_time_sec", quantization_time)
            mlflow.log_param("compression_method", method)
            mlflow.log_param("checkpoint_path", checkpoint_path)

            # Log model size if available
            if os.path.exists(checkpoint_path):
                size_gb = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(checkpoint_path)
                    for filename in filenames
                ) / (1024**3)
                mlflow.log_metric("model_size_gb", size_gb)

        except Exception as e:
            logger.error(f"Failed to log compression results: {e}")

    def log_evaluation_results(
        self,
        benchmark_scores: Dict[str, float],
        average_accuracy: float,
        latency_ms: float,
        memory_gb: float,
        throughput: float,
    ):
        """Log evaluation results.

        Args:
            benchmark_scores: Scores per benchmark
            average_accuracy: Average accuracy
            latency_ms: Inference latency
            memory_gb: Memory usage
            throughput: Throughput in tokens/sec
        """
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            # Log overall metrics
            mlflow.log_metric("average_accuracy", average_accuracy)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("memory_gb", memory_gb)
            mlflow.log_metric("throughput_tokens_per_sec", throughput)

            # Log per-benchmark scores
            for benchmark, score in benchmark_scores.items():
                mlflow.log_metric(f"score_{benchmark}", score)

            # Calculate and log efficiency metrics
            efficiency = average_accuracy / (latency_ms / 100)  # Accuracy per 100ms
            mlflow.log_metric("efficiency_score", efficiency)

        except Exception as e:
            logger.error(f"Failed to log evaluation results: {e}")

    def log_pareto_update(
        self,
        is_pareto_optimal: bool,
        dominated_solutions: List[str],
        frontier_size: int,
    ):
        """Log Pareto frontier update.

        Args:
            is_pareto_optimal: Whether solution is Pareto optimal
            dominated_solutions: Solutions dominated by this one
            frontier_size: Current frontier size
        """
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            mlflow.log_metric("is_pareto_optimal", 1 if is_pareto_optimal else 0)
            mlflow.log_metric("num_dominated", len(dominated_solutions))
            mlflow.log_metric("frontier_size", frontier_size)

            if dominated_solutions:
                mlflow.log_param("dominated_solutions", json.dumps(dominated_solutions))

        except Exception as e:
            logger.error(f"Failed to log Pareto update: {e}")

    def log_reward(self, reward: float, component_scores: Dict[str, float]):
        """Log reward function values.

        Args:
            reward: Total reward
            component_scores: Individual component scores
        """
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            mlflow.log_metric("total_reward", reward)

            for component, score in component_scores.items():
                if isinstance(score, (int, float)):
                    mlflow.log_metric(f"reward_{component}", score)

        except Exception as e:
            logger.error(f"Failed to log reward: {e}")

    def end_episode(self, status: str = "FINISHED", error_message: Optional[str] = None):
        """End tracking for current episode.

        Args:
            status: Episode status
            error_message: Error message if failed
        """
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            mlflow.log_param("episode_status", status)
            if error_message:
                mlflow.log_param("error_message", error_message)

            mlflow.end_run()
            self.active_run = None

        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_model(self, model_path: str, model_name: str = "compressed_model"):
        """Log model to MLflow.

        Args:
            model_path: Path to model checkpoint
            model_name: Name for logged model
        """
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            # Log as PyTorch model if possible
            if os.path.exists(model_path):
                mlflow.log_artifacts(model_path, artifact_path=model_name)

        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def compare_runs(
        self, run_ids: List[str], metrics: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to compare

        Returns:
            Dictionary with comparison data
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return {}

        comparison = {}

        try:
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                }

                # Extract requested metrics
                run_metrics = {}
                for metric in metrics:
                    if metric in run.data.metrics:
                        run_metrics[metric] = run.data.metrics[metric]

                comparison[run_id] = run_metrics

        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")

        return comparison

    def get_best_run(self, metric: str = "average_accuracy", ascending: bool = False) -> Optional[Dict]:
        """Get best run based on metric.

        Args:
            metric: Metric to optimize
            ascending: Whether lower is better

        Returns:
            Best run data or None
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return None

        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
                max_results=1,
            )

            if runs:
                run = runs[0]
                return {
                    "run_id": run.info.run_id,
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                }

        except Exception as e:
            logger.error(f"Failed to get best run: {e}")

        return None

    def log_system_metrics(self, gpu_utilization: float, memory_usage: float, temperature: float):
        """Log system metrics during compression.

        Args:
            gpu_utilization: GPU utilization percentage
            memory_usage: Memory usage in GB
            temperature: GPU temperature
        """
        if not MLFLOW_AVAILABLE or not self.active_run:
            return

        try:
            mlflow.log_metric("gpu_utilization", gpu_utilization)
            mlflow.log_metric("gpu_memory_gb", memory_usage)
            mlflow.log_metric("gpu_temperature", temperature)

        except Exception as e:
            logger.error(f"Failed to log system metrics: {e}")

    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data for dashboard visualization.

        Returns:
            Dictionary with dashboard data
        """
        if not MLFLOW_AVAILABLE or not self.client:
            return {}

        try:
            # Get all runs
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=["metrics.episode_id ASC"],
            )

            dashboard_data = {
                "episodes": [],
                "pareto_frontier": [],
                "metrics_over_time": {
                    "accuracy": [],
                    "latency": [],
                    "compression_ratio": [],
                },
            }

            for run in runs:
                episode_data = {
                    "episode_id": run.data.params.get("episode_id"),
                    "accuracy": run.data.metrics.get("average_accuracy"),
                    "latency": run.data.metrics.get("latency_ms"),
                    "compression_ratio": run.data.metrics.get("compression_ratio"),
                    "is_pareto": run.data.metrics.get("is_pareto_optimal", 0) == 1,
                }

                dashboard_data["episodes"].append(episode_data)

                if episode_data["is_pareto"]:
                    dashboard_data["pareto_frontier"].append(episode_data)

                # Time series data
                dashboard_data["metrics_over_time"]["accuracy"].append(episode_data["accuracy"])
                dashboard_data["metrics_over_time"]["latency"].append(episode_data["latency"])
                dashboard_data["metrics_over_time"]["compression_ratio"].append(
                    episode_data["compression_ratio"]
                )

            return dashboard_data

        except Exception as e:
            logger.error(f"Failed to create dashboard data: {e}")
            return {}


class MockMLflowTracker(MLflowTracker):
    """Mock tracker for when MLflow is not available."""

    def __init__(self, experiment_name: str, **kwargs):
        self.experiment_name = experiment_name
        self.logs = []

    def start_episode(self, episode_id: int, strategy: Dict, model_spec: Dict) -> str:
        self.logs.append({"type": "start", "episode_id": episode_id})
        return f"mock_run_{episode_id}"

    def log_compression_results(self, **kwargs):
        self.logs.append({"type": "compression", **kwargs})

    def log_evaluation_results(self, **kwargs):
        self.logs.append({"type": "evaluation", **kwargs})

    def end_episode(self, status: str = "FINISHED", error_message: Optional[str] = None):
        self.logs.append({"type": "end", "status": status})

    def get_logs(self) -> List[Dict]:
        return self.logs


def create_experiment_tracker(
    experiment_name: str,
    use_mlflow: bool = True,
) -> MLflowTracker:
    """Create an experiment tracker.

    Args:
        experiment_name: Name of experiment
        use_mlflow: Whether to use MLflow

    Returns:
        Experiment tracker instance
    """
    if use_mlflow and MLFLOW_AVAILABLE:
        return MLflowTracker(experiment_name)
    else:
        return MockMLflowTracker(experiment_name)


__all__ = ["MLflowTracker", "MockMLflowTracker", "create_experiment_tracker"]