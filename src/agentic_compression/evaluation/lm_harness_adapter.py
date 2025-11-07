"""
Adapter for lm-evaluation-harness.

Provides a clean interface to the lm-eval library for benchmarking.
"""

import logging
from typing import Dict, List, Optional

import torch
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LMHarnessAdapter:
    """
    Adapter for lm-evaluation-harness library.

    Provides unified interface for evaluating models on multiple benchmarks.

    Supported benchmarks:
    - GSM8K: Mathematical reasoning
    - TruthfulQA: Truthfulness and factual consistency
    - CommonsenseQA: Commonsense reasoning
    - HumanEval: Code generation
    - BigBench: Multi-domain challenging tasks
    """

    # Benchmark task mapping
    BENCHMARK_TASKS = {
        "gsm8k": "gsm8k",
        "truthfulqa": "truthfulqa_mc2",
        "commonsenseqa": "commonsenseqa",
        "humaneval": "humaneval",
        "bigbench": "bigbench_qa_wikidata",  # Use a representative subset
    }

    def __init__(self, batch_size: int = 8, device: str = "cuda"):
        """
        Initialize the adapter.

        Args:
            batch_size: Batch size for evaluation
            device: Device to run evaluation on
        """
        self.batch_size = batch_size
        self.device = device
        logger.info(f"Initialized LMHarnessAdapter with batch_size={batch_size}, device={device}")

    def evaluate_model(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        tasks: List[str],
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        Evaluate a model on specified tasks.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            tasks: List of task names to evaluate on
            num_fewshot: Number of few-shot examples
            limit: Limit number of samples (for testing)

        Returns:
            Dictionary with results per task

        Example:
            >>> adapter = LMHarnessAdapter()
            >>> results = adapter.evaluate_model(model, tokenizer, ["gsm8k", "truthfulqa"])
        """
        logger.info(f"Evaluating model on tasks: {tasks}")

        try:
            # Convert task names to lm-eval format
            eval_tasks = [self.BENCHMARK_TASKS.get(task, task) for task in tasks]

            # Create HF model wrapper for lm-eval
            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=self.batch_size,
                device=self.device,
            )

            # Run evaluation
            results = evaluator.simple_evaluate(
                model=lm,
                tasks=eval_tasks,
                num_fewshot=num_fewshot,
                batch_size=self.batch_size,
                limit=limit,
                log_samples=False,
            )

            # Extract and format results
            formatted_results = self._format_results(results, tasks)

            logger.info("Evaluation completed successfully")
            return formatted_results

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

    def evaluate_single_task(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        task: str,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate a model on a single task.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            task: Task name
            num_fewshot: Number of few-shot examples
            limit: Limit number of samples

        Returns:
            Dictionary with task results
        """
        results = self.evaluate_model(
            model=model,
            tokenizer=tokenizer,
            tasks=[task],
            num_fewshot=num_fewshot,
            limit=limit,
        )
        return results.get(task, {})

    def _format_results(self, raw_results: dict, task_names: List[str]) -> Dict[str, Dict]:
        """
        Format raw lm-eval results into a clean structure.

        Args:
            raw_results: Raw results from lm-eval
            task_names: Original task names

        Returns:
            Formatted results dictionary
        """
        formatted = {}

        results_dict = raw_results.get("results", {})

        for task_name in task_names:
            eval_task = self.BENCHMARK_TASKS.get(task_name, task_name)
            task_result = results_dict.get(eval_task, {})

            # Extract primary metric
            accuracy = self._extract_primary_metric(task_result, eval_task)

            formatted[task_name] = {
                "accuracy": accuracy,
                "task_name": eval_task,
                "raw_results": task_result,
            }

        return formatted

    def _extract_primary_metric(self, task_result: dict, task_name: str) -> float:
        """
        Extract the primary accuracy metric from task results.

        Args:
            task_result: Raw task results
            task_name: Name of the task

        Returns:
            Accuracy value (0.0-1.0)
        """
        # Common metric names in lm-eval
        metric_keys = ["acc", "acc_norm", "exact_match", "pass@1", "accuracy"]

        for key in metric_keys:
            if key in task_result:
                value = task_result[key]
                # Handle both direct values and nested dicts
                if isinstance(value, dict):
                    return value.get("value", 0.0)
                return float(value)

        # Fallback: look for any metric ending in "acc" or containing "accuracy"
        for key, value in task_result.items():
            if "acc" in key.lower() or "accuracy" in key.lower():
                if isinstance(value, dict):
                    return value.get("value", 0.0)
                return float(value)

        logger.warning(f"Could not extract primary metric for task: {task_name}")
        return 0.0

    @staticmethod
    def list_available_tasks() -> List[str]:
        """
        List all available tasks in lm-eval.

        Returns:
            List of task names
        """
        try:
            task_list = tasks.get_task_list()
            return sorted(task_list)
        except Exception as e:
            logger.error(f"Failed to list tasks: {str(e)}")
            return []

    @staticmethod
    def get_task_info(task_name: str) -> Optional[Dict]:
        """
        Get information about a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with task information or None
        """
        try:
            task_dict = tasks.get_task_dict([task_name])
            if task_name in task_dict:
                task_obj = task_dict[task_name]
                return {
                    "name": task_name,
                    "description": getattr(task_obj, "DESCRIPTION", "No description"),
                    "version": getattr(task_obj, "VERSION", "unknown"),
                }
        except Exception as e:
            logger.error(f"Failed to get task info for {task_name}: {str(e)}")
        return None
