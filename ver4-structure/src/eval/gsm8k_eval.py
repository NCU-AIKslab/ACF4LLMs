"""
GSM8K evaluator for model assessment.

Provides comprehensive evaluation functionality including accuracy measurement,
latency profiling, error analysis, and resource monitoring.
"""

import time
import gc
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from collections import defaultdict
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import numpy as np

from ..artifacts import ModelArtifact, DatasetArtifact, EvalArtifact
from .gsm8k_data import extract_answer, compare_answers
from ..monitor.metrics_collector import MetricsCollector


logger = logging.getLogger(__name__)


class GSM8KEvaluator:
    """
    Comprehensive evaluator for GSM8K dataset.

    Handles model evaluation with accurate answer extraction, performance
    profiling, error analysis, and resource monitoring.
    """

    def __init__(self,
                 extract_fn: Optional[Callable[[str], str]] = None,
                 compare_fn: Optional[Callable[[str, str], bool]] = None,
                 device: str = "auto",
                 runtime_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GSM8K evaluator.

        Args:
            extract_fn: Function to extract answers from model output
            compare_fn: Function to compare predicted vs gold answers
            device: Device to run evaluation on
            runtime_config: Runtime optimization configuration
        """
        self.extract_fn = extract_fn or extract_answer
        self.compare_fn = compare_fn or compare_answers
        self.device = device
        self.runtime_config = runtime_config or {}
        self.metrics_collector = MetricsCollector()

    def evaluate(self,
                model: AutoModelForCausalLM,
                tokenizer: AutoTokenizer,
                dataset: Dataset,
                model_artifact: ModelArtifact,
                dataset_artifact: DatasetArtifact,
                batch_size: int = 8,
                max_new_tokens: int = 256,
                temperature: float = 0.7,
                top_p: float = 0.9,
                num_samples: Optional[int] = None,
                run_id: Optional[str] = None) -> EvalArtifact:
        """
        Evaluate model on GSM8K dataset.

        Args:
            model: The model to evaluate
            tokenizer: Associated tokenizer
            dataset: GSM8K dataset split to evaluate on
            model_artifact: Model artifact metadata
            dataset_artifact: Dataset artifact metadata
            batch_size: Batch size for inference
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_samples: Number of samples to evaluate (None = all)
            run_id: Unique identifier for this evaluation run

        Returns:
            EvalArtifact with comprehensive evaluation results
        """
        run_id = run_id or f"eval_{int(time.time())}"
        logger.info(f"Starting GSM8K evaluation with run_id: {run_id}")

        # Limit dataset size if specified
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        start_time = time.time()
        predictions = []
        latencies = []
        correct_count = 0
        error_types = defaultdict(int)

        # Initialize VRAM tracking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch = dataset.select(range(i, batch_end))

            batch_predictions, batch_latencies = self._evaluate_batch(
                model, tokenizer, batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

            predictions.extend(batch_predictions)
            latencies.extend(batch_latencies)

            # Check correctness and analyze errors
            for j, (prediction, example) in enumerate(zip(batch_predictions, batch)):
                # Use compare_fn which handles extraction internally
                is_correct = self.compare_fn(prediction, example["answer"])
                if is_correct:
                    correct_count += 1
                else:
                    error_type = self._classify_error(prediction, example["answer"], example["question"])
                    error_types[error_type] += 1

            # Log progress
            if (i // batch_size + 1) % 10 == 0:
                current_acc = correct_count / len(predictions)
                logger.info(f"Processed {len(predictions)}/{len(dataset)} samples, "
                          f"current accuracy: {current_acc:.3f}")

        end_time = time.time()
        total_duration = end_time - start_time

        # Calculate metrics
        accuracy = correct_count / len(predictions)
        latency_p50 = np.percentile(latencies, 50)
        latency_p90 = np.percentile(latencies, 90)
        latency_p99 = np.percentile(latencies, 99)

        # Calculate throughput
        total_tokens = sum(len(tokenizer.encode(pred)) for pred in predictions)
        tokens_per_sec = total_tokens / total_duration

        # Get VRAM usage
        vram_peak_mb = 0
        if torch.cuda.is_available():
            vram_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Create evaluation artifact
        eval_artifact = EvalArtifact(
            run_id=run_id,
            model_artifact=model_artifact,
            dataset_artifact=dataset_artifact,
            accuracy=accuracy,
            latency_ms_p50=latency_p50 * 1000,  # Convert to milliseconds
            latency_ms_p90=latency_p90 * 1000,
            latency_ms_p99=latency_p99 * 1000,
            tokens_per_sec=tokens_per_sec,
            vram_peak_mb=vram_peak_mb,
            samples_evaluated=len(predictions),
            errors_by_type=dict(error_types),
            duration_seconds=total_duration,
            metadata={
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "correct_predictions": correct_count,
                "total_predictions": len(predictions)
            }
        )

        logger.info(f"Evaluation completed - Accuracy: {accuracy:.3f}, "
                   f"Latency P50: {latency_p50*1000:.1f}ms, "
                   f"VRAM: {vram_peak_mb:.0f}MB")

        return eval_artifact

    def _evaluate_batch(self,
                       model: AutoModelForCausalLM,
                       tokenizer: AutoTokenizer,
                       batch: Dataset,
                       **generation_kwargs) -> Tuple[List[str], List[float]]:
        """
        Evaluate a single batch of examples.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch: Batch of examples
            **generation_kwargs: Generation parameters

        Returns:
            Tuple of (predictions, latencies)
        """
        questions = [example["question"] for example in batch]

        # Create prompts
        prompts = [self._create_prompt(question) for question in questions]

        # Tokenize
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=generation_kwargs.get("temperature", 0.0) > 0,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )

        end_time = time.time()
        batch_latency = (end_time - start_time) / len(batch)

        # Decode predictions
        predictions = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            input_length = inputs["input_ids"].shape[1]
            generated = output[input_length:]
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(prediction)

        latencies = [batch_latency] * len(batch)
        return predictions, latencies

    def _create_prompt(self, question: str) -> str:
        """
        Create a prompt for GSM8K evaluation.

        Args:
            question: The math question

        Returns:
            Formatted prompt
        """
        return f"Problem: {question}\nSolution:"

    def _classify_error(self, prediction: str, gold_answer: str, question: str) -> str:
        """
        Classify the type of error in a prediction.

        Args:
            prediction: Model prediction
            gold_answer: Correct answer
            question: Original question

        Returns:
            Error type classification
        """
        pred_answer = self.extract_fn(prediction)
        gold_numeric = self.extract_fn(gold_answer)

        # No answer extracted
        if pred_answer is None:
            if "####" not in prediction:
                return "no_answer_format"
            else:
                return "answer_extraction_failed"

        # Answer format present but wrong
        if gold_numeric is None:
            return "gold_answer_invalid"

        try:
            pred_numeric = float(pred_answer)
            gold_numeric = float(gold_numeric)

            # Check for common arithmetic errors
            if abs(pred_numeric - gold_numeric) < 1:
                return "arithmetic_slip"
            elif pred_numeric == 0:
                return "zero_answer"
            elif str(pred_numeric).replace(".", "").replace("-", "") in str(gold_numeric).replace(".", "").replace("-", ""):
                return "digit_error"
            else:
                return "wrong_calculation"

        except ValueError:
            return "non_numeric_answer"

    def evaluate_with_carbon_tracking(self,
                                    model: AutoModelForCausalLM,
                                    tokenizer: AutoTokenizer,
                                    dataset: Dataset,
                                    model_artifact: ModelArtifact,
                                    dataset_artifact: DatasetArtifact,
                                    **kwargs) -> EvalArtifact:
        """
        Evaluate model with carbon footprint tracking.

        Args:
            Same as evaluate() method
            **kwargs: Additional arguments for evaluate()

        Returns:
            EvalArtifact with carbon metrics included
        """
        try:
            from codecarbon import EmissionsTracker

            tracker = EmissionsTracker(
                project_name=f"gsm8k_eval_{kwargs.get('run_id', 'default')}",
                output_dir="./carbon_logs",
                save_to_file=True,
                log_level="WARNING"
            )

            tracker.start()
            eval_artifact = self.evaluate(
                model, tokenizer, dataset, model_artifact, dataset_artifact, **kwargs
            )
            emissions = tracker.stop()

            # Update artifact with carbon metrics
            eval_artifact.energy_kwh = emissions or 0.0
            eval_artifact.co2_g = (emissions or 0.0) * 1000  # Convert to grams (rough estimate)

            logger.info(f"Carbon tracking - Energy: {eval_artifact.energy_kwh:.6f} kWh, "
                       f"CO2: {eval_artifact.co2_g:.3f}g")

        except ImportError:
            logger.warning("CodeCarbon not available, skipping carbon tracking")
            eval_artifact = self.evaluate(
                model, tokenizer, dataset, model_artifact, dataset_artifact, **kwargs
            )

        return eval_artifact

    def save_detailed_results(self, eval_artifact: EvalArtifact, output_path: Path) -> None:
        """
        Save detailed evaluation results to file.

        Args:
            eval_artifact: Evaluation results
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "run_id": eval_artifact.run_id,
            "timestamp": eval_artifact.created_at,
            "model_config": eval_artifact.model_artifact.to_dict(),
            "dataset_config": eval_artifact.dataset_artifact.to_dict(),
            "metrics": {
                "accuracy": eval_artifact.accuracy,
                "latency_ms_p50": eval_artifact.latency_ms_p50,
                "latency_ms_p90": eval_artifact.latency_ms_p90,
                "latency_ms_p99": eval_artifact.latency_ms_p99,
                "tokens_per_sec": eval_artifact.tokens_per_sec,
                "vram_peak_mb": eval_artifact.vram_peak_mb,
                "energy_kwh": eval_artifact.energy_kwh,
                "co2_g": eval_artifact.co2_g,
            },
            "errors_by_type": eval_artifact.errors_by_type,
            "samples_evaluated": eval_artifact.samples_evaluated,
            "duration_seconds": eval_artifact.duration_seconds,
            "efficiency_score": eval_artifact.get_efficiency_score(),
            "pareto_objectives": eval_artifact.get_pareto_objectives()
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Detailed results saved to {output_path}")


class ErrorAnalyzer:
    """
    Analyzes and categorizes evaluation errors for insights.
    """

    def __init__(self):
        self.error_patterns = {
            "arithmetic_slip": [
                r"off by [1-5]",
                r"calculation.*error",
                r"arithmetic.*mistake"
            ],
            "reasoning_error": [
                r"wrong approach",
                r"incorrect.*logic",
                r"misunderstood.*problem"
            ],
            "format_error": [
                r"missing.*####",
                r"wrong.*format",
                r"no.*final.*answer"
            ]
        }

    def analyze_errors(self, eval_artifact: EvalArtifact) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis.

        Args:
            eval_artifact: Evaluation results

        Returns:
            Dictionary with error analysis results
        """
        total_errors = sum(eval_artifact.errors_by_type.values())
        total_samples = eval_artifact.samples_evaluated

        analysis = {
            "total_samples": total_samples,
            "total_errors": total_errors,
            "accuracy": eval_artifact.accuracy,
            "error_breakdown": {},
            "error_rates": {},
            "recommendations": []
        }

        # Calculate error rates
        for error_type, count in eval_artifact.errors_by_type.items():
            rate = count / total_samples if total_samples > 0 else 0
            analysis["error_breakdown"][error_type] = count
            analysis["error_rates"][error_type] = rate

        # Generate recommendations
        if analysis["error_rates"].get("arithmetic_slip", 0) > 0.1:
            analysis["recommendations"].append(
                "Consider adding calculator tools or arithmetic verification"
            )

        if analysis["error_rates"].get("no_answer_format", 0) > 0.05:
            analysis["recommendations"].append(
                "Improve prompt engineering to enforce #### answer format"
            )

        if analysis["error_rates"].get("answer_extraction_failed", 0) > 0.05:
            analysis["recommendations"].append(
                "Review answer extraction logic for edge cases"
            )

        return analysis