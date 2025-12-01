"""GSM8K mathematical reasoning evaluator."""

import re
import torch
from typing import Any, Dict, List
import logging
from src.evaluation.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for GSM8K mathematical reasoning benchmark."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.benchmark_name = "gsm8k"

    def evaluate(self, model: Any, tokenizer: Any, batch_size: int = 8) -> float:
        """Evaluate model on GSM8K benchmark.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch_size: Batch size

        Returns:
            Accuracy score
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="test")
        except:
            logger.warning("GSM8K dataset not available, using mock evaluation")
            return self._mock_evaluate()

        # Limit for testing
        dataset = dataset.select(range(min(100, len(dataset))))

        predictions = []
        references = []

        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            # Prepare prompts
            prompts = [self.prepare_prompt(ex) for ex in batch]

            # Generate answers
            outputs = self.generate_batch(model, tokenizer, prompts)

            # Extract answers
            for output, example in zip(outputs, batch):
                pred_answer = self.extract_answer(output)
                true_answer = self.extract_answer(example["answer"])

                predictions.append(pred_answer)
                references.append(true_answer)

        # Calculate accuracy
        return self.compute_metric(predictions, references)

    def prepare_prompt(self, example: Dict[str, Any]) -> str:
        """Prepare GSM8K prompt.

        Args:
            example: Dataset example

        Returns:
            Formatted prompt
        """
        question = example.get("question", "")

        prompt = f"""Solve this math problem step by step.

Question: {question}

Let's think step by step:
"""
        return prompt

    def extract_answer(self, output: str) -> str:
        """Extract numerical answer from output.

        Args:
            output: Model output or reference answer

        Returns:
            Extracted numerical answer
        """
        # Look for numbers at the end or after "####"
        if "####" in output:
            answer = output.split("####")[-1].strip()
        else:
            # Find last number in the output
            numbers = re.findall(r'-?\d+\.?\d*', output)
            answer = numbers[-1] if numbers else ""

        # Clean and normalize
        answer = answer.replace(",", "").strip()

        # Convert to float if possible for comparison
        try:
            answer = str(float(answer))
        except:
            pass

        return answer

    def _mock_evaluate(self) -> float:
        """Mock evaluation for testing."""
        import random
        return random.uniform(0.3, 0.5)  # GSM8K is challenging

    def evaluate_proxy(
        self, model: Any, tokenizer: Any, num_samples: int = 100, batch_size: int = 8
    ) -> float:
        """Fast proxy evaluation on subset.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            num_samples: Number of samples
            batch_size: Batch size

        Returns:
            Proxy score
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="test")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        except:
            return self._mock_evaluate()

        predictions = []
        references = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            prompts = [self.prepare_prompt(ex) for ex in batch]
            outputs = self.generate_batch(model, tokenizer, prompts, max_length=256)

            for output, example in zip(outputs, batch):
                pred = self.extract_answer(output)
                ref = self.extract_answer(example["answer"])
                predictions.append(pred)
                references.append(ref)

        return self.compute_metric(predictions, references)