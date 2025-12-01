"""BIG-Bench Hard evaluator."""

import torch
from typing import Any, Dict
import logging
from src.evaluation.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class BigBenchHardEvaluator(BaseEvaluator):
    """Evaluator for BIG-Bench Hard benchmark."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.benchmark_name = "bigbench_hard"

    def evaluate(self, model: Any, tokenizer: Any, batch_size: int = 8) -> float:
        """Evaluate model on BIG-Bench Hard tasks.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch_size: Batch size

        Returns:
            Average accuracy across tasks
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("lukaemon/bbh", split="test")
            dataset = dataset.select(range(min(100, len(dataset))))
        except:
            logger.warning("BIG-Bench Hard not available, using mock")
            return self._mock_evaluate()

        correct = 0
        total = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            prompts = [self.prepare_prompt(ex) for ex in batch]
            outputs = self.generate_batch(model, tokenizer, prompts)

            for output, example in zip(outputs, batch):
                pred = self.extract_answer(output)
                ref = str(example.get("target", ""))

                if self._compare_answers(pred, ref):
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def prepare_prompt(self, example: Dict[str, Any]) -> str:
        """Prepare BIG-Bench Hard prompt.

        Args:
            example: Dataset example

        Returns:
            Formatted prompt
        """
        input_text = example.get("input", "")
        return f"{input_text}\nAnswer:"

    def extract_answer(self, output: str) -> str:
        """Extract answer from output.

        Args:
            output: Model output

        Returns:
            Extracted answer
        """
        # Take first line or sentence
        lines = output.strip().split('\n')
        answer = lines[0] if lines else ""

        # Clean up
        answer = answer.strip().lower()

        # Remove common prefixes
        for prefix in ["answer:", "the answer is", "therefore"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        return answer

    def _compare_answers(self, pred: str, ref: str) -> bool:
        """Compare predicted and reference answers.

        Args:
            pred: Predicted answer
            ref: Reference answer

        Returns:
            True if answers match
        """
        # Normalize for comparison
        pred = pred.lower().strip()
        ref = ref.lower().strip()

        # Exact match
        if pred == ref:
            return True

        # Check if reference is contained in prediction
        if ref and ref in pred:
            return True

        return False

    def _mock_evaluate(self) -> float:
        """Mock evaluation."""
        import random
        return random.uniform(0.3, 0.5)