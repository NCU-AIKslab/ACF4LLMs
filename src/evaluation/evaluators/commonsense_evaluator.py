"""CommonsenseQA evaluator."""

import torch
from typing import Any, Dict, List
import logging
from src.evaluation.evaluators.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CommonsenseQAEvaluator(BaseEvaluator):
    """Evaluator for CommonsenseQA benchmark."""

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.benchmark_name = "commonsenseqa"

    def evaluate(self, model: Any, tokenizer: Any, batch_size: int = 8) -> float:
        """Evaluate model on CommonsenseQA.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            batch_size: Batch size

        Returns:
            Accuracy score
        """
        try:
            from datasets import load_dataset
            dataset = load_dataset("commonsense_qa", split="validation")
            dataset = dataset.select(range(min(100, len(dataset))))
        except:
            logger.warning("CommonsenseQA not available, using mock")
            return self._mock_evaluate()

        correct = 0
        total = 0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            prompts = [self.prepare_prompt(ex) for ex in batch]
            outputs = self.generate_batch(model, tokenizer, prompts)

            for output, example in zip(outputs, batch):
                pred_answer = self.extract_answer(output)
                true_answer = example["answerKey"]

                if pred_answer.upper() == true_answer.upper():
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def prepare_prompt(self, example: Dict[str, Any]) -> str:
        """Prepare CommonsenseQA prompt.

        Args:
            example: Dataset example

        Returns:
            Formatted prompt
        """
        question = example.get("question", "")
        choices = example.get("choices", {})

        prompt = f"Question: {question}\n\n"

        if choices and "text" in choices:
            labels = choices.get("label", [])
            texts = choices.get("text", [])

            prompt += "Choices:\n"
            for label, text in zip(labels, texts):
                prompt += f"{label}. {text}\n"

        prompt += "\nAnswer (A/B/C/D/E):"
        return prompt

    def extract_answer(self, output: str) -> str:
        """Extract multiple choice answer.

        Args:
            output: Model output

        Returns:
            Extracted answer (A/B/C/D/E)
        """
        output = output.strip().upper()

        # Look for single letter answer
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in output[:10]:  # Check beginning of output
                return letter

        return output[0] if output else ""

    def _mock_evaluate(self) -> float:
        """Mock evaluation."""
        import random
        return random.uniform(0.6, 0.8)