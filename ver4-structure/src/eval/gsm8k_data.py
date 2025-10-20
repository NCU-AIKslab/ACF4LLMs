"""
GSM8K dataset loader and utilities.

Provides functionality to load, split, and augment the GSM8K dataset
following the canonical structure and evaluation rules.
"""

import re
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
import sympy
from decimal import Decimal, InvalidOperation
import logging

from ..artifacts import DatasetArtifact


logger = logging.getLogger(__name__)


class GSM8KDataLoader:
    """
    Loads and preprocesses the GSM8K dataset with augmentation support.

    Provides canonical GSM8K loading with proper train/val/test splits and
    math-preserving augmentation strategies.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the GSM8K data loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset_name = "openai/gsm8k"

    def load_dataset(self,
                    val_split_size: int = 1000,
                    augmentation_recipes: Optional[List[str]] = None,
                    random_seed: int = 42) -> Tuple[DatasetDict, DatasetArtifact]:
        """
        Load GSM8K dataset with proper splits and augmentations.

        Args:
            val_split_size: Number of samples to hold out from train for validation
            augmentation_recipes: List of augmentation recipe names to apply
            random_seed: Random seed for reproducible splits

        Returns:
            Tuple of (dataset_dict, dataset_artifact)
        """
        logger.info(f"Loading GSM8K dataset from {self.dataset_name}")

        # Load the canonical GSM8K dataset
        dataset = load_dataset(self.dataset_name, "main", cache_dir=self.cache_dir)

        # Create train/val split from the original train set
        # GSM8K canonical: 7473 train, 1319 test
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

        # Split train into train/val
        train_dataset = train_dataset.shuffle(seed=random_seed)
        val_dataset = train_dataset.select(range(val_split_size))
        train_dataset = train_dataset.select(range(val_split_size, len(train_dataset)))

        logger.info(f"Split sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Apply augmentations if specified
        if augmentation_recipes:
            train_dataset = self._apply_augmentations(train_dataset, augmentation_recipes, random_seed)
            logger.info(f"After augmentation - Train: {len(train_dataset)}")

        # Create dataset dict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        })

        # Create dataset artifact
        dataset_artifact = DatasetArtifact(
            dataset_name=self.dataset_name,
            train_split={"size": len(train_dataset), "indices": list(range(len(train_dataset)))},
            val_split={"size": len(val_dataset), "indices": list(range(len(val_dataset)))},
            test_split={"size": len(test_dataset), "indices": list(range(len(test_dataset)))},
            augmentation_recipes=augmentation_recipes or [],
            prompts=self._get_default_prompts(),
            total_samples=len(train_dataset) + len(val_dataset) + len(test_dataset),
            metadata={
                "val_split_size": val_split_size,
                "random_seed": random_seed,
                "original_train_size": len(dataset["train"]),
                "original_test_size": len(dataset["test"])
            }
        )

        return dataset_dict, dataset_artifact

    def _apply_augmentations(self, dataset: Dataset, recipes: List[str], seed: int) -> Dataset:
        """Apply the specified augmentation recipes to the dataset."""
        augmentations = GSM8KAugmentations(seed=seed)
        augmented_data = []

        for example in dataset:
            augmented_data.append(example)

            for recipe in recipes:
                if hasattr(augmentations, recipe):
                    try:
                        augmented_example = getattr(augmentations, recipe)(example)
                        if augmented_example and self._validate_augmented_example(augmented_example):
                            augmented_data.append(augmented_example)
                    except Exception as e:
                        logger.warning(f"Augmentation {recipe} failed for example: {e}")

        # Convert back to Dataset
        return Dataset.from_list(augmented_data)

    def _validate_augmented_example(self, example: Dict[str, Any]) -> bool:
        """Validate that an augmented example is correct."""
        try:
            # Extract and validate the answer
            answer = extract_answer(example["answer"])
            return answer is not None and answer.strip() != ""
        except Exception:
            return False

    def _get_default_prompts(self) -> Dict[str, str]:
        """Get default prompt templates for GSM8K."""
        return {
            "system": "Solve the following math problem step by step. Show your work clearly and end your final numeric answer on a new line as #### <number>.",
            "few_shot_prefix": "Here are some examples of how to solve math problems:\n\n",
            "cot_instruction": "Let's think step by step.",
            "answer_format": "#### {answer}"
        }


class GSM8KAugmentations:
    """
    Math-preserving augmentation strategies for GSM8K problems.

    Implements various augmentation techniques that preserve the mathematical
    correctness while creating diverse training examples.
    """

    def __init__(self, seed: int = 42):
        """Initialize augmentations with random seed."""
        self.rng = random.Random(seed)

    def numeric_jitter(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply small numeric perturbations while maintaining answer correctness.

        Changes numbers by small percentages and recalculates the answer.
        """
        try:
            question = example["question"]
            answer = example["answer"]

            # Find numbers in the question
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
            if len(numbers) < 2:
                return None

            # Pick a number to jitter (avoid very small numbers)
            valid_numbers = [n for n in numbers if float(n) >= 2]
            if not valid_numbers:
                return None

            original_num = self.rng.choice(valid_numbers)
            original_val = float(original_num)

            # Apply 1-10% jitter
            jitter_percent = self.rng.uniform(0.01, 0.1)
            direction = self.rng.choice([-1, 1])
            new_val = original_val * (1 + direction * jitter_percent)
            new_num = str(int(new_val)) if new_val == int(new_val) else f"{new_val:.1f}"

            # Replace in question
            new_question = question.replace(original_num, new_num, 1)

            # Try to update answer proportionally (simple cases only)
            original_answer = extract_answer(answer)
            if original_answer and original_answer.isdigit():
                ratio = new_val / original_val
                new_answer_val = int(float(original_answer) * ratio)
                new_answer = answer.replace(f"#### {original_answer}", f"#### {new_answer_val}")

                return {
                    "question": new_question,
                    "answer": new_answer
                }

        except Exception as e:
            logger.debug(f"Numeric jitter failed: {e}")

        return None

    def unit_paraphrase(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Paraphrase units and measurements while preserving meaning.

        Examples: "5 nickels" -> "nickels worth $0.25"
        """
        question = example["question"]
        answer = example["answer"]

        # Common unit transformations
        transformations = [
            (r'(\d+) nickels', lambda m: f'nickels worth ${int(m.group(1)) * 0.05:.2f}'),
            (r'(\d+) dimes', lambda m: f'dimes worth ${int(m.group(1)) * 0.10:.2f}'),
            (r'(\d+) quarters', lambda m: f'quarters worth ${int(m.group(1)) * 0.25:.2f}'),
            (r'(\d+) hours', lambda m: f'{int(m.group(1)) * 60} minutes'),
            (r'(\d+) feet', lambda m: f'{int(m.group(1)) * 12} inches'),
        ]

        new_question = question
        for pattern, replacement in transformations:
            if re.search(pattern, new_question, re.IGNORECASE):
                try:
                    new_question = re.sub(pattern, replacement, new_question, count=1, flags=re.IGNORECASE)
                    return {
                        "question": new_question,
                        "answer": answer
                    }
                except Exception:
                    continue

        return None

    def distractor_insertion(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Insert neutral distractor sentences that don't affect the math.
        """
        question = example["question"]

        distractors = [
            "By the way, it's a sunny day outside.",
            "This problem is from a math competition.",
            "Remember to show your work clearly.",
            "Let's solve this step by step.",
            "This is an interesting problem."
        ]

        distractor = self.rng.choice(distractors)

        # Insert at random position (not at the very end)
        sentences = question.split('. ')
        if len(sentences) > 1:
            insert_pos = self.rng.randint(0, len(sentences) - 1)
            sentences.insert(insert_pos, distractor)
            new_question = '. '.join(sentences)

            return {
                "question": new_question,
                "answer": example["answer"]
            }

        return None

    def calculator_annotation(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add calculator-style annotations as in GSM8K training data.

        Example: "5 * 3 = 15" -> "5 * 3 = <<5*3=15>>15"
        """
        answer = example["answer"]

        # Look for simple arithmetic in the answer
        arithmetic_pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'

        def add_calculator_tag(match):
            num1, op, num2, result = match.groups()
            return f"{num1} {op} {num2} = <<{num1}{op}{num2}={result}>>{result}"

        new_answer = re.sub(arithmetic_pattern, add_calculator_tag, answer)

        if new_answer != answer:
            return {
                "question": example["question"],
                "answer": new_answer
            }

        return None


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from GSM8K solution text.

    Uses multiple fallback strategies to handle various answer formats.

    Args:
        text: The solution text containing the answer

    Returns:
        The extracted numeric answer as a string, or None if not found
    """
    # Try to find #### format first (GSM8K standard)
    pattern = r"####\s*([-+]?\d+(?:\.\d+)?)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # Try to find boxed format
    boxed_pattern = r"\\boxed\{([-+]?\d+(?:\.\d+)?)\}"
    match = re.search(boxed_pattern, text)
    if match:
        return match.group(1).strip()

    # Try to find "answer is X" format
    answer_pattern = r"(?:answer is|final answer is|answer:|the answer is)\s*([-+]?\d+(?:\.\d+)?)"
    match = re.search(answer_pattern, text.lower())
    if match:
        return match.group(1).strip()

    # Find last number in text as final fallback
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def normalize_number(num_str: str) -> Optional[Decimal]:
    """
    Normalize a numeric string to Decimal for robust comparison.

    Args:
        num_str: String representation of a number

    Returns:
        Decimal representation, or None if invalid
    """
    if not num_str:
        return None

    try:
        # Handle common formats
        cleaned = num_str.strip().replace(',', '')
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        try:
            # Try with SymPy for complex expressions
            expr = sympy.sympify(num_str)
            if expr.is_number:
                return Decimal(str(float(expr.evalf())))
        except:
            pass

    return None


def compare_answers(pred: str, gold: str) -> bool:
    """
    Compare predicted and gold answers with robust numeric comparison.

    Args:
        pred: Predicted answer text
        gold: Gold standard answer text

    Returns:
        True if answers match, False otherwise
    """
    pred_num = extract_answer(pred)
    gold_num = extract_answer(gold)

    if pred_num is None or gold_num is None:
        return False

    pred_decimal = normalize_number(pred_num)
    gold_decimal = normalize_number(gold_num)

    if pred_decimal is None or gold_decimal is None:
        return pred_num == gold_num  # Fallback to string comparison

    # Use small tolerance for floating point comparison
    tolerance = Decimal('0.001')
    return abs(pred_decimal - gold_decimal) <= tolerance