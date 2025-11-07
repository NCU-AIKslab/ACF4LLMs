"""
Model evaluation module using lm-evaluation-harness.

Provides benchmarking on multiple datasets.
"""

from .benchmark_runner import BenchmarkRunner
from .lm_harness_adapter import LMHarnessAdapter

__all__ = ["LMHarnessAdapter", "BenchmarkRunner"]
