"""
Agentic Carbon-Efficient LLM Compression Framework

A multi-agent system for optimizing Large Language Models while balancing
accuracy, efficiency, and carbon footprint.

Based on research by Liu et al. (2025).
"""

from .core.config import CompressionConfig
from .core.metrics import EvaluationMetrics, ParetoSolution
from .graph.workflow import create_compression_workflow
from .optimization.agent_driven import AgentDrivenOptimization

__version__ = "2.0.0"

__all__ = [
    "CompressionConfig",
    "EvaluationMetrics",
    "ParetoSolution",
    "AgentDrivenOptimization",
    "create_compression_workflow",
]
