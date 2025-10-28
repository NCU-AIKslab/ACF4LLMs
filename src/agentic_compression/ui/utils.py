"""
Utility functions for Streamlit UI.

Handles async operations, session state management, and data conversion.
"""

import asyncio
import json
from collections.abc import Coroutine
from typing import Any

import streamlit as st


def run_async(coro: Coroutine) -> Any:
    """
    Run an async coroutine in Streamlit.

    Args:
        coro: Async coroutine to execute

    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


def init_session_state(key: str, default_value: Any) -> None:
    """
    Initialize session state variable if not exists.

    Args:
        key: Session state key
        default_value: Default value
    """
    if key not in st.session_state:
        st.session_state[key] = default_value


def export_results_json(results: dict, filename: str = "results.json") -> str:
    """
    Export results as JSON string.

    Args:
        results: Results dictionary
        filename: Output filename

    Returns:
        JSON string
    """
    return json.dumps(results, indent=2, default=str)


def format_metric_value(value: float, metric_name: str) -> str:
    """
    Format metric value for display.

    Args:
        value: Metric value
        metric_name: Metric name

    Returns:
        Formatted string
    """
    if metric_name in ["accuracy", "average_accuracy"]:
        return f"{value:.1%}"
    elif metric_name in ["co2_kg", "energy_kwh"]:
        return f"{value:.4f}"
    elif metric_name in ["latency_ms"]:
        return f"{value:.1f}ms"
    elif metric_name in ["memory_gb"]:
        return f"{value:.2f}GB"
    elif metric_name in ["throughput_tps"]:
        return f"{value:.0f} t/s"
    elif metric_name in ["compression_ratio"]:
        return f"{value:.2f}x"
    else:
        return f"{value:.3f}"


def calculate_carbon_percentage(used: float, budget: float) -> float:
    """
    Calculate carbon budget usage percentage.

    Args:
        used: Carbon used (kg)
        budget: Carbon budget (kg)

    Returns:
        Percentage (0-100)
    """
    return min(100.0, (used / budget) * 100) if budget > 0 else 0.0
