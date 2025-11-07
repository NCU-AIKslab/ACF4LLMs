#!/usr/bin/env python3
"""
Root-level Streamlit entry point.

This wrapper keeps the README instructions (`streamlit run app.py`) working
while the real UI lives in `src/agentic_compression/ui/app.py`.
"""

from importlib import import_module
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

# Ensure src/ is on sys.path so local imports work without `pip install -e .`
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))


def main():
    """Load the actual Streamlit app module."""
    import_module("agentic_compression.ui.app")


if __name__ == "__main__":
    main()
