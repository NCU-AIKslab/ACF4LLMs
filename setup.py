"""Setup script for the Agentic Compression Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="agentic-compression",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Intelligent model compression using DeepAgents and multi-objective optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agentic-compression",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-openai>=0.0.5",
        "langgraph>=0.0.20",
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "pydantic>=2.0.0",
        "click>=8.1.0",
        "numpy>=1.24.0",
        "plotly>=5.18.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        "compression": [
            "deepagents>=0.1.0",
            "auto-round>=0.2.0",
            "gptqmodel>=1.4.3",
            "awq>=0.1.0",
            "bitsandbytes>=0.41.0",
            "peft>=0.7.0",
            "trl>=0.7.0",
        ],
        "evaluation": [
            "datasets>=2.14.0",
            "evaluate>=0.4.0",
            "lm-eval>=0.4.0",
        ],
        "monitoring": [
            "mlflow>=2.9.0",
            "streamlit>=1.30.0",
            "prometheus-client>=0.19.0",
            "codecarbon>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic-compress=scripts.run_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
)
