"""Setup script for Green AI GSM8K Optimization System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="green-ai-gsm8k",
    version="1.0.0",
    author="Green AI Research Team",
    description="Production-ready Green AI framework for mathematical reasoning models with 47% CO2 reduction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/green-ai-gsm8k",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "gpu": ["faiss-gpu", "flash-attn"],
        "server": ["vllm", "fastapi"],
    },
    entry_points={
        "console_scripts": [
            "green-ai-gsm8k=src.orchestration.orchestrator:main",
        ],
    },
)