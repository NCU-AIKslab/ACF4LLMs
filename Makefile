# Makefile for LLM Compressor 2.0
# LLM-driven intelligent multi-agent system for LLM compression and optimization

.PHONY: help install build run clean test lint format docker setup-dev llm-test

# Variables
PYTHON := python3
PIP := pip
DOCKER_IMAGE := llm-compressor
DOCKER_TAG := latest
CONFIG_FILE := llm_compressor/configs/default.yaml
OUTPUT_DIR := reports

# Default target
help:
	@echo "LLM Compressor 2.0 - LLM-driven intelligent optimization system"
	@echo ""
	@echo "Available targets:"
	@echo "  install         Install dependencies (including LangChain/LangGraph)"
	@echo "  build           Build Docker image with LLM support"
	@echo "  run             Run conservative optimization (default)"
	@echo "  run-baseline    Run baseline performance measurement"
	@echo "  run-conservative Run conservative LLM-driven optimization"
	@echo "  run-aggressive  Run aggressive LLM-driven optimization"
	@echo "  run-llm-planned Run LLM-planned optimization portfolio"
	@echo "  test            Run test suite"
	@echo "  llm-test        Test LLM agent system"
	@echo "  docker-*        Docker variants of all run commands"
	@echo "  clean           Clean up generated files"
	@echo "  lint            Run code linting"
	@echo "  format          Format code"
	@echo "  setup-dev       Setup development environment"
	@echo "  export          Export and analyze results"
	@echo ""
	@echo "LLM-driven Docker commands:"
	@echo "  docker-build       Build LLM-enabled Docker image"
	@echo "  docker-baseline    Run baseline in Docker"
	@echo "  docker-conservative Run conservative optimization in Docker"
	@echo "  docker-aggressive  Run aggressive optimization in Docker"
	@echo "  docker-shell       Interactive Docker shell"
	@echo "  docker-test        Run LLM system tests in Docker"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG_FILE=$(CONFIG_FILE)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  DOCKER_IMAGE=$(DOCKER_IMAGE):$(DOCKER_TAG)"
	@echo ""
	@echo "Environment Variables for LLM APIs:"
	@echo "  OPENAI_API_KEY     OpenAI API key (for GPT models)"
	@echo "  ANTHROPIC_API_KEY  Anthropic API key (for Claude models)"
	@echo "  GOOGLE_API_KEY     Google API key (for Gemini models)"

# Installation
install:
	@echo "Installing LLM Compressor dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "Installation completed!"

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install pytest pytest-cov black isort flake8 mypy
	@echo "Development installation completed!"

# Build
build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

# LLM-driven run targets
run: run-conservative
	@echo "Default run completed (conservative optimization)"

run-baseline:
	@echo "Running LLM-driven baseline measurement..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --recipes baseline --output $(OUTPUT_DIR)/baseline

run-conservative:
	@echo "Running LLM-driven conservative optimization..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --recipes conservative --output $(OUTPUT_DIR)/conservative

run-aggressive:
	@echo "Running LLM-driven aggressive optimization..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --recipes aggressive --output $(OUTPUT_DIR)/aggressive

run-llm-planned:
	@echo "Running LLM-planned optimization portfolio..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --recipes llm_planned --output $(OUTPUT_DIR)/llm_planned

# Docker commands using run_docker.sh script
docker-build:
	@echo "Building LLM-enabled Docker image..."
	./run_docker.sh build

docker-baseline:
	@echo "Running baseline measurement in Docker..."
	./run_docker.sh baseline

docker-conservative:
	@echo "Running conservative optimization in Docker..."
	./run_docker.sh conservative

docker-aggressive:
	@echo "Running aggressive optimization in Docker..."
	./run_docker.sh aggressive

docker-llm-planned:
	@echo "Running LLM-planned optimization in Docker..."
	./run_docker.sh llm-planned

docker-shell:
	@echo "Starting interactive Docker shell..."
	./run_docker.sh shell

docker-test:
	@echo "Running LLM system tests in Docker..."
	./run_docker.sh test

# Legacy Docker command (kept for compatibility)
run-docker: docker-conservative
	@echo "Legacy docker run completed"

run-interactive: docker-shell
	@echo "Legacy interactive run completed"

# Export and analysis
export:
	@echo "Exporting and analyzing results..."
	$(PYTHON) scripts/export_report.py --db experiments.db --output analysis_report

export-json:
	@echo "Exporting from JSON summary..."
	$(PYTHON) scripts/export_report.py --input $(OUTPUT_DIR)/execution_summary.json --output analysis_report

# Testing
test:
	@echo "Running test suite..."
	$(PYTHON) -m pytest tests/ -v --cov=llm_compressor --cov-report=html --cov-report=term

test-quick:
	@echo "Running quick tests..."
	$(PYTHON) -m pytest tests/ -v -x --tb=short

llm-test:
	@echo "Testing LLM agent system..."
	$(PYTHON) test_llm_system.py

# Code quality
lint:
	@echo "Running linting checks..."
	flake8 llm_compressor/ scripts/ --max-line-length=100 --ignore=E203,W503
	mypy llm_compressor/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black llm_compressor/ scripts/ --line-length=100
	isort llm_compressor/ scripts/ --profile black

format-check:
	@echo "Checking code formatting..."
	black llm_compressor/ scripts/ --line-length=100 --check
	isort llm_compressor/ scripts/ --profile black --check-only

# Development setup
setup-dev: install-dev
	@echo "Setting up development environment..."
	@mkdir -p reports artifacts logs evals
	@echo "Creating sample evaluation data..."
	@bash scripts/quickstart.sh --create-data
	@echo "Development environment ready!"

setup-data:
	@echo "Setting up sample datasets..."
	@mkdir -p evals
	@bash scripts/quickstart.sh --create-data
	@echo "Sample datasets created!"

# Quickstart
quickstart:
	@echo "Running quickstart setup and optimization..."
	@bash scripts/quickstart.sh setup
	@bash scripts/quickstart.sh baseline

quickstart-docker: build
	@echo "Running quickstart in Docker..."
	docker run --gpus all -it --rm \
		-v $(PWD)/reports:/app/reports \
		$(DOCKER_IMAGE):$(DOCKER_TAG) \
		bash scripts/quickstart.sh baseline

# Cleanup
clean:
	@echo "Cleaning up generated files..."
	rm -rf reports/* artifacts/* logs/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Cleanup completed!"

clean-docker:
	@echo "Cleaning up Docker resources..."
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	docker system prune -f

clean-all: clean clean-docker
	@echo "Full cleanup completed!"

# Development utilities
check: format-check lint test-quick
	@echo "All checks passed!"

validate-config:
	@echo "Validating configuration..."
	$(PYTHON) scripts/run_search.py --config $(CONFIG_FILE) --dry-run

# Environment information
env-info:
	@echo "Environment Information:"
	@echo "========================"
	@$(PYTHON) --version
	@$(PIP) --version
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU detected"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "CUDA: $$(nvcc --version 2>/dev/null | grep 'release' || echo 'Not available')"

# Performance benchmarking
benchmark:
	@echo "Running performance benchmark..."
	$(PYTHON) scripts/run_search.py \
		--config configs/default.yaml \
		--recipes baseline \
		--output reports/benchmark_$$(date +%Y%m%d_%H%M%S) \
		--log-level INFO

# Documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs
	@echo "# LLM Compressor Documentation\n" > docs/index.md
	@echo "Auto-generated documentation would go here..." >> docs/index.md

# Custom recipes
run-custom:
	@if [ -z "$(RECIPE)" ]; then \
		echo "Usage: make run-custom RECIPE=path/to/recipe.yaml"; \
		exit 1; \
	fi
	@echo "Running custom recipe: $(RECIPE)"
	$(PYTHON) scripts/run_search.py --config $(RECIPE) --output $(OUTPUT_DIR)/custom

# Multi-GPU support
run-multi-gpu:
	@echo "Running with multiple GPUs..."
	CUDA_VISIBLE_DEVICES=0,1 $(PYTHON) scripts/run_search.py \
		--config $(CONFIG_FILE) \
		--output $(OUTPUT_DIR)/multi_gpu

# Monitoring and debugging
debug:
	@echo "Running in debug mode..."
	$(PYTHON) scripts/run_search.py \
		--config $(CONFIG_FILE) \
		--log-level DEBUG \
		--output $(OUTPUT_DIR)/debug

monitor:
	@echo "Monitoring system resources..."
	watch -n 5 'nvidia-smi; echo ""; ps aux | grep python | head -10'

# CI/CD targets
ci-test: install-dev format-check lint test
	@echo "CI tests completed!"

ci-build: build
	@echo "CI build completed!"

# Help for specific targets
help-docker:
	@echo "Docker Commands:"
	@echo "  make build         Build Docker image"
	@echo "  make run-docker    Run optimization in container"
	@echo "  make run-interactive  Start interactive container"
	@echo "  make clean-docker  Remove Docker image"

help-dev:
	@echo "Development Commands:"
	@echo "  make setup-dev     Setup development environment"
	@echo "  make test          Run full test suite"
	@echo "  make lint          Run code linting"
	@echo "  make format        Format code"
	@echo "  make check         Run all quality checks"