# Dockerfile for LLM Compressor
# Multi-agent system for LLM compression and optimization

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Use system pip directly

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install essential packages first - using latest PyTorch with CUDA 12.1 (compatible with CUDA 13.0)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install transformers and tokenizers first to avoid conflicts
RUN pip install --no-cache-dir \
    tokenizers>=0.15.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0

# Verify transformers installation
RUN python3 -c "from transformers import AutoTokenizer; print('AutoTokenizer import: SUCCESS')" || \
    (echo "AutoTokenizer import failed, reinstalling..." && \
     pip uninstall -y transformers tokenizers && \
     pip install --no-cache-dir transformers==4.35.2 tokenizers==0.15.0)

# Install additional ML libraries
RUN pip install --no-cache-dir \
    datasets>=2.14.0 \
    bitsandbytes>=0.41.0 \
    peft>=0.7.0 \
    safetensors \
    sentencepiece \
    protobuf

# Install monitoring and visualization libraries
RUN pip install --no-cache-dir \
    plotly \
    matplotlib \
    seaborn \
    nvidia-ml-py \
    GPUtil \
    psutil \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    optuna \
    deap \
    pyyaml \
    tqdm \
    click \
    loguru \
    rich

# Install quantization libraries - try simpler approach first
RUN pip install --no-cache-dir bitsandbytes>=0.41.0

# Try to install quantization libraries (may fail, that's ok for basic testing)
RUN pip install --no-cache-dir auto-gptq>=0.7.0 || echo "auto-gptq installation failed, will use BitsAndBytes" && \
    pip install --no-cache-dir autoawq>=0.1.8 || echo "autoawq installation failed, will use BitsAndBytes"

# Install vLLM for model serving (optional but recommended)
RUN pip install --no-cache-dir vllm>=0.3.0 || echo "vLLM installation failed, continuing..."

# Install FlashAttention for optimized attention
RUN pip install --no-cache-dir ninja packaging && \
    pip install --no-cache-dir flash-attn --no-build-isolation || echo "FlashAttention optional, continuing..."

# Install LLM Agent Framework (LangChain, LangGraph)
RUN pip install --no-cache-dir \
    langchain>=0.1.0 \
    langchain-community>=0.0.20 \
    langchain-core>=0.1.20 \
    langgraph>=0.0.20 \
    langsmith>=0.0.80

# Install LLM Provider APIs
RUN pip install --no-cache-dir \
    openai>=1.12.0 \
    anthropic>=0.8.0 \
    google-generativeai>=0.3.0

# Install development tools
RUN pip install --no-cache-dir \
    pytest \
    black \
    isort

# Copy the project files
COPY llm_compressor/ ./llm_compressor/
COPY scripts/ ./scripts/
COPY start_real_compression.sh .
COPY test_llm_system.py .
COPY Makefile .
COPY README.md .
COPY CLAUDE.md .

# Create necessary directories
RUN mkdir -p reports artifacts logs models

# Set permissions for scripts
RUN chmod +x scripts/*.py

# Create a non-root user (use different UID to avoid conflict)
RUN useradd -m -u 1001 llmuser && \
    chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python3 -c "import llm_compressor; print('LLM Compressor OK')" || exit 1

# Default command
CMD ["python3", "scripts/run_search.py", "--help"]

# Build arguments for customization
ARG MODEL_NAME="google/gemma-3-4b-it"
ARG GPU_TYPE="RTX_4090"
ARG SEQUENCE_LENGTH="4096"
ARG LLM_PROVIDER="openai"

# Environment variables that can be overridden
ENV MODEL_NAME=${MODEL_NAME}
ENV GPU_TYPE=${GPU_TYPE}
ENV SEQUENCE_LENGTH=${SEQUENCE_LENGTH}
ENV DEFAULT_LLM_PROVIDER=${LLM_PROVIDER}

# LLM API key placeholders (to be set at runtime)
ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""
ENV GOOGLE_API_KEY=""

# LangChain tracing (optional)
ENV LANGCHAIN_TRACING_V2="false"
ENV LANGCHAIN_ENDPOINT=""
ENV LANGCHAIN_API_KEY=""

# Labels for metadata
LABEL maintainer="LLM Compressor Team"
LABEL version="2.0.0"
LABEL description="LLM-driven intelligent multi-agent system for LLM compression and optimization"
LABEL gpu.required="true"
LABEL gpu.vendor="nvidia"
LABEL gpu.memory.min="22GB"
LABEL llm.framework="langchain"
LABEL llm.orchestration="langgraph"
LABEL llm.providers="openai,anthropic,google"

# Documentation
LABEL org.opencontainers.image.title="LLM Compressor 2.0"
LABEL org.opencontainers.image.description="LLM-driven intelligent multi-agent system using LangChain and LangGraph"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.vendor="LLM Compressor Team"
LABEL org.opencontainers.image.source="https://github.com/example/llm-compressor"
LABEL org.opencontainers.image.documentation="https://github.com/example/llm-compressor/blob/main/README.md"