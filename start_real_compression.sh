#!/bin/bash
# Script to start LLM compression with real model support

set -e

echo "=========================================="
echo "Starting LLM Compression with Real Models"
echo "=========================================="

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
else
    echo "Running on host system"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "Warning: NVIDIA GPU not detected"
fi

# Set environment variables
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/app/models}
export HF_HOME=${HF_HOME:-/app/models}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Parse command line arguments
CONFIG_FILE=${1:-configs/gemma3_270m.yaml}
RECIPES=${2:-baseline}
LOG_LEVEL=${3:-INFO}

# Start vLLM server if requested
if [ "$START_VLLM" = "true" ]; then
    echo "Starting vLLM server..."
    
    # Extract model name from config file
    MODEL_NAME=$(python -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(config['model']['base_model'])")
    MAX_LEN=$(python -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(config['model']['max_model_len'])")
    
    echo "Using model: $MODEL_NAME with max_len: $MAX_LEN"
    
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --host 0.0.0.0 \
        --port 8000 \
        --dtype float16 \
        --max-model-len "$MAX_LEN" &
    VLLM_PID=$!
    echo "vLLM server started with PID: $VLLM_PID"
    
    # Wait for server to be ready
    echo "Waiting for vLLM server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "vLLM server is ready!"
            break
        fi
        sleep 2
    done
fi

# Run the compression pipeline
echo ""
echo "Running compression pipeline:"
echo "  Config: $CONFIG_FILE"
echo "  Recipes: $RECIPES"
echo "  Log Level: $LOG_LEVEL"
echo ""

python scripts/run_search.py \
    --config "$CONFIG_FILE" \
    --recipes "$RECIPES" \
    --log-level "$LOG_LEVEL"

# Cleanup
if [ ! -z "$VLLM_PID" ]; then
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
fi

echo "=========================================="
echo "Compression pipeline completed!"
echo "=========================================="