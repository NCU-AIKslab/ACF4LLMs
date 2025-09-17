#!/bin/bash

# Docker run script for LLM-driven optimization system
# Usage: ./run_docker.sh [command] [options]

set -e

# Default values
CONTAINER_NAME="llm-compressor"
IMAGE_NAME="llm-compressor:latest"
CONFIG_FILE="llm_compressor/configs/default.yaml"
RECIPE_TYPE="conservative"
OUTPUT_DIR="reports"
GPU_DEVICES="all"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check for NVIDIA Docker support
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        print_warning "NVIDIA Docker support not detected. GPU optimization may not work."
        print_info "To install NVIDIA Container Toolkit:"
        print_info "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
    return 0
}

# Check API keys
check_api_keys() {
    local has_keys=false

    if [[ -n "$OPENAI_API_KEY" ]]; then
        print_success "OpenAI API key found"
        has_keys=true
    fi

    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        print_success "Anthropic API key found"
        has_keys=true
    fi

    if [[ -n "$GOOGLE_API_KEY" ]]; then
        print_success "Google API key found"
        has_keys=true
    fi

    if [[ "$has_keys" == false ]]; then
        print_warning "No LLM API keys found. System will run in mock mode."
        print_info "To use real LLM agents, set one or more of:"
        print_info "  export OPENAI_API_KEY='your-openai-key'"
        print_info "  export ANTHROPIC_API_KEY='your-anthropic-key'"
        print_info "  export GOOGLE_API_KEY='your-google-key'"
    fi
}

# Build Docker image
build_image() {
    print_info "Building LLM-driven optimization Docker image..."

    docker build \
        --build-arg MODEL_NAME="google/gemma-3-4b-it" \
        --build-arg GPU_TYPE="RTX_4090" \
        --build-arg LLM_PROVIDER="openai" \
        -t $IMAGE_NAME \
        .

    if [[ $? -eq 0 ]]; then
        print_success "Docker image built successfully: $IMAGE_NAME"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Run optimization with Docker
run_optimization() {
    local recipe_type=$1
    local config_file=$2
    local output_dir=$3
    local extra_args="${@:4}"

    print_info "Running LLM-driven optimization with Docker..."
    print_info "Recipe: $recipe_type"
    print_info "Config: $config_file"
    print_info "Output: $output_dir"

    # Prepare environment variables for LLM APIs
    local env_args=""
    [[ -n "$OPENAI_API_KEY" ]] && env_args="$env_args -e OPENAI_API_KEY=$OPENAI_API_KEY"
    [[ -n "$ANTHROPIC_API_KEY" ]] && env_args="$env_args -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    [[ -n "$GOOGLE_API_KEY" ]] && env_args="$env_args -e GOOGLE_API_KEY=$GOOGLE_API_KEY"
    [[ -n "$LANGCHAIN_API_KEY" ]] && env_args="$env_args -e LANGCHAIN_API_KEY=$LANGCHAIN_API_KEY"
    [[ -n "$LANGCHAIN_TRACING_V2" ]] && env_args="$env_args -e LANGCHAIN_TRACING_V2=$LANGCHAIN_TRACING_V2"

    # GPU support
    local gpu_args=""
    if check_nvidia_docker; then
        gpu_args="--gpus $GPU_DEVICES"
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Run the container
    docker run \
        $gpu_args \
        --rm \
        --name $CONTAINER_NAME \
        $env_args \
        -v "$(pwd)/$output_dir:/app/reports" \
        -v "$(pwd)/llm_compressor/configs:/app/configs" \
        $IMAGE_NAME \
        python3 scripts/run_search.py \
        --config configs/default.yaml \
        --recipes $recipe_type \
        --output reports \
        $extra_args
}

# Interactive shell
run_shell() {
    print_info "Starting interactive shell in LLM optimization container..."

    local env_args=""
    [[ -n "$OPENAI_API_KEY" ]] && env_args="$env_args -e OPENAI_API_KEY=$OPENAI_API_KEY"
    [[ -n "$ANTHROPIC_API_KEY" ]] && env_args="$env_args -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    [[ -n "$GOOGLE_API_KEY" ]] && env_args="$env_args -e GOOGLE_API_KEY=$GOOGLE_API_KEY"

    local gpu_args=""
    if check_nvidia_docker; then
        gpu_args="--gpus $GPU_DEVICES"
    fi

    docker run \
        $gpu_args \
        --rm \
        -it \
        --name $CONTAINER_NAME \
        $env_args \
        -v "$(pwd)/reports:/app/reports" \
        -v "$(pwd)/llm_compressor/configs:/app/configs" \
        $IMAGE_NAME \
        /bin/bash
}

# Test the system
run_test() {
    print_info "Running LLM system tests in Docker..."

    local env_args=""
    [[ -n "$OPENAI_API_KEY" ]] && env_args="$env_args -e OPENAI_API_KEY=$OPENAI_API_KEY"
    [[ -n "$ANTHROPIC_API_KEY" ]] && env_args="$env_args -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    [[ -n "$GOOGLE_API_KEY" ]] && env_args="$env_args -e GOOGLE_API_KEY=$GOOGLE_API_KEY"

    docker run \
        --rm \
        --name $CONTAINER_NAME-test \
        $env_args \
        $IMAGE_NAME \
        python3 test_llm_system.py
}

# Show usage
show_usage() {
    echo "LLM-Driven Optimization Docker Runner"
    echo "====================================="
    echo
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  build                     Build the Docker image"
    echo "  baseline                  Run baseline performance measurement"
    echo "  conservative              Run conservative optimization"
    echo "  aggressive                Run aggressive optimization"
    echo "  llm-planned               Run LLM-planned optimization portfolio"
    echo "  shell                     Start interactive shell"
    echo "  test                      Run system tests"
    echo "  help                      Show this help message"
    echo
    echo "Examples:"
    echo "  $0 build                                    # Build Docker image"
    echo "  $0 conservative                            # Run conservative optimization"
    echo "  $0 aggressive --output reports/aggressive  # Run aggressive with custom output"
    echo "  $0 shell                                   # Interactive development"
    echo "  $0 test                                    # Verify system works"
    echo
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY           OpenAI API key for GPT models"
    echo "  ANTHROPIC_API_KEY        Anthropic API key for Claude models"
    echo "  GOOGLE_API_KEY           Google API key for Gemini models"
    echo "  LANGCHAIN_API_KEY        LangSmith API key for tracing"
    echo "  LANGCHAIN_TRACING_V2     Enable LangChain tracing (true/false)"
    echo
    echo "GPU Support:"
    echo "  Requires NVIDIA Container Toolkit for GPU acceleration"
    echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
}

# Main script logic
main() {
    local command=${1:-help}
    shift || true

    # Check prerequisites
    check_docker
    check_api_keys

    case $command in
        build)
            build_image
            ;;
        baseline)
            run_optimization "baseline" "$CONFIG_FILE" "$OUTPUT_DIR" "$@"
            ;;
        conservative)
            run_optimization "conservative" "$CONFIG_FILE" "$OUTPUT_DIR" "$@"
            ;;
        aggressive)
            run_optimization "aggressive" "$CONFIG_FILE" "$OUTPUT_DIR" "$@"
            ;;
        llm-planned)
            run_optimization "llm_planned" "$CONFIG_FILE" "$OUTPUT_DIR" "$@"
            ;;
        shell)
            run_shell
            ;;
        test)
            run_test
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"