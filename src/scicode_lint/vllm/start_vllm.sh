#!/bin/bash
# Start vLLM server locally
#
# Preferred usage (after pip install):
#   scicode-lint vllm-server start
#   scicode-lint vllm-server start --restart
#   scicode-lint vllm-server start --model "Qwen/Qwen3-8B-FP8" --port 5001
#
# Direct usage:
#   bash src/scicode_lint/vllm/start_vllm.sh [OPTIONS] [MODEL] [PORT] [MAX_LEN] [GPU_MEM] [MAX_NUM_SEQS]
#
# Options:
#   --restart, --force    Kill any running vLLM server before starting
#
# All parameters read from config.toml by default.
#
# Examples:
#   bash src/scicode_lint/vllm/start_vllm.sh
#   bash src/scicode_lint/vllm/start_vllm.sh --restart
#   bash src/scicode_lint/vllm/start_vllm.sh "meta-llama/Llama-3.1-8B-Instruct"
#   bash src/scicode_lint/vllm/start_vllm.sh --restart "Qwen/Qwen3-8B-FP8" 5001 20096 0.85 256
#
# For background mode:
#   nohup bash src/scicode_lint/vllm/start_vllm.sh > /tmp/vllm.log 2>&1 &
#
# Requirements:
#   pip install scicode-lint[vllm-server]
#
# Note: This configures the vLLM SERVER (what model it loads).
#       config.toml configures the CLIENT (what it expects - auto-detected by default).

set -e

# Parse flags
RESTART=false
while [[ "$1" == --* ]]; do
    case "$1" in
        --restart|--force)
            RESTART=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Read configuration values from config.toml using Python
# Sets: CFG_MODEL, CFG_MAX_LEN, CFG_GPU_MEM, CFG_MAX_NUM_SEQS, CFG_MIN_VRAM_MB
read_config() {
    local values
    values=$(python3 -c "
try:
    import tomllib
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
    llm = config.get('llm', {})
    vllm_cfg = config.get('vllm', {})
    perf = config.get('performance', {})
    max_input = llm.get('max_input_tokens', 16000)
    max_completion = llm.get('max_completion_tokens', 4096)
    print(llm.get('model', ''))
    print(max_input + max_completion)
    print(vllm_cfg.get('gpu_memory_utilization', 0.85))
    print(perf.get('vllm_max_num_seqs', 256))
    print(vllm_cfg.get('min_vram_mb', 15500))
except Exception:
    print('')
    print(20096)
    print(0.85)
    print(256)
    print(15500)
" 2>/dev/null)
    CFG_MODEL=$(echo "$values" | sed -n '1p')
    CFG_MAX_LEN=$(echo "$values" | sed -n '2p')
    CFG_GPU_MEM=$(echo "$values" | sed -n '3p')
    CFG_MAX_NUM_SEQS=$(echo "$values" | sed -n '4p')
    CFG_MIN_VRAM_MB=$(echo "$values" | sed -n '5p')
    # Apply defaults for empty/missing values
    CFG_MODEL="${CFG_MODEL:-RedHatAI/Qwen3-8B-FP8-dynamic}"
    CFG_MAX_LEN="${CFG_MAX_LEN:-20096}"
    CFG_GPU_MEM="${CFG_GPU_MEM:-0.85}"
    CFG_MAX_NUM_SEQS="${CFG_MAX_NUM_SEQS:-256}"
    CFG_MIN_VRAM_MB="${CFG_MIN_VRAM_MB:-15500}"
}

# Load config once
read_config

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo "✗ vLLM not found."
    echo "  Install with: pip install scicode-lint[vllm-server]"
    exit 1
fi

# Check vLLM version (require 0.16+)
VLLM_VERSION=$(vllm --version 2>&1 | grep -oP 'vllm \K[0-9.]+' | head -1)
if [ -n "$VLLM_VERSION" ]; then
    MAJOR=$(echo "$VLLM_VERSION" | cut -d. -f1)
    MINOR=$(echo "$VLLM_VERSION" | cut -d. -f2)

    if [ "$MAJOR" -eq 0 ] && [ "$MINOR" -lt 16 ]; then
        echo "✗ vLLM version $VLLM_VERSION is too old (requires 0.16+)"
        echo "  Upgrade with: pip install --upgrade vllm"
        exit 1
    fi
    echo "✓ vLLM version $VLLM_VERSION"
fi

# Auto-detect GPU capabilities and set defaults if not specified
if [ -z "$3" ] || [ -z "$4" ]; then
    # Detect compute capability for FP8 support
    COMPUTE_CAP=""
    NATIVE_FP8=false
    GPU_AVAILABLE=false

    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$COMPUTE_CAP" ]; then
            GPU_AVAILABLE=true
            # Convert to comparable integer (e.g., "8.9" -> 89, "8.0" -> 80)
            COMPUTE_CAP_MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)
            COMPUTE_CAP_MINOR=$(echo "$COMPUTE_CAP" | cut -d. -f2)
            COMPUTE_CAP_INT=$((COMPUTE_CAP_MAJOR * 10 + COMPUTE_CAP_MINOR))

            # Native FP8 support requires compute capability >= 8.9 (Ada Lovelace, Hopper)
            if [ "$COMPUTE_CAP_INT" -ge 89 ]; then
                NATIVE_FP8=true
            fi
        fi
    fi

    # Allow override for testing (e.g., SCICODE_VRAM_MB=16384 for simulating 16GB)
    if [ -n "$SCICODE_VRAM_MB" ]; then
        VRAM_MB="$SCICODE_VRAM_MB"
        echo "⚙️  Using simulated VRAM: ${VRAM_MB}MB (for testing)"
    else
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    fi

    if [ -n "$VRAM_MB" ]; then
        # Round up to avoid misleading "19GB" for 20GB cards
        VRAM_GB=$(( (VRAM_MB + 512) / 1024 ))

        MIN_VRAM_MB="$CFG_MIN_VRAM_MB"
        MIN_VRAM_GB=$((MIN_VRAM_MB / 1024))

        # Enforce minimum VRAM from config
        if [ $VRAM_MB -lt $MIN_VRAM_MB ]; then
            echo "✗ ERROR: Detected ${VRAM_GB}GB VRAM. Minimum requirement: ${MIN_VRAM_GB}GB VRAM"
            echo ""
            echo "scicode-lint requires ${MIN_VRAM_GB}GB+ VRAM with native FP8 support."
            echo "Supported GPUs:"
            echo "  • Consumer: RTX 4060 Ti 16GB, RTX 4070+ (16GB+), RTX 4090 (24GB)"
            echo "  • Workstation: RTX 4000 Ada (20GB), RTX 5000 Ada (32GB)"
            echo "  • HPC/Cloud inference: L4 (24GB), L40 (48GB), A10 (24GB)"
            echo ""
            echo "Cloud GPU pricing: L4 ~\$0.39/hr, A10 ~\$0.24/hr"
            echo "See INSTALLATION.md for deployment options."
            exit 1
        fi

        # Enforce native FP8 support (compute capability >= 8.9)
        if [ "$NATIVE_FP8" = false ]; then
            echo "✗ ERROR: GPU lacks native FP8 support"
            echo ""
            echo "Detected compute capability: ${COMPUTE_CAP:-unknown}"
            echo "Required: compute capability >= 8.9"
            echo ""
            echo "Your GPU (likely A100, V100, or RTX 30-series) does not support"
            echo "native FP8 operations. scicode-lint requires modern GPUs with"
            echo "native FP8 tensor cores for reliable performance."
            echo ""
            echo "Supported GPUs (compute cap >= 8.9):"
            echo "  • RTX 40-series (4090, 4000 Ada, 5000 Ada)"
            echo "  • L4, L40, A10"
            echo ""
            echo "Cloud alternatives: L4 ~\$0.39/hr, upgrade from A100/V100"
            exit 1
        fi

        # Use config values (already loaded by read_config)
        DEFAULT_MODEL="$CFG_MODEL"
        DEFAULT_MAX_LEN="$CFG_MAX_LEN"
        DEFAULT_GPU_MEM="$CFG_GPU_MEM"
        DEFAULT_MAX_NUM_SEQS="$CFG_MAX_NUM_SEQS"

        echo "✓ Detected ${VRAM_GB}GB VRAM (compute capability: ${COMPUTE_CAP:-unknown})"
        echo "  → Model: $DEFAULT_MODEL"
        echo "     ${DEFAULT_MAX_LEN} context, ~1,500 line files, native FP8 support"
    else
        # No GPU detected
        echo "✗ ERROR: GPU not detected"
        echo ""
        echo "scicode-lint requires a GPU with:"
        echo "  • ${MIN_VRAM_GB:-16}GB+ VRAM"
        echo "  • Native FP8 support (compute capability >= 8.9)"
        echo ""
        echo "CPU-only mode is not supported."
        echo "See INSTALLATION.md for cloud GPU options (L4 ~\$0.39/hr)."
        exit 1
    fi
else
    # User specified custom max_len and gpu_mem - use config values for defaults
    DEFAULT_MODEL="$CFG_MODEL"
    DEFAULT_MAX_LEN="$CFG_MAX_LEN"
    DEFAULT_GPU_MEM="$CFG_GPU_MEM"
    DEFAULT_MAX_NUM_SEQS="$CFG_MAX_NUM_SEQS"
fi

# Configuration (with optional parameters)
# If user didn't specify model, use hardware-detected default
MODEL="${1:-${DEFAULT_MODEL:-RedHatAI/Qwen3-8B-FP8-dynamic}}"
PORT="${2:-5001}"
MAX_LEN="${3:-$DEFAULT_MAX_LEN}"
GPU_MEM="${4:-$DEFAULT_GPU_MEM}"
MAX_NUM_SEQS="${5:-$DEFAULT_MAX_NUM_SEQS}"

# Set short model name - Qwen3 8B FP8 is the standard model
if [[ "$MODEL" == *"Qwen3-8B"* ]]; then
    SERVED_NAME="qwen3-8b-fp8"
elif [[ "$MODEL" == *"gemma"* ]]; then
    SERVED_NAME="gemma-3-12b-fp8"
else
    # Fallback for any other model
    SERVED_NAME=$(basename "$MODEL" | sed 's/-FP8.*//' | tr '[:upper:]' '[:lower:]')
fi

# Handle --restart flag
if [ "$RESTART" = true ]; then
    if pgrep -f "vllm serve" > /dev/null; then
        echo "🔄 Stopping existing vLLM server..."
        pkill -f "vllm serve"
        sleep 3
        echo "✓ Stopped"
    else
        echo "ℹ  No vLLM server running"
    fi
fi

# Check if vLLM is already running on this port
if curl -s "http://localhost:${PORT}/v1/models" &> /dev/null; then
    echo "✓ vLLM server already running on port ${PORT}"
    echo ""
    echo "Current model:"
    curl -s "http://localhost:${PORT}/v1/models" | python3 -c "import sys, json; data=json.load(sys.stdin); print('  ' + data['data'][0]['id'])" 2>/dev/null || echo "  (unable to detect)"
    echo ""
    echo "To restart with different settings, use:"
    echo "  bash $0 --restart [MODEL] [PORT] [MAX_LEN] [GPU_MEM]"
    exit 0
fi

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  Served as: $SERVED_NAME"
echo "  Port: $PORT"
echo "  Max length: $MAX_LEN tokens"
echo "  Max sequences: $MAX_NUM_SEQS"
echo "  GPU memory utilization: ${GPU_MEM}"
echo ""
echo "Model will download automatically on first run (~8GB)"
echo ""

vllm serve \
    --host 0.0.0.0 \
    --port $PORT \
    --model $MODEL \
    --served-model-name $SERVED_NAME \
    --trust-remote-code \
    --gpu-memory-utilization $GPU_MEM \
    --max-model-len $MAX_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill
