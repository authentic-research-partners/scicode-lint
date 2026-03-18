# Installation Guide

**Local/Institutional vLLM Required:** scicode-lint uses vLLM for inference. You need either:
- Local GPU (16GB+ VRAM with native FP8 support)
- Access to institutional GPU cluster with vLLM server

**Hardware requirements:**
- Minimum 16GB VRAM
- Native FP8 support (compute capability >= 8.9)
- Examples: RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, RTX 4000 Ada, L4, L40, A10

**No cloud APIs:** OpenAI, Anthropic, etc. not supported (by design) to prevent accidental costs and keep code private.

## Quick Start

**Option A: Using remote vLLM server** (university/institutional)
```bash
pip install scicode-lint
scicode-lint lint path/to/code.py --vllm-url https://vllm.your-institution.edu

# Or configure once in ~/.config/scicode-lint/config.toml:
# [llm]
# base_url = "https://vllm.your-institution.edu"
```

**Option B: Running vLLM locally** (requires 16GB+ GPU)
```bash
# 1. Install with vLLM server
pip install scicode-lint[vllm-server]

# 2. Start vLLM server (downloads model on first run - see "Model Storage" below)
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic \
    --trust-remote-code --gpu-memory-utilization 0.85 \
    --max-model-len 20000

# 3. Run the linter (in another terminal)
scicode-lint lint path/to/code.py
```

## Installation: Isolated Environment (Recommended)

**⚠️ Important:** The `[vllm-server]` extra has heavy dependencies. Install in an isolated environment to avoid conflicts. If using a remote vLLM server, `pip install scicode-lint` (without extras) has minimal dependencies.

**Safety Note:** scicode-lint only *reads* your code files as text - it never executes or imports your code.

### Option 1: pipx (Recommended for CLI Usage)

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install scicode-lint                  # Remote vLLM server
pipx install scicode-lint[vllm-server]     # Local vLLM server
```

### Option 2: Dedicated Environment (For Python API or Development)

**Using conda:**
```bash
conda create -n scicode python=3.13
conda activate scicode
pip install scicode-lint                   # Remote vLLM server
# or
pip install scicode-lint[vllm-server]      # Local vLLM server

# For development:
pip install -e ".[all]"
```

**Using venv:**
```bash
python -m venv ~/.scicode-venv
source ~/.scicode-venv/bin/activate
pip install scicode-lint                   # Remote vLLM server
# or
pip install scicode-lint[vllm-server]      # Local vLLM server
```

**Note:** Activate the environment in each new terminal session before using scicode-lint or Claude Code:
```bash
conda activate scicode   # or: source ~/.scicode-venv/bin/activate
```

### Option 3: Install in Project Environment (Not Recommended)

May cause dependency conflicts with your project.

---

## Installation Options

### Local vLLM Server (Recommended)

Install the vLLM server locally to host models on your GPU:

```bash
# For GPU (CUDA)
pip install scicode-lint[vllm-server]

# For CPU-only systems
pip install vllm-cpu scicode-lint
```

**Pros:**
- **No external dependencies** - just pip install
- Best performance on GPU (prefix caching, batching)
- Full control over model and settings
- Works offline
- Supports both GPU and CPU

**Cons:**
- GPU: Requires CUDA GPU with 16GB+ VRAM and native FP8 support
- CPU-only: Not supported
- ~2-3GB download for vLLM + ~13GB for model

**Starting vLLM:**
```bash
vllm serve \
    --host 0.0.0.0 \
    --port 5001 \
    --model RedHatAI/Qwen3-8B-FP8-dynamic \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 20000  # 20K context (16K input + 4K response)
```

Or use the CLI command (auto-verifies hardware):
```bash
scicode-lint vllm-server start
```

The script will exit with an error if your hardware doesn't meet requirements (16GB+ VRAM, compute cap >= 8.9).

**Starting vLLM (CPU):**
```bash
vllm serve \
    --model RedHatAI/Qwen3-8B-FP8-dynamic \
    --trust-remote-code \
    --max-model-len 20000 \
    --device cpu
```

**Note:** CPU inference is 10-50x slower than GPU. vLLM supports CPU mode with `--device cpu` flag.

## Google Colab

Colab is not supported. Use cloud providers with L4/A10 GPUs instead.

**Requirements:**
- 16GB+ VRAM
- Native FP8 support (compute capability >= 8.9)

**Setup in Colab:**

```python
# 1. Install scicode-lint with vLLM
!pip install scicode-lint[vllm-server]

# 2. Start vLLM server in background
# NOTE: First run downloads ~13GB model to /root/.cache/huggingface/ (takes 2-5 min)
!nohup vllm serve \
    --model RedHatAI/Qwen3-8B-FP8-dynamic \
    --trust-remote-code \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 5001 > /tmp/vllm.log 2>&1 &

# 3. Wait for server to start (30-60 seconds)
import time
import httpx
for i in range(30):
    try:
        response = httpx.get("http://localhost:5001/health")
        if response.status_code == 200:
            print("✓ vLLM server ready!")
            break
    except:
        pass
    time.sleep(2)
    print(f"Waiting for server... ({i*2}s)")

# 4. Check your code
!scicode-lint lint your_file.py

# Or use Python API
from pathlib import Path
from scicode_lint import SciCodeLinter

linter = SciCodeLinter()
result = linter.check_file(Path("your_file.py"))

for finding in result.findings:
    print(f"{finding.severity} | {finding.id}")
    print(f"  {finding.explanation}\n")
```

**Best for:**
- ✅ Trying scicode-lint without local GPU
- ✅ Demos, tutorials, workshops
- ✅ Quick one-off file checks
- ✅ Learning how the tool works

**Not suitable for:**
- ❌ Production use (session timeouts)
- ❌ CI/CD pipelines (unreliable)
- ❌ Batch processing many files (runtime limits)
- ❌ Long-running analysis

**Tips:**
- Upload your Python files to Colab using the file browser
- Server needs restart after session timeout
- First run downloads model to `/root/.cache/huggingface/` (~13GB, takes 2-5 minutes)
- Subsequent runs use cached model (starts in seconds)

---

## HPC Cluster Usage

**Recommended approaches on HPC:**

1. **Use institutional vLLM server** (best option)
   - Ask your HPC admin if they provide a shared vLLM inference server
   - No GPU allocation needed, fair resource sharing
   - Point scicode-lint to the server URL

2. **Use dedicated inference nodes**
   - Use if your cluster has dedicated inference nodes (L4, A10)
   - Requires 16GB+ VRAM and native FP8 support (compute cap >= 8.9)
   - Check with your HPC admin first

**Example: Using institutional vLLM server**
```bash
# Option 1: CLI flag (per-command)
scicode-lint lint your_code.py --vllm-url https://vllm.your-hpc.edu

# Option 2: Environment variable (per-session)
export OPENAI_BASE_URL="https://vllm.your-hpc.edu/v1"
scicode-lint lint your_code.py

# Option 3: Config file (persistent) - create ~/.config/scicode-lint/config.toml
# [llm]
# base_url = "https://vllm.your-hpc.edu"
```

---

### Option 3: Remote vLLM Server

Connect to a remote vLLM server (institutional or self-hosted):

**Note:** Remote vLLM servers already have a model loaded. You use whatever model the server admin chose - no model selection or hardware detection needed on your end.

**CLI usage:**
```bash
# Install scicode-lint only (no local server needed)
pip install scicode-lint

# Use remote vLLM server
scicode-lint lint path/to/code.py --vllm-url https://your-vllm-server.com

# Or via environment variable
export OPENAI_BASE_URL="https://your-vllm-server.com/v1"
scicode-lint lint path/to/code.py
```

**Python API usage:**
```python
from scicode_lint.vllm import VLLMServer
from scicode_lint import SciCodeLinter

# Connect to remote vLLM (verifies connectivity only, never starts/stops)
with VLLMServer(base_url="http://gpu-cluster.your-institution.edu:5001"):
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
```

**Pros:**
- No local resources needed
- Works on any machine (even CPU-only)
- Scalable across team/institution

**Cons:**
- Requires network connection
- Need access to a vLLM server

**Note:** scicode-lint only supports vLLM servers. It does NOT work with commercial APIs (OpenAI, Anthropic, etc.) to avoid accidental API costs.

## Development Installation

For development with testing and linting tools:

```bash
# Clone the repository
git clone https://github.com/ssamsonau/scicode-lint
cd scicode-lint

# Create isolated environment (recommended)
conda create -n scicode python=3.13
conda activate scicode

# Install all dependencies (dev, vllm, eval, dashboard, etc.)
pip install -e ".[all]"

# Run tests
pytest

# Run linter checks
ruff check . && ruff format .
mypy .
```

## System Requirements

**For FP8 models (default: Qwen3-8B-FP8):**

- **Python:** 3.13+
- **GPU:** NVIDIA with native FP8 support
  - **VRAM:** 16GB minimum (20K context: 16K input + 4K response)
  - **Compute capability:** >= 8.9 (native FP8 tensor cores)
  - **Supported GPUs:**
    - Consumer: RTX 4060 Ti 16GB, RTX 4070+ (16GB+), RTX 4090 (24GB)
    - Workstation: RTX 4000 Ada (20GB), RTX 5000 Ada (32GB)
    - Cloud/HPC inference: L4 (24GB), L40 (48GB), A10 (24GB)
- **Disk space:** ~15GB free for model weights (see "Model Storage" section below)

**Tested on:** Windows/WSL2
**Expected to work on:** Linux

## Model Storage

**⚠️ Important: vLLM downloads models on first launch**

When you first start vLLM with a model, it will automatically download the model weights from HuggingFace. This is a one-time download that requires disk space.

**Default storage location:**
```
~/.cache/huggingface/hub/
```

**Disk space requirements:**
- **FP8 model** (RedHatAI/Qwen3-8B-FP8-dynamic): ~13GB
- **Recommended free space:** ~15GB (allows for updates and cache)

**First run behavior:**
```bash
# First time running vLLM - downloads model (2-5 minutes depending on connection)
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic --trust-remote-code

# Output will show:
# Downloading model from HuggingFace...
# ━━━━━━━━━━━━━━━━━━━━━ 100% 13.3GB/13.3GB

# Subsequent runs - uses cached model (starts in seconds)
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic --trust-remote-code
```

**Customizing cache location:**

Set the `HF_HOME` environment variable to change where models are stored:

```bash
# Store models in custom location (e.g., larger disk partition)
export HF_HOME=/mnt/data/huggingface
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic
```

**Managing cached models:**

Model weights are stored in `~/.cache/huggingface/hub/`. To remove a specific model or clear the cache, delete the corresponding directory.

## Troubleshooting

### vLLM installation fails

Try installing with the specific version:
```bash
pip install vllm==0.16.0
```

### GPU not detected

Verify CUDA and PyTorch are properly installed and detecting your GPU.

### Model download is slow

Model downloads to `~/.cache/huggingface/` on first use (~13GB). You can pre-download:

```bash
# Pre-download FP8 model
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RedHatAI/Qwen3-8B-FP8-dynamic')"
```

To use a custom download location:
```bash
export HF_HOME=/path/to/large/disk
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RedHatAI/Qwen3-8B-FP8-dynamic')"
```

### WSL2 issues

vLLM 0.16.0+ works well on WSL2. If you have issues:
1. Ensure CUDA drivers are up to date
2. Try: `export VLLM_USE_TRITON_FLASH_ATTN=0`

## Configuration

Create a `config.toml` in your project or `~/.config/scicode-lint/`:

```toml
[llm]
# base_url = "http://localhost:5001"  # Optional, auto-detects if not set
# model = "RedHatAI/Qwen3-8B-FP8-dynamic"  # Optional, auto-detects if not set
temperature = 0.3

[linter]
min_confidence = 0.7
enabled_severities = ["critical", "high", "medium"]
```

See [config.toml](src/scicode_lint/config.toml) for all options.
