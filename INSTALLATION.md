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
uv tool install scicode-lint --python 3.13
scicode-lint lint path/to/code.py --vllm-url https://vllm.your-institution.edu

# Or configure once in ~/.config/scicode-lint/config.toml:
# [llm]
# base_url = "https://vllm.your-institution.edu"
```

**Option B: Running vLLM locally** (requires 16GB+ GPU)
```bash
# 1. Install scicode-lint + container runtime
uv tool install scicode-lint --python 3.13
# Install podman (or docker) + nvidia-container-toolkit:
#   sudo apt install podman nvidia-container-toolkit
#   sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 2. Start vLLM container (downloads model on first run - see "Model Storage" below)
scicode-lint vllm-server start

# 3. Run the linter
scicode-lint lint path/to/code.py
```

## Installation: Isolated Environment (Recommended)

**Safety Note:** scicode-lint only *reads* your code files as text - it never executes or imports your code.

### Option 1: uv tool install (Recommended for CLI Usage)

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). uv downloads Python 3.13 automatically (without affecting any system Python you already have) and installs scicode-lint as a globally available command in an isolated environment.

```bash
uv tool install scicode-lint --python 3.13
```

### Option 2: pipx (Alternative for CLI Usage)

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install scicode-lint
```

### Option 3: Plain pip (Advanced)

Works fine if you already have Python 3.13 and manage your own environment. May cause dependency conflicts with other projects — prefer the isolated options above.

```bash
pip install scicode-lint
```

### Option 4: Dedicated Environment (For Python API or Development)

**Using uv:**
```bash
uv venv --python 3.13 ~/.scicode-venv
source ~/.scicode-venv/bin/activate
uv pip install scicode-lint
```

**Using conda:**
```bash
conda create -n scicode python=3.13
conda activate scicode
pip install scicode-lint
```

**Note:** Activate the environment in each new terminal session before using scicode-lint:
```bash
source ~/.scicode-venv/bin/activate    # or: conda activate scicode
```

---

## Installation Options

### Local vLLM Server (Recommended)

vLLM runs in a container (podman or docker) with GPU passthrough:

```bash
# 1. Install scicode-lint
uv tool install scicode-lint --python 3.13

# 2. Install container runtime + GPU support
sudo apt install podman nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Verify GPU passthrough works:
podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi

# 3. Start vLLM container
scicode-lint vllm-server start
```

**Pros:**
- No `pip install vllm` (~2GB) — vLLM runs inside the container
- Pinned vLLM version (v0.18.0) — no version confusion
- Auto-restarts on failure (`--restart unless-stopped`)
- Best performance on GPU (prefix caching, batching)
- Full control over model and settings
- Works offline after first model download

**Cons:**
- Requires CUDA GPU with 16GB+ VRAM and native FP8 support
- Requires podman or docker + nvidia-container-toolkit
- ~13GB model download on first run

**Container management:**
```bash
scicode-lint vllm-server start     # Start container
scicode-lint vllm-server stop      # Stop container
scicode-lint vllm-server status    # Show container + GPU status
scicode-lint vllm-server restart   # Restart (stop + remove + start)
scicode-lint vllm-server logs      # Show vLLM logs
scicode-lint vllm-server logs -f   # Follow logs
scicode-lint vllm-server rm        # Remove container
```


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
uv tool install scicode-lint --python 3.13

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

## Development install

For development with testing and linting tools:

```bash
# Clone the repository
git clone https://github.com/authentic-research-partners/scicode-lint
cd scicode-lint

# Create isolated environment with uv (recommended — fast, manages Python itself)
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install all dependencies (dev, vllm, eval, dashboard, etc.)
uv pip install -e ".[all]"

# Run tests
pytest

# Run linter checks
ruff check . && ruff format .
mypy .
```

**Alternative with conda:**
```bash
conda create -n scicode python=3.13
conda activate scicode
pip install -e ".[all]"
```

### Reproducible Environment (`requirements-pinned.txt`)

`pyproject.toml` declares **minimum** versions (lower bounds), so a plain `pip install` may resolve newer packages as upstream releases ship. For reproducing the exact versions the maintainer develops and tests against, use [`requirements-pinned.txt`](./requirements-pinned.txt).

**What's in it:** top-level packages declared in `pyproject.toml` (runtime + all optional extras) pinned to `==` versions. Transitive dependencies are **not** pinned — pip resolves them within each top-level package's own constraints. This is a deliberate middle ground: it locks the packages scicode-lint code actually imports and tests against, without over-constraining the dep tree.

**When to use which:**
- Library users: `pip install scicode-lint` (resolves against minimums — fine for most use)
- Reproducing benchmarks / debugging a version-drift issue / CI: use the pinned file

**Install from the pinned file:**

> **Note:** Both recipes below create a new virtualenv directory named `.venv` **in your current working directory**. Run them from inside your clone of the scicode-lint repo (so `.venv/` sits next to `pyproject.toml`). `.venv` is already gitignored. If you prefer a different location (e.g. `~/.virtualenvs/scicode-lint`), substitute that path in both the create step and the `source …/bin/activate` step. The resulting venv is independent of any conda env you may already have — it doesn't replace or modify it — but to avoid confusion about which `python` / `pip` is on `PATH`, start from a fresh shell (or `conda deactivate`) before running these commands.

```bash
# With uv (fast, isolated, manages Python itself)
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -r requirements-pinned.txt
uv pip install -e . --no-deps    # install scicode-lint source, deps already pinned

# Or with plain pip
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pinned.txt
pip install -e . --no-deps
```

Activate the venv in each new terminal with `source .venv/bin/activate` (or configure [direnv](https://direnv.net/) for automatic activation).

**Regenerate the pinned file** — run from the environment you actively develop and test in (conda env, venv, whatever), after bumping deps in `pyproject.toml` or upgrading a dependency. The script reads from whatever `python`/`pip` is currently on `PATH`, so *activate first*:
```bash
# Example: conda env named `scicode`
conda activate scicode
python scripts/regenerate_pinned_requirements.py

# Or a venv
source .venv/bin/activate
python scripts/regenerate_pinned_requirements.py
```

The script calls `pip freeze` in the current env, filters to the top-level names declared in `pyproject.toml`, and writes `requirements-pinned.txt`. It exits non-zero if any declared dep is missing from the env — a sanity check that you're running from a fully-installed dev environment. Commit the regenerated file alongside any `pyproject.toml` version bumps in the same PR.

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
# First start — downloads model (2-5 minutes depending on connection)
scicode-lint vllm-server start

# Container logs (scicode-lint vllm-server logs) will show:
# Downloading model from HuggingFace...
# ━━━━━━━━━━━━━━━━━━━━━ 100% 13.3GB/13.3GB

# Subsequent starts — uses cached model (starts in seconds)
scicode-lint vllm-server start
```

**Customizing cache location:**

Set `HF_HOME` before starting the container. The container mounts this volume so
downloads persist across restarts:

```bash
export HF_HOME=/mnt/data/huggingface
scicode-lint vllm-server restart
```

**Managing cached models:**

Model weights are stored in `~/.cache/huggingface/hub/`. To remove a specific model or clear the cache, delete the corresponding directory.

## Troubleshooting

### Container fails to start

Check logs: `scicode-lint vllm-server logs`. Common causes:
- `nvidia-container-toolkit` not installed or CDI not generated — rerun `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
- GPU not visible to container — test with `podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi`

### GPU not detected

Verify `nvidia-smi` works on the host, then verify GPU passthrough into the container (see command above).

### Model download is slow

Model downloads to `~/.cache/huggingface/` on first use (~13GB). You can pre-download outside the container:

```bash
# Pre-download FP8 model
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('RedHatAI/Qwen3-8B-FP8-dynamic')"
```

To use a custom download location:
```bash
export HF_HOME=/path/to/large/disk
scicode-lint vllm-server restart   # container picks up the new cache path
```

### WSL2 issues

Ensure CUDA drivers are up to date and `nvidia-smi` works from WSL2. Then verify container GPU passthrough as above.

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
