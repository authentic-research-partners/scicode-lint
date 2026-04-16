# vLLM Server Utilities

Lightweight utilities for managing vLLM server lifecycle in automated workflows.

**Optional module** - Not required for normal scicode-lint usage. Designed for:
- GenAI coding agents that need automated server management
- Testing and CI/CD pipelines
- Scripted workflows

---

## When to Use

**Use these utilities if:**
- You're a GenAI agent automating the full workflow (start server → lint → stop server)
- You're writing test scripts that need temporary vLLM instances
- You want programmatic control over server lifecycle

**Don't use if:**
- You manually start vLLM once and keep it running (most users)
- You use Docker/systemd/supervisor to manage vLLM
- You connect to a remote vLLM server

---

## Quick Start

### Option 1: Context Manager - Local Server (Recommended)

```python
from pathlib import Path
from scicode_lint import SciCodeLinter
from scicode_lint.vllm import VLLMServer

# Server starts automatically if not running, stops when done
with VLLMServer():
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
    print(f"Found {len(result.findings)} issues")

# Server automatically stopped (only if we started it)
```

**Smart behavior:**
- If server already running → uses it, doesn't stop it
- If server not running → starts it, stops when done
- If running with different model → warns but uses it

### Option 2: Context Manager - Remote Server

```python
from scicode_lint.vllm import VLLMServer

# Connect to remote vLLM server (never starts/stops it)
with VLLMServer(base_url="http://gpu-cluster.example.com:5001"):
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))

# Remote server is left running
```

**Remote server notes:**
- Only verifies connectivity
- Never attempts to start or stop
- Raises error if not reachable

### Option 3: Manual Control (Local Only)

```python
from scicode_lint.vllm import start_server, stop_server, wait_for_ready

# Start server
proc = start_server(wait=True)  # Blocks until ready

try:
    # Use linter
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
finally:
    # Always stop server
    stop_server(proc)
```

---

## API Reference

### `is_running()`

Check if vLLM server is responding.

```python
from scicode_lint.vllm import is_running

if is_running("http://localhost:5001"):
    print("Server is ready")
else:
    print("Server not running")
```

**Parameters:**
- `base_url` (str, optional): Server URL to check. Default: `"http://localhost:5001"`

**Returns:**
- `bool`: True if server is responding, False otherwise

---

### `wait_for_ready()`

Wait for vLLM server to become ready.

```python
from scicode_lint.vllm import wait_for_ready

if wait_for_ready(timeout=30):
    print("Server is ready")
else:
    print("Timeout - server didn't start")
```

**Parameters:**
- `base_url` (str, optional): Server URL to check. Default: `"http://localhost:5001"`
- `timeout` (int, optional): Maximum wait time in seconds. Default: `60`
- `check_interval` (float, optional): Time between checks. Default: `2.0`

**Returns:**
- `bool`: True if server became ready, False if timeout

---

### `start_server()`

Start vLLM server as a subprocess.

```python
from scicode_lint.vllm import start_server

# Start with defaults
proc = start_server()

# Start with custom config
proc = start_server(
    model="RedHatAI/Qwen3-8B-FP8-dynamic",
    port=5001,
    max_model_len=20000,
    gpu_memory_utilization=0.85,
    wait=True,  # Block until ready
)
```

**Parameters:**
- `model` (str, optional): Model name or path. Default: `"RedHatAI/Qwen3-8B-FP8-dynamic"`
- `port` (int, optional): Port number. Default: `5001`
- `max_model_len` (int, optional): Max context length. Default: `20000`
- `gpu_memory_utilization` (float, optional): GPU memory 0.0-1.0. Default: from `config.toml`
- `wait` (bool, optional): Wait for server ready. Default: `False`
- `wait_timeout` (int, optional): Timeout for waiting. Default: `60`

**Returns:**
- `subprocess.Popen`: Server process object

**Raises:**
- `FileNotFoundError`: If vllm command not found
- `RuntimeError`: If server already running on port
- `TimeoutError`: If wait=True and server doesn't start in time

**Note:** Model downloads automatically on first run (~8GB)

---

### `stop_server()`

Stop vLLM server gracefully.

```python
from scicode_lint.vllm import start_server, stop_server

proc = start_server()
# ... use server ...
stop_server(proc)
```

**Parameters:**
- `process` (subprocess.Popen): Process from `start_server()`
- `timeout` (int, optional): Graceful shutdown timeout. Default: `10`

**Behavior:**
- Tries graceful termination (SIGTERM)
- Falls back to kill (SIGKILL) if timeout

---

### `VLLMServer` (Context Manager)

Context manager for automatic server lifecycle. Supports both local and remote servers.

```python
from scicode_lint.vllm import VLLMServer

# Local server (auto-start if needed)
with VLLMServer():
    # Server running here (started if needed)
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
# Server stopped only if we started it

# Remote server
with VLLMServer(base_url="http://gpu-cluster:5001"):
    # Uses remote server (never starts/stops)
    linter = SciCodeLinter()
    # ... use linter ...

# Custom local config
with VLLMServer(port=8000, max_model_len=20000):
    # Server on port 8000
    linter = SciCodeLinter()
    # ... use linter ...
```

**Parameters:**
- `model` (str, optional): Model name. Default: `"RedHatAI/Qwen3-8B-FP8-dynamic"`. Only used for local servers.
- `port` (int, optional): Port number. Default: `5001`. Only used for local servers.
- `base_url` (str, optional): Full URL for remote server (e.g., `"http://10.0.0.5:5001"`). If provided, server is treated as remote (no start/stop).
- `max_model_len` (int, optional): Max context. Default: `20000`. Only used for local servers.
- `gpu_memory_utilization` (float, optional): GPU memory. Default: from `config.toml`. Only used for local servers.
- `wait_timeout` (int, optional): Startup/verification timeout. Default: `60`

**Behavior:**

| Scenario | Action |
|----------|--------|
| Local, not running | Starts server, stops when done |
| Local, already running | Uses it, doesn't stop |
| Local, wrong model | Warns but uses it (stop manually to change) |
| Remote, running | Uses it, never stops |
| Remote, not running | Raises `RuntimeError` |

---

## Complete Examples

### Example 1: GenAI Agent Automated Workflow

```python
from pathlib import Path
from scicode_lint import SciCodeLinter
from scicode_lint.vllm import VLLMServer

def check_and_fix_with_auto_server(file_path: str):
    """
    Complete automated workflow:
    1. Start vLLM server
    2. Check code
    3. Fix issues
    4. Stop server
    """

    with VLLMServer():
        linter = SciCodeLinter()
        result = linter.check_file(Path(file_path))

        if not result.findings:
            print(f"✓ {file_path}: Clean")
            return

        print(f"Found {len(result.findings)} issues:")
        for finding in result.findings:
            print(f"  {finding.id}: {finding.explanation}")
            # AI agent applies fix here
            apply_fix(file_path, finding)

        # Verify fixes
        result = linter.check_file(Path(file_path))
        if not result.findings:
            print(f"✓ All issues fixed")

    # Server automatically stopped
```

### Example 2: Testing Multiple Files

```python
from scicode_lint.vllm import VLLMServer

def test_multiple_files():
    """Test multiple files with one server instance."""

    files = ["pipeline.py", "train.py", "evaluate.py"]

    # Start server once for all files
    with VLLMServer():
        linter = SciCodeLinter()

        for file in files:
            result = linter.check_file(Path(file))
            print(f"{file}: {len(result.findings)} issues")
```

### Example 3: Remote Server on HPC Cluster

```python
from scicode_lint.vllm import VLLMServer
from pathlib import Path

def check_on_remote_gpu():
    """Use remote vLLM server on HPC cluster or remote machine."""

    # Connect to remote server (must be already running)
    with VLLMServer(base_url="http://gpu-node-42.cluster.edu:5001"):
        linter = SciCodeLinter()

        # Process multiple files using remote GPU
        for file in Path("src").glob("**/*.py"):
            result = linter.check_file(file)
            if result.findings:
                print(f"{file}: {len(result.findings)} issues")

    # Remote server is left running
```

**Use cases for remote servers:**
- HPC clusters with GPU nodes
- Shared GPU servers in your lab/organization
- Cloud GPU instances
- Any vLLM server you can't control (start/stop)

### Example 4: Custom Configuration (Local)

```python
from scicode_lint.vllm import start_server, stop_server

# Start with custom config
proc = start_server(
    model="RedHatAI/Qwen3-8B-FP8-dynamic",
    port=5001,
    max_model_len=20000,
    gpu_memory_utilization=0.85,
    wait=True,  # Wait for ready
    wait_timeout=120,  # 2 minutes timeout
)

try:
    # Check large file
    linter = SciCodeLinter()
    result = linter.check_file(Path("large_file.py"))
finally:
    stop_server(proc)
```

### Example 5: Error Handling

```python
from scicode_lint.vllm import start_server, stop_server, is_running

# Check if already running
if is_running():
    print("Server already running, using existing")
    linter = SciCodeLinter()
else:
    print("Starting new server")
    proc = None
    try:
        proc = start_server(wait=True, wait_timeout=60)
        linter = SciCodeLinter()
        result = linter.check_file(Path("myfile.py"))
    except TimeoutError:
        print("Server failed to start")
    except FileNotFoundError:
        print("Container runtime missing. Run: sudo apt install podman nvidia-container-toolkit")
    finally:
        if proc:
            stop_server(proc)
```

---

## Requirements

**Container runtime must be installed** (vLLM runs in a container):
```bash
sudo apt install podman nvidia-container-toolkit
# (docker works too)
```

**System requirements:**
- NVIDIA GPU with native FP8 support (compute capability >= 8.9)
- 16GB+ VRAM (for FP8 models)
- Linux or WSL2

---

## Troubleshooting

### Server fails to start

**Error:** `neither podman nor docker found on PATH`

**Solution:** Install a container runtime: `sudo apt install podman nvidia-container-toolkit`

---

### Timeout waiting for server

**Error:** `TimeoutError: Server failed to start within 60 seconds`

**Possible causes:**
- Model downloading (first run takes time - ~8GB download)
- GPU not available
- Port already in use

**Solutions:**
```python
# Increase timeout for first run
proc = start_server(wait=True, wait_timeout=300)  # 5 minutes

# Check if port in use
from scicode_lint.vllm import is_running
if is_running("http://localhost:5001"):
    print("Port already in use")
```

---

### Server already running

**Error:** `RuntimeError: Server already running on port 5001`

**Solution:**
```python
# Use existing server
if is_running():
    linter = SciCodeLinter()  # Use existing
else:
    with VLLMServer():
        linter = SciCodeLinter()
```

---

### Wrong model running

**Warning:** `Local vLLM server is running with model 'X' but you requested 'Y'`

**What it means:**
- A local vLLM server is already running
- It's serving a different model than you requested
- The context manager will use the running model (not restart)

**Solutions:**

**Option 1: Use the running model** (recommended)
```python
# Just proceed, the warning is informational
with VLLMServer():
    linter = SciCodeLinter()  # Uses running model
```

**Option 2: Restart with desired model**

Stop the running server and start with the desired model using `start_vllm.sh`.

**Option 3: Check what's running first**
```python
from scicode_lint.vllm import get_server_info

info = get_server_info()
if info.is_running:
    print(f"Running model: {info.model}")
```

---

### Remote server not reachable

**Error:** `RuntimeError: Remote vLLM server not reachable at http://...`

**What it means:**
- You specified a `base_url` (remote server)
- The server is not responding

**Solutions:**
1. Check the URL is correct
2. Verify the remote server is running
3. Check network connectivity and firewall rules

---

## Performance Tips

1. **Reuse server for multiple files** - Starting server is slow, keep it running
2. **Use context manager** - Ensures cleanup even on errors
3. **Increase timeout on first run** - Model download takes time
4. **Check if running first** - Avoid starting duplicate servers

---

## Limitations

1. **Only vLLM servers** - Requires vLLM server (local or remote), not other LLM APIs
2. **No output capture** - Local server logs go to stdout/stderr
3. **Simple process management** - Not production-grade (use systemd/Docker for that)
4. **Single server at a time** - No multi-server support per instance
5. **Remote servers** - Can connect but cannot start/stop/manage

---

## When NOT to Use

- **Production deployments** - Use Docker, systemd, or process managers for local servers
- **Long-running services** - Just start vLLM manually once, connect with `base_url`
- **Multiple simultaneous models** - Each port can only serve one model

---

## Summary

**For GenAI agents - Local server:**
```python
from scicode_lint.vllm import VLLMServer

with VLLMServer():
    # Server auto-starts if needed
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
    # Apply fixes...
# Server auto-stops only if we started it
```

**For GenAI agents - Remote server:**
```python
with VLLMServer(base_url="http://gpu-cluster:5001"):
    # Uses remote server (never starts/stops)
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
```

**Key functions:**
- `VLLMServer()` - Context manager (recommended, supports local and remote)
- `start_server()` - Start local vLLM subprocess
- `stop_server()` - Stop local vLLM process
- `is_running()` - Check server status (local or remote)
- `wait_for_ready()` - Wait for startup (local or remote)
- `get_server_info()` - Get server details (model, status)

**Features:**
- Smart local server management (auto-start/stop)
- Remote server support (connectivity only)
- Model mismatch warnings
- GPU info utilities

**Simple, lightweight, focused on automation use cases.**
