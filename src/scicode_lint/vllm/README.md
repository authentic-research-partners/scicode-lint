# vLLM Utilities

Lightweight utilities for vLLM server lifecycle management and system information.

## Contents

### Python Module (`__init__.py`)

Programmatic server management for GenAI agents:
- `VLLMServer()` - Context manager
- `start_server()` / `stop_server()` - Manual control
- `get_gpu_info()` - GPU and VRAM information
- `get_server_info()` - vLLM server status
- `print_system_info()` - Print complete system status

See [VLLM_UTILITIES.md](../../../docs_use_genai/VLLM_UTILITIES.md) for complete documentation.

### CLI Command

Server management for humans:
```bash
# Start with defaults
scicode-lint vllm-server start

# Start with custom model
scicode-lint vllm-server start --model "meta-llama/Llama-3.1-8B-Instruct"

# Restart server
scicode-lint vllm-server start --restart

# Check status
scicode-lint vllm-server status

# Stop server
scicode-lint vllm-server stop
```

## Quick Examples

Use `print_system_info()` to check GPU and vLLM status, or `get_gpu_info()` for programmatic access to VRAM information.

## Files

- `__init__.py` - Python utilities module
- `start_vllm.sh` - Bash script (called by `scicode-lint vllm-server start`)
- `README.md` - This file
