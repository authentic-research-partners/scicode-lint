"""CLI server management commands (start/stop/status)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from loguru import logger


def _run_server(args: argparse.Namespace) -> int:
    """Run the server subcommand (start/stop/status)."""
    if args.server_command == "start":
        return _run_server_start(args)
    elif args.server_command == "stop":
        return _run_server_stop()
    elif args.server_command == "status":
        return _run_server_status()
    else:
        logger.error(f"Unknown server command: {args.server_command}")
        return 1


def _get_start_script_path() -> Path:
    """Get path to the bundled start_vllm.sh script."""
    return Path(__file__).parent / "vllm" / "start_vllm.sh"


def _run_server_start(args: argparse.Namespace) -> int:
    """Start the vLLM server by executing the bundled shell script."""
    script = _get_start_script_path()
    if not script.exists():
        logger.error(f"start_vllm.sh not found at {script}")
        return 1

    cmd = ["bash", str(script)]

    if args.restart:
        cmd.append("--restart")

    # Pass positional args (model, port, max_len, gpu_mem, max_num_seqs)
    if args.model:
        cmd.append(args.model)
    if args.port:
        # If port specified but not model, need empty model to keep positional order
        if not args.model:
            cmd.append("")
        cmd.append(str(args.port))

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        return 130


def _run_server_stop() -> int:
    """Stop any running vLLM server."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm serve"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print("No vLLM server running.")
            return 0

        subprocess.run(["pkill", "-f", "vllm serve"], check=False)
        print("vLLM server stopped.")
        return 0
    except FileNotFoundError:
        logger.error("pgrep/pkill not found (not available on this platform)")
        return 1


def _run_server_status() -> int:
    """Show vLLM server status."""
    from scicode_lint.vllm import get_gpu_info, get_server_info

    server = get_server_info()
    if server.is_running:
        print(f"vLLM server: running at {server.base_url}")
        if server.model:
            print(f"  Model: {server.model}")
        if server.max_model_len:
            print(f"  Max context: {server.max_model_len} tokens")
    else:
        print("vLLM server: not running")

    gpu = get_gpu_info()
    if gpu:
        print(f"\nGPU: {gpu.name}")
        print(
            f"  VRAM: {gpu.used_memory_mb}/{gpu.total_memory_mb} MB ({gpu.free_memory_mb} MB free)"
        )
    else:
        print("\nGPU: not detected")

    return 0
