"""CLI server management commands (start/stop/status/restart/logs/rm/monitor)."""

from __future__ import annotations

import argparse

from loguru import logger


def _run_server(args: argparse.Namespace) -> int:
    """Run the server subcommand (start/stop/status/restart/logs/rm/monitor)."""
    cmd = args.server_command
    if cmd == "start":
        return _run_server_start(args)
    elif cmd == "stop":
        return _run_server_stop()
    elif cmd == "status":
        return _run_server_status()
    elif cmd == "restart":
        return _run_server_restart(args)
    elif cmd == "logs":
        return _run_server_logs(args)
    elif cmd == "rm":
        return _run_server_rm(args)
    elif cmd == "monitor":
        return _run_server_monitor(args)
    else:
        logger.error(f"Unknown server command: {cmd}")
        return 1


def _run_server_start(args: argparse.Namespace) -> int:
    """Start the vLLM container."""
    from scicode_lint.vllm.container import start_container

    return start_container(
        model=getattr(args, "model", None),
        port=getattr(args, "port", None),
        pull=getattr(args, "pull", False),
    )


def _run_server_stop() -> int:
    """Stop the vLLM container."""
    from scicode_lint.vllm.container import stop_container

    return stop_container()


def _run_server_status() -> int:
    """Show vLLM container and GPU status."""
    from scicode_lint.vllm.container import container_status

    rc = container_status()

    from scicode_lint.vllm import get_gpu_info

    gpu = get_gpu_info()
    if gpu:
        print(f"\nGPU: {gpu.name}")
        print(
            f"  VRAM: {gpu.used_memory_mb}/{gpu.total_memory_mb} MB ({gpu.free_memory_mb} MB free)"
        )
    else:
        print("\nGPU: not detected")

    return rc


def _run_server_restart(args: argparse.Namespace) -> int:
    """Restart the vLLM container (stop + remove + start)."""
    from scicode_lint.vllm.container import (
        remove_container,
        start_container,
        stop_container,
    )

    stop_container()
    remove_container(force=True)
    return start_container(
        model=getattr(args, "model", None),
        port=getattr(args, "port", None),
        pull=getattr(args, "pull", False),
    )


def _run_server_logs(args: argparse.Namespace) -> int:
    """Show vLLM container logs."""
    from scicode_lint.vllm.container import container_logs

    return container_logs(
        follow=getattr(args, "follow", False),
        tail=getattr(args, "tail", 50),
    )


def _run_server_rm(args: argparse.Namespace) -> int:
    """Remove the vLLM container."""
    from scicode_lint.vllm.container import remove_container

    return remove_container(force=getattr(args, "force", False))


def _run_server_monitor(args: argparse.Namespace) -> int:
    """Live-refresh monitor for vLLM server metrics."""
    from scicode_lint.vllm.container import container_monitor

    return container_monitor(interval=getattr(args, "interval", 2.0))
