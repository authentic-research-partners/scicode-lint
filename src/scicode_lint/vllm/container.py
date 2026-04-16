"""Container-based vLLM server lifecycle management.

Runs vLLM in a container (podman/docker) with CDI GPU passthrough.
Single-model design — one container named ``scicode-lint-vllm``.

Container command flags:
  model (positional), --host 0.0.0.0, --port 8000 (internal),
  --served-model-name, --trust-remote-code, --gpu-memory-utilization,
  --max-model-len, --max-num-seqs, --kv-cache-dtype fp8,
  --enable-chunked-prefill, --reasoning-parser (conditional).
"""

from __future__ import annotations

import json
import shutil
import signal
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.table import Table

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONTAINER_NAME = "scicode-lint-vllm"
_DEFAULT_PORT = 5001
_STATE_DIR = Path.home() / ".scicode-lint"
_LAST_PORTS_FILE = _STATE_DIR / "last_ports.json"

# ---------------------------------------------------------------------------
# Config accessors (read from config.toml via vllm/__init__.py helpers)
# ---------------------------------------------------------------------------


def _get_vllm_version() -> str:
    """Get vLLM container image version tag from [vllm] section."""
    from scicode_lint.vllm import _get_vllm_config

    return str(_get_vllm_config().get("vllm_version", "v0.18.0"))


def _get_vllm_memory() -> str:
    """Get container RAM limit from [vllm] section."""
    from scicode_lint.vllm import _get_vllm_config

    return str(_get_vllm_config().get("vllm_memory", "4g"))


def _get_vllm_image() -> str:
    """Compute full container image reference from version."""
    return f"docker.io/vllm/vllm-openai:{_get_vllm_version()}"


# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------


def _port_available(host: str, port: int) -> bool:
    """Check whether a port can be bound."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def _load_last_port() -> int | None:
    """Load remembered vLLM port from ~/.scicode-lint/last_ports.json."""
    try:
        data = json.loads(_LAST_PORTS_FILE.read_text())
        port = data.get("vllm")
        return int(port) if port is not None else None
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def _save_last_port(port: int) -> None:
    """Remember which port vLLM is using."""
    try:
        data: dict[str, int] = {}
        if _LAST_PORTS_FILE.exists():
            data = json.loads(_LAST_PORTS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        data = {}

    if data.get("vllm") == port:
        return

    data["vllm"] = port
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_PORTS_FILE.write_text(json.dumps(data, indent=2) + "\n")
    except OSError:
        pass


def _resolve_port(port: int | None) -> int:
    """Resolve vLLM port: explicit > remembered > default."""
    if port is not None:
        return port

    last = _load_last_port()
    if last is not None and _port_available("0.0.0.0", last):  # nosec B104
        return last
    return _DEFAULT_PORT


# ---------------------------------------------------------------------------
# Container helpers
# ---------------------------------------------------------------------------


def _detect_container_runtime() -> str | None:
    """Detect available container runtime (podman preferred)."""
    for rt in ("podman", "docker"):
        if shutil.which(rt):
            return rt
    return None


def _container_exists(runtime: str) -> bool:
    """Check if the scicode-lint-vllm container exists."""
    result = subprocess.run(
        [runtime, "container", "inspect", _CONTAINER_NAME],
        capture_output=True,
    )
    return result.returncode == 0


def _container_running(runtime: str) -> bool:
    """Check if the scicode-lint-vllm container is running."""
    result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Running}}", _CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _find_container_on_port(runtime: str | None, port: int) -> str | None:
    """Return the name of the running container publishing ``port`` on the host.

    Returns None if no runtime is available, no container holds the port,
    or the runtime call fails.
    """
    if not runtime:
        return None
    result = subprocess.run(
        [runtime, "ps", "--format", "{{.Names}}\t{{.Ports}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.strip().splitlines():
        if f":{port}->" in line:
            return line.split("\t")[0]
    return None


def _identify_port_holder(runtime: str | None, port: int) -> str:
    """Try to identify which container holds a port. Returns ' by <name>' or ''."""
    name = _find_container_on_port(runtime, port)
    return f" by container '{name}'" if name else ""


# ---------------------------------------------------------------------------
# Lifecycle functions
# ---------------------------------------------------------------------------


def start_container(
    model: str | None = None,
    port: int | None = None,
    pull: bool = False,
) -> int:
    """Start vLLM in a container (podman/docker).

    Args:
        model: HuggingFace model ID (default: from config.toml [llm].model)
        port: Host port (default: 5001 or last-used port)
        pull: Pull latest image before starting

    Returns:
        Exit code (0 = success, 1 = error)
    """
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error(
            "Neither podman nor docker found on PATH.\n\n"
            "Install one of:\n"
            "  sudo apt install podman  # or: sudo apt install docker.io\n"
            "  sudo apt install nvidia-container-toolkit\n"
            "  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
        )
        return 1

    # Read config
    from scicode_lint.config import load_config_from_toml
    from scicode_lint.vllm import (
        _get_default_max_model_len,
        _get_default_model,
        _get_gpu_memory_utilization,
        _get_llm_config_required,
        _get_reasoning_parser,
    )

    model_name = model or _get_default_model()
    serve_port = _resolve_port(port)
    served_name: str = _get_llm_config_required("model_served_name")
    max_model_len = _get_default_max_model_len()
    gpu_mem = _get_gpu_memory_utilization()
    reasoning_parser = _get_reasoning_parser()
    perf_config = load_config_from_toml().get("performance", {})
    max_num_seqs: int = int(perf_config.get("vllm_max_num_seqs", 256))
    image = _get_vllm_image()
    container_memory = _get_vllm_memory()

    # Already running?
    if _container_running(runtime):
        logger.info(f"Container '{_CONTAINER_NAME}' is already running")
        logger.info("Use 'scicode-lint vllm-server restart' to restart")
        return 0

    # Restart stopped container (fast — weights cached in page cache)
    if _container_exists(runtime) and not pull:
        logger.info(f"Restarting stopped container '{_CONTAINER_NAME}'")
        result = subprocess.run([runtime, "start", _CONTAINER_NAME], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Container restarted: {_CONTAINER_NAME}")
            logger.info(f"API will be available at http://localhost:{serve_port}/v1")
            _save_last_port(serve_port)
            return 0
        # Start failed (e.g. config changed) — fall through to rm + run
        logger.info(f"Restart failed, recreating container '{_CONTAINER_NAME}'")
        subprocess.run([runtime, "rm", "-f", _CONTAINER_NAME], capture_output=True)

    # Check port is free
    if not _port_available("0.0.0.0", serve_port):  # nosec B104
        blocker = _identify_port_holder(runtime, serve_port)
        logger.error(
            f"Port {serve_port} is already in use{blocker}.\n"
            f"Free it with: scicode-lint vllm-server stop\n"
            f"Or check manually: ss -tlnp | grep {serve_port}"
        )
        return 1

    # Pull image
    if pull:
        logger.info(f"Pulling {image}")
        pull_result = subprocess.run([runtime, "pull", image])
        if pull_result.returncode != 0:
            logger.error(f"Failed to pull image: {image}")
            return 1

    # Build container run command
    cmd = [
        runtime,
        "run",
        "-d",
        "--name",
        _CONTAINER_NAME,
        "--device",
        "nvidia.com/gpu=all",
        "--memory",
        container_memory,
        "-p",
        f"{serve_port}:8000",
        "-v",
        f"{Path.home() / '.cache' / 'huggingface'}:/root/.cache/huggingface",
        "--restart",
        "unless-stopped",
        image,
        model_name,  # positional arg (--model deprecated in vLLM v0.13)
        "--host",
        "0.0.0.0",  # nosec B104
        "--port",
        "8000",
        "--served-model-name",
        served_name,
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(gpu_mem),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--kv-cache-dtype",
        "fp8",
        "--enable-chunked-prefill",
    ]

    # Reasoning parser for thinking models (Qwen3).
    # vLLM v0.18+ removed --enable-reasoning; setting --reasoning-parser alone
    # is sufficient and enables reasoning implicitly.
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])

    logger.info(f"Starting vLLM container '{_CONTAINER_NAME}'")
    logger.info(f"Runtime: {runtime}")
    logger.info(f"Image: {image}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Served as: {served_name}")
    logger.info(f"Port: {serve_port}")
    logger.info(f"GPU memory: {gpu_mem}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "CDI" in stderr or "unresolvable" in stderr:
            logger.error(
                "GPU passthrough not configured (CDI spec missing).\n\n"
                "Fix with:\n"
                "  sudo apt install nvidia-container-toolkit\n"
                "  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml\n"
                "  podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi"
            )
        else:
            logger.error(f"Failed to start container:\n{stderr}")
        return 1

    container_id = result.stdout.strip()[:12]
    logger.info(f"Container started: {container_id}")
    logger.info(f"API will be available at http://localhost:{serve_port}/v1")

    # Model cache hint
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_name = f"models--{model_name.replace('/', '--')}"
    if (hf_cache / model_cache_name).exists():
        logger.info("Model weights cached. Loading into GPU (~30-60s)")
    else:
        logger.info("First run: downloading model weights. This may take several minutes")
    logger.info(f"Check logs: {runtime} logs -f {_CONTAINER_NAME}")
    logger.info("Check status: scicode-lint vllm-server status")

    _save_last_port(serve_port)
    return 0


def stop_container() -> int:
    """Stop the vLLM container.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    if not _container_exists(runtime):
        logger.info(f"Container '{_CONTAINER_NAME}' does not exist")
        return 0

    if not _container_running(runtime):
        logger.info(f"Container '{_CONTAINER_NAME}' is not running")
        return 0

    logger.info(f"Stopping container '{_CONTAINER_NAME}'")
    result = subprocess.run([runtime, "stop", _CONTAINER_NAME], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to stop container:\n{result.stderr.strip()}")
        return 1
    logger.info("Container stopped. GPU memory released")
    return 0


def remove_container(force: bool = False) -> int:
    """Remove the vLLM container.

    Args:
        force: Force remove even if running

    Returns:
        Exit code (0 = success, 1 = error)
    """
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    if not _container_exists(runtime):
        logger.info(f"Container '{_CONTAINER_NAME}' does not exist")
        return 0

    if _container_running(runtime) and not force:
        logger.info(f"Container '{_CONTAINER_NAME}' is still running")
        logger.info("Stop it first with 'scicode-lint vllm-server stop', or use --force")
        return 1

    cmd = [runtime, "rm"]
    if force:
        cmd.append("-f")
    cmd.append(_CONTAINER_NAME)

    logger.info(f"Removing container '{_CONTAINER_NAME}'")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to remove container:\n{result.stderr.strip()}")
        return 1
    logger.info("Container removed")
    return 0


def container_status() -> int:
    """Show vLLM container status and API health.

    Returns:
        Exit code (0 = success)
    """
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    if not _container_exists(runtime):
        print(f"Container '{_CONTAINER_NAME}' does not exist")
        print("Start it with: scicode-lint vllm-server start")
        return 0

    # Get container state
    inspect_result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Status}}", _CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    state = inspect_result.stdout.strip() if inspect_result.returncode == 0 else "unknown"
    print(f"Container: {_CONTAINER_NAME}")
    print(f"State: {state}")

    if state != "running":
        print("API: not available (container not running)")
        return 0

    # API health check
    from scicode_lint.vllm import get_server_info

    server = get_server_info()
    if server.is_running:
        print(f"API: ready at {server.base_url}")
        if server.model:
            print(f"Model: {server.model}")
        if server.max_model_len:
            print(f"Max context: {server.max_model_len} tokens")
    else:
        print("API: loading (not responding yet)")

    return 0


def container_logs(follow: bool = False, tail: int = 50) -> int:
    """Show vLLM container logs.

    Args:
        follow: Follow log output (like tail -f)
        tail: Number of lines to show

    Returns:
        Exit code (0 = success, 1 = error)
    """
    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    if not _container_exists(runtime):
        logger.error(f"Container '{_CONTAINER_NAME}' does not exist")
        return 1

    cmd = [runtime, "logs", "--tail", str(tail)]
    if follow:
        cmd.append("-f")
    cmd.append(_CONTAINER_NAME)

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
    return 0


def _resolve_cgroup_dir(runtime: str, container: str = _CONTAINER_NAME) -> Path | None:
    """Find the cgroup v2 directory for a container.

    Reads the container's init PID via inspect, then returns the cgroup path
    accessible via ``/proc/<pid>/root/sys/fs/cgroup``. Returns None if the
    container isn't running, doesn't exist, or cgroup v2 is unavailable
    (e.g. cgroup v1 host).
    """
    result = subprocess.run(
        [runtime, "inspect", container, "--format", "{{.State.Pid}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    pid = result.stdout.strip()
    if not pid or pid == "0":
        return None
    cgroup = Path(f"/proc/{pid}/root/sys/fs/cgroup")
    if not (cgroup / "memory.current").exists():
        return None
    return cgroup


def _read_cgroup_memory(cgroup: Path) -> tuple[int, int | None]:
    """Read container RSS and memory limit from cgroup v2 files.

    Uses ``anon`` from ``memory.stat`` (actual heap/stack RSS) rather than
    ``memory.current`` (which includes page cache and GPU driver mappings
    — misleading for GPU workloads like vLLM).

    Returns (used_bytes, limit_bytes). limit_bytes is None if no limit set.
    """
    try:
        limit_str = (cgroup / "memory.max").read_text().strip()
        stat_text = (cgroup / "memory.stat").read_text()
    except OSError:
        return 0, None
    limit = None if limit_str == "max" else int(limit_str)
    used = 0
    for line in stat_text.splitlines():
        if line.startswith("anon "):
            used = int(line.split()[1])
            break
    return used, limit


def _bar(pct: float, width: int = 30) -> str:
    """Render a compact bar like ``[████████░░░░░░░░]  25%``."""
    filled = int(pct * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1%}"


def _bar_color(pct: float) -> str:
    """Rich color name based on utilization percentage."""
    if pct < 0.6:
        return "green"
    if pct < 0.85:
        return "yellow"
    return "red"


def _build_monitor_table(
    *,
    model_name: str,
    base_url: str,
    metrics: dict[str, float | int],
    vram_used_mb: int | None,
    vram_total_mb: int | None,
    gpu_util_pct: int | None,
    ram_used_bytes: int | None,
    ram_limit_bytes: int | None,
    prompt_rate: float,
    gen_rate: float,
    preemption_rate: float,
) -> Table:
    """Build a rich Table grouping monitor sections for the vLLM panel."""
    from rich.table import Table
    from rich.text import Text

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", min_width=12)
    grid.add_column()

    # --- Model / GPU ---
    max_seq = metrics.get("max_seq")
    max_seq_str = f"  max_seq: {int(max_seq):,}" if max_seq else ""
    grid.add_row("Model", f"{model_name}{max_seq_str}")

    if vram_used_mb is not None and vram_total_mb:
        vram_pct = vram_used_mb / vram_total_mb
        vram_text = Text()
        vram_text.append(_bar(vram_pct), style="cyan")
        vram_text.append(
            f"  {vram_used_mb / 1024:.1f}GB / {vram_total_mb / 1024:.1f}GB", style="dim"
        )
        grid.add_row("VRAM", vram_text)

    if gpu_util_pct is not None:
        color = "green" if gpu_util_pct > 10 else "dim"
        gpu_text = Text()
        gpu_text.append(_bar(gpu_util_pct / 100, width=20), style=color)
        grid.add_row("GPU load", gpu_text)

    if ram_used_bytes is not None:
        used_gb = ram_used_bytes / (1024**3)
        ram_text = Text()
        if ram_limit_bytes:
            limit_gb = ram_limit_bytes / (1024**3)
            pct = ram_used_bytes / ram_limit_bytes
            ram_text.append(_bar(pct), style=_bar_color(pct))
            ram_text.append(f"  {used_gb:.1f}GB / {limit_gb:.1f}GB", style="dim")
        else:
            ram_text.append(f"{used_gb:.1f}GB", style="bold")
            ram_text.append("  (container anon RSS, no limit set)", style="dim")
        grid.add_row("RAM", ram_text)

    grid.add_row("", "")

    # --- Requests ---
    running = int(metrics.get("requests_running", 0))
    waiting = int(metrics.get("requests_waiting", 0))
    swapped = int(metrics.get("requests_swapped", 0))
    req_text = Text()
    req_text.append(f"running: {running}", style="bold" if running else "dim")
    req_text.append(f"   waiting: {waiting}", style="red bold" if waiting else "dim")
    req_text.append(f"   swapped: {swapped}", style="red bold" if swapped else "dim")
    grid.add_row("Requests", req_text)

    # Request outcomes
    stops = int(metrics.get("req_success_stop", 0))
    lengths = int(metrics.get("req_success_length", 0))
    aborts = int(metrics.get("req_success_abort", 0))
    errors = int(metrics.get("req_success_error", 0))
    total_reqs = stops + lengths + aborts + errors
    if total_reqs > 0:
        hist = Text()
        hist.append(f"{stops} ok", style="green")
        if lengths:
            hist.append(f"  {lengths} truncated", style="yellow")
        if aborts:
            hist.append(f"  {aborts} aborted", style="yellow")
        if errors:
            hist.append(f"  {errors} errors", style="red")
        hist.append(f"  ({total_reqs} total)", style="dim")
        grid.add_row("History", hist)

    grid.add_row("", "")

    # --- KV cache ---
    kv_pct_val = metrics.get("kv_cache_pct")
    if isinstance(kv_pct_val, (int, float)):
        kv_pct = float(kv_pct_val)
        blocks = metrics.get("num_gpu_blocks")
        block_size = metrics.get("block_size")
        if isinstance(blocks, (int, float)) and isinstance(block_size, (int, float)):
            tokens = int(blocks) * int(block_size)
            blocks_str = f"  ({tokens:,} tokens, {int(blocks)} blocks)"
        elif isinstance(blocks, (int, float)):
            blocks_str = f"  ({int(blocks)} GPU blocks)"
        else:
            blocks_str = ""
        if kv_pct < 0.85:
            kv_color = "cyan"
        elif kv_pct < 0.95:
            kv_color = "yellow"
        else:
            kv_color = "red"
        bar_text = Text()
        bar_text.append(_bar(kv_pct), style=kv_color)
        bar_text.append(blocks_str, style="dim")
        grid.add_row("KV cache", bar_text)

    # Prefix cache hit rate
    cache_hits = float(metrics.get("prefix_cache_hits", 0.0))
    cache_queries = float(metrics.get("prefix_cache_queries", 0.0))
    if cache_queries > 0:
        hit_rate = cache_hits / cache_queries
        hit_color = "green" if hit_rate > 0.3 else "yellow" if hit_rate > 0.1 else "dim"
        hit_text = Text()
        hit_text.append(_bar(hit_rate), style=hit_color)
        hit_text.append(f"  ({int(cache_hits):,}/{int(cache_queries):,} tokens)", style="dim")
        grid.add_row("Cache hit", hit_text)
    else:
        grid.add_row("Cache hit", Text("no queries yet", style="dim"))

    # Preemptions
    cur_preemptions = int(metrics.get("num_preemptions", 0))
    if cur_preemptions > 0 or preemption_rate > 0:
        evict_text = Text(
            f"{cur_preemptions:,} preemptions  ({preemption_rate:.1f}/s)",
            style="red bold",
        )
    else:
        evict_text = Text("0 preemptions", style="green")
    grid.add_row("Evictions", evict_text)

    grid.add_row("", "")

    # --- Latency ---
    e2e_count = float(metrics.get("e2e_latency_count", 0))
    if e2e_count > 0:
        e2e_avg = float(metrics.get("e2e_latency_sum", 0)) / e2e_count
        ttft_count = float(metrics.get("ttft_count", 0))
        ttft_avg = float(metrics.get("ttft_sum", 0)) / ttft_count if ttft_count > 0 else 0
        itl_count = float(metrics.get("itl_count", 0))
        itl_avg = float(metrics.get("itl_sum", 0)) / itl_count if itl_count > 0 else 0
        lat = Text()
        lat.append(f"e2e: {e2e_avg:.2f}s", style="bold")
        lat.append(f"   TTFT: {ttft_avg:.3f}s")
        lat.append(f"   ITL: {itl_avg * 1000:.1f}ms")
        lat.append(f"  (avg over {int(e2e_count):,} reqs)", style="dim")
        grid.add_row("Latency", lat)
    else:
        grid.add_row("Latency", Text("no completed requests yet", style="dim"))

    # --- Throughput ---
    prompt_tok = float(metrics.get("prompt_tokens_total", 0))
    gen_tok = float(metrics.get("generation_tokens_total", 0))
    tp = Text()
    tp.append(f"prompt: {prompt_rate:,.0f} tok/s", style="bold")
    tp.append(f"   gen: {gen_rate:,.0f} tok/s")
    tp.append(f"   total: {int(prompt_tok + gen_tok):,} tokens served", style="dim")
    grid.add_row("Throughput", tp)

    # --- Health warnings ---
    grid.add_row("", "")
    warnings: list[str] = []
    if isinstance(kv_pct_val, (int, float)) and float(kv_pct_val) > 0.9:
        warnings.append(f"KV cache at {float(kv_pct_val):.0%} — risk of preemptions")
    if waiting > 0:
        warnings.append(f"{waiting} requests waiting — server is at capacity")
    if preemption_rate > 0.5:
        warnings.append(
            f"Preemptions at {preemption_rate:.1f}/s — reduce concurrency or max_model_len"
        )
    if lengths >= 5 and total_reqs >= 10 and lengths / total_reqs > 0.1:
        pct = lengths / total_reqs * 100
        warnings.append(
            f"{lengths} of {total_reqs} ({pct:.0f}%) responses cut short"
            " — model hit output token limit before finishing"
        )
    if warnings:
        warn_text = Text()
        for i, w in enumerate(warnings):
            if i > 0:
                warn_text.append("\n")
            warn_text.append(f"  {w}", style="yellow bold")
        grid.add_row(Text("⚠ Warning", style="yellow bold"), warn_text)
    else:
        grid.add_row("", Text("✓ Load looks healthy", style="green"))

    return grid


def container_monitor(interval: float = 2.0) -> int:
    """Live-refresh terminal monitor for vLLM server.

    Rich-based infinite display loop. Handles three states gracefully:
    container not running, container running but API loading, and API
    ready (full metrics panel). Exits cleanly on Ctrl+C.

    Args:
        interval: Refresh interval in seconds

    Returns:
        Exit code (0 = success, 1 = error)
    """
    import time

    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    from scicode_lint.vllm import get_gpu_info, get_server_info
    from scicode_lint.vllm.metrics import fetch_metrics

    runtime = _detect_container_runtime()
    if not runtime:
        logger.error("Neither podman nor docker found on PATH")
        return 1

    base_url = "http://localhost:5001"
    console = Console()
    model: str | None = None

    prev_prompt = 0.0
    prev_gen = 0.0
    prev_preemptions = 0.0
    prev_time = 0.0

    start_time = time.monotonic()
    elapsed = 0.0

    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                now = time.monotonic()
                elapsed = now - start_time

                container_running = _container_exists(runtime)
                server = get_server_info(base_url) if container_running else None
                api_ready = server is not None and server.is_running

                if server and server.model and model is None:
                    model = server.model

                title = f"vLLM Monitor — {model or 'loading'} — {base_url}  "
                title += f"(elapsed {elapsed:.0f}s, Ctrl+C to exit)"

                if not container_running:
                    msg = Text()
                    msg.append("Container not running\n", style="red bold")
                    msg.append("Start with: ", style="dim")
                    msg.append("scicode-lint vllm-server start", style="bold")
                    panel = Panel(msg, title=title, border_style="red", padding=(1, 2))
                elif not api_ready:
                    msg = Text()
                    msg.append("Container running, API loading\n", style="yellow bold")
                    msg.append("Tail logs: ", style="dim")
                    msg.append(f"{runtime} logs -f {_CONTAINER_NAME}", style="bold")
                    panel = Panel(msg, title=title, border_style="yellow", padding=(1, 2))
                else:
                    metrics = fetch_metrics(base_url)
                    if server and server.max_model_len:
                        metrics["max_seq"] = server.max_model_len

                    # Throughput deltas
                    prompt_tok = float(metrics.get("prompt_tokens_total", 0.0))
                    gen_tok = float(metrics.get("generation_tokens_total", 0.0))
                    cur_preempt = float(metrics.get("num_preemptions", 0.0))
                    prompt_rate = gen_rate = preemption_rate = 0.0
                    if prev_time > 0:
                        dt = now - prev_time
                        if dt > 0:
                            prompt_rate = (prompt_tok - prev_prompt) / dt
                            gen_rate = (gen_tok - prev_gen) / dt
                            preemption_rate = (cur_preempt - prev_preemptions) / dt
                    prev_prompt, prev_gen, prev_preemptions, prev_time = (
                        prompt_tok,
                        gen_tok,
                        cur_preempt,
                        now,
                    )

                    gpu = get_gpu_info()
                    ram_used: int | None = None
                    ram_limit: int | None = None
                    port_holder = _find_container_on_port(runtime, 5001)
                    if port_holder:
                        cgroup = _resolve_cgroup_dir(runtime, port_holder)
                        if cgroup:
                            ram_used, ram_limit = _read_cgroup_memory(cgroup)

                    table = _build_monitor_table(
                        model_name=model or "unknown",
                        base_url=base_url,
                        metrics=metrics,
                        vram_used_mb=gpu.used_memory_mb if gpu else None,
                        vram_total_mb=gpu.total_memory_mb if gpu else None,
                        gpu_util_pct=gpu.utilization_percent if gpu else None,
                        ram_used_bytes=ram_used,
                        ram_limit_bytes=ram_limit,
                        prompt_rate=prompt_rate,
                        gen_rate=gen_rate,
                        preemption_rate=preemption_rate,
                    )
                    panel = Panel(table, title=title, border_style="blue", padding=(1, 2))

                live.update(panel)
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print(f"\nStopped after {elapsed:.0f}s")
        return 0
