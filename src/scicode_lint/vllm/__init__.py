"""Lightweight utilities for managing vLLM server lifecycle.

Optional utilities for GenAI agents and automated testing.
Not required for normal scicode-lint usage.

Supports both local and remote vLLM servers:
    - **Local servers**: Can start/stop programmatically
    - **Remote servers**: Can verify connectivity only

Features:
    - Start/stop local vLLM servers programmatically
    - Check server status and GPU information
    - Context manager for automatic lifecycle management
    - Support for remote vLLM servers

Example - Local Server Management:
    >>> from scicode_lint.vllm import start_server, stop_server, wait_for_ready
    >>> proc = start_server()
    >>> if wait_for_ready():
    ...     linter = SciCodeLinter()
    ...     result = linter.check_file(Path("myfile.py"))
    >>> stop_server(proc)

Example - Context Manager (Local):
    >>> from scicode_lint.vllm import VLLMServer
    >>> with VLLMServer():  # Auto-starts if needed
    ...     linter = SciCodeLinter()
    ...     result = linter.check_file(Path("myfile.py"))

Example - Remote Server:
    >>> with VLLMServer(base_url="http://gpu-cluster:5001"):
    ...     linter = SciCodeLinter()  # Uses remote server
    ...     result = linter.check_file(Path("myfile.py"))

Example - System Info:
    >>> from scicode_lint.vllm import get_gpu_info, print_system_info
    >>> gpu = get_gpu_info()
    >>> print(f"Free VRAM: {gpu.free_memory_mb} MB")
    >>> print_system_info()  # Print complete system status
"""

import asyncio
import subprocess
import time
import types
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import httpx


def _get_vllm_config() -> dict[str, Any]:
    """Get vLLM config from config.toml. Fails if config cannot be loaded."""
    from scicode_lint.config import load_config_from_toml

    config = load_config_from_toml()
    result: dict[str, Any] = config.get("vllm", {})
    return result


def _get_llm_config_required(key: str) -> Any:
    """Get a required value from [llm] section of config.toml.

    Fails loudly if the key is missing — no hardcoded fallback.
    """
    from scicode_lint.config import load_config_from_toml

    config = load_config_from_toml()
    llm_section = config.get("llm", {})
    if key not in llm_section:
        raise RuntimeError(
            f"Required key '[llm].{key}' missing from config.toml. "
            "Bundled config should define this; reinstall scicode-lint if config is corrupted."
        )
    return llm_section[key]


def _get_default_model() -> str:
    """Get model name from config.toml. Fails loudly if not set."""
    result: str = _get_llm_config_required("model")
    return result


def _get_default_max_model_len() -> int:
    """Get max_model_len from [vllm] section of config.toml.

    This is the vLLM server's sequence-length ceiling, not a pre-allocation.
    PagedAttention allocates KV cache on demand, so the value only matters when
    a request's total tokens would exceed it (vLLM rejects the request).
    Default: 40960 (Qwen3-8B native max_position_embeddings).
    """
    vllm_config = _get_vllm_config()
    return int(vllm_config.get("max_model_len", 40960))


def _get_min_vram_mb() -> int:
    """Get minimum VRAM requirement from config.toml."""
    vllm_config = _get_vllm_config()
    if "min_vram_mb" not in vllm_config:
        raise KeyError("min_vram_mb not found in [vllm] section of config.toml")
    return int(vllm_config["min_vram_mb"])


def _get_gpu_memory_utilization() -> float:
    """Get GPU memory utilization from config.toml."""
    vllm_config = _get_vllm_config()
    if "gpu_memory_utilization" not in vllm_config:
        raise KeyError("gpu_memory_utilization not found in [vllm] section of config.toml")
    return float(vllm_config["gpu_memory_utilization"])


def _get_reasoning_parser() -> str:
    """Get reasoning parser from config.toml.

    Returns:
        Parser name (e.g., "qwen3") or empty string if disabled.
        Qwen3-specific: separates <think> blocks from JSON server-side.
        Other thinking models need their own parser (e.g., "deepseek").
    """
    vllm_config = _get_vllm_config()
    return str(vllm_config.get("reasoning_parser", "qwen3"))


def is_running(base_url: str = "http://localhost:5001") -> bool:
    """Check if vLLM server is running.

    Args:
        base_url: Server URL to check (default: http://localhost:5001)

    Returns:
        True if server is responding, False otherwise

    Example:
        >>> from scicode_lint.vllm import is_running
        >>> if is_running():
        ...     print("Server is ready")
    """
    try:
        response = httpx.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except (httpx.HTTPError, ConnectionError):
        return False


def wait_for_ready(
    base_url: str = "http://localhost:5001",
    timeout: int = 60,
    check_interval: float = 2.0,
) -> bool:
    """Wait for vLLM server to be ready.

    Args:
        base_url: Server URL to check (default: http://localhost:5001)
        timeout: Maximum time to wait in seconds (default: 60)
        check_interval: Time between checks in seconds (default: 2.0)

    Returns:
        True if server became ready, False if timeout reached

    Example:
        >>> from scicode_lint.vllm import wait_for_ready
        >>> if wait_for_ready(timeout=30):
        ...     print("Server is ready")
        ... else:
        ...     print("Timeout waiting for server")
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_running(base_url):
            return True
        time.sleep(check_interval)
    return False


class VLLMServer:
    """Context manager for vLLM server lifecycle.

    Supports both local and remote vLLM servers:
    - **Local server (default)**: Starts container if not running, stops if we started it
    - **Remote server**: Only verifies it's reachable, never starts/stops

    Args:
        model: Model name or path (only used for local servers)
        port: Port to run on (only used for local servers)
        base_url: Full URL for remote server (e.g., "http://10.0.0.5:5001")
                  If provided, port is ignored and no start/stop attempted
        wait_timeout: Timeout for server startup/verification in seconds

    Example - Local server (auto-start):
        >>> from scicode_lint.vllm import VLLMServer
        >>> from scicode_lint import SciCodeLinter
        >>>
        >>> # Starts local container if not running, stops when done
        >>> with VLLMServer():
        ...     linter = SciCodeLinter()
        ...     result = linter.check_file(Path("myfile.py"))

    Example - Remote server:
        >>> # Just verifies remote server is reachable
        >>> with VLLMServer(base_url="http://gpu-cluster.example.com:5001"):
        ...     linter = SciCodeLinter()
        ...     result = linter.check_file(Path("myfile.py"))
    """

    def __init__(
        self,
        model: str | None = None,
        port: int = 5001,
        base_url: str | None = None,
        wait_timeout: int = 120,
    ):
        """Initialize VLLMServer context manager.

        Uses settings from config.toml. Model defaults to config value.
        """
        self.model = model if model is not None else _get_default_model()
        self.port = port
        self.base_url = base_url
        self.wait_timeout = wait_timeout
        self.we_started = False
        self.was_already_running = False
        self.is_remote = base_url is not None

    def __enter__(self) -> "VLLMServer":
        """Start vLLM container if local and not already running."""
        server_url = self.base_url or f"http://localhost:{self.port}"

        # Check if server already running
        if is_running(server_url):
            self.was_already_running = True

            # Warn if running model differs from requested (local only)
            if not self.is_remote:
                try:
                    response = httpx.get(f"{server_url}/v1/models", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if "data" in data and len(data["data"]) > 0:
                            running_model = data["data"][0].get("id", "unknown")
                            if self.model not in running_model and running_model not in self.model:
                                import warnings

                                msg = (
                                    f"Local vLLM server is running with model "
                                    f"'{running_model}' but you requested '{self.model}'. "
                                    f"Using existing server. To use the requested model, "
                                    f"stop the server first: scicode-lint vllm-server stop"
                                )
                                warnings.warn(msg, RuntimeWarning)
                except (httpx.HTTPError, KeyError, IndexError):
                    pass  # Can't verify model, proceed anyway

            return self

        # Remote server not reachable
        if self.is_remote:
            raise RuntimeError(
                f"Remote vLLM server not reachable at {server_url}. "
                "Cannot start remote servers - please start it manually."
            )

        # Start local container
        from scicode_lint.vllm.container import start_container

        exit_code = start_container(model=self.model, port=self.port)
        if exit_code != 0:
            raise RuntimeError(
                "Failed to start vLLM container. Check logs with: scicode-lint vllm-server logs"
            )

        # Wait for API to be ready
        if not wait_for_ready(server_url, timeout=self.wait_timeout):
            from scicode_lint.vllm.container import stop_container

            stop_container()
            raise TimeoutError(
                f"Container started but vLLM server not ready within {self.wait_timeout}s. "
                "Check logs with: scicode-lint vllm-server logs"
            )

        self.we_started = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Stop vLLM container only if we started it (local only)."""
        if self.we_started and not self.was_already_running and not self.is_remote:
            from scicode_lint.vllm.container import stop_container

            stop_container()


@dataclass
class GPUInfo:
    """GPU information from nvidia-smi.

    Attributes:
        name: GPU name (e.g., "NVIDIA RTX 4000 Ada")
        total_memory_mb: Total VRAM in MB
        used_memory_mb: Used VRAM in MB
        free_memory_mb: Free VRAM in MB
        utilization_percent: GPU utilization percentage
        cuda_version: CUDA version
    """

    name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: int
    cuda_version: str


@dataclass
class ServerInfo:
    """vLLM server information.

    Attributes:
        model: Model name being served
        is_running: Whether server is responding
        base_url: Server URL
        max_model_len: Maximum context length configured on server
    """

    model: str | None
    is_running: bool
    base_url: str
    max_model_len: int | None = None


def get_gpu_info() -> GPUInfo | None:
    """Get GPU information using nvidia-smi.

    Returns:
        GPUInfo object with GPU details, or None if nvidia-smi fails

    Example:
        >>> from scicode_lint.vllm import get_gpu_info
        >>> gpu = get_gpu_info()
        >>> if gpu:
        ...     print(f"GPU: {gpu.name}")
        ...     print(f"VRAM: {gpu.used_memory_mb}/{gpu.total_memory_mb} MB")
        ...     print(f"Free: {gpu.free_memory_mb} MB")
        ...     print(f"Utilization: {gpu.utilization_percent}%")
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse first GPU (most systems have one)
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]

        # Get CUDA version
        cuda_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        cuda_version = cuda_result.stdout.strip()

        return GPUInfo(
            name=parts[0],
            total_memory_mb=int(parts[1]),
            used_memory_mb=int(parts[2]),
            free_memory_mb=int(parts[3]),
            utilization_percent=int(parts[4]),
            cuda_version=cuda_version,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return None


def get_server_info(base_url: str = "http://localhost:5001") -> ServerInfo:
    """Get vLLM server information.

    Args:
        base_url: Server URL (default: http://localhost:5001)

    Returns:
        ServerInfo object with server details

    Example:
        >>> from scicode_lint.vllm import get_server_info
        >>> info = get_server_info()
        >>> if info.is_running:
        ...     print(f"Model: {info.model}")
        ...     print(f"Max context: {info.max_model_len}")
        >>> else:
        ...     print("Server not running")
    """
    model_name = None
    max_model_len = None
    running = is_running(base_url)

    if running:
        try:
            # Try to get model info from /v1/models endpoint
            response = httpx.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    model_info = data["data"][0]
                    model_name = model_info.get("id")
                    max_model_len = model_info.get("max_model_len")
        except (httpx.HTTPError, KeyError, IndexError):
            pass

    return ServerInfo(
        model=model_name,
        is_running=running,
        base_url=base_url,
        max_model_len=max_model_len,
    )


class VLLMMetricsMonitor:
    """Background monitor for vLLM metrics during long-running operations.

    Periodically fetches vLLM server metrics (running requests, queued requests,
    throughput) and writes them to a file for later analysis.

    Args:
        base_url: vLLM server URL (default: http://localhost:5001)
        interval: Seconds between metric checks (default: 5.0)
        output_file: Path to write metrics (default: evals/reports/vllm_metrics.log)
        console: Also print summary to console (default: True)

    Example:
        >>> import asyncio
        >>> from scicode_lint.vllm import VLLMMetricsMonitor
        >>>
        >>> async def run_with_monitoring():
        ...     monitor = VLLMMetricsMonitor()
        ...     monitor.start()
        ...     try:
        ...         await asyncio.sleep(30)
        ...     finally:
        ...         await monitor.stop()
        ...         print(f"Metrics saved to {monitor.output_file}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:5001",
        interval: float = 5.0,
        output_file: str | None = None,
        console: bool = True,
    ):
        self.metrics_url = f"{base_url}/metrics"
        self.interval = interval
        self.output_file = output_file or "evals/reports/vllm_metrics.log"
        self.console = console
        self._task: asyncio.Task[Any] | None = None
        self._stop = False
        self._start_time = 0.0
        self._peak_running = 0
        self._peak_waiting = 0
        self._total_requests = 0

    async def _fetch_metrics(self) -> dict[str, float] | None:
        """Fetch and parse vLLM metrics."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(self.metrics_url)
                if resp.status_code != 200:
                    return None

                metrics = {}
                for line in resp.text.split("\n"):
                    if line.startswith("vllm:num_requests_running"):
                        metrics["running"] = float(line.split()[-1])
                    elif line.startswith("vllm:num_requests_waiting"):
                        metrics["waiting"] = float(line.split()[-1])
                    elif line.startswith("vllm:num_requests_finished"):
                        metrics["finished"] = float(line.split()[-1])
                return metrics
        except Exception:
            return None

    async def _monitor_loop(self) -> None:
        """Background loop collecting metrics."""
        import os
        from datetime import datetime

        self._start_time = time.time()
        last_finished = 0.0

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, "w") as f:
            # Write CSV header
            f.write("timestamp,elapsed_s,running,waiting,finished,throughput_req_s\n")
            f.flush()

            while not self._stop:
                metrics = await self._fetch_metrics()
                if metrics:
                    now = datetime.now(UTC).isoformat(timespec="seconds")
                    elapsed = time.time() - self._start_time
                    running = int(metrics.get("running", 0))
                    waiting = int(metrics.get("waiting", 0))
                    finished = metrics.get("finished", 0)

                    # Track peaks
                    self._peak_running = max(self._peak_running, running)
                    self._peak_waiting = max(self._peak_waiting, waiting)
                    self._total_requests = int(finished)

                    # Calculate throughput
                    completed_delta = finished - last_finished
                    last_finished = finished
                    throughput = completed_delta / self.interval if self.interval > 0 else 0

                    # Write to file (CSV format with timestamp)
                    f.write(
                        f"{now},{elapsed:.1f},{running},{waiting},{finished:.0f},{throughput:.2f}\n"
                    )
                    f.flush()

                    # Optional console output
                    if self.console:
                        print(
                            f"[{elapsed:5.0f}s] vLLM: {running} running, {waiting} queued, "
                            f"{throughput:.1f} req/s"
                        )

                await asyncio.sleep(self.interval)

    def start(self) -> None:
        """Start background monitoring."""
        self._stop = False
        self._peak_running = 0
        self._peak_waiting = 0
        self._total_requests = 0
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop background monitoring and return summary."""
        self._stop = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        elapsed = time.time() - self._start_time
        if self.console and elapsed > 0:
            avg_throughput = self._total_requests / elapsed if elapsed > 0 else 0
            print(
                f"Metrics summary: peak {self._peak_running} concurrent, "
                f"{self._peak_waiting} max queued, {avg_throughput:.1f} avg req/s"
            )
            print(f"Full metrics: {self.output_file}")


def print_system_info() -> None:
    """Print system information (GPU, server status).

    Useful for GenAI agents to check system readiness before starting work.

    Example:
        >>> from scicode_lint.vllm import print_system_info
        >>> print_system_info()
        === GPU Information ===
        GPU: NVIDIA RTX 4000 Ada
        VRAM: 4096/20480 MB (20% used, 16384 MB free)
        Utilization: 15%
        CUDA: 535.183.01

        === vLLM Server ===
        Status: Running
        Model: RedHatAI/Qwen3-8B-FP8-dynamic
        URL: http://localhost:5001
    """
    print("=== GPU Information ===")
    gpu = get_gpu_info()
    if gpu:
        print(f"GPU: {gpu.name}")
        print(
            f"VRAM: {gpu.used_memory_mb}/{gpu.total_memory_mb} MB "
            f"({gpu.used_memory_mb * 100 // gpu.total_memory_mb}% used, "
            f"{gpu.free_memory_mb} MB free)"
        )
        print(f"Utilization: {gpu.utilization_percent}%")
        print(f"CUDA: {gpu.cuda_version}")
    else:
        print("GPU information not available (nvidia-smi failed)")

    print("\n=== vLLM Server ===")
    server = get_server_info()
    if server.is_running:
        print("Status: Running")
        print(f"Model: {server.model or 'Unknown'}")
        print(f"URL: {server.base_url}")
    else:
        print("Status: Not running")


__all__ = [
    "wait_for_ready",
    "is_running",
    "VLLMServer",
    "VLLMMetricsMonitor",
    "get_gpu_info",
    "get_server_info",
    "print_system_info",
    "GPUInfo",
    "ServerInfo",
]
