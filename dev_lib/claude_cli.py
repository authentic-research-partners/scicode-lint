"""Unified async wrapper for Claude CLI subprocess calls.

Provides consistent error handling, timeout management, flag construction,
and **global rate limiting** for all dev tooling that calls the Claude CLI.

Rate limiting is shared across ALL ClaudeCLI instances in the same process:
- asyncio.Semaphore caps parallel subprocess count (memory protection)
- aiolimiter.AsyncLimiter caps requests/minute (API rate limit protection)

Both limits are configured in config.toml under [claude_cli].

Usage:
    from dev_lib.claude_cli import ClaudeCLI, ClaudeCLITimeoutError

    cli = ClaudeCLI(model="sonnet", effort="low", timeout=120)

    # Basic async call (rate limiting applied automatically)
    result = await cli.arun("Analyze this code...")

    # With agent and tool restrictions
    result = await cli.arun(
        "Review pattern pt-001",
        agent="pattern-reviewer",
        disallowed_tools=DEFAULT_DISALLOWED_TOOLS,
    )

    # JSON output (double-parsed: envelope → inner JSON)
    data = await cli.arun_json("Return JSON with fields x, y")
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aiolimiter import AsyncLimiter
from loguru import logger

if TYPE_CHECKING:
    from asyncio.subprocess import Process

# Transient retry budget for Claude CLI JSON calls: 3 total attempts.
# Covers rare JSON-envelope glitches observed in sciwrite-lint production.
# Does NOT cover timeouts (same prompt = same latency) or process errors
# (usually auth/config issues, not transient).
_CLAUDE_TRANSIENT_RETRIES = 2

# Tools blocked for all automated Claude calls (prevent runaway agents)
DEFAULT_DISALLOWED_TOOLS = [
    "Task",
    "WebSearch",
    "WebFetch",
    "Bash",
    "Write",
    "Edit",
    "NotebookEdit",
]

# All tools blocked (read-only not needed either)
DISALLOWED_TOOLS_ALL = DEFAULT_DISALLOWED_TOOLS + ["Read", "Glob", "Grep"]


# --- Rate limiting defaults (overridden by config.toml [claude_cli]) ---

_DEFAULT_CLAUDE_MAX_PARALLEL_PROCESSES = 32
_DEFAULT_CLAUDE_MAX_REQUESTS_PER_MINUTE = 40


# --- Global rate limiting state (shared across all ClaudeCLI instances) ---

_global_semaphore: asyncio.Semaphore | None = None
_global_rate_limiter: AsyncLimiter | None = None
_global_limits_initialized = False


def _load_claude_cli_config() -> tuple[int, int]:
    """Load [claude_cli] config from config.toml.

    Returns:
        Tuple of (claude_max_parallel_processes, claude_max_requests_per_minute).
    """
    try:
        from dev_lib.config import load_project_config

        config = load_project_config()
        cli_config = config.get("claude_cli", {})
        max_parallel = cli_config.get(
            "claude_max_parallel_processes",
            _DEFAULT_CLAUDE_MAX_PARALLEL_PROCESSES,
        )
        max_rpm = cli_config.get(
            "claude_max_requests_per_minute",
            _DEFAULT_CLAUDE_MAX_REQUESTS_PER_MINUTE,
        )
        return int(max_parallel), int(max_rpm)
    except (FileNotFoundError, RuntimeError, ImportError):
        # Config not available (e.g., in tests) — use defaults
        return _DEFAULT_CLAUDE_MAX_PARALLEL_PROCESSES, _DEFAULT_CLAUDE_MAX_REQUESTS_PER_MINUTE


def _get_global_semaphore() -> asyncio.Semaphore:
    """Get or create the global process semaphore (lazy init)."""
    global _global_semaphore, _global_rate_limiter, _global_limits_initialized
    if _global_semaphore is None:
        max_parallel, max_rpm = _load_claude_cli_config()
        _global_semaphore = asyncio.Semaphore(max_parallel)
        _global_rate_limiter = AsyncLimiter(max_rpm, 60)
        _global_limits_initialized = True
        logger.info(
            "Claude CLI global rate limits initialized: "
            "max_parallel_processes={}, max_requests_per_minute={}",
            max_parallel,
            max_rpm,
        )
    return _global_semaphore


def _get_global_rate_limiter() -> AsyncLimiter:
    """Get or create the global rate limiter (lazy init)."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _get_global_semaphore()  # initializes both
    assert _global_rate_limiter is not None
    return _global_rate_limiter


def reset_global_limits(
    *,
    claude_max_parallel_processes: int | None = None,
    claude_max_requests_per_minute: int | None = None,
) -> None:
    """Reset global rate limits. Useful for tests or runtime reconfiguration.

    Args:
        claude_max_parallel_processes: Override max parallel processes.
        claude_max_requests_per_minute: Override max requests per minute.
            If None, loads from config.toml.
    """
    global _global_semaphore, _global_rate_limiter, _global_limits_initialized

    if claude_max_parallel_processes is None or claude_max_requests_per_minute is None:
        cfg_parallel, cfg_rpm = _load_claude_cli_config()
        if claude_max_parallel_processes is None:
            claude_max_parallel_processes = cfg_parallel
        if claude_max_requests_per_minute is None:
            claude_max_requests_per_minute = cfg_rpm

    _global_semaphore = asyncio.Semaphore(claude_max_parallel_processes)
    _global_rate_limiter = AsyncLimiter(claude_max_requests_per_minute, 60)
    _global_limits_initialized = True
    logger.info(
        "Claude CLI global rate limits reset: "
        "max_parallel_processes={}, max_requests_per_minute={}",
        claude_max_parallel_processes,
        claude_max_requests_per_minute,
    )


# --- Error types ---


class ClaudeCLIError(Exception):
    """Base error for Claude CLI operations."""


class ClaudeCLINotFoundError(ClaudeCLIError):
    """Claude CLI binary not found. Install: npm install -g @anthropic-ai/claude-code"""


class ClaudeCLITimeoutError(ClaudeCLIError):
    """Claude CLI process timed out."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"Claude CLI timed out after {timeout}s")
        self.timeout = timeout


class ClaudeCLIProcessError(ClaudeCLIError):
    """Claude CLI returned non-zero exit code."""

    def __init__(self, returncode: int, stderr: str) -> None:
        super().__init__(f"Claude CLI exited with code {returncode}: {stderr[:200]}")
        self.returncode = returncode
        self.stderr = stderr


class ClaudeCLIParseError(ClaudeCLIError):
    """Failed to parse Claude CLI output as JSON."""

    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


# --- Result type ---


@dataclass(frozen=True)
class ClaudeCLIResult:
    """Raw result from a Claude CLI invocation."""

    stdout: str
    stderr: str


# --- Main class ---


class ClaudeCLI:
    """Unified async wrapper for Claude CLI subprocess calls.

    All instances share global rate limits (semaphore + RPM limiter)
    configured in config.toml under [claude_cli]. Callers do NOT need
    to manage their own semaphores.

    Args:
        model: Default Claude model (sonnet, opus).
        effort: Default thinking effort (low, medium, high).
        timeout: Default timeout in seconds.
    """

    def __init__(
        self,
        model: str = "sonnet",
        effort: str = "medium",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.effort = effort
        self.timeout = timeout

    def _build_args(
        self,
        prompt: str,
        *,
        agent: str | None = None,
        disallowed_tools: list[str] | None = None,
        effort: str | None = None,
        output_format: str | None = None,
    ) -> list[str]:
        """Build the CLI argument list.

        Args:
            prompt: The prompt text.
            agent: Optional agent name (e.g., "pattern-reviewer").
            disallowed_tools: Optional list of tools to block.
            effort: Override effort level for this call.
            output_format: Optional output format (e.g., "json").

        Returns:
            List of CLI arguments (without "claude" binary name).
        """
        args: list[str] = []

        if agent:
            args.extend(["--agent", agent])

        args.extend(["--model", self.model])
        args.extend(["--effort", effort or self.effort])
        args.append("--print")

        if output_format:
            args.extend(["--output-format", output_format])

        if disallowed_tools:
            args.extend(["--disallowed-tools", ",".join(disallowed_tools)])

        if output_format == "json":
            # JSON mode uses -p flag for prompt
            args.extend(["-p", prompt])
        else:
            # Text mode uses -- separator then prompt as positional arg
            args.extend(["--", prompt])

        return args

    async def _exec(
        self,
        args: list[str],
        *,
        timeout: float,
        prompt_len: int,
        label: str,
    ) -> ClaudeCLIResult:
        """Execute Claude CLI subprocess with rate limiting and logging.

        Acquires global semaphore (parallel limit) and rate limiter (RPM limit)
        before spawning the subprocess.

        Args:
            args: CLI arguments (without "claude" binary).
            timeout: Timeout in seconds.
            prompt_len: Length of the prompt in characters (for token estimation).
            label: Human-readable label for logging (e.g., agent name or "json").

        Returns:
            ClaudeCLIResult with stdout and stderr.

        Raises:
            ClaudeCLINotFoundError: If claude binary not found.
            ClaudeCLITimeoutError: If process exceeds timeout.
            ClaudeCLIProcessError: If non-zero exit code.
        """
        est_input_tokens = prompt_len // 4  # rough char-to-token estimate

        semaphore = _get_global_semaphore()
        rate_limiter = _get_global_rate_limiter()

        async with semaphore:
            await rate_limiter.acquire()

            logger.debug(
                "claude call start | model={} effort={} label={} est_input_tokens=~{}",
                self.model,
                self.effort,
                label,
                est_input_tokens,
            )

            t0 = time.monotonic()
            try:
                proc: Process = await asyncio.create_subprocess_exec(
                    "claude",
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                raise ClaudeCLINotFoundError(
                    "Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "claude call timeout | label={} elapsed={:.1f}s timeout={}s",
                    label,
                    elapsed,
                    timeout,
                )
                proc.kill()
                await proc.wait()
                raise ClaudeCLITimeoutError(timeout)

        elapsed = time.monotonic() - t0
        stdout = stdout_bytes.decode() if stdout_bytes else ""
        stderr = stderr_bytes.decode() if stderr_bytes else ""

        if proc.returncode != 0:
            logger.error(
                "claude call failed | label={} code={} elapsed={:.1f}s stderr={}",
                label,
                proc.returncode,
                elapsed,
                stderr[:200],
            )
            raise ClaudeCLIProcessError(proc.returncode or 1, stderr)

        est_output_tokens = len(stdout) // 4
        logger.info(
            "claude call done | model={} label={} elapsed={:.1f}s "
            "est_tokens=~{}in/~{}out output_chars={}",
            self.model,
            label,
            elapsed,
            est_input_tokens,
            est_output_tokens,
            len(stdout),
        )

        return ClaudeCLIResult(stdout=stdout, stderr=stderr)

    async def arun(
        self,
        prompt: str,
        *,
        agent: str | None = None,
        disallowed_tools: list[str] | None = None,
        effort: str | None = None,
        timeout: float | None = None,
    ) -> ClaudeCLIResult:
        """Run Claude CLI asynchronously and return raw result.

        Rate limiting is applied automatically via global semaphore + RPM limiter.

        Args:
            prompt: The prompt text.
            agent: Optional agent name.
            disallowed_tools: Optional list of tools to block.
            effort: Override effort level for this call.
            timeout: Override timeout for this call.

        Returns:
            ClaudeCLIResult with stdout and stderr.

        Raises:
            ClaudeCLINotFoundError: If claude binary not found.
            ClaudeCLITimeoutError: If process exceeds timeout.
            ClaudeCLIProcessError: If non-zero exit code.
        """
        args = self._build_args(
            prompt,
            agent=agent,
            disallowed_tools=disallowed_tools,
            effort=effort,
        )
        label = agent or "text"
        return await self._exec(
            args,
            timeout=timeout or self.timeout,
            prompt_len=len(prompt),
            label=label,
        )

    @staticmethod
    def _parse_json_response(stdout: str) -> dict[str, Any]:
        """Parse Claude CLI --output-format=json output (two layers).

        Layer 1: outer envelope ``{"result": "..."}``.
        Layer 2: inner JSON object extracted from the result string (or
        returned directly if already a dict).

        Raises ClaudeCLIParseError on any failure.
        """
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCLIParseError(f"Failed to parse outer JSON envelope: {e}", stdout) from e

        content = response.get("result", response.get("content", ""))

        if isinstance(content, dict):
            return content

        if not isinstance(content, str):
            raise ClaudeCLIParseError(f"Unexpected content type: {type(content)}", stdout)

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ClaudeCLIParseError(f"No JSON object found in response: {content[:200]}", stdout)

        try:
            return json.loads(json_match.group())  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise ClaudeCLIParseError(f"Failed to parse inner JSON: {e}", stdout) from e

    async def arun_json(
        self,
        prompt: str,
        *,
        effort: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Run Claude CLI with --output-format json and parse the response.

        Retries on ``ClaudeCLIParseError`` up to ``_CLAUDE_TRANSIENT_RETRIES``
        times with exponential backoff — mirrors the vLLM client's transient
        retry policy. Does NOT retry on timeout or non-zero exit (those are
        not transient).

        Rate limiting is applied automatically via global semaphore + RPM limiter.

        Args:
            prompt: The prompt text.
            effort: Override effort level for this call.
            timeout: Override timeout for this call.

        Returns:
            Parsed inner JSON dict.

        Raises:
            ClaudeCLINotFoundError: If claude binary not found.
            ClaudeCLITimeoutError: If process exceeds timeout.
            ClaudeCLIProcessError: If non-zero exit code.
            ClaudeCLIParseError: If JSON parsing fails after all retries.
        """
        args = self._build_args(
            prompt,
            effort=effort,
            output_format="json",
        )
        effective_timeout = timeout or self.timeout

        last_error: ClaudeCLIParseError | None = None
        for attempt in range(_CLAUDE_TRANSIENT_RETRIES + 1):
            result = await self._exec(
                args,
                timeout=effective_timeout,
                prompt_len=len(prompt),
                label="json",
            )
            try:
                return self._parse_json_response(result.stdout)
            except ClaudeCLIParseError as e:
                last_error = e
                if attempt < _CLAUDE_TRANSIENT_RETRIES:
                    delay = 0.5 * (attempt + 1)
                    logger.warning(
                        "claude JSON parse failed (attempt {}/{}), retrying in {}s: {}",
                        attempt + 1,
                        _CLAUDE_TRANSIENT_RETRIES + 1,
                        delay,
                        str(e)[:200],
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

        # Unreachable: the loop either returns or raises, but mypy needs this.
        assert last_error is not None
        raise last_error
