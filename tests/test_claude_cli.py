"""Tests for dev_lib.claude_cli — unified Claude CLI wrapper."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dev_lib.claude_cli import (
    DEFAULT_DISALLOWED_TOOLS,
    DISALLOWED_TOOLS_ALL,
    ClaudeCLI,
    ClaudeCLINotFoundError,
    ClaudeCLIParseError,
    ClaudeCLIProcessError,
    ClaudeCLITimeoutError,
    reset_global_limits,
)


@pytest.fixture(autouse=True)
def _reset_rate_limits() -> None:
    """Reset global rate limits before each test to avoid cross-test interference."""
    reset_global_limits(
        claude_max_parallel_processes=100,
        claude_max_requests_per_minute=1000,
    )


class TestBuildArgs:
    """Tests for ClaudeCLI._build_args."""

    def test_basic_text_mode(self) -> None:
        """Basic call should use -- separator for prompt."""
        cli = ClaudeCLI(model="sonnet", effort="low")
        args = cli._build_args("hello")
        assert args == [
            "--model",
            "sonnet",
            "--effort",
            "low",
            "--print",
            "--",
            "hello",
        ]

    def test_with_agent(self) -> None:
        """Agent flag should come before model."""
        cli = ClaudeCLI(model="sonnet", effort="medium")
        args = cli._build_args("review", agent="pattern-reviewer")
        assert "--agent" in args
        assert args[args.index("--agent") + 1] == "pattern-reviewer"

    def test_with_disallowed_tools(self) -> None:
        """Disallowed tools should be comma-separated."""
        cli = ClaudeCLI()
        args = cli._build_args("test", disallowed_tools=["Task", "Bash"])
        assert "--disallowed-tools" in args
        assert args[args.index("--disallowed-tools") + 1] == "Task,Bash"

    def test_json_output_format_uses_p_flag(self) -> None:
        """JSON output format should use -p flag instead of -- separator."""
        cli = ClaudeCLI()
        args = cli._build_args("test", output_format="json")
        assert "--output-format" in args
        assert "json" in args
        assert "-p" in args
        assert "--" not in args

    def test_effort_override(self) -> None:
        """Per-call effort should override default."""
        cli = ClaudeCLI(effort="low")
        args = cli._build_args("test", effort="high")
        assert args[args.index("--effort") + 1] == "high"

    def test_default_disallowed_tools_constant(self) -> None:
        """DEFAULT_DISALLOWED_TOOLS should contain expected tools."""
        assert "Task" in DEFAULT_DISALLOWED_TOOLS
        assert "Bash" in DEFAULT_DISALLOWED_TOOLS
        assert "Write" in DEFAULT_DISALLOWED_TOOLS

    def test_all_disallowed_tools_superset(self) -> None:
        """DISALLOWED_TOOLS_ALL should be superset of DEFAULT."""
        assert set(DEFAULT_DISALLOWED_TOOLS).issubset(set(DISALLOWED_TOOLS_ALL))
        assert "Read" in DISALLOWED_TOOLS_ALL
        assert "Glob" in DISALLOWED_TOOLS_ALL
        assert "Grep" in DISALLOWED_TOOLS_ALL


def _make_mock_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    """Create a mock async subprocess with given outputs."""
    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (stdout, stderr)
    mock_proc.returncode = returncode
    return mock_proc


class TestArun:
    """Tests for ClaudeCLI.arun (async execution)."""

    async def test_successful_call(self) -> None:
        """Successful call should return ClaudeCLIResult."""
        mock_proc = _make_mock_proc(stdout=b"output text")

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI()
            result = await cli.arun("hello")

        assert result.stdout == "output text"
        assert result.stderr == ""

    async def test_not_found_raises(self) -> None:
        """FileNotFoundError should become ClaudeCLINotFoundError."""
        with patch(
            "dev_lib.claude_cli.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError,
        ):
            cli = ClaudeCLI()
            with pytest.raises(ClaudeCLINotFoundError):
                await cli.arun("hello")

    async def test_timeout_raises(self) -> None:
        """Timeout should kill process and raise ClaudeCLITimeoutError."""
        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = TimeoutError
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI(timeout=5)
            with pytest.raises(ClaudeCLITimeoutError) as exc_info:
                await cli.arun("hello")
            assert exc_info.value.timeout == 5
            mock_proc.kill.assert_called_once()

    async def test_nonzero_exit_raises(self) -> None:
        """Non-zero exit code should raise ClaudeCLIProcessError."""
        mock_proc = _make_mock_proc(stderr=b"something went wrong", returncode=1)

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI()
            with pytest.raises(ClaudeCLIProcessError) as exc_info:
                await cli.arun("hello")
            assert exc_info.value.returncode == 1
            assert "something went wrong" in exc_info.value.stderr


class TestArunJson:
    """Tests for ClaudeCLI.arun_json (JSON output parsing)."""

    async def test_valid_json_response(self) -> None:
        """Should parse two-layer JSON correctly."""
        inner = {"patterns": ["ml-001"], "scenario_type": "data_pipeline"}
        outer = {"result": json.dumps(inner)}
        mock_proc = _make_mock_proc(stdout=json.dumps(outer).encode())

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI()
            result = await cli.arun_json("select patterns")

        assert result == inner

    async def test_json_with_surrounding_text(self) -> None:
        """Should extract JSON from text with surrounding content."""
        inner = {"key": "value"}
        outer = {"result": f"Here is the result:\n{json.dumps(inner)}\nDone."}
        mock_proc = _make_mock_proc(stdout=json.dumps(outer).encode())

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI()
            result = await cli.arun_json("test")

        assert result == inner

    async def test_invalid_outer_json_raises(self) -> None:
        """Invalid outer JSON should raise ClaudeCLIParseError after retries."""
        mock_proc = _make_mock_proc(stdout=b"not json")

        with (
            patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("dev_lib.claude_cli.asyncio.sleep", new_callable=AsyncMock),
        ):
            cli = ClaudeCLI()
            with pytest.raises(ClaudeCLIParseError, match="outer JSON"):
                await cli.arun_json("test")

    async def test_no_inner_json_raises(self) -> None:
        """Missing inner JSON should raise ClaudeCLIParseError after retries."""
        outer = {"result": "No JSON here, just text."}
        mock_proc = _make_mock_proc(stdout=json.dumps(outer).encode())

        with (
            patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("dev_lib.claude_cli.asyncio.sleep", new_callable=AsyncMock),
        ):
            cli = ClaudeCLI()
            with pytest.raises(ClaudeCLIParseError, match="No JSON object"):
                await cli.arun_json("test")

    async def test_dict_content_returned_directly(self) -> None:
        """If content is already a dict, return it directly."""
        inner = {"key": "value"}
        outer = {"result": inner}
        mock_proc = _make_mock_proc(stdout=json.dumps(outer).encode())

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI()
            result = await cli.arun_json("test")

        assert result == inner

    async def test_timeout_propagates(self) -> None:
        """Timeout in arun_json should raise ClaudeCLITimeoutError."""
        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = TimeoutError
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI(timeout=10)
            with pytest.raises(ClaudeCLITimeoutError):
                await cli.arun_json("test")

    async def test_parse_error_chains_cause(self) -> None:
        """ClaudeCLIParseError should chain the original JSONDecodeError."""
        mock_proc = _make_mock_proc(stdout=b"not valid json")

        with (
            patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("dev_lib.claude_cli.asyncio.sleep", new_callable=AsyncMock),
        ):
            cli = ClaudeCLI()
            with pytest.raises(ClaudeCLIParseError) as exc_info:
                await cli.arun_json("test")
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


class TestArunJsonTransientRetry:
    """Tests for arun_json retry on transient JSON parse failures."""

    async def test_retries_bad_then_succeeds(self) -> None:
        """Should retry on parse error and succeed when next attempt is valid."""
        inner = {"patterns": ["pt-001"]}
        good_stdout = json.dumps({"result": json.dumps(inner)}).encode()
        bad = _make_mock_proc(stdout=b"not json")
        good = _make_mock_proc(stdout=good_stdout)

        with (
            patch(
                "dev_lib.claude_cli.asyncio.create_subprocess_exec",
                side_effect=[bad, good],
            ),
            patch("dev_lib.claude_cli.asyncio.sleep", new_callable=AsyncMock),
        ):
            cli = ClaudeCLI()
            result = await cli.arun_json("test")

        assert result == inner

    async def test_retries_exhausted_raises(self) -> None:
        """Should raise after all retry attempts return bad JSON."""
        from dev_lib.claude_cli import _CLAUDE_TRANSIENT_RETRIES

        bad = _make_mock_proc(stdout=b"not json")

        with (
            patch(
                "dev_lib.claude_cli.asyncio.create_subprocess_exec",
                side_effect=[bad] * (_CLAUDE_TRANSIENT_RETRIES + 1),
            ),
            patch("dev_lib.claude_cli.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            cli = ClaudeCLI()
            with pytest.raises(ClaudeCLIParseError):
                await cli.arun_json("test")

        # Should have slept between attempts (N retries = N sleeps)
        assert mock_sleep.call_count == _CLAUDE_TRANSIENT_RETRIES

    async def test_no_retry_on_success(self) -> None:
        """No retry when the first attempt succeeds."""
        inner = {"key": "value"}
        good = _make_mock_proc(stdout=json.dumps({"result": json.dumps(inner)}).encode())

        with (
            patch(
                "dev_lib.claude_cli.asyncio.create_subprocess_exec",
                return_value=good,
            ) as mock_exec,
            patch("dev_lib.claude_cli.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            cli = ClaudeCLI()
            result = await cli.arun_json("test")

        assert result == inner
        assert mock_exec.call_count == 1
        assert mock_sleep.call_count == 0

    async def test_timeout_not_retried(self) -> None:
        """Timeout errors should NOT trigger retries (not transient)."""
        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = TimeoutError
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with (
            patch(
                "dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc
            ) as mock_exec,
            patch("dev_lib.claude_cli.asyncio.sleep", new_callable=AsyncMock),
        ):
            cli = ClaudeCLI(timeout=10)
            with pytest.raises(ClaudeCLITimeoutError):
                await cli.arun_json("test")

        # Only one exec call — no retry on timeout
        assert mock_exec.call_count == 1


class TestGlobalRateLimiting:
    """Tests for global rate limiting (semaphore + RPM limiter)."""

    async def test_semaphore_limits_parallel(self) -> None:
        """Global semaphore should limit concurrent subprocess count."""
        reset_global_limits(
            claude_max_parallel_processes=2,
            claude_max_requests_per_minute=1000,
        )

        current_concurrent = 0
        max_concurrent_seen = 0

        async def mock_exec(*args: object, **kwargs: object) -> AsyncMock:
            nonlocal current_concurrent, max_concurrent_seen
            current_concurrent += 1
            max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
            proc = AsyncMock()
            proc.communicate.return_value = (b"ok", b"")
            proc.returncode = 0
            # Simulate some work
            import asyncio

            await asyncio.sleep(0.05)
            current_concurrent -= 1
            return proc

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", side_effect=mock_exec):
            cli = ClaudeCLI()
            tasks = [cli.arun("test") for _ in range(6)]
            import asyncio

            await asyncio.gather(*tasks)

        assert max_concurrent_seen <= 2, f"Max concurrent was {max_concurrent_seen}, expected <= 2"

    async def test_reset_changes_limits(self) -> None:
        """reset_global_limits should update the shared state."""
        reset_global_limits(
            claude_max_parallel_processes=5,
            claude_max_requests_per_minute=100,
        )

        from dev_lib.claude_cli import _global_rate_limiter, _global_semaphore

        assert _global_semaphore is not None
        assert _global_rate_limiter is not None
        assert _global_semaphore._value == 5
        assert _global_rate_limiter.max_rate == 100

    async def test_multiple_instances_share_limits(self) -> None:
        """Different ClaudeCLI instances should share the same global limits."""
        reset_global_limits(
            claude_max_parallel_processes=1,
            claude_max_requests_per_minute=1000,
        )

        call_order: list[str] = []

        async def mock_exec(*args: object, **kwargs: object) -> AsyncMock:
            proc = AsyncMock()
            proc.communicate.return_value = (b"ok", b"")
            proc.returncode = 0
            # Record which "instance" ran and ensure serialization
            import asyncio

            call_order.append("start")
            await asyncio.sleep(0.05)
            call_order.append("end")
            return proc

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", side_effect=mock_exec):
            cli_a = ClaudeCLI(model="sonnet")
            cli_b = ClaudeCLI(model="opus")
            import asyncio

            await asyncio.gather(cli_a.arun("test"), cli_b.arun("test"))

        # With semaphore=1, calls must be serialized: start,end,start,end
        assert call_order == ["start", "end", "start", "end"]

    async def test_lazy_init_without_reset(self) -> None:
        """Lazy init (no prior reset_global_limits) should work without AssertionError."""
        import dev_lib.claude_cli as mod

        # Force lazy init by clearing global state
        mod._global_semaphore = None
        mod._global_rate_limiter = None
        mod._global_limits_initialized = False

        mock_proc = _make_mock_proc(stdout=b"ok")

        with patch("dev_lib.claude_cli.asyncio.create_subprocess_exec", return_value=mock_proc):
            cli = ClaudeCLI()
            result = await cli.arun("test")

        assert result.stdout == "ok"
        # Both globals should be initialized
        assert mod._global_semaphore is not None
        assert mod._global_rate_limiter is not None
