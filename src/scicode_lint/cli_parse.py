"""CLI argument parsing and logging configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from scicode_lint.config import Severity

_SEVERITY_MAP = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
}


def _parse_filters(
    args: argparse.Namespace,
) -> tuple[set[Severity], set[str] | None, set[str] | None]:
    """Parse severity, pattern, and category filters from CLI args.

    Returns:
        Tuple of (enabled_severities, enabled_patterns, enabled_categories).
    """
    enabled_severities: set[Severity] = set()
    for s in args.severity.split(","):
        s = s.strip().lower()
        if s in _SEVERITY_MAP:
            enabled_severities.add(_SEVERITY_MAP[s])
        elif s:
            logger.warning(f"Unknown severity '{s}', ignoring (valid: critical, high, medium)")

    enabled_patterns = None
    if args.pattern:
        enabled_patterns = {p.strip() for p in args.pattern.split(",")}

    enabled_categories = None
    if args.category:
        enabled_categories = {c.strip() for c in args.category.split(",")}

    return enabled_severities, enabled_patterns, enabled_categories


def _configure_logging(verbose: int) -> None:
    """Configure logging based on verbosity level."""
    logger.remove()  # Remove default handler
    if verbose == 0:
        # Only show warnings and errors to stderr
        logger.add(sys.stderr, level="WARNING", format="<level>{message}</level>")
    elif verbose == 1:
        # Show info messages (includes timing)
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )
    elif verbose == 2:
        # Show debug messages
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
        )
    else:  # verbose >= 3
        # Show everything with full context
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan> - {message}"
            ),
        )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-powered linter for scientific Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments for LLM-based commands
    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--vllm-url",
            type=str,
            help="vLLM server URL (default: auto-detect on ports 5001 or 8000)",
        )
        subparser.add_argument(
            "--model",
            type=str,
            help="Model name (default: auto-detect from vLLM server)",
        )
        subparser.add_argument(
            "--verbose",
            "-v",
            action="count",
            default=0,
            help="Increase verbosity (use -v, -vv, or -vvv)",
        )

    # Lint command
    lint_parser = subparsers.add_parser("lint", help="Lint files for issues")
    lint_parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to lint",
    )
    lint_parser.add_argument(
        "--severity",
        type=str,
        help="Comma-separated severity levels to check (critical,high,medium)",
        default="critical,high,medium",
    )
    lint_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (text for humans, json for GenAI agents)",
    )
    lint_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (0.0-1.0)",
    )
    lint_parser.add_argument(
        "--pattern",
        type=str,
        help=(
            "Comma-separated pattern IDs to check (e.g., ml-001,ml-002). "
            "If not specified, all patterns are checked."
        ),
    )
    lint_parser.add_argument(
        "--category",
        type=str,
        help=(
            "Comma-separated categories to check (e.g., ml-correctness,pytorch-training). "
            "If not specified, all categories are checked."
        ),
    )
    lint_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run in benchmark mode (shows detailed timing statistics)",
    )
    add_common_args(lint_parser)

    # Filter-repo command
    filter_parser = subparsers.add_parser(
        "filter-repo",
        help="Filter repository for self-contained ML files",
    )
    filter_parser.add_argument(
        "repo_path",
        type=Path,
        help="Repository path to filter",
    )
    filter_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file with found files",
    )
    filter_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    filter_parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Include files with uncertain classification",
    )
    filter_parser.add_argument(
        "--filter-concurrency",
        type=int,
        default=None,
        help="Max concurrent LLM requests during file filtering (default: 50)",
    )
    filter_parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Store results to SQLite database (for real_world_demo integration)",
    )
    filter_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite database (implies --save-to-db)",
    )
    add_common_args(filter_parser)

    # Analyze command (full pipeline: clone -> filter -> lint)
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Full analysis pipeline: clone repo, filter ML files, run lint",
    )
    analyze_parser.add_argument(
        "repo",
        type=str,
        help="Repository URL (https/git) or local path",
    )
    analyze_parser.add_argument(
        "--severity",
        type=str,
        help="Comma-separated severity levels to check (critical,high,medium)",
        default="critical,high,medium",
    )
    analyze_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    analyze_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (0.0-1.0)",
    )
    analyze_parser.add_argument(
        "--pattern",
        type=str,
        help="Comma-separated pattern IDs to check",
    )
    analyze_parser.add_argument(
        "--category",
        type=str,
        help="Comma-separated categories to check",
    )
    analyze_parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Include files with uncertain classification in analysis",
    )
    analyze_parser.add_argument(
        "--keep-clone",
        action="store_true",
        help="Keep cloned repo after analysis (default: delete)",
    )
    analyze_parser.add_argument(
        "--clone-dir",
        type=Path,
        default=None,
        help="Directory to clone repo into (default: temp directory)",
    )
    analyze_parser.add_argument(
        "--filter-concurrency",
        type=int,
        default=None,
        help="Max concurrent LLM requests during file filtering (default: 50)",
    )
    analyze_parser.add_argument(
        "--lint-concurrency",
        type=int,
        default=None,
        help="Max concurrent pattern checks per file during linting (default: 60)",
    )
    add_common_args(analyze_parser)

    # vllm-server command
    server_parser = subparsers.add_parser(
        "vllm-server",
        help="Manage vLLM container (start/stop/status/restart/logs/rm)",
    )
    server_subparsers = server_parser.add_subparsers(dest="server_command", help="Server action")

    # server start
    server_start = server_subparsers.add_parser("start", help="Start vLLM container")
    server_start.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: from config.toml)",
    )
    server_start.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run on (default: 5001)",
    )
    server_start.add_argument(
        "--pull",
        action="store_true",
        help="Pull latest container image before starting",
    )

    # server stop
    server_subparsers.add_parser("stop", help="Stop vLLM container")

    # server status
    server_subparsers.add_parser("status", help="Show container and GPU status")

    # server restart
    server_restart = server_subparsers.add_parser(
        "restart", help="Restart vLLM container (stop + remove + start)"
    )
    server_restart.add_argument("--model", type=str, default=None)
    server_restart.add_argument("--port", type=int, default=None)
    server_restart.add_argument("--pull", action="store_true", help="Pull latest image")

    # server logs
    server_logs = server_subparsers.add_parser("logs", help="Show vLLM container logs")
    server_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    server_logs.add_argument(
        "--tail", type=int, default=50, help="Number of lines to show (default: 50)"
    )

    # server rm
    server_rm = server_subparsers.add_parser("rm", help="Remove vLLM container")
    server_rm.add_argument("--force", action="store_true", help="Force remove even if running")

    # server monitor
    server_monitor = server_subparsers.add_parser(
        "monitor", help="Live-refresh monitor for vLLM metrics"
    )
    server_monitor.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2)",
    )

    parsed = parser.parse_args(args)

    # Show help if no command provided
    if parsed.command is None:
        parser.print_help()
        sys.exit(1)

    # Show server help if no server subcommand
    if parsed.command == "vllm-server" and getattr(parsed, "server_command", None) is None:
        server_parser.print_help()
        sys.exit(1)

    return parsed
