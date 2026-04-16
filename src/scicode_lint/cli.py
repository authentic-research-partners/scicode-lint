"""Command-line interface."""

from __future__ import annotations

import sys

from loguru import logger

from .cli_commands import _run_analyze, _run_filter_repo, _run_lint
from .cli_parse import _configure_logging, parse_args
from .cli_server import _run_server
from .exceptions import SciCodeLintError


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Exit codes follow linter convention (see docs/USAGE.md):
        0 = clean run (no findings)
        1 = findings detected
        2 = tool/runtime error (e.g., vLLM unreachable, config invalid)
    """
    args = parse_args(argv)

    # Configure logging based on verbosity
    if hasattr(args, "verbose"):
        _configure_logging(args.verbose)

    try:
        # Dispatch to appropriate command handler
        if args.command == "lint":
            return _run_lint(args)
        elif args.command == "filter-repo":
            return _run_filter_repo(args)
        elif args.command == "analyze":
            return _run_analyze(args)
        elif args.command == "vllm-server":
            return _run_server(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 2
    except SciCodeLintError as e:
        # Any documented scicode-lint error reaching this point is a tool
        # failure (per-file errors are already captured into LintResult.error
        # by the command handlers). Print concise message to stderr and exit 2
        # so CI can distinguish "findings" from "tool broken".
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
