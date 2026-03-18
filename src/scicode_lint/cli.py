"""Command-line interface."""

from __future__ import annotations

from loguru import logger

from .cli_commands import _run_analyze, _run_filter_repo, _run_lint
from .cli_parse import _configure_logging, parse_args
from .cli_server import _run_server


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    args = parse_args(argv)

    # Configure logging based on verbosity
    if hasattr(args, "verbose"):
        _configure_logging(args.verbose)

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
        return 1
