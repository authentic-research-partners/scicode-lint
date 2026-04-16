"""CLI command implementations (lint, filter-repo, analyze)."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from scicode_lint.config import LinterConfig, LLMConfig, get_default_config
from scicode_lint.linter import SciCodeLinter
from scicode_lint.output.formatter import LintError, LintResult, format_findings

from .cli_parse import _parse_filters

if TYPE_CHECKING:
    from scicode_lint.repo_filter.scan import RepoScanSummary


def _check_file_capturing_errors(linter: SciCodeLinter, file_path: Path) -> LintResult:
    """Lint one file; if it raises, return a LintResult carrying a LintError.

    Used by both ``lint`` and ``analyze`` so per-file error handling is in one
    place. Systemic errors (e.g. ``LLMConnectionError``) should be re-raised
    inside the linter itself — this helper only captures per-file failures
    (context overflow, notebook parse, etc.) that the CLI surfaces via
    ``LintResult.error`` and uses to drive exit code ``2``.
    """
    try:
        return linter.check_file(file_path)
    except Exception as e:
        logger.error(f"Error checking {file_path}: {e}")
        error = LintError(
            file=file_path,
            error_type=type(e).__name__,
            message=str(e),
            details=e.to_dict() if hasattr(e, "to_dict") else None,
        )
        return LintResult(file=file_path, findings=[], error=error)


def _run_lint(args: argparse.Namespace) -> int:
    """Run the lint command."""
    start_time = time.time()

    enabled_severities, enabled_patterns, enabled_categories = _parse_filters(args)
    if enabled_patterns:
        logger.info(f"Filtering to patterns: {enabled_patterns}")
    if enabled_categories:
        logger.info(f"Filtering to categories: {enabled_categories}")

    # Configure LLM (auto-detects vLLM if no URL specified)
    llm_config = LLMConfig(
        base_url=args.vllm_url or "",
        model=args.model or "",
    )

    # Configure linter
    linter_config = LinterConfig(
        # patterns_dir will use default from get_default_patterns_dir()
        llm_config=llm_config,
        min_confidence=args.min_confidence,
        enabled_severities=enabled_severities,
        enabled_patterns=enabled_patterns,
        enabled_categories=enabled_categories,
    )

    # Create linter
    linter = SciCodeLinter(linter_config)

    # Collect files to check
    files_to_check: list[Path] = []
    for path in args.paths:
        if path.is_file():
            if path.suffix in {".py", ".ipynb"}:
                files_to_check.append(path)
        elif path.is_dir():
            # Recursively find Python files
            files_to_check.extend(path.rglob("*.py"))
            files_to_check.extend(path.rglob("*.ipynb"))

    if not files_to_check:
        logger.error("No Python files found to check.")
        return 1

    logger.info(f"Found {len(files_to_check)} files to check")

    # Check all files
    results = []
    for idx, file_path in enumerate(files_to_check, 1):
        logger.info(f"Processing file {idx}/{len(files_to_check)}")
        results.append(_check_file_capturing_errors(linter, file_path))

    # Calculate timing statistics
    elapsed = time.time() - start_time
    total_findings = sum(len(r.findings) for r in results)
    errors_occurred = any(r.error is not None for r in results)

    # Format and print results
    output = format_findings(results, output_format=args.format)
    print(output)

    # Show benchmark/timing summary
    if args.benchmark or args.verbose > 0:
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files checked: {len(results)}")
        logger.info(f"Total findings: {total_findings}")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Average time per file: {elapsed / len(results):.2f}s")
        logger.info(f"Files per minute: {len(results) / (elapsed / 60):.1f}")

        # Show per-file breakdown in benchmark mode
        if args.benchmark and results:
            logger.info("\nPer-file breakdown:")
            for result in results:
                logger.info(f"  {result.file.name}: {len(result.findings)} findings")

        logger.info("=" * 60)

    logger.success(f"Linting completed in {elapsed:.2f}s")

    # Exit codes follow linter convention (same as ruff, mypy, shellcheck):
    #   0 = clean, 1 = findings detected, 2 = tool/runtime error
    if errors_occurred:
        return 2
    return 1 if total_findings > 0 else 0


def _run_filter_repo(args: argparse.Namespace) -> int:
    """Run the filter-repo command."""
    from scicode_lint.config import get_filter_concurrency
    from scicode_lint.llm.client import create_client
    from scicode_lint.repo_filter.scan import filter_scan_results, scan_repo_for_ml_files

    start_time = time.time()

    # Validate repo path
    if not args.repo_path.is_dir():
        logger.error(f"Repository path does not exist: {args.repo_path}")
        return 1

    # Configure LLM
    llm_config = LLMConfig(
        base_url=args.vllm_url or "",
        model=args.model or "",
    )

    # Create LLM client
    llm_client = create_client(llm_config)

    # Get max_concurrent from args or config
    max_concurrent = args.filter_concurrency or get_filter_concurrency()

    # Run scan (returns ALL results)
    logger.info(f"Scanning repository: {args.repo_path}")
    summary = asyncio.run(
        scan_repo_for_ml_files(
            repo_path=args.repo_path,
            llm_client=llm_client,
            max_concurrent=max_concurrent,
        )
    )

    elapsed = time.time() - start_time

    # Store to database (if --save-to-db or --db-path specified)
    # DB gets ALL results from summary.results
    scan_id = None
    if args.save_to_db or args.db_path:
        scan_id = _store_scan_to_db(args, summary, elapsed, llm_config)

    # Filter results for display
    filtered_results = filter_scan_results(summary, include_uncertain=args.include_uncertain)

    # Output results
    if args.format == "json":
        output_data = summary.to_dict()
        # Replace with filtered results for JSON output
        output_data["files"] = [r.to_dict() for r in filtered_results]
        output_data["elapsed_seconds"] = round(elapsed, 2)
        if scan_id:
            output_data["scan_id"] = scan_id
        output = json.dumps(output_data, indent=2)
    else:
        lines = [
            f"Scan completed in {elapsed:.2f}s",
            "",
            f"Total files: {summary.total_files}",
            f"Passed ML import filter: {summary.passed_ml_import_filter}",
            f"Failed ML import filter: {summary.failed_ml_import_filter}",
            "",
            "After LLM classification:",
            f"  Self-contained: {summary.self_contained}",
            f"  Fragments: {summary.fragments}",
            f"  Uncertain: {summary.uncertain}",
            "",
        ]
        if scan_id:
            lines.append(f"Stored to database (scan_id: {scan_id})")
            lines.append("")
        if filtered_results:
            lines.append("Self-contained ML files:")
            for result in filtered_results:
                rel_path = result.filepath.relative_to(args.repo_path)
                if result.details:
                    lines.append(f"  {rel_path} (confidence: {result.details.confidence:.2f})")
                else:
                    lines.append(f"  {rel_path}")
        else:
            lines.append("No self-contained ML files found.")
        output = "\n".join(lines)

    print(output)

    # Save to file if requested
    if args.output:
        output_data = summary.to_dict()
        output_data["elapsed_seconds"] = round(elapsed, 2)
        if scan_id:
            output_data["scan_id"] = scan_id
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return 0


def _store_scan_to_db(
    args: argparse.Namespace,
    summary: RepoScanSummary,
    elapsed: float,
    llm_config: LLMConfig,
) -> int | None:
    """Store scan results to database.

    Args:
        args: CLI arguments.
        summary: Scan summary with results.
        elapsed: Scan duration in seconds.
        llm_config: LLM configuration used.

    Returns:
        Scan ID if stored successfully, None otherwise.
    """
    try:
        # Import here to avoid circular imports and make real_world_demo optional
        from real_world_demo.database import (
            complete_repo_scan,
            get_or_create_repo,
            init_db,
            insert_file,
            start_repo_scan,
            update_file_classification,
        )
    except ImportError:
        logger.warning("real_world_demo not available, skipping database storage")
        return None

    # Determine database path
    if args.db_path:
        db_path = args.db_path
    else:
        # Default: real_world_demo/data/analysis.db
        db_path = Path(__file__).parent.parent.parent / "real_world_demo" / "data" / "analysis.db"

    try:
        conn = init_db(db_path)

        # Get or create repo record
        repo_name = args.repo_path.name
        repo_data = {
            "repo_name": repo_name,
            "repo_url": str(args.repo_path.absolute()),
            "data_source": "scan_repo_cli",
        }
        repo_id = get_or_create_repo(conn, repo_data)

        if not repo_id:
            logger.warning("Failed to create repo record")
            return None

        # Start scan record
        scan_id = start_repo_scan(
            conn,
            repo_id=repo_id,
            total_files=summary.total_files,
            model_name=llm_config.model_served_name,
        )

        # Store individual file classifications (all results)
        for result in summary.results:
            # Insert or get file record
            file_data = {
                "file_path": str(result.filepath.relative_to(args.repo_path)),
                "original_path": str(result.filepath),
                "is_notebook": result.filepath.suffix == ".ipynb",
            }
            file_id = insert_file(conn, repo_id, file_data)

            # Determine has_ml_imports from skip_reason
            has_ml = result.skip_reason != "no_ml_imports_found"

            if file_id:
                update_file_classification(
                    conn,
                    file_id=file_id,
                    scan_id=scan_id,
                    has_ml_imports=has_ml,
                    classification=result.classification if has_ml else None,
                    confidence=result.details.confidence if result.details else None,
                    reasoning=result.details.reasoning if result.details else None,
                )

        # Complete scan record
        # skipped includes both no_ml_imports and too_large
        skipped = summary.failed_ml_import_filter + summary.skipped_too_large
        complete_repo_scan(
            conn,
            scan_id=scan_id,
            passed_ml_import_filter=summary.passed_ml_import_filter,
            self_contained=summary.self_contained,
            fragments=summary.fragments,
            uncertain=summary.uncertain,
            skipped=skipped,
            duration_seconds=elapsed,
        )

        conn.close()
        logger.info(f"Scan results stored to {db_path} (scan_id: {scan_id})")
        return scan_id

    except Exception as e:
        logger.error(f"Failed to store scan results: {e}")
        return None


def _is_git_url(repo: str) -> bool:
    """Check if repo is a git URL (not a local path).

    Works with any git host (GitHub, GitLab, Bitbucket, self-hosted):
    - https://github.com/user/repo
    - https://gitlab.com/user/repo
    - git@gitlab.company.com:group/repo
    - git://bitbucket.org/user/repo
    """
    return repo.startswith(("https://", "http://", "git@", "git://"))


def _clone_repo(url: str, target_dir: Path) -> bool:
    """Clone a git repository.

    Args:
        url: Git repository URL.
        target_dir: Directory to clone into.

    Returns:
        True if clone succeeded, False otherwise.
    """
    try:
        logger.info(f"Cloning {url} to {target_dir}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        logger.error("git command not found. Please install git.")
        return False


def _run_analyze(args: argparse.Namespace) -> int:
    """Run the analyze command (full pipeline)."""
    from scicode_lint.config import get_filter_concurrency
    from scicode_lint.llm.client import create_client
    from scicode_lint.repo_filter.scan import filter_scan_results, scan_repo_for_ml_files

    start_time = time.time()

    # Determine if repo is URL or local path
    is_url = _is_git_url(args.repo)
    temp_dir = None
    repo_path: Path

    if is_url:
        # Clone to temp or specified directory
        if args.clone_dir:
            repo_path = args.clone_dir
            repo_path.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp(prefix="scicode-lint-")
            repo_path = Path(temp_dir)

        if not _clone_repo(args.repo, repo_path):
            if temp_dir and not args.keep_clone:
                shutil.rmtree(temp_dir, ignore_errors=True)
            return 1
    else:
        # Local path
        repo_path = Path(args.repo)
        if not repo_path.is_dir():
            logger.error(f"Repository path does not exist: {repo_path}")
            return 1

    # Quick check: does the directory contain any Python files?
    py_files = list(repo_path.rglob("*.py"))[:1]  # Just check if any exist
    ipynb_files = list(repo_path.rglob("*.ipynb"))[:1]
    if not py_files and not ipynb_files:
        logger.error(f"No Python files (.py or .ipynb) found in {repo_path}")
        if temp_dir and not args.keep_clone:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return 1

    try:
        # Configure LLM
        llm_config = LLMConfig(
            base_url=args.vllm_url or "",
            model=args.model or "",
        )
        llm_client = create_client(llm_config)

        # Phase 1: Scan and filter (concurrent LLM calls with semaphore)
        # Uses asyncio.Semaphore internally for concurrent file classification
        logger.info("Phase 1: Scanning repository for self-contained ML files...")
        max_concurrent = args.filter_concurrency or get_filter_concurrency()

        scan_summary = asyncio.run(
            scan_repo_for_ml_files(
                repo_path=repo_path,
                llm_client=llm_client,
                max_concurrent=max_concurrent,
            )
        )

        # Filter results based on include_uncertain flag
        filtered_results = filter_scan_results(
            scan_summary, include_uncertain=args.include_uncertain
        )
        ml_files = [r.filepath for r in filtered_results]
        scan_elapsed = time.time() - start_time

        logger.info(
            f"Scan complete: {scan_summary.total_files} files, "
            f"{scan_summary.self_contained} self-contained"
        )

        if not ml_files:
            logger.warning("No self-contained ML files found to analyze")
            if args.format == "json":
                output = json.dumps(
                    {
                        "scan": scan_summary.to_dict(),
                        "findings": [],
                        "summary": {"total_files_analyzed": 0, "total_findings": 0},
                    },
                    indent=2,
                )
            else:
                output = (
                    f"Scan completed in {scan_elapsed:.2f}s\n\n"
                    f"Total files: {scan_summary.total_files}\n"
                    f"Passed ML import filter: {scan_summary.passed_ml_import_filter}\n"
                    f"Self-contained: {scan_summary.self_contained}\n\n"
                    "No self-contained ML files found to analyze."
                )
            print(output)
            return 0

        # Phase 2: Run linter on self-contained files (sequential)
        # Linter checks each file one at a time (multiple patterns per file)
        # Both phases use the same vLLM server; phases run sequentially
        logger.info(f"Phase 2: Analyzing {len(ml_files)} self-contained ML files...")

        enabled_severities, enabled_patterns, enabled_categories = _parse_filters(args)

        # Configure linter
        lint_concurrency = (
            args.lint_concurrency
            if args.lint_concurrency is not None
            else get_default_config().max_concurrent
        )
        linter_config = LinterConfig(
            llm_config=llm_config,
            min_confidence=args.min_confidence,
            enabled_severities=enabled_severities,
            enabled_patterns=enabled_patterns,
            enabled_categories=enabled_categories,
            max_concurrent=lint_concurrency,
        )
        linter = SciCodeLinter(linter_config)

        # Check all self-contained files
        results = []
        for idx, file_path in enumerate(ml_files, 1):
            logger.info(f"Analyzing file {idx}/{len(ml_files)}: {file_path.name}")
            results.append(_check_file_capturing_errors(linter, file_path))

        # Calculate stats
        elapsed = time.time() - start_time
        total_findings = sum(len(r.findings) for r in results)
        errors_occurred = any(r.error is not None for r in results)

        # Format output
        if args.format == "json":
            findings_output = json.loads(format_findings(results, output_format="json"))
            output = json.dumps(
                {
                    "repo": args.repo,
                    "scan": scan_summary.to_dict(),
                    "findings": findings_output,
                    "summary": {
                        "total_files_scanned": scan_summary.total_files,
                        "self_contained_files": scan_summary.self_contained,
                        "files_analyzed": len(ml_files),
                        "total_findings": total_findings,
                        "scan_time_seconds": round(scan_elapsed, 2),
                        "total_time_seconds": round(elapsed, 2),
                    },
                },
                indent=2,
            )
        else:
            findings_text = format_findings(results, output_format="text")
            lines = [
                f"Repository: {args.repo}",
                "",
                "=== Scan Results ===",
                f"Total files: {scan_summary.total_files}",
                f"Passed ML import filter: {scan_summary.passed_ml_import_filter}",
                f"Self-contained: {scan_summary.self_contained}",
                f"Fragments: {scan_summary.fragments}",
                "",
                "=== Analysis Results ===",
                f"Files analyzed: {len(ml_files)}",
                f"Total findings: {total_findings}",
                "",
            ]
            if findings_text.strip():
                lines.append(findings_text)
            lines.extend(
                [
                    "",
                    f"Scan time: {scan_elapsed:.2f}s",
                    f"Total time: {elapsed:.2f}s",
                ]
            )
            output = "\n".join(lines)

        print(output)
        # Exit codes follow linter convention: 0=clean, 1=findings, 2=error
        if errors_occurred:
            return 2
        return 1 if total_findings > 0 else 0

    finally:
        # Cleanup temp directory if we created one
        if temp_dir and not args.keep_clone:
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif is_url and args.keep_clone:
            logger.info(f"Repository kept at: {repo_path}")
