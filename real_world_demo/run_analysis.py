"""Run scicode-lint on collected files and aggregate findings.

Analyzes collected ML code files and generates impact statistics.
Uses direct async calls to SciCodeLinter (no subprocess overhead).

Usage:
    python run_analysis.py [--max-files 100]
"""

import argparse
import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.analysis_reporting import (
    aggregate_findings,
    generate_report,
    print_summary,
    save_raw_results,
)
from real_world_demo.config import (
    ANALYSIS_CONCURRENCY,
    COLLECTED_DIR,
    DATA_SOURCE_CONFIGS,
    DEFAULT_ANALYSIS_CONCURRENT,
    REPORTS_DIR,
)
from real_world_demo.database import (
    complete_analysis_run,
    get_analyzed_file_ids,
    get_incomplete_run,
    get_or_create_repo,
    get_prefilter_run,
    get_prefilter_run_files,
    get_timed_out_patterns,
    init_db,
    insert_file,
    insert_file_analysis,
    insert_findings,
    insert_pattern_runs,
    print_stats,
    start_analysis_run,
    update_pattern_run,
)
from scicode_lint import SciCodeLinter
from scicode_lint.config import get_default_config

# Re-export for backward compatibility (tests import from here)
__all__ = ["aggregate_findings"]


def load_manifest(manifest_file: Path) -> list[dict[str, Any]]:
    """Load manifest CSV file.

    Args:
        manifest_file: Path to manifest.csv.

    Returns:
        List of manifest records.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
    """
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_file}. Run generate_manifest.py first."
        )

    with open(manifest_file, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


async def run_scicode_lint(
    file_path: Path,
    linter: SciCodeLinter,
    pattern_timeout: float = 120.0,
) -> dict[str, Any]:
    """Run scicode-lint on a single file using direct async call.

    Args:
        file_path: Path to file to analyze.
        linter: Shared SciCodeLinter instance.
        pattern_timeout: Timeout in seconds per pattern, applied AFTER work starts
            (queue wait time doesn't count toward timeout).

    Returns:
        Dict with file_path, success, findings, and error fields.
    """
    try:
        logger.debug(f"Analyzing: {file_path}")

        # Timeout is handled per-pattern INSIDE the linter (after semaphore acquired)
        lint_result = await linter._check_file_async(file_path, pattern_timeout=pattern_timeout)

        # Check for errors in result
        if lint_result.error:
            return {
                "file_path": str(file_path),
                "success": False,
                "error": f"{lint_result.error.error_type}: {lint_result.error.message}",
                "findings": [],
            }

        # Check if all patterns failed (e.g., all timed out)
        total_patterns = len(lint_result.checked_patterns) + lint_result.patterns_failed
        if total_patterns > 0 and lint_result.patterns_failed == total_patterns:
            # Summarize failure reasons
            error_types: dict[str, int] = {}
            for fp in lint_result.failed_patterns:
                error_types[fp.error_type] = error_types.get(fp.error_type, 0) + 1
            error_summary = ", ".join(f"{count} {t}" for t, count in error_types.items())
            return {
                "file_path": str(file_path),
                "success": False,
                "error": f"All {lint_result.patterns_failed} patterns failed ({error_summary})",
                "findings": [],
            }

        # Convert findings to dict format
        findings = []
        for finding in lint_result.findings:
            loc = finding.location
            findings.append(
                {
                    "pattern_id": finding.id,
                    "category": finding.category,
                    "severity": finding.severity,
                    "confidence": finding.confidence,
                    "issue": finding.issue,
                    "explanation": finding.explanation,
                    "suggestion": finding.suggestion,
                    "reasoning": finding.reasoning,
                    "location_name": loc.name if loc else None,
                    "location_type": loc.location_type if loc else None,
                    "lines": loc.lines if loc else [],
                    "focus_line": loc.focus_line if loc else None,
                    "snippet": loc.snippet if loc else "",
                }
            )

        # Convert pattern results to dict format
        checked_patterns = [
            {
                "pattern_id": p.pattern_id,
                "detected": p.detected,
                "confidence": p.confidence,
                "reasoning": p.reasoning,
            }
            for p in lint_result.checked_patterns
        ]
        failed_patterns = [
            {
                "pattern_id": p.pattern_id,
                "error_type": p.error_type,
                "error_message": p.error_message,
            }
            for p in lint_result.failed_patterns
        ]

        return {
            "file_path": str(file_path),
            "success": True,
            "error": None,
            "findings": findings,
            "checked_patterns": checked_patterns,
            "failed_patterns": failed_patterns,
        }

    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return {
            "file_path": str(file_path),
            "success": False,
            "error": str(e),
            "findings": [],
        }


def calculate_timeout(record: dict[str, Any], base_timeout: float) -> float:
    """Calculate timeout based on file size.

    Larger files need more time to analyze. Scale timeout based on line count.

    Args:
        record: Manifest record with file metadata.
        base_timeout: Base timeout in seconds.

    Returns:
        Scaled timeout in seconds.
    """
    line_count = int(record.get("line_count", 0))

    # Scale: base timeout for files up to 500 lines
    # Add 60s per 500 additional lines, cap at 5x base
    if line_count <= 500:
        return base_timeout

    extra_blocks = (line_count - 500) // 500
    scaled = base_timeout + (extra_blocks * 60)
    max_timeout = base_timeout * 5

    return min(scaled, max_timeout)


async def retry_pattern(
    file_path: Path,
    pattern_id: str,
    linter: SciCodeLinter,
    pattern_timeout: float,
) -> dict[str, Any]:
    """Retry a single pattern on a file.

    Args:
        file_path: Path to file to analyze.
        pattern_id: Pattern ID to check.
        linter: Shared SciCodeLinter instance (must have pattern enabled).
        pattern_timeout: Timeout in seconds.

    Returns:
        Dict with pattern_id, success, and result details.
    """
    try:
        logger.debug(f"Retrying pattern {pattern_id} on {file_path.name}")

        lint_result = await linter._check_file_async(file_path, pattern_timeout=pattern_timeout)

        # Find result for the specific pattern
        for checked in lint_result.checked_patterns:
            if checked.pattern_id == pattern_id:
                return {
                    "pattern_id": pattern_id,
                    "success": True,
                    "status": "success",
                    "detected": checked.detected,
                    "confidence": checked.confidence,
                    "reasoning": checked.reasoning,
                }

        # Check if it failed again
        for failed in lint_result.failed_patterns:
            if failed.pattern_id == pattern_id:
                return {
                    "pattern_id": pattern_id,
                    "success": False,
                    "status": failed.error_type,
                    "error_message": failed.error_message,
                }

        # Pattern not found in results
        return {
            "pattern_id": pattern_id,
            "success": False,
            "status": "error",
            "error_message": "Pattern not found in results",
        }

    except Exception as e:
        logger.error(f"Error retrying {pattern_id} on {file_path}: {e}")
        return {
            "pattern_id": pattern_id,
            "success": False,
            "status": "error",
            "error_message": str(e),
        }


async def retry_timed_out_patterns(
    conn: Any,
    run_id: int,
    base_dir: Path,
    pattern_timeout: float,
    max_concurrent: int = 10,
) -> tuple[int, int]:
    """Retry all timed-out patterns from a run.

    Args:
        conn: Database connection.
        run_id: Analysis run ID to retry timeouts from.
        base_dir: Base directory containing files.
        pattern_timeout: New timeout in seconds.
        max_concurrent: Maximum concurrent requests.

    Returns:
        Tuple of (retried_count, success_count).
    """
    from collections import defaultdict

    # Get all timed-out patterns
    timeouts = get_timed_out_patterns(conn, run_id)
    if not timeouts:
        logger.info("No timed-out patterns to retry")
        return 0, 0

    logger.info(f"Found {len(timeouts)} timed-out patterns to retry")

    # Group by file for efficient processing
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in timeouts:
        by_file[t["file_path"]].append(t)

    logger.info(f"Across {len(by_file)} files")

    # Process files
    config = get_default_config()
    config.max_concurrent = max_concurrent
    retried = 0
    succeeded = 0

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_file_patterns(
        file_path_str: str, patterns: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Process all timed-out patterns for one file."""
        async with semaphore:
            results: list[tuple[dict[str, Any], dict[str, Any]]] = []
            file_path = base_dir / file_path_str

            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return results

            # Create linter with only these patterns enabled
            pattern_ids = {p["pattern_id"] for p in patterns}
            file_config = get_default_config()
            file_config.enabled_patterns = pattern_ids
            file_config.max_concurrent = 1  # Already controlled by outer semaphore
            file_linter = SciCodeLinter(file_config)

            for p in patterns:
                result = await retry_pattern(
                    file_path, p["pattern_id"], file_linter, pattern_timeout
                )
                results.append((p, result))

            return results

    # Run all retries
    tasks = [process_file_patterns(fp, patterns) for fp, patterns in by_file.items()]
    all_results = await asyncio.gather(*tasks)

    # Update database
    for file_results in all_results:
        for timeout_info, result in file_results:
            retried += 1
            file_analysis_id = timeout_info["file_analysis_id"]
            pattern_id = timeout_info["pattern_id"]

            if result["success"]:
                succeeded += 1
                update_pattern_run(
                    conn,
                    file_analysis_id,
                    pattern_id,
                    status="success",
                    detected=result.get("detected"),
                    confidence=result.get("confidence"),
                    reasoning=result.get("reasoning"),
                )
                logger.info(
                    f"✓ {pattern_id} on {timeout_info['file_path']}: "
                    f"detected={result.get('detected')}"
                )
            else:
                update_pattern_run(
                    conn,
                    file_analysis_id,
                    pattern_id,
                    status=result.get("status", "error"),
                    error_message=result.get("error_message"),
                )
                logger.warning(
                    f"✗ {pattern_id} on {timeout_info['file_path']}: "
                    f"{result.get('status')} - {result.get('error_message', '')[:50]}"
                )

    return retried, succeeded


async def analyze_single_file(
    record: dict[str, Any],
    base_dir: Path,
    linter: SciCodeLinter,
    pattern_timeout: float,
) -> dict[str, Any]:
    """Analyze a single file and enrich result with metadata.

    Args:
        record: Manifest record for the file.
        base_dir: Base directory containing collected files.
        linter: Shared SciCodeLinter instance.
        pattern_timeout: Base timeout in seconds per pattern (scaled by file size).

    Returns:
        Analysis result enriched with metadata including duration.
    """
    start_time = time.time()
    file_path = base_dir / record["file_path"]

    # Scale timeout based on file size
    scaled_timeout = calculate_timeout(record, pattern_timeout)
    result = await run_scicode_lint(file_path, linter, scaled_timeout)
    duration = time.time() - start_time

    # Enrich with manifest metadata and timing
    result["domain"] = record.get("domain", "")
    result["repo_name"] = record.get("repo_name", "")
    result["paper_url"] = record.get("paper_url", "")
    result["is_notebook"] = record.get("is_notebook", "")
    result["duration_seconds"] = duration

    return result


async def analyze_all_files_incremental(
    manifest: list[dict[str, Any]],
    base_dir: Path,
    conn: Any,
    run_id: int,
    file_ids: dict[str, int],
    linter: SciCodeLinter,
    max_files: int | None = None,
    pattern_timeout: float = 120.0,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Run scicode-lint on all files, saving results incrementally.

    Args:
        manifest: List of manifest records.
        base_dir: Base directory containing collected files.
        conn: SQLite database connection.
        run_id: Analysis run ID.
        file_ids: Mapping of file paths to database IDs.
        linter: Shared SciCodeLinter instance (has internal semaphore for vLLM).
        max_files: Optional limit on number of files to analyze.
        pattern_timeout: Timeout in seconds per pattern (applied after work starts).

    Returns:
        Tuple of (results, analyzed_count, files_with_findings, total_findings).
    """
    files_to_analyze = manifest[:max_files] if max_files else manifest
    total_files = len(files_to_analyze)
    logger.info(f"Analyzing {total_files:,} files...")

    # Create tasks - linter's internal semaphore handles vLLM concurrency
    # Timeout is applied per-pattern AFTER semaphore acquired (inside linter)
    tasks = []
    for record in files_to_analyze:
        tasks.append(analyze_single_file(record, base_dir, linter, pattern_timeout))

    # Process results as they complete
    results = []
    analyzed_count = 0
    files_with_findings = 0
    total_findings_count = 0
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1

        # Save to database immediately
        file_path = result["file_path"]
        maybe_file_id = file_ids.get(file_path)

        if maybe_file_id is not None:
            file_id = maybe_file_id
            duration = result.get("duration_seconds", 0)

            if result["success"]:
                file_analysis_id = insert_file_analysis(
                    conn, run_id, file_id, "success", duration=duration
                )
                analyzed_count += 1

                # Store pattern-level results (success, timeout, etc.)
                checked_patterns = result.get("checked_patterns", [])
                failed_patterns = result.get("failed_patterns", [])
                if checked_patterns or failed_patterns:
                    insert_pattern_runs(conn, file_analysis_id, checked_patterns, failed_patterns)

                findings = result.get("findings", [])
                if findings:
                    files_with_findings += 1
                    count = insert_findings(conn, file_analysis_id, findings)
                    total_findings_count += count
            else:
                insert_file_analysis(
                    conn, run_id, file_id, "error", result.get("error"), duration=duration
                )

        # Progress log every 50 files
        if completed % 50 == 0:
            pct = 100 * completed / total_files
            logger.info(f"Progress: {completed}/{total_files} files ({pct:.1f}%)")

    return results, analyzed_count, files_with_findings, total_findings_count


async def analyze_all_files(
    manifest: list[dict[str, Any]],
    base_dir: Path,
    linter: SciCodeLinter,
    max_files: int | None = None,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Run scicode-lint on all files in manifest (legacy, no incremental save).

    Args:
        manifest: List of manifest records.
        base_dir: Base directory containing collected files.
        linter: Shared SciCodeLinter instance.
        max_files: Optional limit on number of files to analyze.
        timeout: Timeout in seconds per file.

    Returns:
        List of analysis results.
    """
    files_to_analyze = manifest[:max_files] if max_files else manifest
    logger.info(f"Analyzing {len(files_to_analyze):,} files...")

    tasks = []
    for record in files_to_analyze:
        tasks.append(analyze_single_file(record, base_dir, linter, timeout))

    results = await asyncio.gather(*tasks)
    return list(results)


def main() -> None:
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(description="Run scicode-lint on collected files")
    parser.add_argument(
        "--source",
        type=str,
        choices=list(DATA_SOURCE_CONFIGS.keys()),
        help="Data source shortcut (sets manifest, base-dir, and patterns automatically)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to manifest.csv file (default: from --source or collected_code/manifest.csv)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory for files (default: from --source or collected_code/)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Limit number of files to analyze",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Maximum concurrent vLLM requests (default: from config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout per pattern in seconds, applied after work starts (default: 120)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        help="Comma-separated pattern IDs to check (e.g., ml-001,ml-007,ml-009,ml-010)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for reports (default: auto-detect from manifest)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis even if results exist",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest incomplete run (skip already-analyzed files)",
    )
    parser.add_argument(
        "--retry-timeouts",
        type=int,
        metavar="RUN_ID",
        help="Retry timed-out patterns from a previous run with larger timeout",
    )
    parser.add_argument(
        "--from-prefilter-run",
        type=int,
        metavar="RUN_ID",
        help="Use files from a specific prefilter run instead of manifest.csv",
    )
    args = parser.parse_args()

    # Apply --source config if provided
    if args.source:
        source_config = DATA_SOURCE_CONFIGS[args.source]
        if args.manifest is None:
            args.manifest = source_config.manifest
        if args.base_dir is None:
            args.base_dir = source_config.base_dir
        if args.patterns is None and source_config.default_patterns:
            args.patterns = ",".join(source_config.default_patterns)

    # Apply defaults if not set by --source or explicitly
    if args.manifest is None:
        args.manifest = COLLECTED_DIR / "manifest.csv"
    if args.base_dir is None:
        args.base_dir = COLLECTED_DIR

    # Handle retry-timeouts mode separately
    if args.retry_timeouts:
        conn = init_db()
        logger.info(f"Retrying timed-out patterns from run {args.retry_timeouts}")
        logger.info(f"Using timeout: {args.timeout}s")

        retried, succeeded = asyncio.run(
            retry_timed_out_patterns(
                conn,
                args.retry_timeouts,
                args.base_dir,
                pattern_timeout=args.timeout,
                max_concurrent=args.max_concurrent or DEFAULT_ANALYSIS_CONCURRENT,
            )
        )

        logger.info("=" * 50)
        logger.info(f"Retry complete: {succeeded}/{retried} patterns succeeded")
        conn.close()
        return

    # Handle --from-prefilter-run: get files from prefilter run instead of manifest
    from real_world_demo.models import PrefilterFileResult

    prefilter_run_id: int | None = None
    prefilter_files: list[PrefilterFileResult] | None = None

    if args.from_prefilter_run:
        conn = init_db()
        prefilter_run = get_prefilter_run(conn, args.from_prefilter_run)
        if not prefilter_run:
            logger.error(f"Prefilter run {args.from_prefilter_run} not found")
            conn.close()
            return

        # Get self-contained files from the prefilter run
        prefilter_files = get_prefilter_run_files(
            conn, args.from_prefilter_run, classification="self_contained"
        )
        if not prefilter_files:
            logger.error(
                f"No self-contained files found in prefilter run {args.from_prefilter_run}"
            )
            conn.close()
            return

        prefilter_run_id = args.from_prefilter_run
        logger.info(
            f"Using {len(prefilter_files)} self-contained files from "
            f"prefilter run {args.from_prefilter_run}"
        )
        data_source = prefilter_run.data_source
        conn.close()
    else:
        # Auto-detect data source from manifest path
        # e.g., "leakage_paper" from ".../leakage_paper/manifest.csv"
        manifest_parent = args.manifest.parent.name
        if manifest_parent in ("data", "collected_code"):
            # Fallback: use grandparent if parent is generic
            data_source = args.manifest.parent.parent.name
        else:
            data_source = manifest_parent

    logger.info(f"Data source: {data_source}")

    # Set output directory based on data source
    if args.output_dir is None:
        args.output_dir = REPORTS_DIR / data_source

    # Generate timestamped filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    report_filename = f"findings_{timestamp}.md"
    results_filename = f"raw_results_{timestamp}.json"

    # Check if results already exist (use latest non-timestamped file for --force check)
    report_file = args.output_dir / report_filename
    if report_file.exists() and not args.force:
        logger.info(f"Report already exists: {report_file}")
        logger.info("Use --force to re-analyze")
        return

    # Load manifest or use prefilter files
    if prefilter_files:
        # Convert prefilter files (Pydantic models) to manifest format (dicts)
        manifest = []
        for pf in prefilter_files:
            manifest.append(
                {
                    "file_path": pf.file_path,
                    "file_id": pf.file_id,
                    "repo_id": pf.repo_id,
                    "classification": pf.classification,
                    "confidence": pf.confidence,
                }
            )
        logger.info(f"Using {len(manifest):,} files from prefilter run {prefilter_run_id}")
    else:
        manifest = load_manifest(args.manifest)
        logger.info(f"Loaded {len(manifest):,} files from manifest")

    # Validate that all files exist on disk before starting analysis
    missing_files = [
        str(entry["file_path"]) for entry in manifest if not Path(str(entry["file_path"])).exists()
    ]
    if missing_files:
        logger.error(
            f"{len(missing_files)}/{len(manifest)} files are missing on disk. "
            f"Re-clone repos first (e.g., clone_repos.py --repos-file <repos.json>)."
        )
        for f in missing_files[:10]:
            logger.error(f"  Missing: {f}")
        if len(missing_files) > 10:
            logger.error(f"  ... and {len(missing_files) - 10} more")
        raise SystemExit(1)

    # Create shared linter instance with configured concurrency
    config = get_default_config()
    if args.max_concurrent:
        config.max_concurrent = args.max_concurrent
    else:
        # Use source-specific default concurrency
        config.max_concurrent = ANALYSIS_CONCURRENCY.get(data_source, DEFAULT_ANALYSIS_CONCURRENT)
    if args.patterns:
        config.enabled_patterns = set(args.patterns.split(","))
    linter = SciCodeLinter(config)
    pattern_count = len(linter.list_patterns())
    if config.enabled_patterns:
        # Filter count to only enabled patterns
        pattern_count = len([p for p in linter.list_patterns() if p.id in config.enabled_patterns])
    logger.info(
        f"Initialized linter (max_concurrent={config.max_concurrent}, patterns={pattern_count})"
    )

    # Initialize database
    conn = init_db()
    logger.info("Initialized SQLite database")

    # Check for resume mode
    run_id: int | None = None
    already_analyzed: set[int] = set()

    if args.resume:
        run_id = get_incomplete_run(conn)
        if run_id:
            already_analyzed = get_analyzed_file_ids(conn, run_id)
            logger.info(f"Resuming run {run_id} ({len(already_analyzed)} files already analyzed)")
        else:
            logger.info("No incomplete run to resume, starting fresh")

    # Prepare files for analysis - insert into database first
    files_to_analyze = manifest[: args.max_files] if args.max_files else manifest
    file_ids: dict[str, int] = {}

    for record in files_to_analyze:
        # If from prefilter run, file_id is already available
        record_file_id = record.get("file_id")
        if prefilter_files and record_file_id is not None:
            file_path = str(args.base_dir / record["file_path"])
            file_ids[file_path] = int(str(record_file_id))
        else:
            # Insert repo
            repo_id = get_or_create_repo(conn, record)

            # Insert file
            file_id = insert_file(conn, repo_id, record)
            file_path = str(args.base_dir / record["file_path"])
            file_ids[file_path] = file_id

    # Start new run if not resuming
    if run_id is None:
        run_id = start_analysis_run(
            conn,
            len(files_to_analyze),
            data_source=data_source,
            prefilter_run_id=prefilter_run_id,
        )
        logger.info(f"Started analysis run {run_id} (data_source={data_source})")

    # Filter out already-analyzed files when resuming
    if already_analyzed:
        files_to_analyze = [
            r
            for r in files_to_analyze
            if file_ids.get(str(args.base_dir / r["file_path"])) not in already_analyzed
        ]
        logger.info(f"Remaining files to analyze: {len(files_to_analyze)}")

    # Run analysis with incremental saving
    results, analyzed_count, files_with_findings, total_findings_count = asyncio.run(
        analyze_all_files_incremental(
            files_to_analyze,
            args.base_dir,
            conn,
            run_id,
            file_ids,
            linter=linter,
            max_files=None,  # Already filtered
            pattern_timeout=args.timeout,
        )
    )

    # Complete the run
    complete_analysis_run(conn, run_id, analyzed_count, files_with_findings, total_findings_count)

    # Save raw results to JSON with timestamp
    save_raw_results(results, args.output_dir / results_filename)

    # Aggregate and report
    stats = aggregate_findings(results)
    generate_report(stats, report_file)
    print_summary(stats)

    # Print database stats
    print_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
