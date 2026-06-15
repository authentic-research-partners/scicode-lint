"""Filter repositories for self-contained ML files using repo_filter.

Two-stage filtering:
1. ML import presence check (deterministic, instant)
2. LLM classification (self-contained vs fragment)

Replaces the old filter_files.py with proper self-contained detection.
Results are saved to both database and JSON file.

Usage:
    python -m real_world_demo.sources.papers_with_code.filter_repos
    python -m real_world_demo.sources.papers_with_code.filter_repos --force
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.config import CLONED_DIR, DATA_DIR
from real_world_demo.database import (
    complete_repo_scan,
    get_or_create_repo,
    init_db,
    insert_file,
    start_repo_scan,
    update_file_classification,
)
from scicode_lint.config import get_filter_concurrency, load_llm_config
from scicode_lint.llm.client import create_client
from scicode_lint.repo_filter import filter_scan_results, scan_repo_for_ml_files
from scicode_lint.repo_filter.scan import RepoScanSummary


def save_scan_to_db(
    repo_path: Path,
    summary: RepoScanSummary,
    duration_seconds: float,
    model_name: str,
) -> int | None:
    """Save scan results to database.

    Args:
        repo_path: Path to the repository.
        summary: Scan summary with ALL results.
        duration_seconds: Time taken for the scan.
        model_name: LLM model used for classification.

    Returns:
        Scan ID if successful, None otherwise.
    """
    conn = init_db()

    try:
        # Get or create repo record
        repo_name = repo_path.name
        repo_id = get_or_create_repo(
            conn,
            {
                "repo_name": repo_name,
                "repo_url": f"local://{repo_path}",
                "data_source": "papersWithCode",
            },
        )

        if not repo_id:
            logger.warning(f"Could not get/create repo: {repo_name}")
            return None

        # Start scan record
        scan_id = start_repo_scan(
            conn,
            repo_id=repo_id,
            total_files=summary.total_files,
            model_name=model_name,
        )

        # Store all file classifications
        for result in summary.results:
            try:
                relative_path = str(result.filepath.relative_to(repo_path))
            except ValueError:
                relative_path = str(result.filepath)

            file_id = insert_file(
                conn,
                repo_id=repo_id,
                file_data={
                    "file_path": relative_path,
                    "original_path": str(result.filepath),
                    "is_notebook": result.filepath.suffix == ".ipynb",
                },
            )

            has_ml = result.skip_reason != "no_ml_imports_found"
            update_file_classification(
                conn,
                file_id=file_id,
                scan_id=scan_id,
                has_ml_imports=has_ml,
                classification=result.classification if has_ml else None,
                confidence=result.details.confidence if result.details else None,
                reasoning=result.details.reasoning if result.details else None,
            )

        # Complete scan
        skipped = summary.failed_ml_import_filter + summary.skipped_too_large
        complete_repo_scan(
            conn,
            scan_id=scan_id,
            passed_ml_import_filter=summary.passed_ml_import_filter,
            self_contained=summary.self_contained,
            fragments=summary.fragments,
            uncertain=summary.uncertain,
            skipped=skipped,
            duration_seconds=duration_seconds,
        )

        return scan_id

    finally:
        conn.close()


async def filter_all_repos(
    cloned_dir: Path,
    max_concurrent: int = 50,
    include_uncertain: bool = False,
) -> dict[str, Any]:
    """Filter all repos in cloned_dir for self-contained ML files.

    Results are saved to database for each repo.

    Args:
        cloned_dir: Directory containing cloned repos.
        max_concurrent: Max concurrent LLM requests.
        include_uncertain: Include files with uncertain classification in JSON output.

    Returns:
        Dict with summary and list of self-contained files.
    """
    # One-shot directory listing before the concurrent section.
    repo_dirs = [d for d in cloned_dir.iterdir() if d.is_dir() and (d / ".git").exists()]  # noqa: ASYNC240
    logger.info(f"Found {len(repo_dirs)} cloned repos")

    llm_config = load_llm_config()
    llm_client = create_client(llm_config)
    model_name = llm_config.model_served_name

    # Aggregate stats
    total_files = 0
    total_passed_ml_filter = 0
    total_failed_ml_filter = 0
    total_skipped_too_large = 0
    total_self_contained = 0
    total_fragments = 0
    total_uncertain = 0
    all_self_contained_files: list[dict[str, Any]] = []

    for i, repo_path in enumerate(repo_dirs):
        logger.info(f"[{i + 1}/{len(repo_dirs)}] Filtering: {repo_path.name}")
        start_time = time.time()

        try:
            # Scan returns ALL results
            summary = await scan_repo_for_ml_files(
                repo_path=repo_path,
                llm_client=llm_client,
                max_concurrent=max_concurrent,
            )

            duration = time.time() - start_time

            # Save ALL results to database
            save_scan_to_db(
                repo_path=repo_path,
                summary=summary,
                duration_seconds=duration,
                model_name=model_name,
            )

            # Aggregate stats
            total_files += summary.total_files
            total_passed_ml_filter += summary.passed_ml_import_filter
            total_failed_ml_filter += summary.failed_ml_import_filter
            total_skipped_too_large += summary.skipped_too_large
            total_self_contained += summary.self_contained
            total_fragments += summary.fragments
            total_uncertain += summary.uncertain

            # Filter for JSON output
            filtered = filter_scan_results(summary, include_uncertain=include_uncertain)
            for result in filtered:
                all_self_contained_files.append(
                    {
                        "file_path": str(result.filepath),
                        "relative_path": str(result.filepath.relative_to(repo_path)),
                        "repo_path": str(repo_path),
                        "repo_name": repo_path.name,
                        "classification": result.classification,
                        "confidence": result.details.confidence if result.details else None,
                        "reasoning": result.details.reasoning if result.details else None,
                    }
                )

            logger.info(
                f"  {summary.total_files} files, "
                f"{summary.passed_ml_import_filter} ML imports, "
                f"{summary.self_contained} self-contained "
                f"({duration:.1f}s)"
            )

        except Exception as e:
            logger.error(f"Error filtering {repo_path.name}: {e}")
            continue

    return {
        "summary": {
            "total_repos": len(repo_dirs),
            "total_files": total_files,
            "passed_ml_import_filter": total_passed_ml_filter,
            "failed_ml_import_filter": total_failed_ml_filter,
            "skipped_too_large": total_skipped_too_large,
            "self_contained": total_self_contained,
            "fragments": total_fragments,
            "uncertain": total_uncertain,
        },
        "files": all_self_contained_files,
    }


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Filter repos for self-contained ML files")
    parser.add_argument(
        "--cloned-dir",
        type=Path,
        default=CLONED_DIR,
        help="Directory containing cloned repos",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DATA_DIR / "self_contained_files.json",
        help="Output file for self-contained files list",
    )
    parser.add_argument(
        "--filter-concurrency",
        type=int,
        default=None,
        help="Max concurrent LLM requests (default: from config)",
    )
    parser.add_argument(
        "--include-uncertain",
        action="store_true",
        help="Include files with uncertain classification",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-filtering even if output file exists",
    )
    args = parser.parse_args()

    if args.output_file.exists() and not args.force:
        logger.info(f"Output already exists: {args.output_file}")
        logger.info("Use --force to re-filter")
        with open(args.output_file) as f:
            existing = json.load(f)
        logger.info(f"Existing self-contained files: {len(existing.get('files', []))}")
        return

    if not args.cloned_dir.exists():
        logger.error(f"Cloned repos directory not found: {args.cloned_dir}")
        return

    max_concurrent = args.filter_concurrency or get_filter_concurrency()

    logger.info("Starting repo filtering...")
    logger.info(f"  Cloned dir: {args.cloned_dir}")
    logger.info(f"  Concurrency: {max_concurrent}")
    logger.info("  Results will be saved to database and JSON")

    results = asyncio.run(
        filter_all_repos(
            cloned_dir=args.cloned_dir,
            max_concurrent=max_concurrent,
            include_uncertain=args.include_uncertain,
        )
    )

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    logger.info("=" * 60)
    logger.info("Filter Summary:")
    logger.info(f"  Total repos: {s['total_repos']}")
    logger.info(f"  Total files: {s['total_files']}")
    logger.info(f"  Passed ML import filter: {s['passed_ml_import_filter']}")
    logger.info(f"  Failed ML import filter: {s['failed_ml_import_filter']}")
    logger.info(f"  Skipped (too large): {s['skipped_too_large']}")
    logger.info("  After LLM classification:")
    logger.info(f"    Self-contained: {s['self_contained']}")
    logger.info(f"    Fragments: {s['fragments']}")
    logger.info(f"    Uncertain: {s['uncertain']}")
    logger.info(f"  Saved to DB and: {args.output_file}")


if __name__ == "__main__":
    main()
