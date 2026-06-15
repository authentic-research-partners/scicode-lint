"""Prefilter files using vLLM to identify self-contained ML pipeline code.

Uses repo_filter/classify.py to classify files as self-contained vs fragments.
Only self-contained files (complete ML workflows) are kept for analysis.

Usage:
    python prefilter_files.py [--max-concurrent 10]

    # With DB integration (default):
    python prefilter_files.py --max-concurrent 50
    # Creates prefilter run 52, saves results incrementally

    # Sample 30 papers (all their files) with reproducible seed:
    python prefilter_files.py --sample 30 --seed 42

    # Re-classify files from a previous prefilter run (with current model):
    python prefilter_files.py --reclassify-from-run 43

    # Use files from repos in a previous analysis run:
    python prefilter_files.py --from-analysis-run 49

    # Resume an interrupted run:
    python prefilter_files.py --resume

    # Exclude papers from previous improvement loop runs (prevents data contamination):
    python prefilter_files.py --reclassify-from-run 43 --exclude-from-prefilter-run 4 5
"""

import argparse
import asyncio
import json
import random
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.config import DATA_DIR
from real_world_demo.database import (
    complete_prefilter_run,
    get_analysis_run_data_source,
    get_classified_file_ids,
    get_file_id,
    get_files_from_analysis_run_repos,
    get_incomplete_prefilter_run,
    get_or_create_repo,
    get_paper_ids_from_prefilter_runs,
    get_prefilter_run,
    get_prefilter_run_files,
    init_db,
    insert_file,
    insert_prefilter_result,
    start_prefilter_run,
)
from scicode_lint.config import load_llm_config
from scicode_lint.llm.client import create_client
from scicode_lint.repo_filter.classify import FileClassification, classify_file


async def check_file_with_classifier(
    file_path: Path,
    semaphore: asyncio.Semaphore,
    llm_client: Any,
    file_id: int | None = None,
) -> dict[str, Any]:
    """Classify a file as self-contained or fragment.

    Args:
        file_path: Path to Python file.
        semaphore: Semaphore for concurrency control.
        llm_client: LLM client for classification.
        file_id: Optional file ID for DB tracking.

    Returns:
        Dict with file_path, is_pipeline, classification, and reasoning.
    """
    async with semaphore:
        try:
            # Read file content
            try:
                content = await asyncio.to_thread(
                    file_path.read_text, encoding="utf-8", errors="ignore"
                )
            except Exception as e:
                logger.warning(f"Cannot read {file_path}: {e}")
                return {
                    "file_path": str(file_path),
                    "file_id": file_id,
                    "is_pipeline": False,
                    "classification": "error",
                    "error": f"read_error: {e}",
                }

            # Check context length before sending to LLM
            from scicode_lint.llm.exceptions import ContextLengthError
            from scicode_lint.llm.tokens import check_context_length
            from scicode_lint.repo_filter.classify import (
                CLASSIFY_SYSTEM_PROMPT,
                CLASSIFY_USER_PROMPT,
            )

            user_prompt = CLASSIFY_USER_PROMPT.format(code=content)
            try:
                check_context_length(
                    system_prompt=CLASSIFY_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=llm_client.get_max_model_len(),
                    file_path=str(file_path),
                )
            except ContextLengthError as e:
                logger.debug(f"{file_path.name}: skipped, {e}")
                return {
                    "file_path": str(file_path),
                    "file_id": file_id,
                    "is_pipeline": False,
                    "classification": "fragment",
                    "confidence": 1.0,
                    "reasoning": f"File too large for analysis: {e}",
                }

            # Classify the file
            result: FileClassification = await classify_file(content, llm_client)

            # Self-contained files are kept for analysis
            is_pipeline = result.classification == "self_contained"

            logger.debug(
                f"{file_path.name}: {result.classification} (confidence={result.confidence:.2f})"
            )
            return {
                "file_path": str(file_path),
                "file_id": file_id,
                "is_pipeline": is_pipeline,
                "classification": result.classification,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "entry_point_indicators": result.entry_point_indicators,
                "missing_components": result.missing_components,
            }

        except TimeoutError:
            logger.warning(f"Timeout classifying {file_path}")
            return {
                "file_path": str(file_path),
                "file_id": file_id,
                "is_pipeline": False,
                "classification": "error",
                "error": "timeout",
            }
        except Exception as e:
            logger.warning(f"Error classifying {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "file_id": file_id,
                "is_pipeline": False,
                "classification": "error",
                "error": str(e),
            }


async def prefilter_all_files(
    qualifying_files: list[dict[str, Any]],
    max_concurrent: int = 10,
    file_ids: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prefilter all qualifying files using LLM classification.

    Args:
        qualifying_files: List of file records from filter_files.
        max_concurrent: Maximum concurrent LLM requests.
        file_ids: Optional mapping of file paths to DB file IDs.

    Returns:
        Tuple of (pipeline_files, filtered_out_files).
    """
    # Create LLM client using scicode_lint infrastructure
    llm_config = load_llm_config()
    llm_client = create_client(llm_config)

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for file_info in qualifying_files:
        file_path = Path(file_info["file_path"])
        file_id = file_ids.get(file_info["file_path"]) if file_ids else None
        tasks.append(check_file_with_classifier(file_path, semaphore, llm_client, file_id))

    logger.info(f"Classifying {len(tasks)} files (self-contained vs fragment)...")
    results = await asyncio.gather(*tasks)

    # Partition into self-contained and fragment files
    pipeline_files = []
    filtered_out = []

    for file_info, result in zip(qualifying_files, results, strict=True):
        # Add classification result to file_info
        file_info["self_contained_class"] = result.get("classification", "unknown")
        file_info["prefilter_response"] = result.get("reasoning", "")
        file_info["prefilter_confidence"] = result.get("confidence")

        if result.get("is_pipeline", False):
            pipeline_files.append(file_info)
        else:
            filtered_out.append(file_info)

    return pipeline_files, filtered_out


async def prefilter_all_files_incremental(
    qualifying_files: list[dict[str, Any]],
    conn: sqlite3.Connection,
    run_id: int,
    file_ids: dict[str, int],
    max_concurrent: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Prefilter files with incremental DB saving.

    Args:
        qualifying_files: List of file records from filter_files.
        conn: Database connection.
        run_id: Prefilter run ID.
        file_ids: Mapping of file paths to DB file IDs.
        max_concurrent: Maximum concurrent LLM requests.

    Returns:
        Tuple of (pipeline_files, filtered_out_files, counts_dict).
    """
    # Create LLM client using scicode_lint infrastructure
    llm_config = load_llm_config()
    llm_client = create_client(llm_config)

    semaphore = asyncio.Semaphore(max_concurrent)
    total_files = len(qualifying_files)

    logger.info(f"Classifying {total_files} files (self-contained vs fragment)...")

    # Process results as they complete for incremental saving
    pipeline_files: list[dict[str, Any]] = []
    filtered_out: list[dict[str, Any]] = []
    counts = {"self_contained": 0, "fragment": 0, "uncertain": 0, "error": 0}
    completed = 0

    # Create tasks with index to track file_info mapping
    async def classify_and_save(idx: int, file_info: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        file_path = Path(file_info["file_path"])
        file_id = file_ids.get(file_info["file_path"])
        result = await check_file_with_classifier(file_path, semaphore, llm_client, file_id)
        return idx, result

    tasks = [classify_and_save(i, f) for i, f in enumerate(qualifying_files)]

    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        completed += 1

        # Get corresponding file_info
        file_info = qualifying_files[idx]

        # Get file_id
        file_id = result.get("file_id")
        if file_id is None:
            file_id = file_ids.get(file_info["file_path"])

        # Save to DB immediately
        classification = result.get("classification", "error")
        if file_id:
            insert_prefilter_result(
                conn,
                run_id,
                file_id,
                classification,
                result.get("confidence"),
                result.get("reasoning"),
            )

        # Update counts
        if classification == "self_contained":
            counts["self_contained"] += 1
        elif classification == "fragment":
            counts["fragment"] += 1
        elif classification == "uncertain":
            counts["uncertain"] += 1
        else:
            counts["error"] += 1

        # Add classification result to file_info
        file_info["self_contained_class"] = classification
        file_info["prefilter_response"] = result.get("reasoning", "")
        file_info["prefilter_confidence"] = result.get("confidence")

        if result.get("is_pipeline", False):
            pipeline_files.append(file_info)
        else:
            filtered_out.append(file_info)

        # Progress log every 100 files
        if completed % 100 == 0:
            pct = 100 * completed / total_files
            logger.info(
                f"Progress: {completed}/{total_files} ({pct:.1f}%) - "
                f"self_contained={counts['self_contained']}, fragments={counts['fragment']}"
            )

    return pipeline_files, filtered_out, counts


def load_qualifying_files(input_file: Path) -> list[dict[str, Any]]:
    """Load qualifying files from JSON.

    Args:
        input_file: Path to qualifying_files.json.

    Returns:
        List of file records.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Qualifying files not found: {input_file}. Run filter_files.py first."
        )

    with open(input_file) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def save_results(
    pipeline_files: list[dict[str, Any]],
    filtered_out: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save prefilter results.

    Args:
        pipeline_files: Files identified as self-contained ML code.
        filtered_out: Files identified as fragments.
        output_dir: Output directory.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save pipeline files (these will be analyzed)
    pipeline_file = output_dir / "pipeline_files.json"
    with open(pipeline_file, "w") as f:
        json.dump(pipeline_files, f, indent=2)
    logger.info(f"Saved {len(pipeline_files)} self-contained files to {pipeline_file}")

    # Save filtered out files (for review/debugging)
    filtered_file = output_dir / "prefilter_excluded.json"
    with open(filtered_file, "w") as f:
        json.dump(filtered_out, f, indent=2)
    logger.info(f"Saved {len(filtered_out)} fragment files to {filtered_file}")


def print_summary(pipeline_files: list[dict[str, Any]], filtered_out: list[dict[str, Any]]) -> None:
    """Print prefilter summary.

    Args:
        pipeline_files: Files kept.
        filtered_out: Files excluded.
    """
    total = len(pipeline_files) + len(filtered_out)
    kept = len(pipeline_files)
    excluded = len(filtered_out)

    logger.info("=" * 50)
    logger.info("Prefilter Summary:")
    logger.info(f"  Total files checked: {total}")
    if total > 0:
        logger.info(f"  Self-contained (kept): {kept} ({100 * kept / total:.1f}%)")
        logger.info(f"  Fragments (excluded): {excluded} ({100 * excluded / total:.1f}%)")
    else:
        logger.info("  Self-contained (kept): 0")
        logger.info("  Fragments (excluded): 0")


def ensure_files_in_db(
    conn: sqlite3.Connection,
    qualifying_files: list[dict[str, Any]],
) -> dict[str, int]:
    """Ensure all files are in database and return file_id mapping.

    Derives repo_name from repo_path if not present in file records,
    so that files are correctly linked to repos in the database.

    Args:
        conn: Database connection.
        qualifying_files: List of file records from filter_files.

    Returns:
        Mapping of file_path to file_id.
    """
    file_ids: dict[str, int] = {}
    for file_info in qualifying_files:
        # First try to get existing file
        file_path = file_info["file_path"]
        existing_id = get_file_id(conn, file_path)
        if existing_id:
            file_ids[file_path] = existing_id
        else:
            # Derive repo_name from repo_path if missing
            # repo_path is like: .../cloned_repos/owner__repo
            if not file_info.get("repo_name") and file_info.get("repo_path"):
                file_info["repo_name"] = Path(file_info["repo_path"]).name

            # Insert repo and file
            repo_id = get_or_create_repo(conn, file_info)
            file_id = insert_file(conn, repo_id, file_info)
            file_ids[file_path] = file_id
    return file_ids


def _get_file_to_paper_map(
    conn: sqlite3.Connection,
    file_ids: dict[str, int],
) -> dict[int, int | None]:
    """Build file_id -> paper_id mapping from database.

    Traces: files → repos → papers.

    Args:
        conn: Database connection.
        file_ids: Mapping of file_path to file_id.

    Returns:
        Mapping of file_id to paper_id (None if repo has no paper).
    """
    if not file_ids:
        return {}
    file_id_list = list(file_ids.values())
    placeholders = ",".join("?" * len(file_id_list))
    cursor = conn.execute(
        f"""
        SELECT f.id, r.paper_id
        FROM files f
        JOIN repos r ON r.id = f.repo_id
        WHERE f.id IN ({placeholders})
        """,
        file_id_list,
    )
    return {row["id"]: row["paper_id"] for row in cursor.fetchall()}


def _sample_by_paper(
    conn: sqlite3.Connection,
    qualifying_files: list[dict[str, Any]],
    file_ids: dict[str, int],
    sample_n: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample N papers and return all files from those papers.

    Groups files by paper_id (via files → repos → papers), randomly selects
    N papers, returns all their files. Papers with multiple repos are kept together.

    Args:
        conn: Database connection.
        qualifying_files: Files to sample from.
        file_ids: Mapping of file_path to file_id.
        sample_n: Number of papers to sample.

    Returns:
        Tuple of (sampled_files, sampled_file_ids).
    """
    file_to_paper = _get_file_to_paper_map(conn, file_ids)

    # Group files by paper
    paper_to_files: dict[int | None, list[dict[str, Any]]] = {}
    for f in qualifying_files:
        fid = file_ids.get(f["file_path"])
        paper_id = file_to_paper.get(fid) if fid else None
        paper_to_files.setdefault(paper_id, []).append(f)

    # Separate papers with IDs from orphans (no paper_id)
    paper_keys = [k for k in paper_to_files if k is not None]
    orphan_files = paper_to_files.get(None, [])
    if orphan_files:
        logger.warning(f"{len(orphan_files)} files have no paper_id, excluded from sampling")

    if sample_n >= len(paper_keys):
        logger.info(f"Requested {sample_n} papers but only {len(paper_keys)} available, using all")
        return qualifying_files, file_ids

    random.shuffle(paper_keys)
    sampled_keys = paper_keys[:sample_n]

    sampled_files = []
    for key in sampled_keys:
        sampled_files.extend(paper_to_files[key])

    sampled_ids = {f["file_path"]: file_ids[f["file_path"]] for f in sampled_files}

    logger.info(
        f"Sampled {sample_n} papers → {len(sampled_files)} files "
        f"(from {len(paper_keys)} total papers)"
    )
    return sampled_files, sampled_ids


def _exclude_papers_from_runs(
    conn: sqlite3.Connection,
    qualifying_files: list[dict[str, Any]],
    file_ids: dict[str, int],
    exclude_run_ids: list[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Remove files whose papers appeared in the given prefilter runs.

    Excludes at paper level: if a paper had any file in the excluded runs,
    ALL files from that paper (across all its repos) are removed.

    Args:
        conn: Database connection.
        qualifying_files: Files to filter.
        file_ids: Mapping of file_path to file_id.
        exclude_run_ids: Prefilter run IDs whose papers should be excluded.

    Returns:
        Filtered (qualifying_files, file_ids).
    """
    excluded_paper_ids = get_paper_ids_from_prefilter_runs(conn, exclude_run_ids)
    if not excluded_paper_ids:
        logger.warning(f"No papers found in runs {exclude_run_ids}, nothing to exclude")
        return qualifying_files, file_ids

    file_to_paper = _get_file_to_paper_map(conn, file_ids)

    before_count = len(qualifying_files)
    filtered_files = []
    filtered_ids: dict[str, int] = {}
    for f in qualifying_files:
        fid = file_ids.get(f["file_path"])
        paper_id = file_to_paper.get(fid) if fid else None
        if paper_id is not None and paper_id in excluded_paper_ids:
            continue
        filtered_files.append(f)
        if fid:
            filtered_ids[f["file_path"]] = fid

    excluded_count = before_count - len(filtered_files)
    logger.info(
        f"Excluded {excluded_count} files from {len(excluded_paper_ids)} papers "
        f"(from runs {exclude_run_ids}), {len(filtered_files)} files remaining"
    )
    return filtered_files, filtered_ids


def main() -> None:
    """Main entry point for prefiltering."""
    parser = argparse.ArgumentParser(
        description="Prefilter files: classify as self-contained vs fragment"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DATA_DIR / "qualifying_files.json",
        help="Input file with qualifying files list",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent LLM requests (default: 50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-filtering even if output exists",
    )
    # New DB integration flags
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        default=True,
        help="Save results to database (default: True)",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable database saving (legacy mode)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip JSON output files (DB only)",
    )
    parser.add_argument(
        "--reclassify-from-run",
        type=int,
        metavar="RUN_ID",
        help="Re-classify files from a previous prefilter run with the current model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest incomplete prefilter run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling (use with --sample)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Randomly sample N papers, including all their files (use --seed for reproducibility)",
    )
    parser.add_argument(
        "--from-analysis-run",
        type=int,
        metavar="RUN_ID",
        help="Use files from repos that were used in a previous analysis run",
    )
    parser.add_argument(
        "--exclude-from-prefilter-run",
        type=int,
        nargs="+",
        metavar="RUN_ID",
        help="Exclude papers that appeared in these prefilter runs (prevents data contamination)",
    )
    args = parser.parse_args()

    # Handle --no-db flag
    if args.no_db:
        args.save_to_db = False

    # Set random seed if provided (for --sample reproducibility)
    if args.seed is not None and args.sample:
        random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    elif args.seed is not None and not args.sample:
        logger.warning("--seed has no effect without --sample")
    elif args.sample and args.seed is None:
        logger.warning("--sample without --seed: results will not be reproducible")

    # Initialize database if needed
    conn: sqlite3.Connection | None = None
    run_id: int | None = None
    file_ids: dict[str, int] = {}
    already_classified: set[int] = set()

    if args.save_to_db:
        conn = init_db()
        logger.info("Initialized SQLite database")

    # Handle --reclassify-from-run: get files from previous run, re-classify
    if args.reclassify_from_run:
        if not conn:
            conn = init_db()

        parent_run = get_prefilter_run(conn, args.reclassify_from_run)
        if not parent_run:
            logger.error(f"Prefilter run {args.reclassify_from_run} not found")
            return

        logger.info(
            f"Reclassifying files from run {args.reclassify_from_run} "
            f"({parent_run.self_contained} self-contained, "
            f"{parent_run.fragments} fragments)"
        )

        # Get all files from that run (regardless of classification - we reclassify)
        parent_files = get_prefilter_run_files(conn, args.reclassify_from_run)
        if not parent_files:
            logger.error(f"No files found in run {args.reclassify_from_run}")
            return

        # Convert to qualifying_files format
        qualifying_files = []
        for pf in parent_files:
            qualifying_files.append({"file_path": pf.file_path})
            file_ids[pf.file_path] = pf.file_id

        logger.info(f"Loaded {len(qualifying_files)} files from run {args.reclassify_from_run}")

        # Exclude papers from specified runs
        if args.exclude_from_prefilter_run:
            qualifying_files, file_ids = _exclude_papers_from_runs(
                conn, qualifying_files, file_ids, args.exclude_from_prefilter_run
            )

        # Sample N papers (including all their files)
        if args.sample:
            qualifying_files, file_ids = _sample_by_paper(
                conn, qualifying_files, file_ids, args.sample
            )

        # Start new run with parent reference
        llm_config = load_llm_config()
        run_id = start_prefilter_run(
            conn,
            total_files=len(qualifying_files),
            seed=args.seed,
            config={
                "max_concurrent": args.max_concurrent,
                "reclassify_from_run": args.reclassify_from_run,
                "sample": args.sample,
                "exclude_from_prefilter_run": args.exclude_from_prefilter_run,
            },
            model_name=llm_config.model_served_name,
            parent_run_id=args.reclassify_from_run,
        )
        logger.info(
            f"Created prefilter run {run_id} (reclassifying from run {args.reclassify_from_run})"
        )

    # Handle --from-analysis-run: get files from repos used in an analysis run
    elif args.from_analysis_run:
        if not conn:
            conn = init_db()

        data_source = get_analysis_run_data_source(conn, args.from_analysis_run)
        if not data_source:
            logger.error(f"Analysis run {args.from_analysis_run} not found")
            return

        logger.info(f"Getting files from repos in analysis run {args.from_analysis_run}")

        # Get all files from repos that were used in that analysis run
        analysis_files = get_files_from_analysis_run_repos(conn, args.from_analysis_run)
        if not analysis_files:
            logger.error(f"No files found for repos in analysis run {args.from_analysis_run}")
            return

        # Convert to qualifying_files format
        qualifying_files = []
        for af in analysis_files:
            qualifying_files.append({"file_path": af.file_path})
            file_ids[af.file_path] = af.file_id

        repo_count = len(set(af.repo_id for af in analysis_files))
        logger.info(f"Found {len(qualifying_files)} files from {repo_count} repos")

        # Exclude papers from specified runs
        if args.exclude_from_prefilter_run:
            qualifying_files, file_ids = _exclude_papers_from_runs(
                conn, qualifying_files, file_ids, args.exclude_from_prefilter_run
            )

        # Sample N papers (including all their files)
        if args.sample:
            qualifying_files, file_ids = _sample_by_paper(
                conn, qualifying_files, file_ids, args.sample
            )

        # Start new run
        llm_config = load_llm_config()
        run_id = start_prefilter_run(
            conn,
            total_files=len(qualifying_files),
            seed=args.seed,
            data_source=data_source,
            config={
                "max_concurrent": args.max_concurrent,
                "from_analysis_run": args.from_analysis_run,
                "sample": args.sample,
                "exclude_from_prefilter_run": args.exclude_from_prefilter_run,
            },
            model_name=llm_config.model_served_name,
        )
        logger.info(
            f"Created prefilter run {run_id} (files from analysis run {args.from_analysis_run})"
        )

    # Handle --resume: continue incomplete run
    elif args.resume:
        if not conn:
            conn = init_db()

        run_id = get_incomplete_prefilter_run(conn)
        if run_id:
            already_classified = get_classified_file_ids(conn, run_id)
            logger.info(
                f"Resuming run {run_id} ({len(already_classified)} files already classified)"
            )
            # Load original input file to get remaining files
            qualifying_files = load_qualifying_files(args.input_file)
            # Ensure all files in DB and get IDs
            file_ids = ensure_files_in_db(conn, qualifying_files)
            # Filter out already classified
            qualifying_files = [
                f
                for f in qualifying_files
                if file_ids.get(f["file_path"]) not in already_classified
            ]
            logger.info(f"Remaining files to classify: {len(qualifying_files)}")
        else:
            logger.info("No incomplete run to resume, starting fresh")
            args.resume = False

    # Normal flow: load from input file
    if not args.reclassify_from_run and not args.resume:
        # Check if output already exists
        pipeline_file = args.output_dir / "pipeline_files.json"
        if pipeline_file.exists() and not args.force and not args.no_json:
            logger.info(f"Output already exists: {pipeline_file}")
            logger.info("Use --force to re-filter")
            with open(pipeline_file) as f:
                existing = json.load(f)
            logger.info(f"Existing self-contained files: {len(existing)}")
            return

        # Load qualifying files
        qualifying_files = load_qualifying_files(args.input_file)
        logger.info(f"Loaded {len(qualifying_files)} qualifying files")

        if not qualifying_files:
            logger.warning("No files to prefilter!")
            return

        # Ensure files in DB and start run
        if args.save_to_db and conn:
            file_ids = ensure_files_in_db(conn, qualifying_files)

            # Exclude papers from specified runs
            if args.exclude_from_prefilter_run:
                qualifying_files, file_ids = _exclude_papers_from_runs(
                    conn, qualifying_files, file_ids, args.exclude_from_prefilter_run
                )

            # Sample N papers (including all their files)
            if args.sample:
                qualifying_files, file_ids = _sample_by_paper(
                    conn, qualifying_files, file_ids, args.sample
                )

            llm_config = load_llm_config()
            run_id = start_prefilter_run(
                conn,
                total_files=len(qualifying_files),
                seed=args.seed,
                config={
                    "max_concurrent": args.max_concurrent,
                    "sample": args.sample,
                    "exclude_from_prefilter_run": args.exclude_from_prefilter_run,
                },
                model_name=llm_config.model_served_name,
            )
            logger.info(f"Created prefilter run {run_id}")

    # Run prefilter
    if args.save_to_db and conn and run_id:
        # Incremental DB saving
        pipeline_files, filtered_out, counts = asyncio.run(
            prefilter_all_files_incremental(
                qualifying_files,
                conn,
                run_id,
                file_ids,
                max_concurrent=args.max_concurrent,
            )
        )

        # Complete the run
        complete_prefilter_run(
            conn,
            run_id,
            counts["self_contained"],
            counts["fragment"],
            counts["uncertain"],
            counts["error"],
        )
        logger.info(f"Completed prefilter run {run_id}")
    else:
        # Legacy mode without DB
        pipeline_files, filtered_out = asyncio.run(
            prefilter_all_files(
                qualifying_files,
                max_concurrent=args.max_concurrent,
                file_ids=file_ids if file_ids else None,
            )
        )

    # Save JSON results (unless --no-json)
    if not args.no_json:
        save_results(pipeline_files, filtered_out, args.output_dir)

    # Print summary
    print_summary(pipeline_files, filtered_out)

    if conn:
        conn.close()


if __name__ == "__main__":
    main()
