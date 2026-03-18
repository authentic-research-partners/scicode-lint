"""Prefilter run operations."""

import json
import sqlite3
from datetime import datetime
from typing import Any

from .db_core import get_current_git_commit
from .models import PrefilterFileResult, PrefilterRun, RunStatus

__all__ = [
    "start_prefilter_run",
    "complete_prefilter_run",
    "insert_prefilter_result",
    "get_paper_ids_from_prefilter_runs",
    "get_paper_urls_from_prefilter_runs",
    "get_prefilter_run_files",
    "get_incomplete_prefilter_run",
    "get_classified_file_ids",
    "get_prefilter_run",
    "list_prefilter_runs",
    "get_files_from_analysis_run_repos",
    "get_prefilter_summary",
]


def start_prefilter_run(
    conn: sqlite3.Connection,
    total_files: int,
    data_source: str = "papers_with_code",
    seed: int | None = None,
    config: dict[str, Any] | None = None,
    model_name: str = "",
    parent_run_id: int | None = None,
) -> int:
    """Start a new prefilter run.

    Args:
        conn: Database connection.
        total_files: Total files to classify.
        data_source: Data source identifier.
        seed: Random seed for reproducibility.
        config: Run configuration dict.
        model_name: LLM model being used.
        parent_run_id: ID of parent run if reusing files.

    Returns:
        Run ID.
    """
    cursor = conn.execute(
        """
        INSERT INTO prefilter_runs
        (total_files, data_source, seed, config, model_name, git_commit, parent_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            total_files,
            data_source,
            seed,
            json.dumps(config or {}),
            model_name,
            get_current_git_commit(),
            parent_run_id,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def complete_prefilter_run(
    conn: sqlite3.Connection,
    run_id: int,
    self_contained: int,
    fragments: int,
    uncertain: int,
    errors: int,
    status: str = "completed",
) -> None:
    """Complete a prefilter run.

    Args:
        conn: Database connection.
        run_id: Run ID.
        self_contained: Files classified as self-contained.
        fragments: Files classified as fragments.
        uncertain: Files with uncertain classification.
        errors: Files that errored during classification.
        status: Final status.
    """
    conn.execute(
        """
        UPDATE prefilter_runs
        SET completed_at = ?, status = ?, self_contained = ?,
            fragments = ?, uncertain = ?, errors = ?
        WHERE id = ?
        """,
        (
            datetime.now().isoformat(),
            status,
            self_contained,
            fragments,
            uncertain,
            errors,
            run_id,
        ),
    )
    conn.commit()


def insert_prefilter_result(
    conn: sqlite3.Connection,
    run_id: int,
    file_id: int,
    classification: str,
    confidence: float | None = None,
    reasoning: str | None = None,
) -> int:
    """Insert a single prefilter classification result.

    Args:
        conn: Database connection.
        run_id: Prefilter run ID.
        file_id: File ID.
        classification: Classification result (self_contained, fragment, uncertain, error).
        confidence: Classification confidence.
        reasoning: LLM reasoning for classification.

    Returns:
        Result ID.
    """
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO prefilter_file_results
        (prefilter_run_id, file_id, classification, confidence, reasoning)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, file_id, classification, confidence, reasoning),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_paper_ids_from_prefilter_runs(
    conn: sqlite3.Connection,
    run_ids: list[int],
) -> set[int]:
    """Get all paper IDs that had files in the given prefilter runs.

    Traces: prefilter_file_results -> files -> repos -> papers.

    Args:
        conn: Database connection.
        run_ids: List of prefilter run IDs to get papers from.

    Returns:
        Set of paper IDs.
    """
    if not run_ids:
        return set()
    placeholders = ",".join("?" * len(run_ids))
    cursor = conn.execute(
        f"""
        SELECT DISTINCT r.paper_id
        FROM prefilter_file_results pfr
        JOIN files f ON f.id = pfr.file_id
        JOIN repos r ON r.id = f.repo_id
        WHERE pfr.prefilter_run_id IN ({placeholders})
          AND r.paper_id IS NOT NULL
        """,
        run_ids,
    )
    return {row["paper_id"] for row in cursor.fetchall()}


def get_paper_urls_from_prefilter_runs(
    conn: sqlite3.Connection,
    run_ids: list[int],
) -> set[str]:
    """Get paper URLs that had files in the given prefilter runs.

    Traces: prefilter_file_results -> files -> repos -> papers.

    Args:
        conn: Database connection.
        run_ids: List of prefilter run IDs to get papers from.

    Returns:
        Set of paper URL strings.
    """
    paper_ids = get_paper_ids_from_prefilter_runs(conn, run_ids)
    if not paper_ids:
        return set()
    placeholders = ",".join("?" * len(paper_ids))
    cursor = conn.execute(
        f"SELECT paper_url FROM papers WHERE id IN ({placeholders})",
        list(paper_ids),
    )
    return {row["paper_url"] for row in cursor.fetchall()}


def get_prefilter_run_files(
    conn: sqlite3.Connection,
    run_id: int,
    classification: str | None = None,
) -> list[PrefilterFileResult]:
    """Get files from a specific prefilter run.

    Args:
        conn: Database connection.
        run_id: Prefilter run ID.
        classification: Optional filter by classification (e.g., 'self_contained').

    Returns:
        List of PrefilterFileResult models.
    """
    if classification:
        cursor = conn.execute(
            """
            SELECT f.id, f.file_path, f.repo_id, pfr.classification,
                   pfr.confidence, pfr.reasoning
            FROM prefilter_file_results pfr
            JOIN files f ON f.id = pfr.file_id
            WHERE pfr.prefilter_run_id = ? AND pfr.classification = ?
            ORDER BY pfr.confidence DESC
            """,
            (run_id, classification),
        )
    else:
        cursor = conn.execute(
            """
            SELECT f.id, f.file_path, f.repo_id, pfr.classification,
                   pfr.confidence, pfr.reasoning
            FROM prefilter_file_results pfr
            JOIN files f ON f.id = pfr.file_id
            WHERE pfr.prefilter_run_id = ?
            ORDER BY pfr.classification, pfr.confidence DESC
            """,
            (run_id,),
        )
    return [
        PrefilterFileResult(
            file_id=row["id"],
            file_path=row["file_path"],
            repo_id=row["repo_id"],
            classification=row["classification"],
            confidence=row["confidence"],
            reasoning=row["reasoning"],
        )
        for row in cursor.fetchall()
    ]


def get_incomplete_prefilter_run(
    conn: sqlite3.Connection,
    data_source: str | None = None,
) -> int | None:
    """Get the latest incomplete (running) prefilter run.

    Args:
        conn: Database connection.
        data_source: Optional filter by data source.

    Returns:
        Run ID if there's an incomplete run, None otherwise.
    """
    if data_source:
        cursor = conn.execute(
            """
            SELECT id FROM prefilter_runs
            WHERE status = 'running' AND data_source = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (data_source,),
        )
    else:
        cursor = conn.execute(
            """
            SELECT id FROM prefilter_runs
            WHERE status = 'running'
            ORDER BY started_at DESC
            LIMIT 1
            """
        )
    row = cursor.fetchone()
    return row["id"] if row else None


def get_classified_file_ids(conn: sqlite3.Connection, run_id: int) -> set[int]:
    """Get set of file IDs already classified in a prefilter run.

    Args:
        conn: Database connection.
        run_id: Prefilter run ID.

    Returns:
        Set of file IDs that have been classified.
    """
    cursor = conn.execute(
        "SELECT file_id FROM prefilter_file_results WHERE prefilter_run_id = ?",
        (run_id,),
    )
    return {row["file_id"] for row in cursor.fetchall()}


def get_prefilter_run(conn: sqlite3.Connection, run_id: int) -> PrefilterRun | None:
    """Get a prefilter run by ID.

    Args:
        conn: Database connection.
        run_id: Prefilter run ID.

    Returns:
        PrefilterRun or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM prefilter_runs WHERE id = ?",
        (run_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    return PrefilterRun(
        run_id=row["id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=(datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None),
        status=RunStatus(row["status"]) if row["status"] else RunStatus.RUNNING,
        data_source=row["data_source"] or "papers_with_code",
        total_files=row["total_files"] or 0,
        self_contained=row["self_contained"] or 0,
        fragments=row["fragments"] or 0,
        uncertain=row["uncertain"] or 0,
        errors=row["errors"] or 0,
        seed=row["seed"],
        config=json.loads(row["config"]) if row["config"] else {},
        model_name=row["model_name"] or "",
        git_commit=row["git_commit"] or "",
        parent_run_id=row["parent_run_id"],
    )


def list_prefilter_runs(
    conn: sqlite3.Connection,
    limit: int = 10,
    data_source: str | None = None,
) -> list[PrefilterRun]:
    """List recent prefilter runs.

    Args:
        conn: Database connection.
        limit: Maximum runs to return.
        data_source: Optional filter by data source.

    Returns:
        List of PrefilterRun models.
    """
    if data_source:
        cursor = conn.execute(
            """
            SELECT * FROM prefilter_runs
            WHERE data_source = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (data_source, limit),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM prefilter_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    runs = []
    for row in cursor.fetchall():
        runs.append(
            PrefilterRun(
                run_id=row["id"],
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
                status=RunStatus(row["status"]) if row["status"] else RunStatus.RUNNING,
                data_source=row["data_source"] or "papers_with_code",
                total_files=row["total_files"] or 0,
                self_contained=row["self_contained"] or 0,
                fragments=row["fragments"] or 0,
                uncertain=row["uncertain"] or 0,
                errors=row["errors"] or 0,
                seed=row["seed"],
                config=json.loads(row["config"]) if row["config"] else {},
                model_name=row["model_name"] or "",
                git_commit=row["git_commit"] or "",
                parent_run_id=row["parent_run_id"],
            )
        )
    return runs


def get_files_from_analysis_run_repos(
    conn: sqlite3.Connection,
    analysis_run_id: int,
) -> list[PrefilterFileResult]:
    """Get all files from repos that were used in an analysis run.

    This allows reusing the same paper/repo selection while re-running
    the prefilter classification with vLLM.

    Args:
        conn: Database connection.
        analysis_run_id: ID of the analysis run to get repos from.

    Returns:
        List of PrefilterFileResult models (without classification - to be filled).
    """
    # Get all files from repos that had files in the analysis run
    cursor = conn.execute(
        """
        SELECT DISTINCT f.id, f.file_path, f.repo_id
        FROM files f
        WHERE f.repo_id IN (
            SELECT DISTINCT f2.repo_id
            FROM file_analyses fa
            JOIN files f2 ON f2.id = fa.file_id
            WHERE fa.run_id = ?
        )
        ORDER BY f.repo_id, f.file_path
        """,
        (analysis_run_id,),
    )

    return [
        PrefilterFileResult(
            file_id=row["id"],
            file_path=row["file_path"],
            repo_id=row["repo_id"],
            classification="pending",  # Will be filled by prefilter
            confidence=None,
            reasoning=None,
        )
        for row in cursor.fetchall()
    ]


def get_prefilter_summary(
    conn: sqlite3.Connection,
    analysis_run_id: int,
) -> dict[str, Any] | None:
    """Get prefilter summary for an analysis run.

    Returns stats about original vs filtered papers/repos/files.

    Args:
        conn: Database connection.
        analysis_run_id: Analysis run ID.

    Returns:
        Dict with prefilter stats, or None if no prefilter was used.
    """
    # Check if analysis run has a prefilter_run_id
    cursor = conn.execute(
        "SELECT prefilter_run_id FROM analysis_runs WHERE id = ?",
        (analysis_run_id,),
    )
    row = cursor.fetchone()
    if not row or not row["prefilter_run_id"]:
        return None

    prefilter_run_id = row["prefilter_run_id"]

    # Get prefilter run stats
    prefilter_run = get_prefilter_run(conn, prefilter_run_id)
    if not prefilter_run:
        return None

    # Get original counts (all files in prefilter run)
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT f.repo_id) as original_repos,
            COUNT(DISTINCT r.paper_id) as original_papers
        FROM prefilter_file_results pfr
        JOIN files f ON f.id = pfr.file_id
        JOIN repos r ON r.id = f.repo_id
        WHERE pfr.prefilter_run_id = ?
        """,
        (prefilter_run_id,),
    )
    original = cursor.fetchone()

    # Get filtered counts (self-contained only)
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT f.repo_id) as filtered_repos,
            COUNT(DISTINCT r.paper_id) as filtered_papers
        FROM prefilter_file_results pfr
        JOIN files f ON f.id = pfr.file_id
        JOIN repos r ON r.id = f.repo_id
        WHERE pfr.prefilter_run_id = ? AND pfr.classification = 'self_contained'
        """,
        (prefilter_run_id,),
    )
    filtered = cursor.fetchone()

    # Get dropped repos (all files were fragments)
    cursor = conn.execute(
        """
        SELECT r.repo_name
        FROM repos r
        WHERE r.id IN (
            SELECT DISTINCT f.repo_id
            FROM prefilter_file_results pfr
            JOIN files f ON f.id = pfr.file_id
            WHERE pfr.prefilter_run_id = ?
        )
        AND r.id NOT IN (
            SELECT DISTINCT f.repo_id
            FROM prefilter_file_results pfr
            JOIN files f ON f.id = pfr.file_id
            WHERE pfr.prefilter_run_id = ? AND pfr.classification = 'self_contained'
        )
        ORDER BY r.repo_name
        """,
        (prefilter_run_id, prefilter_run_id),
    )
    dropped_repos = [row["repo_name"] for row in cursor.fetchall()]

    return {
        "prefilter_run_id": prefilter_run_id,
        "total_files": prefilter_run.total_files,
        "self_contained": prefilter_run.self_contained,
        "fragments": prefilter_run.fragments,
        "uncertain": prefilter_run.uncertain,
        "errors": prefilter_run.errors,
        "original_repos": original["original_repos"] or 0,
        "original_papers": original["original_papers"] or 0,
        "filtered_repos": filtered["filtered_repos"] or 0,
        "filtered_papers": filtered["filtered_papers"] or 0,
        "dropped_repos": dropped_repos,
        "model_name": prefilter_run.model_name,
    }
