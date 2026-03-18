"""Repository scan operations."""

import sqlite3
from datetime import datetime
from typing import Any

from .db_core import get_current_git_commit
from .models import RepoScan, RunStatus, ScanStats

__all__ = [
    "start_repo_scan",
    "complete_repo_scan",
    "update_file_classification",
    "get_self_contained_files_for_repo",
    "get_scan_stats",
    "get_latest_scan_for_repo",
]


def start_repo_scan(
    conn: sqlite3.Connection,
    repo_id: int,
    total_files: int,
    model_name: str = "",
) -> int:
    """Start a new repository scan.

    Args:
        conn: Database connection.
        repo_id: Repository being scanned.
        total_files: Total Python files found.
        model_name: LLM model used for classification.

    Returns:
        Scan ID.
    """
    cursor = conn.execute(
        """
        INSERT INTO repo_scans (repo_id, total_files, model_name, git_commit)
        VALUES (?, ?, ?, ?)
        """,
        (repo_id, total_files, model_name, get_current_git_commit()),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def complete_repo_scan(
    conn: sqlite3.Connection,
    scan_id: int,
    passed_ml_import_filter: int,
    self_contained: int,
    fragments: int,
    uncertain: int,
    skipped: int,
    duration_seconds: float,
    status: str = "completed",
) -> None:
    """Complete a repository scan.

    Args:
        conn: Database connection.
        scan_id: Scan ID.
        passed_ml_import_filter: Files passing deterministic ML filter.
        self_contained: Files classified as self-contained.
        fragments: Files classified as fragments.
        uncertain: Files with uncertain classification.
        skipped: Files skipped (no ML indicators).
        duration_seconds: Total scan duration.
        status: Final status.
    """
    conn.execute(
        """
        UPDATE repo_scans
        SET completed_at = ?, status = ?, passed_ml_import_filter = ?,
            self_contained = ?, fragments = ?, uncertain = ?, skipped = ?,
            duration_seconds = ?
        WHERE id = ?
        """,
        (
            datetime.now().isoformat(),
            status,
            passed_ml_import_filter,
            self_contained,
            fragments,
            uncertain,
            skipped,
            duration_seconds,
            scan_id,
        ),
    )
    conn.commit()


def update_file_classification(
    conn: sqlite3.Connection,
    file_id: int,
    scan_id: int,
    has_ml_imports: bool,
    classification: str | None = None,
    confidence: float | None = None,
    reasoning: str | None = None,
) -> None:
    """Update file with classification results.

    Args:
        conn: Database connection.
        file_id: File ID.
        scan_id: Scan ID that classified this file.
        has_ml_imports: Whether file has ML indicators.
        classification: Classification result (self_contained, fragment, uncertain).
        confidence: Classification confidence.
        reasoning: LLM reasoning for classification.
    """
    conn.execute(
        """
        UPDATE files
        SET scan_id = ?, has_ml_imports = ?, self_contained_class = ?,
            self_contained_confidence = ?, self_contained_reasoning = ?
        WHERE id = ?
        """,
        (scan_id, has_ml_imports, classification, confidence, reasoning, file_id),
    )
    conn.commit()


def get_self_contained_files_for_repo(
    conn: sqlite3.Connection,
    repo_id: int,
) -> list[dict[str, Any]]:
    """Get self-contained files for a repository.

    Args:
        conn: Database connection.
        repo_id: Repository ID.

    Returns:
        List of file dicts with path, confidence, and reasoning.
    """
    cursor = conn.execute(
        """
        SELECT id, file_path, original_path, self_contained_confidence, self_contained_reasoning
        FROM files
        WHERE repo_id = ? AND self_contained_class = 'self_contained'
        ORDER BY self_contained_confidence DESC
        """,
        (repo_id,),
    )
    return [
        {
            "id": row["id"],
            "file_path": row["file_path"],
            "original_path": row["original_path"],
            "confidence": row["self_contained_confidence"],
            "reasoning": row["self_contained_reasoning"],
        }
        for row in cursor.fetchall()
    ]


def get_scan_stats(conn: sqlite3.Connection) -> ScanStats:
    """Get aggregated scan statistics.

    Args:
        conn: Database connection.

    Returns:
        ScanStats with aggregated data.
    """
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total_scans,
            COUNT(DISTINCT repo_id) as total_repos,
            SUM(total_files) as total_files,
            SUM(self_contained) as total_self_contained,
            SUM(fragments) as total_fragments,
            SUM(skipped) as total_skipped,
            AVG(duration_seconds) as avg_duration
        FROM repo_scans
        WHERE status = 'completed'
        """
    )
    row = cursor.fetchone()
    if not row or row["total_scans"] == 0:
        return ScanStats()

    total_repos = row["total_repos"] or 1
    return ScanStats(
        total_scans=row["total_scans"] or 0,
        total_repos_scanned=total_repos,
        total_files_scanned=row["total_files"] or 0,
        total_self_contained=row["total_self_contained"] or 0,
        total_fragments=row["total_fragments"] or 0,
        total_skipped=row["total_skipped"] or 0,
        avg_self_contained_per_repo=(row["total_self_contained"] or 0) / total_repos,
        avg_scan_duration=row["avg_duration"] or 0,
    )


def get_latest_scan_for_repo(conn: sqlite3.Connection, repo_id: int) -> RepoScan | None:
    """Get the latest scan for a repository.

    Args:
        conn: Database connection.
        repo_id: Repository ID.

    Returns:
        RepoScan or None if no scans found.
    """
    cursor = conn.execute(
        """
        SELECT * FROM repo_scans
        WHERE repo_id = ?
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (repo_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    return RepoScan(
        scan_id=row["id"],
        repo_id=row["repo_id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        completed_at=(datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None),
        status=RunStatus(row["status"]) if row["status"] else RunStatus.RUNNING,
        total_files=row["total_files"] or 0,
        passed_ml_import_filter=row["passed_ml_import_filter"] or 0,
        self_contained=row["self_contained"] or 0,
        fragments=row["fragments"] or 0,
        uncertain=row["uncertain"] or 0,
        skipped=row["skipped"] or 0,
        duration_seconds=row["duration_seconds"] or 0,
        model_name=row["model_name"] or "",
    )
