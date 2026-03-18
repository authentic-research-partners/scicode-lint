"""Statistics and query functions for analysis runs."""

import json
import sqlite3
from datetime import datetime

from loguru import logger

from .models import (
    AnalysisRun,
    AnalysisStats,
    CategoryStats,
    DomainStats,
    PatternStats,
    RunStatus,
)

__all__ = [
    "get_run_stats",
    "list_runs",
    "get_analyzed_file_ids",
    "get_incomplete_run",
    "print_stats",
]


def get_run_stats(conn: sqlite3.Connection, run_id: int | None = None) -> AnalysisStats:
    """Get statistics for a specific run or the latest run.

    Args:
        conn: Database connection.
        run_id: Specific run ID, or None for latest.

    Returns:
        AnalysisStats with aggregated data.
    """
    # Get run info
    if run_id is None:
        cursor = conn.execute("SELECT * FROM analysis_runs ORDER BY started_at DESC LIMIT 1")
    else:
        cursor = conn.execute("SELECT * FROM analysis_runs WHERE id = ?", (run_id,))

    run_row = cursor.fetchone()
    if not run_row:
        return AnalysisStats(run_id=0, run_date=datetime.now())

    run_id = run_row["id"]

    # Compute actual counts from file_analyses (more reliable than run record)
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT fa.id) as analyzed_files,
            COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN fa.file_id END) as files_with_findings,
            COUNT(fn.id) as total_findings
        FROM file_analyses fa
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ?
        """,
        (run_id,),
    )
    counts = cursor.fetchone()

    stats = AnalysisStats(
        run_id=run_id,
        run_date=datetime.fromisoformat(run_row["started_at"]),
        data_source=run_row["data_source"] or "papers_with_code",
        total_files=run_row["total_files"],
        analyzed_files=counts["analyzed_files"] or 0,
        files_with_findings=counts["files_with_findings"] or 0,
        total_findings=counts["total_findings"] or 0,
    )

    # Total repos and repos with findings for this run
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT f.repo_id) as total_repos,
            COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN f.repo_id END) as repos_with_findings
        FROM files f
        JOIN file_analyses fa ON fa.file_id = f.id
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ?
        """,
        (run_id,),
    )
    repo_counts = cursor.fetchone()
    stats.total_repos = repo_counts["total_repos"] or 0
    stats.repos_with_findings = repo_counts["repos_with_findings"] or 0

    # Total papers and papers with findings (scoped to this run)
    cursor = conn.execute(
        """
        SELECT
            COUNT(DISTINCT r.paper_id) as total_papers,
            COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN r.paper_id END) as papers_with_findings
        FROM repos r
        JOIN files f ON f.repo_id = r.id
        JOIN file_analyses fa ON fa.file_id = f.id
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        WHERE fa.run_id = ? AND r.paper_id IS NOT NULL
        """,
        (run_id,),
    )
    paper_counts = cursor.fetchone()
    stats.total_papers = paper_counts["total_papers"] or 0
    stats.papers_with_findings = paper_counts["papers_with_findings"] or 0

    # Success rate
    if stats.total_files > 0:
        stats.analysis_success_rate = 100 * stats.analyzed_files / stats.total_files

    # Finding rate
    if stats.analyzed_files > 0:
        stats.finding_rate = 100 * stats.files_with_findings / stats.analyzed_files

    # By domain
    cursor = conn.execute(
        """
        SELECT r.domain,
               COUNT(DISTINCT f.id) as total_files,
               COUNT(DISTINCT CASE WHEN fa.status = 'success' THEN f.id END) as analyzed,
               COUNT(DISTINCT CASE WHEN fn.id IS NOT NULL THEN f.id END) as with_findings,
               COUNT(fn.id) as total_findings
        FROM repos r
        JOIN files f ON f.repo_id = r.id
        JOIN file_analyses fa ON fa.file_id = f.id AND fa.run_id = ?
        LEFT JOIN findings fn ON fn.file_analysis_id = fa.id
        GROUP BY r.domain
        ORDER BY total_files DESC
        """,
        (run_id,),
    )
    for row in cursor.fetchall():
        analyzed = row["analyzed"] or 0
        stats.by_domain.append(
            DomainStats(
                domain=row["domain"] or "unknown",
                total_files=row["total_files"],
                analyzed_files=analyzed,
                files_with_findings=row["with_findings"],
                total_findings=row["total_findings"],
                finding_rate=100 * row["with_findings"] / analyzed if analyzed > 0 else 0,
            )
        )

    # By category
    cursor = conn.execute(
        """
        SELECT category,
               COUNT(*) as count,
               COUNT(DISTINCT fa.file_id) as unique_files,
               COUNT(DISTINCT f.repo_id) as unique_repos
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        WHERE fa.run_id = ?
        GROUP BY category
        ORDER BY count DESC
        """,
        (run_id,),
    )
    for row in cursor.fetchall():
        stats.by_category.append(
            CategoryStats(
                category=row["category"],
                count=row["count"],
                unique_files=row["unique_files"],
                unique_repos=row["unique_repos"],
            )
        )

    # By pattern (top 15)
    cursor = conn.execute(
        """
        SELECT pattern_id, category,
               COUNT(*) as count,
               COUNT(DISTINCT fa.file_id) as unique_files,
               AVG(confidence) as avg_confidence
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        GROUP BY pattern_id
        ORDER BY count DESC
        LIMIT 15
        """,
        (run_id,),
    )
    for row in cursor.fetchall():
        stats.by_pattern.append(
            PatternStats(
                pattern_id=row["pattern_id"],
                category=row["category"],
                count=row["count"],
                unique_files=row["unique_files"],
                avg_confidence=row["avg_confidence"],
            )
        )

    # By severity
    cursor = conn.execute(
        """
        SELECT severity, COUNT(*) as count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        GROUP BY severity
        """,
        (run_id,),
    )
    stats.by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}

    return stats


def list_runs(
    conn: sqlite3.Connection,
    limit: int = 10,
    data_source: str | None = None,
) -> list[AnalysisRun]:
    """List recent analysis runs.

    Args:
        conn: Database connection.
        limit: Maximum runs to return.
        data_source: Optional filter by data source.

    Returns:
        List of AnalysisRun models.
    """
    if data_source:
        cursor = conn.execute(
            """
            SELECT * FROM analysis_runs
            WHERE data_source = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (data_source, limit),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM analysis_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    runs = []
    for row in cursor.fetchall():
        runs.append(
            AnalysisRun(
                run_id=row["id"],
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ),
                status=RunStatus(row["status"]) if row["status"] else RunStatus.RUNNING,
                data_source=row["data_source"] or "papers_with_code",
                total_files=row["total_files"],
                analyzed_files=row["analyzed_files"],
                files_with_findings=row["files_with_findings"],
                total_findings=row["total_findings"],
                config=json.loads(row["config"]) if row["config"] else {},
                git_commit=row["git_commit"] or "",
                model_name=row["model_name"] or "",
            )
        )
    return runs


def get_analyzed_file_ids(conn: sqlite3.Connection, run_id: int) -> set[int]:
    """Get set of file IDs already analyzed in a run.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        Set of file IDs that have been analyzed.
    """
    cursor = conn.execute(
        "SELECT file_id FROM file_analyses WHERE run_id = ?",
        (run_id,),
    )
    return {row["file_id"] for row in cursor.fetchall()}


def get_incomplete_run(conn: sqlite3.Connection) -> int | None:
    """Get the latest incomplete (running) analysis run.

    Args:
        conn: Database connection.

    Returns:
        Run ID if there's an incomplete run, None otherwise.
    """
    cursor = conn.execute(
        """
        SELECT id FROM analysis_runs
        WHERE status = 'running'
        ORDER BY started_at DESC
        LIMIT 1
        """
    )
    row = cursor.fetchone()
    return row["id"] if row else None


def print_stats(conn: sqlite3.Connection, run_id: int | None = None) -> None:
    """Print statistics from database.

    Args:
        conn: Database connection.
        run_id: Specific run ID, or None for latest.
    """
    stats = get_run_stats(conn, run_id)

    logger.info("=" * 60)
    logger.info(f"Analysis Run #{stats.run_id} - {stats.run_date.strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)
    logger.info(f"  Total repos: {stats.total_repos:,}")
    logger.info(f"  Total files: {stats.total_files:,}")
    pct = stats.analysis_success_rate
    logger.info(f"  Analyzed successfully: {stats.analyzed_files:,} ({pct:.1f}%)")
    logger.info(f"  Files with findings: {stats.files_with_findings:,} ({stats.finding_rate:.1f}%)")
    logger.info(f"  Total findings: {stats.total_findings:,}")

    if stats.by_category:
        logger.info("  By category:")
        for cat in stats.by_category:
            logger.info(f"    {cat.category}: {cat.count:,} ({cat.unique_files} files)")

    if stats.by_severity:
        logger.info("  By severity:")
        for sev in ["critical", "high", "medium", "low"]:
            if sev in stats.by_severity:
                logger.info(f"    {sev}: {stats.by_severity[sev]:,}")

    if stats.by_domain:
        logger.info("  By domain:")
        for domain in stats.by_domain:
            logger.info(
                f"    {domain.domain}: {domain.files_with_findings}/{domain.analyzed_files} "
                f"({domain.finding_rate:.1f}%)"
            )
