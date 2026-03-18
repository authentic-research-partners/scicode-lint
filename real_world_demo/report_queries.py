"""Database query functions for report generation.

Provides statistical queries and finding lookups used by generate_report.py.
"""

import sqlite3
import statistics
from typing import Any


class DistributionStats:
    """Statistics for a distribution of values."""

    def __init__(self, values: list[int]) -> None:
        self.count = len(values)
        if values:
            self.min = min(values)
            self.max = max(values)
            self.mean = statistics.mean(values)
            self.stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            self.median = statistics.median(values)
        else:
            self.min = self.max = 0
            self.mean = self.stdev = self.median = 0.0


def get_findings_distribution(
    conn: sqlite3.Connection, run_id: int, *, verified_only: bool = False
) -> dict[str, DistributionStats]:
    """Get distribution stats for findings per paper and per severity.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Dict with 'per_paper' and 'per_paper_by_severity' distribution stats.
    """
    result: dict[str, DistributionStats] = {}

    # Build verification join clause
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    # Findings per paper (total)
    cursor = conn.execute(
        f"""
        SELECT r.paper_id, COUNT(fn.id) as finding_count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        JOIN repos r ON r.id = f.repo_id
        {verification_join}
        WHERE fa.run_id = ? AND r.paper_id IS NOT NULL
        GROUP BY r.paper_id
        """,
        (run_id,),
    )
    counts = [row[1] for row in cursor.fetchall()]
    result["per_paper"] = DistributionStats(counts)

    # Findings per paper by severity
    for severity in ["critical", "high", "medium", "low"]:
        cursor = conn.execute(
            f"""
            SELECT r.paper_id, COUNT(fn.id) as finding_count
            FROM findings fn
            JOIN file_analyses fa ON fa.id = fn.file_analysis_id
            JOIN files f ON f.id = fa.file_id
            JOIN repos r ON r.id = f.repo_id
            {verification_join}
            WHERE fa.run_id = ? AND r.paper_id IS NOT NULL AND fn.severity = ?
            GROUP BY r.paper_id
            """,
            (run_id, severity),
        )
        counts = [row[1] for row in cursor.fetchall()]
        result[severity] = DistributionStats(counts)

    return result


def get_papers_by_severity(
    conn: sqlite3.Connection, run_id: int, *, verified_only: bool = False
) -> dict[str, int]:
    """Get paper counts by severity (papers with at least one finding of that severity).

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Dict mapping severity to paper count. A paper with both critical and medium
        findings will be counted in both categories.
    """
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    # Count distinct papers per severity level
    cursor = conn.execute(
        f"""
        SELECT
            fn.severity,
            COUNT(DISTINCT r.paper_id) as paper_count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        JOIN repos r ON r.id = f.repo_id
        {verification_join}
        WHERE fa.run_id = ? AND r.paper_id IS NOT NULL
        GROUP BY fn.severity
        """,
        (run_id,),
    )

    counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for row in cursor.fetchall():
        sev = row[0] or "low"
        if sev in counts:
            counts[sev] = row[1]

    return counts


def get_verification_by_severity(
    conn: sqlite3.Connection, run_id: int
) -> dict[str, dict[str, int]]:
    """Get verification stats broken down by severity.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        Dict with severities as keys, each containing {valid, invalid, uncertain, pending}.
    """
    # Get all findings with their verification status (if any)
    cursor = conn.execute(
        """
        SELECT
            fn.severity,
            fv.status as verification_status,
            COUNT(*) as count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        LEFT JOIN finding_verifications fv ON fv.finding_id = fn.id
        WHERE fa.run_id = ?
        GROUP BY fn.severity, fv.status
        """,
        (run_id,),
    )

    result: dict[str, dict[str, int]] = {}
    for sev in ["critical", "high", "medium", "low"]:
        result[sev] = {"valid": 0, "invalid": 0, "uncertain": 0, "pending": 0, "total": 0}

    for row in cursor.fetchall():
        sev = row[0] or "low"
        status = row[1]  # None if not verified
        count = row[2]

        if sev not in result:
            result[sev] = {"valid": 0, "invalid": 0, "uncertain": 0, "pending": 0, "total": 0}

        result[sev]["total"] += count
        if status is None:
            result[sev]["pending"] += count
        elif status in result[sev]:
            result[sev][status] += count

    return result


def get_example_findings(
    conn: sqlite3.Connection,
    run_id: int,
    limit_per_category: int = 3,
    *,
    verified_only: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Get example findings grouped by category.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        limit_per_category: Max findings per category.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Dict mapping category to list of finding dicts.
    """
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    cursor = conn.execute(
        f"""
        SELECT DISTINCT fn.category
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        {verification_join}
        WHERE fa.run_id = ?
        ORDER BY fn.category
        """,
        (run_id,),
    )
    categories = [row[0] for row in cursor.fetchall()]

    examples: dict[str, list[dict[str, Any]]] = {}
    for category in categories:
        cursor = conn.execute(
            f"""
            SELECT
                fn.id,
                fn.pattern_id,
                fn.severity,
                fn.confidence,
                fn.issue,
                fn.explanation,
                fn.suggestion,
                fn.snippet,
                fn.lines,
                fn.location_name,
                fn.location_type,
                fn.focus_line,
                f.file_path,
                f.original_path,
                r.repo_name,
                r.repo_url,
                r.domain,
                p.title as paper_title,
                p.arxiv_id,
                p.authors as paper_authors
            FROM findings fn
            JOIN file_analyses fa ON fa.id = fn.file_analysis_id
            JOIN files f ON f.id = fa.file_id
            JOIN repos r ON r.id = f.repo_id
            LEFT JOIN papers p ON p.id = r.paper_id
            {verification_join}
            WHERE fa.run_id = ? AND fn.category = ?
            ORDER BY fn.confidence DESC
            LIMIT ?
            """,
            (run_id, category, limit_per_category),
        )
        examples[category] = [
            {
                "finding_id": row[0],
                "pattern_id": row[1],
                "severity": row[2],
                "confidence": row[3],
                "issue": row[4],
                "explanation": row[5],
                "suggestion": row[6],
                "snippet": row[7],
                "lines": row[8],
                "location_name": row[9],
                "location_type": row[10],
                "focus_line": row[11],
                "file_path": row[12],
                "original_path": row[13],
                "repo_name": row[14],
                "repo_url": row[15],
                "domain": row[16],
                "paper_title": row[17],
                "arxiv_id": row[18],
                "paper_authors": row[19],
            }
            for row in cursor.fetchall()
        ]

    return examples


def get_top_patterns(
    conn: sqlite3.Connection, run_id: int, limit: int = 10, *, verified_only: bool = False
) -> list[dict[str, Any]]:
    """Get most frequent patterns.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        limit: Max patterns to return.
        verified_only: If True, only include findings with completed verification.

    Returns:
        List of pattern stats dicts.
    """
    verification_join = (
        "INNER JOIN finding_verifications fv ON fv.finding_id = fn.id AND fv.status != 'error'"
        if verified_only
        else ""
    )

    cursor = conn.execute(
        f"""
        SELECT
            fn.pattern_id,
            fn.category,
            COUNT(*) as count,
            COUNT(DISTINCT fa.file_id) as unique_files,
            COUNT(DISTINCT f.repo_id) as unique_repos,
            AVG(fn.confidence) as avg_confidence
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        {verification_join}
        WHERE fa.run_id = ?
        GROUP BY fn.pattern_id
        ORDER BY count DESC
        LIMIT ?
        """,
        (run_id, limit),
    )
    return [
        {
            "pattern_id": row[0],
            "category": row[1],
            "count": row[2],
            "unique_files": row[3],
            "unique_repos": row[4],
            "avg_confidence": row[5],
        }
        for row in cursor.fetchall()
    ]
