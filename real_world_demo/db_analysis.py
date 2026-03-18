"""Analysis run, file analysis, findings, and verification operations."""

import json
import sqlite3
from datetime import datetime
from typing import Any

from .db_core import get_current_git_commit

__all__ = [
    "start_analysis_run",
    "get_latest_run_id",
    "complete_analysis_run",
    "insert_file_analysis",
    "get_timed_out_patterns",
    "update_pattern_run",
    "insert_pattern_runs",
    "insert_findings",
    "save_verification",
    "get_verification_stats",
    "get_finding_verification",
    "get_analysis_run_data_source",
]


def start_analysis_run(
    conn: sqlite3.Connection,
    total_files: int,
    data_source: str = "papers_with_code",
    config: dict[str, Any] | None = None,
    model_name: str = "",
    notes: str = "",
    prefilter_run_id: int | None = None,
) -> int:
    """Start a new analysis run.

    Args:
        conn: Database connection.
        total_files: Total files to analyze.
        data_source: Data source identifier (e.g., 'papers_with_code', 'leakage_paper').
        config: Run configuration dict.
        model_name: LLM model being used.
        notes: Optional run notes.
        prefilter_run_id: Optional ID of prefilter run that selected these files.

    Returns:
        Run ID.
    """
    cursor = conn.execute(
        """
        INSERT INTO analysis_runs
        (total_files, data_source, config, git_commit, model_name, notes, prefilter_run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            total_files,
            data_source,
            json.dumps(config or {}),
            get_current_git_commit(),
            model_name,
            notes,
            prefilter_run_id,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_latest_run_id(
    conn: sqlite3.Connection,
    data_source: str | None = None,
) -> int:
    """Get the most recent analysis run ID.

    Args:
        conn: Database connection.
        data_source: Optional filter by data source.

    Returns:
        Run ID, or 0 if no runs found.
    """
    if data_source:
        result = conn.execute(
            "SELECT MAX(id) FROM analysis_runs WHERE data_source = ?",
            (data_source,),
        ).fetchone()
    else:
        result = conn.execute("SELECT MAX(id) FROM analysis_runs").fetchone()
    return result[0] if result and result[0] else 0


def complete_analysis_run(
    conn: sqlite3.Connection,
    run_id: int,
    analyzed: int,
    with_findings: int,
    total_findings: int,
    status: str = "completed",
) -> None:
    """Complete an analysis run.

    Args:
        conn: Database connection.
        run_id: Run ID.
        analyzed: Number of files analyzed.
        with_findings: Number of files with findings.
        total_findings: Total findings count.
        status: Final status.
    """
    conn.execute(
        """
        UPDATE analysis_runs
        SET completed_at = ?, analyzed_files = ?, files_with_findings = ?,
            total_findings = ?, status = ?
        WHERE id = ?
        """,
        (datetime.now().isoformat(), analyzed, with_findings, total_findings, status, run_id),
    )
    conn.commit()


def insert_file_analysis(
    conn: sqlite3.Connection,
    run_id: int,
    file_id: int,
    status: str,
    error: str | None = None,
    duration: float = 0,
) -> int:
    """Insert file analysis result.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        file_id: File ID.
        status: Analysis status.
        error: Error message if failed.
        duration: Analysis duration in seconds.

    Returns:
        File analysis ID.
    """
    cursor = conn.execute(
        """
        INSERT OR REPLACE INTO file_analyses
        (run_id, file_id, status, error, duration_seconds)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, file_id, status, error, duration),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_timed_out_patterns(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict[str, Any]]:
    """Get patterns that timed out in a run.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        List of dicts with file_path, file_id, file_analysis_id, and pattern_id.
    """
    query = """
        SELECT f.file_path, f.id as file_id, fa.id as file_analysis_id, pr.pattern_id
        FROM pattern_runs pr
        JOIN file_analyses fa ON pr.file_analysis_id = fa.id
        JOIN files f ON fa.file_id = f.id
        WHERE fa.run_id = ? AND pr.status = 'timeout'
        ORDER BY f.file_path, pr.pattern_id
    """
    results = []
    for row in conn.execute(query, (run_id,)):
        results.append(
            {
                "file_path": row[0],
                "file_id": row[1],
                "file_analysis_id": row[2],
                "pattern_id": row[3],
            }
        )
    return results


def update_pattern_run(
    conn: sqlite3.Connection,
    file_analysis_id: int,
    pattern_id: str,
    status: str,
    detected: str | None = None,
    confidence: float | None = None,
    reasoning: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update an existing pattern run result.

    Args:
        conn: Database connection.
        file_analysis_id: Parent file analysis ID.
        pattern_id: Pattern ID.
        status: New status (success, timeout, etc.).
        detected: Detection result (yes, no, context-dependent).
        confidence: Confidence score.
        reasoning: LLM reasoning.
        error_message: Error message if failed.
    """
    conn.execute(
        """
        UPDATE pattern_runs
        SET status = ?, detected = ?, confidence = ?, reasoning = ?, error_message = ?
        WHERE file_analysis_id = ? AND pattern_id = ?
        """,
        (status, detected, confidence, reasoning, error_message, file_analysis_id, pattern_id),
    )
    conn.commit()


def insert_pattern_runs(
    conn: sqlite3.Connection,
    file_analysis_id: int,
    checked_patterns: list[dict[str, Any]],
    failed_patterns: list[dict[str, Any]],
) -> int:
    """Insert pattern run results for a file analysis.

    Args:
        conn: Database connection.
        file_analysis_id: Parent file analysis ID.
        checked_patterns: List of successful pattern check results.
        failed_patterns: List of failed pattern results.

    Returns:
        Number of pattern runs inserted.
    """
    count = 0

    # Insert successful pattern runs
    for pattern in checked_patterns:
        conn.execute(
            """
            INSERT OR REPLACE INTO pattern_runs
            (file_analysis_id, pattern_id, status, detected, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_analysis_id,
                pattern.get("pattern_id", ""),
                "success",
                pattern.get("detected", ""),
                pattern.get("confidence", 0),
                pattern.get("reasoning", ""),
            ),
        )
        count += 1

    # Insert failed pattern runs
    for pattern in failed_patterns:
        conn.execute(
            """
            INSERT OR REPLACE INTO pattern_runs
            (file_analysis_id, pattern_id, status, error_message)
            VALUES (?, ?, ?, ?)
            """,
            (
                file_analysis_id,
                pattern.get("pattern_id", ""),
                pattern.get("error_type", "api_error"),
                pattern.get("error_message", ""),
            ),
        )
        count += 1

    conn.commit()
    return count


def insert_findings(
    conn: sqlite3.Connection,
    file_analysis_id: int,
    findings: list[dict[str, Any]],
) -> int:
    """Insert findings for a file analysis.

    Args:
        conn: Database connection.
        file_analysis_id: Parent file analysis ID.
        findings: List of finding dicts from scicode-lint.

    Returns:
        Number of findings inserted.
    """
    count = 0
    for finding in findings:
        # Location fields can be at top level or nested in location dict
        # run_analysis.py puts them at top level, other code may nest them
        location = finding.get("location", {})
        location_name = finding.get("location_name") or location.get("name")
        location_type = finding.get("location_type") or location.get("location_type")
        lines = finding.get("lines") or location.get("lines", [])
        focus_line = finding.get("focus_line") or location.get("focus_line")
        snippet = finding.get("snippet") or location.get("snippet", "")

        conn.execute(
            """
            INSERT INTO findings
            (file_analysis_id, pattern_id, category, severity, confidence,
             issue, explanation, suggestion, reasoning,
             location_name, location_type, lines, focus_line, snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_analysis_id,
                finding.get("pattern_id", finding.get("id", "")),
                finding.get("category", ""),
                finding.get("severity", ""),
                finding.get("confidence", 0),
                finding.get("issue", ""),
                finding.get("explanation", ""),
                finding.get("suggestion", ""),
                finding.get("reasoning", ""),
                location_name,
                location_type,
                json.dumps(lines),
                focus_line,
                snippet,
            ),
        )
        count += 1

    conn.commit()
    return count


def save_verification(
    conn: sqlite3.Connection,
    finding_id: int,
    status: str,
    reasoning: str,
    model: str,
) -> None:
    """Save or update a finding verification result.

    Args:
        conn: Database connection.
        finding_id: Finding ID being verified.
        status: Verification status (valid, invalid, uncertain, error).
        reasoning: Claude's reasoning.
        model: Model used for verification.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO finding_verifications
        (finding_id, status, reasoning, model)
        VALUES (?, ?, ?, ?)
        """,
        (finding_id, status, reasoning, model),
    )
    conn.commit()


def get_verification_stats(conn: sqlite3.Connection, run_id: int | None = None) -> dict[str, Any]:
    """Get verification statistics for a run.

    Args:
        conn: Database connection.
        run_id: Analysis run ID, or None for latest.

    Returns:
        Dict with verification stats.
    """
    if run_id is None:
        cursor = conn.execute("SELECT id FROM analysis_runs ORDER BY started_at DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return {}
        run_id = row["id"]

    # Get counts by status
    cursor = conn.execute(
        """
        SELECT fv.status, COUNT(*) as count
        FROM finding_verifications fv
        JOIN findings fn ON fn.id = fv.finding_id
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        GROUP BY fv.status
        """,
        (run_id,),
    )
    stats: dict[str, Any] = {"by_status": {}}
    for row in cursor.fetchall():
        stats["by_status"][row["status"]] = row["count"]

    # Calculate totals
    valid = stats["by_status"].get("valid", 0)
    invalid = stats["by_status"].get("invalid", 0)
    uncertain = stats["by_status"].get("uncertain", 0)
    total = valid + invalid + uncertain

    stats["total_verified"] = total
    stats["valid"] = valid
    stats["invalid"] = invalid
    stats["uncertain"] = uncertain
    stats["precision"] = (valid / total * 100) if total > 0 else 0

    # Get verification date
    cursor = conn.execute(
        """
        SELECT MAX(fv.verified_at) as last_verified, fv.model
        FROM finding_verifications fv
        JOIN findings fn ON fn.id = fv.finding_id
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        WHERE fa.run_id = ?
        """,
        (run_id,),
    )
    row = cursor.fetchone()
    if row and row["last_verified"]:
        stats["verified_at"] = row["last_verified"]
        stats["model"] = row["model"]

    return stats


def get_finding_verification(conn: sqlite3.Connection, finding_id: int) -> dict[str, Any] | None:
    """Get verification result for a specific finding.

    Args:
        conn: Database connection.
        finding_id: Finding ID.

    Returns:
        Dict with status and reasoning, or None if not verified.
    """
    cursor = conn.execute(
        "SELECT status, reasoning, model FROM finding_verifications WHERE finding_id = ?",
        (finding_id,),
    )
    row = cursor.fetchone()
    if row:
        return {"status": row["status"], "reasoning": row["reasoning"], "model": row["model"]}
    return None


def get_analysis_run_data_source(
    conn: sqlite3.Connection,
    analysis_run_id: int,
) -> str | None:
    """Get data source for an analysis run.

    Args:
        conn: Database connection.
        analysis_run_id: Analysis run ID.

    Returns:
        Data source string or None if not found.
    """
    cursor = conn.execute(
        "SELECT data_source FROM analysis_runs WHERE id = ?",
        (analysis_run_id,),
    )
    row = cursor.fetchone()
    return row["data_source"] if row else None
