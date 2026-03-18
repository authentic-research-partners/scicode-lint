"""Valid/critical findings report generation.

Queries and formats verified critical+high findings for quick manual review.
"""

import json
import sqlite3
from typing import Any


def _query_valid_findings(
    conn: sqlite3.Connection, run_id: int, severity: str
) -> list[dict[str, Any]]:
    """Query valid findings for a given severity level.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        severity: Severity level ('critical' or 'high').

    Returns:
        List of finding dicts ordered by confidence.
    """
    cursor = conn.execute(
        """
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
            r.paper_id,
            p.title as paper_title,
            p.arxiv_id,
            p.authors as paper_authors,
            fv.reasoning as verification_reasoning
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN files f ON f.id = fa.file_id
        JOIN repos r ON r.id = f.repo_id
        LEFT JOIN papers p ON p.id = r.paper_id
        INNER JOIN finding_verifications fv ON fv.finding_id = fn.id
        WHERE fa.run_id = ?
          AND fn.severity = ?
          AND fv.status = 'valid'
          AND r.paper_id IS NOT NULL
        GROUP BY r.paper_id
        ORDER BY fn.confidence DESC
        """,
        (run_id, severity),
    )

    return [
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
            "paper_id": row[17],
            "paper_title": row[18],
            "arxiv_id": row[19],
            "paper_authors": row[20],
            "verification_reasoning": row[21],
        }
        for row in cursor.fetchall()
    ]


def get_valid_critical_findings(
    conn: sqlite3.Connection, run_id: int, limit: int = 10
) -> list[dict[str, Any]]:
    """Get valid findings with pattern diversity across severity levels.

    Selection strategy for maximum diversity:
    1. Critical findings: pick one per unique pattern
    2. High findings: pick one per unique pattern (not already seen)
    3. Fill remaining slots with duplicates by confidence

    Args:
        conn: Database connection.
        run_id: Analysis run ID.
        limit: Max findings to return.

    Returns:
        List of finding dicts with paper info.
    """
    selected: list[dict[str, Any]] = []
    seen_patterns: set[str] = set()
    remaining: list[dict[str, Any]] = []

    # Process critical findings first, then high
    for severity in ("critical", "high"):
        findings = _query_valid_findings(conn, run_id, severity)

        for f in findings:
            if len(selected) >= limit:
                break

            pattern = f["pattern_id"]
            if pattern not in seen_patterns:
                selected.append(f)
                seen_patterns.add(pattern)
            else:
                remaining.append(f)

    # Fill remaining slots with duplicates (by confidence, critical first)
    for f in remaining:
        if len(selected) >= limit:
            break
        selected.append(f)

    return selected


def generate_valid_critical_report(conn: sqlite3.Connection, run_id: int) -> str:
    """Generate markdown report of valid findings for quick verification.

    Args:
        conn: Database connection.
        run_id: Analysis run ID.

    Returns:
        Markdown report string.
    """
    findings = get_valid_critical_findings(conn, run_id, limit=10)

    # Count by severity
    critical_count = sum(1 for f in findings if f["severity"] == "critical")
    high_count = sum(1 for f in findings if f["severity"] == "high")

    lines = []
    lines.append("# Valid Findings - Quick Verification Sample")
    lines.append("")

    # Build summary with counts
    counts = []
    if critical_count:
        counts.append(f"{critical_count} critical")
    if high_count:
        counts.append(f"{high_count} high")
    count_str = " + ".join(counts) if counts else "0"

    lines.append(
        f"**{len(findings)} verified findings** ({count_str}) "
        "with pattern diversity for fast manual verification."
    )
    lines.append("")

    for i, f in enumerate(findings, 1):
        lines.append(f"## {i}. {f['pattern_id']} ({f['severity']})")
        lines.append("")

        # Build GitHub link
        repo_url = f["repo_url"] or ""
        original_path = f["original_path"] or ""
        finding_lines = f["lines"] or "[]"

        if repo_url and original_path:
            try:
                line_nums = json.loads(finding_lines) if finding_lines else []
            except (json.JSONDecodeError, TypeError):
                line_nums = []

            github_url = f"{repo_url}/blob/main/{original_path}"
            if line_nums:
                if len(line_nums) == 1:
                    github_url += f"#L{line_nums[0]}"
                else:
                    github_url += f"#L{min(line_nums)}-L{max(line_nums)}"

            lines.append(f"**File:** [{original_path}]({github_url})")
        else:
            lines.append(f"**File:** {f['original_path']}")

        lines.append(f"**Repo:** [{f['repo_name']}]({repo_url})")

        # Add function/class location if available
        if f.get("location_name"):
            loc_type = f.get("location_type") or "function"
            loc_info = f"**Location:** {loc_type} `{f['location_name']}`"
            if f.get("focus_line"):
                loc_info += f" (line {f['focus_line']})"
            lines.append(loc_info)

        if f["paper_title"]:
            paper_ref = f["paper_title"]
            if f["arxiv_id"]:
                paper_ref += f" ([arXiv:{f['arxiv_id']}](https://arxiv.org/abs/{f['arxiv_id']}))"
            lines.append(f"**Paper:** {paper_ref}")

            # Add authors if available
            if f.get("paper_authors"):
                try:
                    authors = json.loads(f["paper_authors"])
                    if authors:
                        # Format: "First Author et al." for >2 authors, otherwise list all
                        if len(authors) > 2:
                            authors_str = f"{authors[0]} et al."
                        else:
                            authors_str = ", ".join(authors)
                        lines.append(f"**Authors:** {authors_str}")
                except (json.JSONDecodeError, TypeError):
                    pass

        lines.append("")
        lines.append(f"**Issue:** {f['issue']}")
        lines.append("")

        if f["explanation"]:
            lines.append(f"**Explanation:** {f['explanation']}")
            lines.append("")

        if f["snippet"]:
            lines.append("**Code:**")
            lines.append("```python")
            lines.append(f["snippet"].strip())
            lines.append("```")
            lines.append("")

        if f["verification_reasoning"]:
            lines.append(f"**Verification reasoning:** {f['verification_reasoning']}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
