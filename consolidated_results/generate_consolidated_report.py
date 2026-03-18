"""Generate consolidated performance report across all eval layers.

Reads from JSON, SQLite DB, and markdown sources to produce a single
report with every key metric.

Data sources (in priority order):
  1. JSON files  — controlled tests, integration eval
  2. SQLite DB   — Kaggle labeled, PapersWithCode (feedback + holdout)
  3. Markdown    — paper set definitions (manually authored)

Usage:
    python consolidated_results/generate_consolidated_report.py
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Source data locations
# ---------------------------------------------------------------------------

# 01 — Controlled tests (JSON)
_JUDGE_DIR = "src/scicode_lint/evals/reports/judge/20260317_173544_all"
JUDGE_REPORT_JSON = ROOT / _JUDGE_DIR / "llm_judge_report.json"

# 02 — Integration eval (JSON + log files)
_INTEG_DIR = "evals/integration/reports/20260316_172513_generate_50"
INTEGRATION_REPORT_JSON = ROOT / _INTEG_DIR / "report.json"
INTEGRATION_SCENARIOS_DIR = ROOT / _INTEG_DIR / "scenarios"

# 03–05 — Real-world results (SQLite DB)
ANALYSIS_DB = ROOT / "real_world_demo/data/analysis.db"

# Run IDs for paper results
KAGGLE_RUN_ID = 57  # leakage_paper dataset
FEEDBACK_RUN_ID = 65  # PapersWithCode feedback set
HOLDOUT_RUN_ID = 66  # PapersWithCode holdout set

# Paper set definitions (markdown — manually authored, not generated)
FEEDBACK_PAPERSET_MD = ROOT / "real_world_demo/paper_sets/meta_loop_set.md"
HOLDOUT_PAPERSET_MD = ROOT / "real_world_demo/paper_sets/holdout_set.md"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ControlledTestMetrics:
    total_patterns: int = 0
    total_tests: int = 0
    overall_accuracy: float = 0.0
    positive_accuracy: float = 0.0
    negative_accuracy: float = 0.0
    context_accuracy: float = 0.0
    semantic_alignment: float = 0.0
    quality_issue_rate: float = 0.0
    patterns_at_100: int = 0
    focus_line_accuracy: float = 0.0
    focus_line_eligible: int = 0
    focus_line_matched: int = 0
    source: str = ""


@dataclass
class IntegrationMetrics:
    scenarios: int = 0
    bugs_intended: int = 0
    tp_intended: int = 0
    tp_bonus: int = 0
    total_fp: int = 0
    total_fn: int = 0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    source: str = ""


@dataclass
class KaggleMetrics:
    labels: dict[str, dict[str, float]] = field(default_factory=dict)
    files_analyzed: int = 0
    files_with_findings: int = 0
    excluded_notebooks: int = 0
    source: str = ""


@dataclass
class RealWorldMetrics:
    run_id: int = 0
    papers_total: int = 0
    papers_with_sc_files: int = 0
    papers_with_verified_bugs: int = 0
    sc_files: int = 0
    total_findings: int = 0
    valid: int = 0
    invalid: int = 0
    uncertain: int = 0
    precision: float = 0.0
    by_category: dict[str, dict[str, int | float]] = field(default_factory=dict)
    by_severity: dict[str, dict[str, int | float]] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction: JSON sources
# ---------------------------------------------------------------------------


def extract_controlled_tests() -> ControlledTestMetrics:
    """Extract from llm_judge_report.json."""
    src = str(JUDGE_REPORT_JSON.relative_to(ROOT))
    m = ControlledTestMetrics(source=src)

    if not JUDGE_REPORT_JSON.exists():
        return m

    with open(JUDGE_REPORT_JSON) as f:
        d = json.load(f)

    m.total_patterns = d["total_patterns"]
    m.total_tests = d["total_tests"]
    m.overall_accuracy = d["overall_accuracy"] * 100
    m.positive_accuracy = d["positive_accuracy"] * 100
    m.negative_accuracy = d["negative_accuracy"] * 100
    m.context_accuracy = d["context_accuracy"] * 100
    m.semantic_alignment = d["semantic_alignment"] * 100
    m.quality_issue_rate = d["quality_issue_rate"] * 100
    m.patterns_at_100 = sum(1 for p in d["patterns"] if p["overall_accuracy"] == 1.0)
    m.focus_line_accuracy = d.get("focus_line_accuracy", 0.0) * 100
    m.focus_line_eligible = d.get("focus_line_eligible", 0)
    m.focus_line_matched = d.get("focus_line_matched", 0)

    return m


def extract_integration() -> IntegrationMetrics:
    """Extract from report.json + count FPs from scenario logs."""
    src = str(INTEGRATION_REPORT_JSON.relative_to(ROOT))
    m = IntegrationMetrics(source=src)

    if not INTEGRATION_REPORT_JSON.exists():
        return m

    with open(INTEGRATION_REPORT_JSON) as f:
        data = json.load(f)

    s = data["summary"]
    m.scenarios = s["scenarios"]
    m.bugs_intended = s["total_bugs_intended"]
    m.tp_intended = s["total_tp_intended"]
    m.tp_bonus = s["total_tp_bonus"]
    m.total_fn = s["total_fn"]
    m.recall = s["recall"]

    # Count FPs from scenario judge logs
    if INTEGRATION_SCENARIOS_DIR.exists():
        for log_file in INTEGRATION_SCENARIOS_DIR.glob("*.log"):
            text = log_file.read_text()
            m.total_fp += text.count('"category": "fp"')

    total_tp = m.tp_intended + m.tp_bonus
    total_findings = total_tp + m.total_fp
    if total_findings > 0:
        m.precision = total_tp / total_findings
    if m.precision + m.recall > 0:
        m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall)

    return m


# ---------------------------------------------------------------------------
# Extraction: SQLite DB
# ---------------------------------------------------------------------------


def _get_db_conn() -> sqlite3.Connection | None:
    """Get DB connection or None if DB doesn't exist."""
    if not ANALYSIS_DB.exists() or ANALYSIS_DB.stat().st_size == 0:
        return None
    conn = sqlite3.connect(str(ANALYSIS_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _db_verification_stats(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    """Get valid/invalid/uncertain counts for a run."""
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
    counts: dict[str, int] = {}
    for row in cursor.fetchall():
        counts[row["status"]] = row["count"]

    valid = counts.get("valid", 0)
    invalid = counts.get("invalid", 0)
    uncertain = counts.get("uncertain", 0)
    total = valid + invalid + uncertain
    precision = (valid / total * 100) if total > 0 else 0.0

    return {
        "valid": valid,
        "invalid": invalid,
        "uncertain": uncertain,
        "total": total,
        "precision": precision,
    }


def _db_verification_by_category(
    conn: sqlite3.Connection, run_id: int
) -> dict[str, dict[str, int | float]]:
    """Get verification breakdown by category."""
    cursor = conn.execute(
        """
        SELECT fn.category, fv.status, COUNT(*) as count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN finding_verifications fv ON fv.finding_id = fn.id
        WHERE fa.run_id = ?
        GROUP BY fn.category, fv.status
        """,
        (run_id,),
    )
    result: dict[str, dict[str, int | float]] = {}
    for row in cursor.fetchall():
        cat = row["category"]
        status = row["status"]
        count = row["count"]
        if cat not in result:
            result[cat] = {"valid": 0, "invalid": 0, "uncertain": 0}
        if status in result[cat]:
            result[cat][status] = count

    # Compute precision per category
    for cat, d in result.items():
        v = d["valid"]
        total = v + d["invalid"] + d["uncertain"]
        d["precision"] = round(v / total * 100) if total > 0 else 0

    return result


def _db_verification_by_severity(
    conn: sqlite3.Connection, run_id: int
) -> dict[str, dict[str, int | float]]:
    """Get verification breakdown by severity."""
    cursor = conn.execute(
        """
        SELECT fn.severity, fv.status, COUNT(*) as count
        FROM findings fn
        JOIN file_analyses fa ON fa.id = fn.file_analysis_id
        JOIN finding_verifications fv ON fv.finding_id = fn.id
        WHERE fa.run_id = ?
        GROUP BY fn.severity, fv.status
        """,
        (run_id,),
    )
    result: dict[str, dict[str, int | float]] = {}
    for row in cursor.fetchall():
        sev = row["severity"] or "low"
        status = row["status"]
        count = row["count"]
        if sev not in result:
            result[sev] = {"valid": 0, "invalid": 0, "uncertain": 0, "total": 0}
        if status in ("valid", "invalid", "uncertain"):
            result[sev][status] = count
            result[sev]["total"] = int(result[sev]["total"]) + count

    for d in result.values():
        v = d["valid"]
        total = d["total"]
        d["precision"] = round(int(v) / int(total) * 100) if total else 0

    return result


def _db_run_file_counts(conn: sqlite3.Connection, run_id: int) -> tuple[int, int]:
    """Get (files_analyzed, files_with_findings) for a run."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) as analyzed,
            COUNT(CASE WHEN fa.id IN (
                SELECT DISTINCT file_analysis_id FROM findings
            ) THEN 1 END) as with_findings
        FROM file_analyses fa
        WHERE fa.run_id = ? AND fa.status = 'success'
        """,
        (run_id,),
    ).fetchone()
    return (row["analyzed"], row["with_findings"]) if row else (0, 0)


def extract_kaggle_from_db(conn: sqlite3.Connection) -> KaggleMetrics:
    """Extract Kaggle ground truth metrics from DB."""
    # Import the compare function that already does the heavy lifting
    sys.path.insert(0, str(ROOT))
    from real_world_demo.sources.leakage_paper.compare_ground_truth import (
        compare,
    )

    m = KaggleMetrics(source=f"analysis.db run_id={KAGGLE_RUN_ID}")

    results, excluded = compare(KAGGLE_RUN_ID)
    m.excluded_notebooks = excluded

    analyzed, with_findings = _db_run_file_counts(conn, KAGGLE_RUN_ID)
    m.files_analyzed = analyzed
    m.files_with_findings = with_findings

    for r in results:
        m.labels[r.label] = {
            "tp": r.true_positives,
            "fp": r.false_positives,
            "fn": r.false_negatives,
            "tn": r.true_negatives,
            "precision": r.precision * 100,
            "recall": r.recall * 100,
            "f1": r.f1 * 100,
        }

    return m


def extract_real_world_from_db(
    conn: sqlite3.Connection,
    run_id: int,
    paperset_md: Path,
) -> RealWorldMetrics:
    """Extract real-world metrics from DB + paper set definition."""
    m = RealWorldMetrics(run_id=run_id)
    m.sources = [
        f"analysis.db run_id={run_id}",
        str(paperset_md.relative_to(ROOT)),
    ]

    # Verification stats from DB
    vs = _db_verification_stats(conn, run_id)
    m.valid = vs["valid"]
    m.invalid = vs["invalid"]
    m.uncertain = vs["uncertain"]
    m.total_findings = vs["total"]
    m.precision = vs["precision"]

    m.by_category = _db_verification_by_category(conn, run_id)
    m.by_severity = _db_verification_by_severity(conn, run_id)

    # Paper set metadata from markdown (manually authored)
    m.sc_files, _ = _db_run_file_counts(conn, run_id)
    if paperset_md.exists():
        text = paperset_md.read_text()

        match = re.search(r"Papers sampled\s*\|\s*(\d+)", text)
        if match:
            m.papers_total = int(match.group(1))

        match = re.search(r"Papers with self-contained files\*\*\s*\|\s*\*\*(\d+)", text)
        if match:
            m.papers_with_sc_files = int(match.group(1))

        match = re.search(r"Papers with verified real bugs\*\*\s*\|\s*\*\*(\d+)", text)
        if match:
            m.papers_with_verified_bugs = int(match.group(1))

    return m


# ---------------------------------------------------------------------------
# Report table helpers
# ---------------------------------------------------------------------------


def _append_data_funnel(lines: list[str], m: RealWorldMetrics) -> None:
    """Append data funnel table for a real-world eval set."""
    lines.append("### Data Funnel")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Papers sampled | {m.papers_total} |")
    lines.append(f"| Papers with self-contained files | {m.papers_with_sc_files} |")
    lines.append(f"| Self-contained files | {m.sc_files} |")
    lines.append(f"| Papers with verified real bugs | {m.papers_with_verified_bugs} |")
    lines.append(f"| Total findings | {m.total_findings} |")
    lines.append(f"| Valid | {m.valid} |")
    lines.append(f"| Invalid | {m.invalid} |")
    lines.append(f"| Uncertain | {m.uncertain} |")
    lines.append(f"| **Precision** | **{m.precision:.1f}%** |")
    lines.append("")


def _append_severity_table(
    lines: list[str],
    by_severity: dict[str, dict[str, int | float]],
) -> None:
    """Append severity breakdown table."""
    if not by_severity:
        return
    lines.append("### By Severity")
    lines.append("")
    lines.append("| Severity | Findings | Valid | Invalid | Precision |")
    lines.append("|----------|----------|-------|---------|-----------|")
    for sev in ["critical", "high", "medium", "low"]:
        d = by_severity.get(sev)
        if d:
            cap = sev.capitalize()
            t, v, inv = d["total"], d["valid"], d["invalid"]
            p = d["precision"]
            lines.append(f"| {cap} | {t} | {v} | {inv} | {p}% |")
    lines.append("")


def _append_category_table(
    lines: list[str],
    by_category: dict[str, dict[str, int | float]],
) -> None:
    """Append category breakdown table."""
    if not by_category:
        return
    lines.append("### By Category")
    lines.append("")
    lines.append("| Category | Valid | Invalid | Uncertain | Precision |")
    lines.append("|----------|-------|---------|-----------|-----------|")

    def _sort_key(c: str) -> int | float:
        return by_category[c].get("precision", 0)

    for cat in sorted(by_category, key=_sort_key, reverse=True):
        d = by_category[cat]
        v, inv, unc = d["valid"], d["invalid"], d["uncertain"]
        p = d["precision"]
        lines.append(f"| {cat} | {v} | {inv} | {unc} | {p}% |")
    lines.append("")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report() -> str:
    """Generate unified paper report."""
    lines: list[str] = []
    warnings: list[str] = []

    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    # Get version and git commit
    version = "unknown"
    try:
        from scicode_lint import __version__

        version = __version__
    except ImportError:
        pass

    git_commit = "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except FileNotFoundError:
        pass

    lines.append("# Consolidated Performance Report")
    lines.append("")
    lines.append(f"- **Generated:** {now}")
    lines.append(f"- **scicode-lint version:** {version}")
    lines.append(f"- **Git commit:** `{git_commit}`")
    lines.append("")
    lines.append("Every number in the paper should trace to a row in this report.")
    lines.append("")

    # Check DB connection upfront (needed for source validation)
    conn = _get_db_conn()
    db_available = conn is not None
    if not db_available:
        warnings.append(
            "Database not available — sections 3–5 will be empty. "
            "Expected: real_world_demo/data/analysis.db"
        )

    # Check data sources
    def _src(p: Path) -> str:
        return str(p.relative_to(ROOT))

    db_ok = db_available
    sources_status: list[tuple[str, str, bool]] = [
        ("Controlled tests", _src(JUDGE_REPORT_JSON), JUDGE_REPORT_JSON.exists()),
        ("Integration eval", _src(INTEGRATION_REPORT_JSON), INTEGRATION_REPORT_JSON.exists()),
        ("Integration logs", _src(INTEGRATION_SCENARIOS_DIR), INTEGRATION_SCENARIOS_DIR.exists()),
        ("Analysis DB", _src(ANALYSIS_DB), db_ok),
        ("Feedback paper set", _src(FEEDBACK_PAPERSET_MD), FEEDBACK_PAPERSET_MD.exists()),
        ("Holdout paper set", _src(HOLDOUT_PAPERSET_MD), HOLDOUT_PAPERSET_MD.exists()),
    ]

    # Collect git commits from DB runs
    db_commits: dict[str, str] = {}
    if db_available:
        assert conn is not None
        for label, run_id in [
            ("Kaggle (run 57)", KAGGLE_RUN_ID),
            ("Feedback (run 56)", FEEDBACK_RUN_ID),
            ("Holdout (run 60)", HOLDOUT_RUN_ID),
        ]:
            row = conn.execute(
                "SELECT git_commit FROM analysis_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row and row["git_commit"]:
                db_commits[label] = row["git_commit"]

    lines.append("## Data Sources")
    lines.append("")
    lines.append("| Source | File | Status |")
    lines.append("|--------|------|--------|")
    for name, path, exists in sources_status:
        status = "OK" if exists else "**MISSING**"
        lines.append(f"| {name} | `{path}` | {status} |")
        if not exists:
            warnings.append(f"MISSING: {path}")

    # Show git commits from source data
    if db_commits:
        lines.append("")
        lines.append("### Source Data Git Commits")
        lines.append("")
        unique_commits = set(db_commits.values())
        if len(unique_commits) == 1:
            lines.append(f"All DB runs at commit `{unique_commits.pop()}`.")
        else:
            lines.append("**WARNING: DB runs were generated at different commits:**")
            lines.append("")
            for label, commit in db_commits.items():
                lines.append(f"- {label}: `{commit}`")
            warnings.append(
                "DB runs were generated at different git commits "
                f"({', '.join(sorted(unique_commits))})"
            )
    lines.append("")

    # Validate run IDs exist
    if db_available:
        assert conn is not None
        for label, run_id in [
            ("Kaggle", KAGGLE_RUN_ID),
            ("Feedback", FEEDBACK_RUN_ID),
            ("Holdout", HOLDOUT_RUN_ID),
        ]:
            row = conn.execute("SELECT id FROM analysis_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                warnings.append(f"Run ID {run_id} ({label}) not found in DB")

    # -----------------------------------------------------------------------
    # 01 — Controlled tests (JSON)
    # -----------------------------------------------------------------------
    ct = extract_controlled_tests()

    lines.append("## 1. Controlled Tests (LLM-as-Judge)")
    lines.append("")
    lines.append(f"Source: `{ct.source}`")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Patterns | {ct.total_patterns} |")
    lines.append(f"| Total tests | {ct.total_tests} |")
    lines.append(f"| **Overall accuracy** | **{ct.overall_accuracy:.2f}%** |")
    lines.append(f"| Positive accuracy | {ct.positive_accuracy:.2f}% |")
    lines.append(f"| Negative accuracy | {ct.negative_accuracy:.2f}% |")
    lines.append(f"| Context-dependent accuracy | {ct.context_accuracy:.1f}% |")
    lines.append(f"| Semantic alignment | {ct.semantic_alignment:.1f}% |")
    lines.append(f"| Quality issue rate | {ct.quality_issue_rate:.1f}% |")
    lines.append(f"| Patterns at 100% | {ct.patterns_at_100}/{ct.total_patterns} |")
    lines.append(
        f"| Focus line accuracy | {ct.focus_line_accuracy:.1f}% "
        f"({ct.focus_line_matched}/{ct.focus_line_eligible}) |"
    )
    lines.append("")

    # -----------------------------------------------------------------------
    # 02 — Integration eval (JSON + logs)
    # -----------------------------------------------------------------------
    ig = extract_integration()
    total_tp = ig.tp_intended + ig.tp_bonus

    lines.append("## 2. Integration Evaluation (Generated Scenarios)")
    lines.append("")
    lines.append(f"Source: `{ig.source}` + `{_INTEG_DIR}/scenarios/*.log`")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Scenarios | {ig.scenarios} |")
    lines.append(f"| Bugs intended | {ig.bugs_intended} |")
    lines.append(f"| TP-intended (expected bugs found) | {ig.tp_intended} |")
    lines.append(f"| TP-bonus (verified extra bugs) | {ig.tp_bonus} |")
    lines.append(f"| False positives | {ig.total_fp} |")
    lines.append(f"| False negatives | {ig.total_fn} |")
    lines.append(f"| Total TP | {total_tp} |")
    lines.append(f"| **Precision** | **{ig.precision:.1%}** |")
    lines.append(f"| **Recall** | **{ig.recall:.1%}** |")
    lines.append(f"| **F1** | **{ig.f1:.1%}** |")
    lines.append("")

    # -----------------------------------------------------------------------
    # 03 — Kaggle labeled (DB)
    # -----------------------------------------------------------------------
    lines.append("## 3. Kaggle Labeled Notebooks (Yang et al. ASE'22)")
    lines.append("")

    if db_available:
        assert conn is not None
        kg = extract_kaggle_from_db(conn)
        lines.append(f"Source: `{kg.source}`")
        lines.append("")
        lines.append(f"- Files analyzed: {kg.files_analyzed}")
        lines.append(f"- Files with findings: {kg.files_with_findings}")
        lines.append(f"- Excluded (timeouts): {kg.excluded_notebooks}")
        lines.append("")
        lines.append("| Label | TP | FP | FN | TN | Precision | Recall | F1 |")
        lines.append("|-------|---:|---:|---:|---:|----------:|-------:|---:|")
        for label in ["pre", "overlap", "multi"]:
            d = kg.labels.get(label, {})
            if d:
                lines.append(
                    f"| {label} | {d['tp']:.0f} | {d['fp']:.0f} "
                    f"| {d['fn']:.0f} | {d['tn']:.0f} "
                    f"| {d['precision']:.1f}% | {d['recall']:.1f}% "
                    f"| {d['f1']:.1f}% |"
                )
        lines.append("")
    else:
        lines.append("*Database not available — skipped.*")
        lines.append("")

    # -----------------------------------------------------------------------
    # 04 — PapersWithCode feedback (DB)
    # -----------------------------------------------------------------------
    lines.append("## 4. PapersWithCode — Feedback Set")
    lines.append("")

    if db_available:
        assert conn is not None
        fb = extract_real_world_from_db(conn, FEEDBACK_RUN_ID, FEEDBACK_PAPERSET_MD)
        lines.append("Sources:")
        for sf in fb.sources:
            lines.append(f"- `{sf}`")
        lines.append("")
        _append_data_funnel(lines, fb)
        _append_severity_table(lines, fb.by_severity)
        _append_category_table(lines, fb.by_category)
    else:
        fb = RealWorldMetrics()
        lines.append("*Database not available — skipped.*")
        lines.append("")

    # -----------------------------------------------------------------------
    # 05 — PapersWithCode holdout (DB)
    # -----------------------------------------------------------------------
    lines.append("## 5. PapersWithCode — Holdout Set")
    lines.append("")

    if db_available:
        assert conn is not None
        ho = extract_real_world_from_db(conn, HOLDOUT_RUN_ID, HOLDOUT_PAPERSET_MD)
        lines.append("Sources:")
        for sf in ho.sources:
            lines.append(f"- `{sf}`")
        lines.append("")
        _append_data_funnel(lines, ho)
        _append_severity_table(lines, ho.by_severity)
        _append_category_table(lines, ho.by_category)
    else:
        ho = RealWorldMetrics()
        lines.append("*Database not available — skipped.*")
        lines.append("")

    # -----------------------------------------------------------------------
    # Generalization gap
    # -----------------------------------------------------------------------
    if fb.precision > 0 and ho.precision > 0:
        gap = fb.precision - ho.precision
        lines.append("### Generalization Gap (Feedback vs Holdout)")
        lines.append("")
        lines.append(f"- Feedback precision: {fb.precision:.1f}%")
        lines.append(f"- Holdout precision: {ho.precision:.1f}%")
        lines.append(f"- Gap: {gap:.1f} percentage points")
        lines.append("")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    lines.append("## Summary Table (for paper)")
    lines.append("")
    lines.append("| Layer | Precision | Recall | Key Detail |")
    lines.append("|-------|-----------|--------|------------|")
    lines.append(
        f"| Controlled tests | — | — | "
        f"{ct.total_tests} tests, {ct.total_patterns} patterns, "
        f"{ct.overall_accuracy:.1f}% overall accuracy |"
    )
    lines.append(
        f"| Integration (n={ig.scenarios}) | {ig.precision:.1%} "
        f"| {ig.recall:.1%} | "
        f"{ig.bugs_intended} intended bugs, {ig.tp_bonus} bonus TPs |"
    )
    if db_available and "pre" in kg.labels:
        pre = kg.labels["pre"]
        lines.append(
            f"| Kaggle labeled (`pre`) | {pre['precision']:.1f}% "
            f"| {pre['recall']:.1f}% | "
            f"Human ground truth (Yang et al. ASE'22) |"
        )
    lines.append(
        f"| PapersWithCode (feedback) | {fb.precision:.1f}% | — | "
        f"{fb.papers_total} papers, {fb.sc_files} files |"
    )
    lines.append(
        f"| PapersWithCode (holdout) | {ho.precision:.1f}% | — | "
        f"{ho.papers_total} papers, {ho.sc_files} files |"
    )
    lines.append("")

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    if conn:
        conn.close()

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate unified paper results report")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "consolidated_results" / "CONSOLIDATED_REPORT.md",
        help="Output file path",
    )
    args = parser.parse_args()

    report = generate_report()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Report written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
