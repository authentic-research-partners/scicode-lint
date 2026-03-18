"""Generate markdown report from SQLite analysis database.

Creates a committable report with statistics and example findings
from real-world scientific ML code analysis.
"""

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .config import REPORTS_DIR
from .database import (
    get_db_path,
    get_prefilter_summary,
    get_run_stats,
    get_scan_stats,
    get_verification_stats,
    init_db,
    list_runs,
)
from .report_critical import generate_valid_critical_report
from .report_queries import (
    get_example_findings,
    get_findings_distribution,
    get_papers_by_severity,
    get_top_patterns,
    get_verification_by_severity,
)
from .sources.leakage_paper.compare_ground_truth import compare as compare_ground_truth

# Re-export for backward compatibility (tests import from this module)
__all__ = ["generate_markdown_report", "get_example_findings", "main", "save_report"]


def generate_markdown_report(
    conn: sqlite3.Connection, run_id: int | None = None, *, verified_only: bool = False
) -> str:
    """Generate markdown report for an analysis run.

    Args:
        conn: Database connection.
        run_id: Specific run ID, or None for latest.
        verified_only: If True, only include findings with completed verification.

    Returns:
        Markdown report string.
    """
    stats = get_run_stats(conn, run_id)
    if stats.run_id == 0:
        return "# No Analysis Data\n\nNo analysis runs found in database.\n"

    run_id = stats.run_id
    examples = get_example_findings(conn, run_id, verified_only=verified_only)
    top_patterns = get_top_patterns(conn, run_id, verified_only=verified_only)
    papers_by_severity = get_papers_by_severity(conn, run_id, verified_only=verified_only)
    verification_by_severity = get_verification_by_severity(conn, run_id)
    verification_stats = get_verification_stats(conn, run_id)
    findings_distribution = get_findings_distribution(conn, run_id, verified_only=verified_only)
    scan_stats = get_scan_stats(conn)
    prefilter_summary = get_prefilter_summary(conn, run_id)

    lines: list[str] = []

    # Header with data source-specific description
    title = "# Real-World Scientific ML Code Analysis Report"
    if verified_only:
        title += " (Verified Findings Only)"
    lines.append(title)
    lines.append("")
    if stats.data_source == "leakage_paper":
        lines.append(
            "Analysis of 100 Python files with **ground truth labels** for data leakage issues, "
            "from Yang et al. ASE'22 study. scicode-lint findings are compared against "
            "human-verified labels to measure detection accuracy."
        )
    else:
        lines.append(
            "Analysis of Python code from **AI applications to scientific domains**. "
            "Papers sourced from PapersWithCode, filtered to include only scientific domains "
            "(biology, chemistry, physics, medicine, earth science, astronomy, materials, etc.) "
            "where ML/AI is applied to scientific discovery and domain-specific research."
        )
    lines.append("")

    # Run metadata
    lines.append("## Analysis Summary")
    lines.append("")
    import scicode_lint

    lines.append(f"- **Analysis Date:** {stats.run_date.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- **scicode-lint Version:** {scicode_lint.__version__}")
    # Skip papers/repos stats for leakage_paper (single source, not meaningful)
    if stats.data_source != "leakage_paper":
        if stats.total_papers > 0:
            paper_pct = 100 * stats.papers_with_findings / stats.total_papers
            lines.append(
                f"- **Papers with Findings:** {stats.papers_with_findings:,} / "
                f"{stats.total_papers:,} ({paper_pct:.1f}%)"
            )
        if stats.total_repos > 0:
            repo_pct = 100 * stats.repos_with_findings / stats.total_repos
            lines.append(
                f"- **Repos with Findings:** {stats.repos_with_findings:,} / "
                f"{stats.total_repos:,} ({repo_pct:.1f}%)"
            )
    lines.append(f"- **Files Analyzed:** {stats.analyzed_files:,} / {stats.total_files:,}")
    finding_pct = stats.finding_rate
    lines.append(f"- **Files with Findings:** {stats.files_with_findings:,} ({finding_pct:.1f}%)")
    lines.append(f"- **Total Findings:** {stats.total_findings:,}")

    # Ground truth comparison (leakage_paper only)
    gt_results: list[Any] | None = None
    gt_excluded = 0
    if stats.data_source == "leakage_paper":
        gt_results, gt_excluded = compare_ground_truth(run_id)
        if gt_results:
            _append_ground_truth_summary(lines, gt_results, gt_excluded)
    lines.append("")

    if gt_results:
        _append_ground_truth_table(lines, gt_results, gt_excluded)

    # File filtering pipeline stats (if scans were performed)
    if scan_stats.total_scans > 0:
        _append_file_filtering_pipeline(lines, scan_stats)

    # Prefilter summary (if prefilter was used)
    if prefilter_summary:
        _append_prefilter_summary(lines, prefilter_summary)

    # Papers by severity (skip for leakage_paper - single source)
    show_papers_by_severity = (
        stats.data_source != "leakage_paper"
        and stats.total_papers > 0
        and any(papers_by_severity.values())
    )
    if show_papers_by_severity:
        _append_papers_by_severity(lines, papers_by_severity, stats.total_papers)

    # Findings distribution per paper (skip for leakage_paper - single source)
    show_dist = (
        stats.data_source != "leakage_paper"
        and findings_distribution
        and findings_distribution["per_paper"].count > 0
    )
    if show_dist:
        _append_findings_distribution(lines, findings_distribution)

    # Verification summary (if any verifications exist)
    total_verified = verification_stats.get("total_verified", 0)
    total_pending = sum(v["pending"] for v in verification_by_severity.values())
    if total_verified > 0 or total_pending > 0:
        _append_verification_summary(
            lines, verification_stats, verification_by_severity, total_verified, total_pending
        )

    # Findings by domain (skip for leakage_paper - all data_science)
    if stats.data_source != "leakage_paper" and stats.by_domain:
        _append_findings_by_domain(lines, stats.by_domain)

    # Findings by category
    if stats.by_category:
        _append_findings_by_category(lines, stats.by_category, stats.data_source)

    # Findings by severity (as table with percentages)
    if stats.by_severity:
        _append_findings_by_severity(lines, stats.by_severity, stats.total_findings)

    # Top patterns
    if top_patterns:
        _append_top_patterns(lines, top_patterns, stats.data_source)

    # Example findings
    if examples:
        _append_example_findings(lines, examples)

    # Footer
    lines.append("---")
    lines.append("")
    analysis_date = stats.run_date.strftime("%Y-%m-%d")
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"*Analysis conducted: {analysis_date} | Report generated: {report_date}*")
    lines.append("")

    return "\n".join(lines)


def _append_ground_truth_summary(lines: list[str], gt_results: list[Any], gt_excluded: int) -> None:
    """Append aggregate ground truth metrics to the summary section."""
    total_tp = sum(r.true_positives for r in gt_results)
    total_fp = sum(r.false_positives for r in gt_results)
    total_fn = sum(r.false_negatives for r in gt_results)
    total_tn = sum(r.true_negatives for r in gt_results)
    agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0.0
    total_evaluated = total_tp + total_fp + total_fn + total_tn
    agg_acc = (total_tp + total_tn) / total_evaluated if total_evaluated > 0 else 0.0
    lines.append(
        f"- **Ground Truth (aggregate):** "
        f"Precision {agg_prec:.0%}, Recall {agg_rec:.0%}, "
        f"F1 {agg_f1:.0%}, Accuracy {agg_acc:.0%}"
    )
    if gt_excluded > 0:
        lines.append(f"- **Excluded:** {gt_excluded} notebooks (pattern timeouts)")


def _append_ground_truth_table(lines: list[str], gt_results: list[Any], gt_excluded: int) -> None:
    """Append detailed ground truth comparison tables."""
    lines.append("## Ground Truth Comparison")
    lines.append("")
    lines.append(
        "Comparison against Yang et al. ASE'22 manual labels "
        "(notebooks with pattern timeouts are excluded)."
    )
    if gt_excluded > 0:
        lines.append(f" Excluded: {gt_excluded} notebooks due to timeouts.")
    lines.append("")

    lines.append("| Label | TP | FP | FN | TN | Precision | Recall | F1 |")
    lines.append("|-------|---:|---:|---:|---:|----------:|-------:|---:|")
    for r in gt_results:
        lines.append(
            f"| {r.label} | {r.true_positives} | {r.false_positives} | "
            f"{r.false_negatives} | {r.true_negatives} | "
            f"{r.precision:.1%} | {r.recall:.1%} | {r.f1:.1%} |"
        )
    lines.append("")

    lines.append("| Label | Correct | Total | Accuracy |")
    lines.append("|-------|--------:|------:|---------:|")
    for r in gt_results:
        total = r.true_positives + r.false_positives + r.false_negatives + r.true_negatives
        if total > 0:
            correct = r.true_positives + r.true_negatives
            lines.append(f"| {r.label} | {correct} | {total} | {correct / total:.1%} |")
    lines.append("")


def _append_file_filtering_pipeline(lines: list[str], scan_stats: Any) -> None:
    """Append file filtering pipeline stats."""
    lines.append("## File Filtering Pipeline")
    lines.append("")
    lines.append("Two stages: ML import presence check -> LLM classification:")
    lines.append("")
    has_ml_imports = scan_stats.total_self_contained + scan_stats.total_fragments
    no_ml_imports = scan_stats.total_skipped
    lines.append("```")
    lines.append(f"Total Python files:       {scan_stats.total_files_scanned:,}")
    lines.append("  |")
    lines.append(f"  |- Has ML imports:      {has_ml_imports:,}  (sent to LLM)")
    lines.append("  |    |")
    lines.append(f"  |    |- Self-contained: {scan_stats.total_self_contained:,}  <- analyzed")
    lines.append(f"  |    |- Fragments:      {scan_stats.total_fragments:,}  <- skipped")
    lines.append("  |")
    lines.append(f"  |- No ML imports:       {no_ml_imports:,}  <- skipped")
    lines.append("```")
    lines.append("")
    if scan_stats.total_files_scanned > 0:
        filter_rate = 100 * scan_stats.total_self_contained / scan_stats.total_files_scanned
        lines.append(
            f"*{filter_rate:.1f}% of files passed both filters and were analyzed. "
            f"Avg {scan_stats.avg_self_contained_per_repo:.1f} self-contained files per repo.*"
        )
        lines.append("")


def _append_prefilter_summary(lines: list[str], prefilter_summary: dict[str, Any]) -> None:
    """Append prefilter classification summary."""
    lines.append("## Prefilter Summary")
    lines.append("")
    lines.append(
        "Files were classified by LLM as self-contained ML pipelines vs code fragments. "
        "Only self-contained files (complete training/inference workflows) were analyzed."
    )
    lines.append("")

    # File classification breakdown
    total = prefilter_summary["total_files"]
    self_contained = prefilter_summary["self_contained"]
    fragments = prefilter_summary["fragments"]
    uncertain = prefilter_summary["uncertain"]
    errors = prefilter_summary["errors"]

    lines.append("| Classification | Files | % |")
    lines.append("|----------------|-------|---|")
    lines.append(
        f"| Self-contained (analyzed) | {self_contained:,} | {100 * self_contained / total:.1f}% |"
    )
    lines.append(f"| Fragment (skipped) | {fragments:,} | {100 * fragments / total:.1f}% |")
    if uncertain > 0:
        lines.append(f"| Uncertain | {uncertain:,} | {100 * uncertain / total:.1f}% |")
    if errors > 0:
        lines.append(f"| Error | {errors:,} | {100 * errors / total:.1f}% |")
    lines.append(f"| **Total** | **{total:,}** | |")
    lines.append("")

    # Paper/repo impact
    orig_papers = prefilter_summary["original_papers"]
    filt_papers = prefilter_summary["filtered_papers"]
    orig_repos = prefilter_summary["original_repos"]
    filt_repos = prefilter_summary["filtered_repos"]
    dropped_repos = prefilter_summary["dropped_repos"]

    if orig_papers > filt_papers:
        lines.append("### Papers/Repos Filtered")
        lines.append("")
        lines.append(
            f"- **Papers:** {filt_papers} analyzed / {orig_papers} original "
            f"({orig_papers - filt_papers} dropped)"
        )
        lines.append(
            f"- **Repos:** {filt_repos} analyzed / {orig_repos} original "
            f"({orig_repos - filt_repos} dropped)"
        )
        lines.append("")

        if dropped_repos:
            lines.append(
                f"**{len(dropped_repos)} repos dropped** (all files classified as fragments):"
            )
            lines.append("")
            for repo in dropped_repos[:10]:  # Limit to 10
                lines.append(f"- `{repo}`")
            if len(dropped_repos) > 10:
                lines.append(f"- ... and {len(dropped_repos) - 10} more")
            lines.append("")

    if prefilter_summary["model_name"]:
        lines.append(f"*Prefilter model: {prefilter_summary['model_name']}*")
        lines.append("")


def _append_papers_by_severity(
    lines: list[str], papers_by_severity: dict[str, int], total_papers: int
) -> None:
    """Append papers-by-severity table."""
    lines.append("## Papers by Severity")
    lines.append("")
    lines.append(
        "Papers with at least one finding of each severity level "
        "(a paper may appear in multiple rows):"
    )
    lines.append("")
    lines.append("| Severity | Papers | % of Papers Analyzed |")
    lines.append("|----------|--------|----------------------|")
    for sev in ["critical", "high", "medium", "low"]:
        count = papers_by_severity.get(sev, 0)
        if count > 0:
            pct = 100 * count / total_papers if total_papers > 0 else 0
            lines.append(f"| {sev.capitalize()} | {count:,} | {pct:.1f}% |")
    lines.append("")
    lines.append(f"*Total papers analyzed: {total_papers:,}*")
    lines.append("")


def _append_findings_distribution(lines: list[str], findings_distribution: dict[str, Any]) -> None:
    """Append findings-distribution-per-paper table."""
    lines.append("## Findings Distribution (per paper)")
    lines.append("")
    lines.append("| Metric | All | Critical | High | Medium | Low |")
    lines.append("|--------|-----|----------|------|--------|-----|")

    dist = findings_distribution

    # Papers with findings row
    papers_row = f"| Papers | {dist['per_paper'].count}"
    for sev in ["critical", "high", "medium", "low"]:
        papers_row += f" | {dist[sev].count}"
    papers_row += " |"
    lines.append(papers_row)

    # Min row
    min_row = f"| Min | {dist['per_paper'].min}"
    for sev in ["critical", "high", "medium", "low"]:
        min_row += f" | {dist[sev].min}" if dist[sev].count > 0 else " | -"
    min_row += " |"
    lines.append(min_row)

    # Max row
    max_row = f"| Max | {dist['per_paper'].max}"
    for sev in ["critical", "high", "medium", "low"]:
        max_row += f" | {dist[sev].max}" if dist[sev].count > 0 else " | -"
    max_row += " |"
    lines.append(max_row)

    # Mean row
    mean_row = f"| Mean | {dist['per_paper'].mean:.1f}"
    for sev in ["critical", "high", "medium", "low"]:
        mean_row += f" | {dist[sev].mean:.1f}" if dist[sev].count > 0 else " | -"
    mean_row += " |"
    lines.append(mean_row)

    # Median row
    median_row = f"| Median | {dist['per_paper'].median:.1f}"
    for sev in ["critical", "high", "medium", "low"]:
        median_row += f" | {dist[sev].median:.1f}" if dist[sev].count > 0 else " | -"
    median_row += " |"
    lines.append(median_row)

    # Std dev row
    std_row = f"| Std Dev | {dist['per_paper'].stdev:.1f}"
    for sev in ["critical", "high", "medium", "low"]:
        std_row += f" | {dist[sev].stdev:.1f}" if dist[sev].count > 1 else " | -"
    std_row += " |"
    lines.append(std_row)

    lines.append("")


def _append_verification_summary(
    lines: list[str],
    verification_stats: dict[str, Any],
    verification_by_severity: dict[str, dict[str, int]],
    total_verified: int,
    total_pending: int,
) -> None:
    """Append verification summary section."""
    lines.append("## Verification Summary")
    lines.append("")

    if total_verified > 0:
        valid = verification_stats.get("valid", 0)
        invalid = verification_stats.get("invalid", 0)
        uncertain = verification_stats.get("uncertain", 0)
        precision = verification_stats.get("precision", 0)
        prec_str = f"{precision:.1f}% ({valid:,} valid / {total_verified:,} verified)"
        lines.append(f"**Overall Precision:** {prec_str}")
        lines.append("")
        lines.append("| Status | Count | % |")
        lines.append("|--------|-------|---|")
        valid_pct = 100 * valid / total_verified
        invalid_pct = 100 * invalid / total_verified
        uncertain_pct = 100 * uncertain / total_verified
        lines.append(f"| Valid (confirmed) | {valid:,} | {valid_pct:.1f}% |")
        lines.append(f"| Invalid (false positive) | {invalid:,} | {invalid_pct:.1f}% |")
        lines.append(f"| Uncertain | {uncertain:,} | {uncertain_pct:.1f}% |")
        if total_pending > 0:
            lines.append(f"| Pending verification | {total_pending:,} | - |")
        lines.append("")

    elif total_pending > 0:
        lines.append(f"**{total_pending:,} findings pending verification.**")
        lines.append("")

    # Verification by severity table
    if total_verified > 0:
        lines.append("### Verified Findings by Severity")
        lines.append("")
        lines.append("| Severity | Total | Valid | Invalid | Uncertain | Pending | Precision |")
        lines.append("|----------|-------|-------|---------|-----------|---------|-----------|")
        for sev in ["critical", "high", "medium", "low"]:
            v = verification_by_severity.get(sev, {})
            total = v.get("total", 0)
            if total == 0:
                continue
            valid = v.get("valid", 0)
            invalid = v.get("invalid", 0)
            uncertain = v.get("uncertain", 0)
            pending = v.get("pending", 0)
            verified = valid + invalid + uncertain
            prec = f"{100 * valid / verified:.0f}%" if verified > 0 else "-"
            sev_cap = sev.capitalize()
            row = f"| {sev_cap} | {total:,} | {valid:,} | {invalid:,} "
            row += f"| {uncertain:,} | {pending:,} | {prec} |"
            lines.append(row)
        lines.append("")


def _append_findings_by_domain(lines: list[str], by_domain: list[Any]) -> None:
    """Append findings-by-domain table."""
    lines.append("## Findings by Scientific Domain")
    lines.append("")
    lines.append("| Domain | Files Analyzed | With Findings | Finding Rate | Total Findings |")
    lines.append("|--------|---------------|---------------|--------------|----------------|")
    for d in sorted(by_domain, key=lambda x: x.total_findings, reverse=True):
        lines.append(
            f"| {d.domain} | {d.analyzed_files:,} | {d.files_with_findings:,} | "
            f"{d.finding_rate:.1f}% | {d.total_findings:,} |"
        )
    lines.append("")


def _append_findings_by_category(
    lines: list[str], by_category: list[Any], data_source: str
) -> None:
    """Append findings-by-category table."""
    lines.append("## Findings by Category")
    lines.append("")
    # Skip Unique Repos for leakage_paper (always 1)
    if data_source == "leakage_paper":
        lines.append("| Category | Count | Unique Files |")
        lines.append("|----------|-------|--------------|")
        for c in by_category:
            row = f"| {c.category} | {c.count:,} | {c.unique_files:,} |"
            lines.append(row)
    else:
        lines.append("| Category | Count | Unique Files | Unique Repos |")
        lines.append("|----------|-------|--------------|--------------|")
        for c in by_category:
            row = f"| {c.category} | {c.count:,} | {c.unique_files:,} | {c.unique_repos:,} |"
            lines.append(row)
    lines.append("")


def _append_findings_by_severity(
    lines: list[str], by_severity: dict[str, int], total_findings: int
) -> None:
    """Append findings-by-severity table."""
    lines.append("## Findings by Severity")
    lines.append("")
    lines.append("| Severity | Count | % of Total |")
    lines.append("|----------|-------|------------|")
    for sev in ["critical", "high", "medium", "low"]:
        if sev in by_severity:
            count = by_severity[sev]
            pct = 100 * count / total_findings if total_findings > 0 else 0
            lines.append(f"| {sev.capitalize()} | {count:,} | {pct:.1f}% |")
    lines.append("")


def _append_top_patterns(
    lines: list[str], top_patterns: list[dict[str, Any]], data_source: str
) -> None:
    """Append most-common-patterns table."""
    lines.append("## Most Common Patterns")
    lines.append("")
    # Simplified columns for leakage_paper (no repos/confidence)
    if data_source == "leakage_paper":
        lines.append("| Pattern | Category | Count | Files |")
        lines.append("|---------|----------|-------|-------|")
        for p in top_patterns:
            lines.append(
                f"| {p['pattern_id']} | {p['category']} | {p['count']:,} | {p['unique_files']:,} |"
            )
    else:
        lines.append("| Pattern | Category | Count | Files | Repos | Avg Confidence |")
        lines.append("|---------|----------|-------|-------|-------|----------------|")
        for p in top_patterns:
            conf = p["avg_confidence"] or 0
            lines.append(
                f"| {p['pattern_id']} | {p['category']} | {p['count']:,} | "
                f"{p['unique_files']:,} | {p['unique_repos']:,} | {conf:.0%} |"
            )
    lines.append("")


def _append_example_findings(lines: list[str], examples: dict[str, list[dict[str, Any]]]) -> None:
    """Append example findings section."""
    lines.append("## Example Findings")
    lines.append("")
    lines.append("Representative findings from each category (with links to source):")
    lines.append("")

    for category, findings in examples.items():
        if not findings:
            continue

        lines.append(f"### {category}")
        lines.append("")

        for f in findings:
            conf = f["confidence"]
            header = f"**{f['pattern_id']}** ({f['severity']}, {conf:.0%} confidence)"
            lines.append(header)
            lines.append("")

            # Build GitHub link to file with line numbers
            repo_url = f["repo_url"] or ""
            original_path = f["original_path"] or ""
            finding_lines = f["lines"] or "[]"

            if repo_url and original_path:
                # Parse line numbers from JSON string
                try:
                    line_nums = json.loads(finding_lines) if finding_lines else []
                except (json.JSONDecodeError, TypeError):
                    line_nums = []

                # Build GitHub blob URL
                github_url = f"{repo_url}/blob/main/{original_path}"
                if line_nums:
                    if len(line_nums) == 1:
                        github_url += f"#L{line_nums[0]}"
                    else:
                        github_url += f"#L{min(line_nums)}-L{max(line_nums)}"

                file_link = f"[{original_path}]({github_url})"
                lines.append(f"- **File:** {file_link} in [{f['repo_name']}]({repo_url})")
            else:
                lines.append(f"- **Repo:** {f['repo_name']} ({f['domain']})")

            # Add function/class location if available
            if f.get("location_name"):
                loc_type = f.get("location_type") or "function"
                loc_info = f"- **Location:** {loc_type} `{f['location_name']}`"
                if f.get("focus_line"):
                    loc_info += f" (line {f['focus_line']})"
                lines.append(loc_info)

            if f["paper_title"]:
                paper_ref = f["paper_title"]
                if f["arxiv_id"]:
                    paper_ref += (
                        f" ([arXiv:{f['arxiv_id']}](https://arxiv.org/abs/{f['arxiv_id']}))"
                    )
                lines.append(f"- **Paper:** {paper_ref}")

                # Add authors if available
                if f.get("paper_authors"):
                    try:
                        authors = json.loads(f["paper_authors"])
                        if authors:
                            if len(authors) > 2:
                                authors_str = f"{authors[0]} et al."
                            else:
                                authors_str = ", ".join(authors)
                            lines.append(f"- **Authors:** {authors_str}")
                    except (json.JSONDecodeError, TypeError):
                        pass
            lines.append(f"- **Issue:** {f['issue']}")
            if f["explanation"]:
                lines.append(f"- **Explanation:** {f['explanation']}")
            if f["suggestion"]:
                lines.append(f"- **Suggestion:** {f['suggestion']}")
            if f["snippet"]:
                lines.append("")
                lines.append("```python")
                lines.append(f["snippet"].strip())
                lines.append("```")
            lines.append("")


def save_report(
    content: str,
    analysis_date: datetime,
    output_path: Path | None = None,
    data_source: str | None = None,
) -> Path:
    """Save report to file.

    Args:
        content: Markdown content.
        analysis_date: Date when analysis was conducted.
        output_path: Output path, or None for default (includes date).
        data_source: Data source name for subdirectory (e.g., 'leakage_paper').

    Returns:
        Path to saved file.
    """
    if output_path is None:
        date_str = analysis_date.strftime("%Y-%m-%d_%H%M")
        base_dir = REPORTS_DIR / data_source if data_source else REPORTS_DIR
        output_path = base_dir / f"FINDINGS_REPORT_{date_str}.md"

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text(content)
    return output_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate markdown report from analysis database")
    parser.add_argument("--run-id", type=int, help="Specific run ID (default: latest)")
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    parser.add_argument("--db", type=Path, help="Database path (default: data/analysis.db)")
    parser.add_argument(
        "--data-source", type=str, help="Filter by data source (e.g., 'leakage_paper')"
    )
    parser.add_argument("--list-runs", action="store_true", help="List available runs and exit")
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Only include findings with completed verification",
    )
    args = parser.parse_args()

    # Connect to database
    db_path = args.db or get_db_path()
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Run the pipeline first: python -m real_world_demo.run_pipeline --pilot 100")
        raise SystemExit(1)

    conn = init_db(db_path)

    # List runs mode
    if args.list_runs:
        runs = list_runs(conn, data_source=args.data_source)
        if not runs:
            msg = "No analysis runs found"
            if args.data_source:
                msg += f" for data_source='{args.data_source}'"
            logger.info(msg)
        else:
            logger.info("Available runs:")
            for run in runs:
                status = run.status.value if run.status else "unknown"
                logger.info(
                    f"  #{run.run_id}: [{run.data_source}] "
                    f"{run.started_at.strftime('%Y-%m-%d %H:%M')} - "
                    f"{run.analyzed_files}/{run.total_files} files, "
                    f"{run.total_findings} findings ({status})"
                )
        conn.close()
        return

    # Generate report
    logger.info(f"Generating report from {db_path}")

    # Get stats first to extract analysis date
    stats = get_run_stats(conn, args.run_id)
    if stats.run_id == 0:
        logger.error("No analysis runs found in database")
        conn.close()
        raise SystemExit(1)

    report = generate_markdown_report(conn, args.run_id, verified_only=args.verified_only)

    # Save with analysis date in filename, organized by data source
    data_source = stats.data_source
    output_path = save_report(report, stats.run_date, args.output, data_source=data_source)
    logger.info(f"Report saved to {output_path}")

    # Generate valid critical findings sample when using --verified-only
    if args.verified_only:
        critical_report = generate_valid_critical_report(conn, stats.run_id)
        date_str = stats.run_date.strftime("%Y-%m-%d_%H%M")
        base_dir = REPORTS_DIR / data_source if data_source else REPORTS_DIR
        critical_path = base_dir / f"VALID_FINDINGS_SAMPLE_{date_str}.md"
        critical_path.write_text(critical_report)
        logger.info(f"Valid findings sample saved to {critical_path}")

    conn.close()


if __name__ == "__main__":
    main()
