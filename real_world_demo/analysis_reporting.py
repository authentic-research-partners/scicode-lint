"""Reporting and aggregation functions for analysis results.

Extracts findings statistics, generates markdown reports, saves raw JSON,
and prints console summaries.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


def aggregate_findings(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate findings across all analyzed files.

    Args:
        results: List of analysis results.

    Returns:
        Aggregated statistics dict.
    """
    stats: dict[str, Any] = {
        "total_files": len(results),
        "analyzed_successfully": sum(1 for r in results if r["success"]),
        "files_with_findings": 0,
        "total_findings": 0,
        "findings_by_pattern": defaultdict(int),
        "findings_by_category": defaultdict(int),
        "findings_by_domain": defaultdict(int),
        "papers_with_findings": set(),
        "repos_with_findings": set(),
        "domain_summary": {},
    }

    for result in results:
        if not result["success"]:
            continue

        findings = result.get("findings", [])
        if findings:
            stats["files_with_findings"] += 1
            stats["total_findings"] += len(findings)

            domain = result.get("domain", "unknown")
            repo_name = result.get("repo_name", "")
            paper_url = result.get("paper_url", "")

            if paper_url:
                stats["papers_with_findings"].add(paper_url)
            if repo_name:
                stats["repos_with_findings"].add(repo_name)

            for finding in findings:
                pattern_id = finding.get("pattern_id", "unknown")
                category = finding.get("category", "unknown")

                stats["findings_by_pattern"][pattern_id] += 1
                stats["findings_by_category"][category] += 1
                stats["findings_by_domain"][domain] += 1

    # Convert sets to counts
    stats["papers_with_findings"] = len(stats["papers_with_findings"])
    stats["repos_with_findings"] = len(stats["repos_with_findings"])

    # Convert defaultdicts to regular dicts for JSON serialization
    stats["findings_by_pattern"] = dict(stats["findings_by_pattern"])
    stats["findings_by_category"] = dict(stats["findings_by_category"])
    stats["findings_by_domain"] = dict(stats["findings_by_domain"])

    # Calculate domain-specific statistics
    domain_files: dict[str, int] = defaultdict(int)
    domain_files_with_findings: dict[str, int] = defaultdict(int)

    for result in results:
        if result["success"]:
            domain = result.get("domain", "unknown")
            domain_files[domain] += 1
            if result.get("findings"):
                domain_files_with_findings[domain] += 1

    stats["domain_summary"] = {
        domain: {
            "total_files": domain_files[domain],
            "files_with_findings": domain_files_with_findings[domain],
            "percentage_with_findings": (
                100 * domain_files_with_findings[domain] / domain_files[domain]
                if domain_files[domain] > 0
                else 0
            ),
        }
        for domain in domain_files
    }

    return stats


def generate_report(stats: dict[str, Any], output_file: Path) -> None:
    """Generate markdown report from aggregated statistics.

    Args:
        stats: Aggregated statistics dict.
        output_file: Path to output markdown file.
    """
    analyzed = stats["analyzed_successfully"]
    with_findings = stats["files_with_findings"]

    lines = [
        "# Real-World Demo: scicode-lint Findings",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total files analyzed:** {analyzed:,}",
        f"- **Files with findings:** {with_findings:,} ({100 * with_findings / analyzed:.1f}%)"
        if analyzed > 0
        else "- **Files with findings:** 0",
        f"- **Total findings:** {stats['total_findings']:,}",
        f"- **Repos with findings:** {stats['repos_with_findings']:,}",
        f"- **Papers with findings:** {stats['papers_with_findings']:,}",
        "",
    ]

    # Findings by category
    if stats["findings_by_category"]:
        lines.extend(
            [
                "## Findings by Category",
                "",
            ]
        )
        for cat, count in sorted(stats["findings_by_category"].items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}:** {count:,}")
        lines.append("")

    # Top patterns
    if stats["findings_by_pattern"]:
        lines.extend(
            [
                "## Top Patterns Detected",
                "",
            ]
        )
        top_patterns = sorted(stats["findings_by_pattern"].items(), key=lambda x: -x[1])[:15]
        for pattern, count in top_patterns:
            lines.append(f"- **{pattern}:** {count:,}")
        lines.append("")

    # Domain breakdown
    if stats["domain_summary"]:
        lines.extend(
            [
                "## Findings by Scientific Domain",
                "",
                "| Domain | Files Analyzed | Files with Issues | Percentage |",
                "|--------|---------------|-------------------|------------|",
            ]
        )
        for domain, domain_stats in sorted(
            stats["domain_summary"].items(),
            key=lambda x: -x[1]["files_with_findings"],
        ):
            lines.append(
                f"| {domain} | {domain_stats['total_files']:,} | "
                f"{domain_stats['files_with_findings']:,} | "
                f"{domain_stats['percentage_with_findings']:.1f}% |"
            )
        lines.append("")

    # Write report
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text("\n".join(lines))
    logger.info(f"Generated report: {output_file}")


def save_raw_results(results: list[dict[str, Any]], output_file: Path) -> None:
    """Save raw analysis results to JSON.

    Args:
        results: List of analysis results.
        output_file: Path to output JSON file.
    """
    output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved raw results: {output_file}")


def print_summary(stats: dict[str, Any]) -> None:
    """Print summary statistics to console.

    Args:
        stats: Aggregated statistics dict.
    """
    total = stats["total_files"]
    analyzed = stats["analyzed_successfully"]
    with_findings = stats["files_with_findings"]

    logger.info("=" * 50)
    logger.info("Analysis Summary:")
    logger.info(f"  Total files: {total:,}")
    logger.info(f"  Analyzed successfully: {analyzed:,}")
    logger.info(
        f"  Files with findings: {with_findings:,} ({100 * with_findings / analyzed:.1f}%)"
        if analyzed > 0
        else ""
    )
    logger.info(f"  Total findings: {stats['total_findings']:,}")

    if stats["findings_by_category"]:
        logger.info("  By category:")
        for cat, count in sorted(stats["findings_by_category"].items(), key=lambda x: -x[1])[:5]:
            logger.info(f"    {cat}: {count:,}")
