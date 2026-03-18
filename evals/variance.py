"""Variance analysis for multi-run evaluation consistency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from .judge_models import PatternJudgeMetrics


def compute_variance_report(
    all_run_results: list[list[PatternJudgeMetrics]], pattern_ids: list[str]
) -> dict[str, Any]:
    """Compute variance across multiple runs to identify unstable patterns.

    Returns a report with:
    - Per-pattern variance in accuracy
    - Unstable patterns (where results differ across runs)
    - Overall variance statistics
    """
    import statistics

    # Build pattern_id -> list of accuracies across runs
    pattern_accuracies: dict[str, list[float]] = {pid: [] for pid in pattern_ids}
    overall_accuracies: list[float] = []

    for run_results in all_run_results:
        run_total_correct = 0
        run_total_tests = 0
        for pm in run_results:
            pattern_accuracies[pm.pattern_id].append(pm.overall_accuracy)
            run_total_correct += pm.correct_count
            run_total_tests += pm.total_tests

        if run_total_tests > 0:
            overall_accuracies.append(run_total_correct / run_total_tests)

    # Compute per-pattern stats
    pattern_stats: list[dict[str, Any]] = []
    unstable_patterns: list[str] = []

    for pid, accuracies in pattern_accuracies.items():
        if len(accuracies) < 2:
            continue

        mean_acc = statistics.mean(accuracies)
        stdev_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        variance = max_acc - min_acc

        stat = {
            "pattern_id": pid,
            "mean_accuracy": mean_acc,
            "stdev": stdev_acc,
            "min": min_acc,
            "max": max_acc,
            "variance": variance,
            "accuracies": accuracies,
        }
        pattern_stats.append(stat)

        # Flag as unstable if variance > 10% or stdev > 5%
        if variance > 0.10 or stdev_acc > 0.05:
            unstable_patterns.append(pid)

    # Sort by variance (most unstable first)
    pattern_stats.sort(key=lambda x: x["variance"], reverse=True)

    # Overall stats
    overall_mean = statistics.mean(overall_accuracies) if overall_accuracies else 0.0
    overall_stdev = statistics.stdev(overall_accuracies) if len(overall_accuracies) > 1 else 0.0

    return {
        "num_runs": len(all_run_results),
        "overall_mean_accuracy": overall_mean,
        "overall_stdev": overall_stdev,
        "overall_accuracies": overall_accuracies,
        "pattern_stats": pattern_stats,
        "unstable_patterns": unstable_patterns,
    }


def print_variance_report(report: dict[str, Any], output_dir: Path) -> None:
    """Print and save variance report from multi-run evaluation."""
    lines = [
        "",
        "=" * 70,
        f"VARIANCE REPORT ({report['num_runs']} runs)",
        "=" * 70,
        "",
        f"Overall Mean Accuracy: {report['overall_mean_accuracy']:.2%}",
        f"Overall Std Dev:       {report['overall_stdev']:.2%}",
        f"Per-run accuracies:    {[f'{a:.2%}' for a in report['overall_accuracies']]}",
        "",
    ]

    if report["unstable_patterns"]:
        lines.extend(
            [
                "UNSTABLE PATTERNS (variance > 10% or stdev > 5%)",
                "-" * 70,
            ]
        )
        for pid in report["unstable_patterns"]:
            stat = next(s for s in report["pattern_stats"] if s["pattern_id"] == pid)
            accs = [f"{a:.0%}" for a in stat["accuracies"]]
            lines.append(
                f"  {pid}: {stat['mean_accuracy']:.1%} mean, "
                f"{stat['variance']:.0%} range [{stat['min']:.0%}-{stat['max']:.0%}], "
                f"runs: {accs}"
            )
        lines.extend(
            [
                "",
                ">> These patterns have ambiguous detection questions.",
                ">> Run with --verbose to see LLM reasoning differences.",
                "",
            ]
        )
    else:
        lines.append("All patterns are stable across runs (variance < 10%).")
        lines.append("")

    lines.append("=" * 70)

    # Print to console
    print("\n".join(lines))

    # Save variance report
    variance_path = output_dir / "variance_report.json"
    variance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(variance_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Variance report saved to {variance_path}")
