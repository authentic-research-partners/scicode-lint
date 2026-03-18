"""Report generation for LLM-as-judge evaluations."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from .judge_models import OverallJudgeMetrics


class JudgeReportGenerator:
    """Generate reports from LLM-as-judge evaluations."""

    @staticmethod
    def generate_summary_text(metrics: OverallJudgeMetrics) -> str:
        """Generate human-readable summary."""
        lines = [
            "",
            "=" * 70,
            "LLM-AS-JUDGE EVALUATION SUMMARY",
            "=" * 70,
            "",
            f"Total Patterns: {metrics.total_patterns}",
            f"Total Tests: {metrics.total_tests}",
            "",
            "ACCURACY BY TEST TYPE",
            "-" * 70,
            f"Positive Tests:  {metrics.positive_accuracy:.2%}",
            f"Negative Tests:  {metrics.negative_accuracy:.2%}",
            f"Context Tests:   {metrics.context_accuracy:.2%}",
            "",
            f"Overall Accuracy: {metrics.overall_accuracy:.2%}",
            f"Avg Judge Confidence: {metrics.avg_judge_confidence:.2f}",
            "",
            f"Patterns Above Threshold (≥85%): "
            f"{metrics.patterns_above_threshold}/{metrics.total_patterns}",
            "",
            "ALIGNMENT METRICS (Direct vs Judge)",
            "-" * 70,
            f"Semantic Alignment:           {metrics.semantic_alignment:.1%}",
        ]
        # Handle conditional formatting for counts
        if metrics.total_tests > 0:
            both_pass_pct = metrics.both_pass_count / metrics.total_tests
            both_fail_pct = metrics.both_fail_count / metrics.total_tests
            cnt = metrics.both_pass_count
            lines.append(f"  - Both Pass:                {cnt} ({both_pass_pct:.1%})")
            cnt = metrics.both_fail_count
            lines.append(f"  - Both Fail:                {cnt} ({both_fail_pct:.1%})")
        else:
            lines.append("  - Both Pass:                0 (0.0%)")
            lines.append("  - Both Fail:                0 (0.0%)")

        lines.extend(
            [
                "",
                f"Quality Issue Rate:           {metrics.quality_issue_rate:.1%} "
                f"({metrics.quality_issue_count} cases)",
                "  (Direct passes, Judge fails - right location, wrong explanation)",
                "",
                f"Ground Truth Strictness Rate: {metrics.ground_truth_strictness_rate:.1%} "
                f"({metrics.overly_strict_count} cases)",
                "  (Direct fails, Judge passes - ground truth too rigid)",
                "",
                "FOCUS LINE ACCURACY",
                "-" * 70,
                f"Focus Line Accuracy:  {metrics.focus_line_accuracy:.1%} "
                f"({metrics.focus_line_matched}/{metrics.focus_line_eligible} eligible)",
                "  (Linter's focus_line lands on one of the expected buggy lines)",
                "",
                "=" * 70,
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def save_json_report(metrics: OverallJudgeMetrics, output_path: Path) -> None:
        """Save metrics as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)
        logger.info(f"JSON report saved to {output_path}")

    @staticmethod
    def save_markdown_report(metrics: OverallJudgeMetrics, output_path: Path) -> None:
        """Save metrics as Markdown."""
        lines = [
            "# LLM-as-Judge Evaluation Report",
            "",
            f"**Total Patterns:** {metrics.total_patterns}  ",
            f"**Total Tests:** {metrics.total_tests}  ",
            f"**Overall Accuracy:** {metrics.overall_accuracy:.2%}  ",
            "",
            "## Accuracy by Test Type",
            "",
            "| Test Type | Accuracy |",
            "|-----------|----------|",
            f"| Positive | {metrics.positive_accuracy:.2%} |",
            f"| Negative | {metrics.negative_accuracy:.2%} |",
            f"| Context-Dependent | {metrics.context_accuracy:.2%} |",
            "",
            "## Alignment Metrics (Direct vs Judge)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Semantic Alignment | {metrics.semantic_alignment:.1%} |",
            f"| Quality Issue Rate | {metrics.quality_issue_rate:.1%} |",
            f"| Ground Truth Strictness Rate | {metrics.ground_truth_strictness_rate:.1%} |",
            "",
            "**Interpretation:**",
            "- **Quality Issues**: Direct metrics pass but judge fails "
            "(right location, wrong explanation)",
            "- **Overly Strict**: Direct metrics fail but judge passes (ground truth too rigid)",
            "",
            "## Focus Line Accuracy",
            "",
            f"**Accuracy:** {metrics.focus_line_accuracy:.1%} "
            f"({metrics.focus_line_matched}/{metrics.focus_line_eligible} eligible)",
            "",
            "Checks whether the linter's `focus_line` lands on one of the expected buggy "
            "lines from pattern.toml. Only for positive tests where linter detected an issue.",
            "",
        ]

        lines.extend(
            [
                "## Per-Pattern Results",
                "",
                "| Pattern ID | Tests | Accuracy | Aligned | Quality Issues |",
                "|------------|-------|----------|---------|----------------|",
            ]
        )

        for pattern in sorted(metrics.patterns, key=lambda p: p.overall_accuracy, reverse=True):
            lines.append(
                f"| {pattern.pattern_id} | {pattern.total_tests} | "
                f"{pattern.overall_accuracy:.2%} | {pattern.alignment_rate:.0%} | "
                f"{pattern.quality_issue_count} |"
            )

        # Add divergent cases section if any exist
        divergent_patterns = [
            p for p in metrics.patterns if p.quality_issue_count > 0 or p.overly_strict_count > 0
        ]
        if divergent_patterns:
            lines.extend(
                [
                    "",
                    "## Divergent Cases (Need Attention)",
                    "",
                ]
            )
            for pattern in divergent_patterns:
                divergent_cases = [e for e in pattern.evaluations if not e.aligned]
                if divergent_cases:
                    lines.append(f"### {pattern.pattern_id}")
                    lines.append("")
                    for case in divergent_cases:
                        emoji = "!!" if case.alignment == "quality_issue" else ">>"
                        lines.append(f"- {emoji} **[{case.alignment}]** `{case.test_file}`")
                        direct_status = "PASS" if case.direct_passed else "FAIL"
                        lines.append(f"  - Direct: {direct_status} - {case.direct_reason}")
                        reason_truncated = case.judge_reasoning[:80]
                        lines.append(f"  - Judge: {case.judge_verdict} - {reason_truncated}...")
                    lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown report saved to {output_path}")
