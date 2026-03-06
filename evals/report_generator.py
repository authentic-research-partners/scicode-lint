"""Report generation for evaluation results.

Generates JSON, Markdown, and optionally HTML reports.
"""

import json
from datetime import datetime
from pathlib import Path

from .metrics import OverallMetrics


class ReportGenerator:
    """Generate evaluation reports in various formats."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json_report(
        self, metrics: OverallMetrics, filename: str = "evaluation_report.json"
    ) -> Path:
        """
        Generate JSON report with complete metrics.

        Args:
            metrics: OverallMetrics to include in report
            filename: Output filename

        Returns:
            Path to generated report file
        """
        report_path = self.output_dir / filename

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "version": "0.1.2",
            "metrics": metrics.to_dict(),
        }

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        return report_path

    def generate_markdown_report(
        self, metrics: OverallMetrics, filename: str = "evaluation_report.md"
    ) -> Path:
        """
        Generate Markdown report for human readability.

        Args:
            metrics: OverallMetrics to include in report
            filename: Output filename

        Returns:
            Path to generated report file
        """
        report_path = self.output_dir / filename

        lines = []
        lines.append("# Scicode-Lint Evaluation Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Overall metrics
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append(f"- **Patterns Evaluated**: {metrics.overall.pattern_count}")
        lines.append(f"- **True Positives**: {metrics.overall.true_positives}")
        lines.append(f"- **False Positives**: {metrics.overall.false_positives}")
        lines.append(f"- **False Negatives**: {metrics.overall.false_negatives}")
        lines.append("")
        lines.append(f"- **Precision**: {metrics.overall.precision:.3f}")
        lines.append(f"- **Recall**: {metrics.overall.recall:.3f}")
        lines.append(f"- **F1 Score**: {metrics.overall.f1_score:.3f}")
        lines.append("")

        # Threshold checks
        lines.append("## Threshold Checks")
        lines.append("")
        overall_status = "✅ PASS" if metrics.meets_overall_thresholds else "❌ FAIL"
        lines.append(f"- **Overall Thresholds**: {overall_status}")
        prec_ok = metrics.overall.precision is not None and metrics.overall.precision >= 0.90
        lines.append(f"  - Precision ≥ 0.90: {'✅' if prec_ok else '❌'}")
        recall_ok = metrics.overall.recall is not None and metrics.overall.recall >= 0.80
        lines.append(f"  - Recall ≥ 0.80: {'✅' if recall_ok else '❌'}")

        if "critical" in metrics.by_severity:
            critical_status = "✅ PASS" if metrics.meets_critical_threshold else "❌ FAIL"
            critical_prec = metrics.by_severity["critical"].precision
            lines.append(f"- **Critical Severity Precision**: {critical_status}")
            crit_ok = critical_prec is not None and critical_prec >= 0.95
            lines.append(f"  - Precision ≥ 0.95: {'✅' if crit_ok else '❌'}")
        lines.append("")

        # By category
        lines.append("## Metrics by Category")
        lines.append("")
        lines.append("| Category | Precision | Recall | F1 | Patterns |")
        lines.append("|----------|-----------|--------|----|----|")
        for category, cat_metrics in sorted(metrics.by_category.items()):
            lines.append(
                f"| {category} | {cat_metrics.precision:.3f} | "
                f"{cat_metrics.recall:.3f} | {cat_metrics.f1_score:.3f} | "
                f"{cat_metrics.pattern_count} |"
            )
        lines.append("")

        # By severity
        lines.append("## Metrics by Severity")
        lines.append("")
        lines.append("| Severity | Precision | Recall | F1 | Patterns |")
        lines.append("|----------|-----------|--------|----|----|")
        severity_order = ["critical", "high", "medium"]
        for severity in severity_order:
            if severity in metrics.by_severity:
                sev_metrics = metrics.by_severity[severity]
                lines.append(
                    f"| {severity} | {sev_metrics.precision:.3f} | "
                    f"{sev_metrics.recall:.3f} | {sev_metrics.f1_score:.3f} | "
                    f"{sev_metrics.pattern_count} |"
                )
        lines.append("")

        # Pattern details
        lines.append("## Pattern Details")
        lines.append("")
        lines.append("| Pattern ID | Category | Severity | Precision | Recall | F1 | Status |")
        lines.append("|------------|----------|----------|-----------|--------|----|----|")
        for pattern in sorted(metrics.patterns, key=lambda p: p.pattern_id):
            status = "✅" if pattern.passes_thresholds else "❌"
            lines.append(
                f"| {pattern.pattern_id} | {pattern.category} | "
                f"{pattern.severity} | {pattern.precision:.3f} | "
                f"{pattern.recall:.3f} | {pattern.f1_score:.3f} | {status} |"
            )
        lines.append("")

        # Failed patterns
        failed_patterns = [p for p in metrics.patterns if not p.passes_thresholds]
        if failed_patterns:
            lines.append("## Failed Patterns")
            lines.append("")
            for pattern in failed_patterns:
                lines.append(f"### {pattern.pattern_id} ({pattern.category})")
                lines.append("")
                lines.append(f"- **Severity**: {pattern.severity}")
                lines.append(f"- **Precision**: {pattern.precision:.3f} (need ≥ 0.90)")
                lines.append(f"- **Recall**: {pattern.recall:.3f} (need ≥ 0.80)")
                lines.append(f"- **True Positives**: {pattern.true_positives}")
                lines.append(f"- **False Positives**: {pattern.false_positives}")
                lines.append(f"- **False Negatives**: {pattern.false_negatives}")
                lines.append("")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        return report_path

    def generate_summary_text(self, metrics: OverallMetrics) -> str:
        """
        Generate a brief text summary for console output.

        Args:
            metrics: OverallMetrics to summarize

        Returns:
            Multi-line string summary
        """
        lines = []
        lines.append("=" * 60)
        lines.append("EVALUATION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Patterns evaluated: {metrics.overall.pattern_count}")
        lines.append(f"Overall precision:  {metrics.overall.precision:.3f} (target: ≥ 0.90)")
        lines.append(f"Overall recall:     {metrics.overall.recall:.3f} (target: ≥ 0.80)")
        lines.append(f"Overall F1 score:   {metrics.overall.f1_score:.3f}")
        lines.append("")

        if "critical" in metrics.by_severity:
            critical_prec = metrics.by_severity["critical"].precision
            lines.append(f"Critical precision: {critical_prec:.3f} (target: ≥ 0.95)")
            lines.append("")

        overall_status = "PASS ✅" if metrics.meets_overall_thresholds else "FAIL ❌"
        lines.append(f"Overall: {overall_status}")

        if "critical" in metrics.by_severity:
            critical_status = "PASS ✅" if metrics.meets_critical_threshold else "FAIL ❌"
            lines.append(f"Critical: {critical_status}")

        lines.append("=" * 60)

        # Failed patterns summary
        failed = [p for p in metrics.patterns if not p.passes_thresholds]
        if failed:
            lines.append(f"\n{len(failed)} pattern(s) failed thresholds:")
            for pattern in failed:
                lines.append(
                    f"  - {pattern.pattern_id}: P={pattern.precision:.3f}, R={pattern.recall:.3f}"
                )

        return "\n".join(lines)

    def generate_all_reports(
        self,
        metrics: OverallMetrics,
        json_filename: str = "latest.json",
        markdown_filename: str = "latest.md",
    ) -> dict[str, Path]:
        """
        Generate all report formats.

        Args:
            metrics: OverallMetrics to report
            json_filename: JSON report filename
            markdown_filename: Markdown report filename

        Returns:
            Dictionary mapping format to report path
        """
        reports = {}
        reports["json"] = self.generate_json_report(metrics, json_filename)
        reports["markdown"] = self.generate_markdown_report(metrics, markdown_filename)
        return reports
