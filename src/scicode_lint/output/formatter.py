"""Format linter findings for output."""

import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from scicode_lint.config import Severity


class Location(BaseModel):
    """Location of a finding in code - name-based with verified line numbers.

    The detection flow produces verified location data:
    1. LLM identifies the function/method name and approximate line (near_line)
    2. AST resolution verifies the name exists and gets exact boundaries
    3. Output includes both context (full function) and focus (specific line)

    This approach eliminates line number variance across LLM runs while
    providing users with specific, actionable line numbers.

    Note: Each pattern produces at most ONE finding per file. If the same bug
    pattern appears multiple times, only the most clear instance is reported.
    Users should re-run the linter after fixing to catch additional instances.
    """

    lines: list[int] = Field(
        description="Full line range of the function/method (for context)",
        examples=[[15, 16, 17, 18, 19], [42, 43, 44]],
    )
    focus_line: int | None = Field(
        default=None,
        description="Specific line to look at (verified from LLM's near_line hint)",
    )
    snippet: str = Field(default="", description="Code snippet from the function/method")
    name: str | None = Field(
        default=None,
        description="Function/class/method name where issue occurs",
    )
    location_type: str | None = Field(
        default=None,
        description="Type of code construct: 'function', 'method', 'class', or 'module'",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "lines": [10, 11, 12, 13, 14],
                    "focus_line": 12,
                    "snippet": "def train_model(data):\\n    scaler = ...",
                    "name": "train_model",
                    "location_type": "function",
                }
            ]
        }
    )


class Finding(BaseModel):
    """A single linter finding.

    Designed for both human and GenAI agent consumption.
    Includes structured location data and actionable explanations.
    """

    id: str = Field(description="Pattern ID (e.g., 'ml-001')")
    category: str = Field(description="Pattern category (e.g., 'ai-training')")
    severity: Severity = Field(description="Issue severity level")
    location: Location = Field(description="Where the issue was found")
    issue: str = Field(description="Brief issue description")
    explanation: str = Field(description="Detailed explanation for humans and agents")
    suggestion: str = Field(description="Actionable suggestion to fix the issue")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    reasoning: str = Field(default="", description="LLM's reasoning for this detection")
    detection_type: Literal["yes", "context-dependent"] = Field(
        default="yes", description="Whether this is a definite issue or context-dependent"
    )
    thinking: str | None = Field(
        default=None, description="Model's internal thinking (from <think> tags, if present)"
    )

    model_config = ConfigDict(
        use_enum_values=True  # Serialize Severity enum as string value
    )


class LintError(BaseModel):
    """An error that occurred during linting.

    Designed for both human and GenAI agent consumption.
    """

    file: Path = Field(description="File where error occurred")
    error_type: str = Field(description="Type of error (e.g., 'ContextLengthError')")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Structured error details (optional)"
    )


class PatternFailure(BaseModel):
    """Record of a pattern that failed to execute."""

    pattern_id: str = Field(description="Pattern ID that failed")
    error_type: str = Field(description="Type of error (e.g., 'timeout', 'api_error')")
    error_message: str = Field(default="", description="Error message")


class PatternCheckResult(BaseModel):
    """Result of checking a single pattern (includes 'no' detections for eval)."""

    pattern_id: str = Field(description="Pattern ID (e.g., 'ml-001')")
    detected: Literal["yes", "no", "context-dependent"] = Field(description="Detection result")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(default="", description="LLM's reasoning for this decision")
    thinking: str | None = Field(
        default=None, description="Model's internal thinking (from <think> tags)"
    )


class LintResult(BaseModel):
    """Result of linting a file."""

    file: Path = Field(description="Path to the linted file")
    findings: list[Finding] = Field(
        default_factory=list, description="List of findings detected in the file"
    )
    checked_patterns: list[PatternCheckResult] = Field(
        default_factory=list,
        description="All patterns checked with their results (for eval/debugging)",
    )
    patterns_failed: int = Field(
        default=0, description="Number of patterns that failed (timeout/error)"
    )
    failed_patterns: list[PatternFailure] = Field(
        default_factory=list, description="Details of patterns that failed"
    )
    error: LintError | None = Field(
        default=None, description="Error that occurred during linting (if any)"
    )

    model_config = ConfigDict(use_enum_values=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> dict[str, Any]:
        """Generate summary statistics (computed on-the-fly)."""
        severity_counts = Counter(f.severity for f in self.findings)
        category_counts = Counter(f.category for f in self.findings)

        return {
            "total_findings": len(self.findings),
            "by_severity": {
                (k.value if hasattr(k, "value") else k): v for k, v in severity_counts.items()
            },
            "by_category": dict(category_counts),
        }


def format_findings(
    results: list[LintResult],
    output_format: Literal["text", "json"] = "text",
) -> str:
    """
    Format linter results for output.

    Supports both human-readable text and structured JSON output.
    Designed for dual audience (humans and GenAI agents).

    Args:
        results: List of lint results
        output_format: 'text' or 'json'

    Returns:
        Formatted output string

    Raises:
        ValueError: If output_format is not 'text' or 'json'

    Example:
        >>> results = [LintResult(...)]
        >>> print(format_findings(results, "json"))
    """
    if output_format == "json":
        return _format_json(results)
    elif output_format == "text":
        return _format_text(results)
    else:
        raise ValueError(f"Unknown output format: {output_format!r}. Use 'text' or 'json'.")


def _format_json(results: list[LintResult]) -> str:
    """Format as JSON for GenAI agent consumption.

    Includes both findings and errors in structured format.

    Args:
        results: List of lint results

    Returns:
        JSON string with findings and errors
    """
    data = [r.model_dump(mode="json") for r in results]
    return json.dumps(data, indent=2)


_SEVERITY_STYLE: dict[Severity, tuple[str, str]] = {
    # (label, rich color) — label keeps emoji so piped plain-text still signals severity
    Severity.CRITICAL: ("🔴 CRITICAL", "bold red"),
    Severity.HIGH: ("🟠 HIGH", "bold dark_orange"),
    Severity.MEDIUM: ("🟡 MEDIUM", "bold yellow"),
}


def _format_location(loc: "Location") -> str:
    """Human-readable location string (e.g., 'in train_model (lines 10-14, focus: 12)')."""
    if loc.name and loc.name != "<module>":
        if loc.lines:
            line_range = f"lines {loc.lines[0]}-{loc.lines[-1]}"
            if loc.focus_line and loc.focus_line != loc.lines[0]:
                return f"in {loc.name} ({line_range}, focus: {loc.focus_line})"
            return f"in {loc.name} ({line_range})"
        return f"in {loc.name}"
    if loc.focus_line:
        return f"line {loc.focus_line}"
    if loc.lines:
        if len(loc.lines) == 1:
            return f"line {loc.lines[0]}"
        return f"lines {loc.lines[0]}-{loc.lines[-1]}"
    return "unknown location"


def _render_finding(finding: "Finding") -> Panel:
    """Render a single finding as a Rich Panel."""
    severity: Severity = (
        finding.severity if isinstance(finding.severity, Severity) else Severity(finding.severity)
    )
    label, color = _SEVERITY_STYLE[severity]
    qualifier = "?" if finding.detection_type == "context-dependent" else ""

    title = Text()
    title.append(f"{label}{qualifier}", style=color)
    title.append(" · ", style="dim")
    title.append(finding.id, style="bold")
    title.append(" · ", style="dim")
    title.append(_format_location(finding.location))

    body_parts: list[Any] = [Text(finding.issue, style="bold")]
    body_parts.append(Text(finding.explanation))
    if finding.reasoning:
        reasoning = Text()
        reasoning.append("Reasoning: ", style="dim")
        reasoning.append(finding.reasoning)
        body_parts.append(reasoning)

    snippet = finding.location.snippet
    if snippet:
        # Display the snippet with original file line numbers in the gutter.
        # `start_line` renumbers the displayed gutter to match the source file;
        # `highlight_lines` uses the same renumbered scheme. No `line_range` —
        # the snippet is already the exact extracted slice.
        start_line = finding.location.lines[0] if finding.location.lines else 1
        highlight = (
            {finding.location.focus_line} if finding.location.focus_line is not None else None
        )
        body_parts.append(
            Syntax(
                snippet,
                "python",
                line_numbers=True,
                start_line=start_line,
                highlight_lines=highlight,
                background_color="default",
                word_wrap=True,
            )
        )

    return Panel(Group(*body_parts), title=title, title_align="left", border_style=color)


def _format_text(results: list[LintResult]) -> str:
    """Format as human-readable text using Rich panels.

    Color is enabled only when stdout is a TTY. When piped/redirected, Rich
    strips ANSI but keeps Unicode box drawing and syntax-highlighted code
    structure (without colors). Agents should use ``--format json`` for
    structured output regardless.

    Args:
        results: List of lint results

    Returns:
        Human-readable text output (with ANSI codes iff stdout is a TTY)
    """
    use_color = sys.stdout.isatty()
    # Route Console output to an in-memory buffer; export_text() below returns
    # the rendered string. `record=True` alone still writes to stdout, which
    # would double-print when the caller invokes `print(format_findings(...))`.
    console = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=use_color,
        no_color=not use_color,
        width=100,
        highlight=False,
        emoji=True,
    )

    for result in results:
        if result.error:
            err = Text()
            err.append("⚠️  ", style="bold yellow")
            err.append(f"{result.file}", style="bold")
            err.append(" — Error during linting\n", style="bold")
            err.append(f"    {result.error.error_type}: {result.error.message}")
            console.print(err)
            console.print()
            continue

        if not result.findings:
            continue

        header = Text()
        header.append(f"{result.file}", style="bold")
        count = len(result.findings)
        header.append(f" — {count} issue{'s' if count != 1 else ''} found")
        console.print(header)
        console.print()

        for finding in result.findings:
            console.print(_render_finding(finding))
            console.print()

    return console.export_text(styles=use_color)


def get_json_schemas() -> dict[str, Any]:
    """
    Get JSON schemas for all output models.

    Useful for API documentation, OpenAPI specs, and GenAI agent integration.

    Returns:
        Dictionary with schema for each model type

    Example:
        >>> from scicode_lint.output.formatter import get_json_schemas
        >>> schemas = get_json_schemas()
        >>> print(schemas["Finding"]["properties"]["confidence"])
    """
    return {
        "Location": Location.model_json_schema(),
        "Finding": Finding.model_json_schema(),
        "LintError": LintError.model_json_schema(),
        "LintResult": LintResult.model_json_schema(),
    }
