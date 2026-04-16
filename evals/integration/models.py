"""Data models for integration evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

# Short identifier (e.g. pattern ID "pt-001"), bounded for vLLM schema enforcement
_PatternId = Annotated[str, StringConstraints(max_length=32)]

# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================


class PatternSelection(BaseModel):
    """Result of pattern selection by LLM."""

    patterns: list[_PatternId] = Field(
        max_length=20,
        description="List of selected pattern IDs (e.g., pt-001, ml-002); at most 20",
    )
    scenario_type: str = Field(
        max_length=200,
        description="Brief description of the scenario type (one line)",
    )
    reasoning: str = Field(
        max_length=500,
        description="Why these patterns work together (2-3 sentences)",
    )


class ManifestEntry(BaseModel):
    """Single bug entry in the manifest."""

    pattern_id: _PatternId = Field(description="Pattern ID (e.g., pt-001)")
    line: int = Field(description="Line number where the bug occurs")
    description: str = Field(
        max_length=300,
        description="Brief description of the bug instance",
    )


class GeneratedScenario(BaseModel):
    """Result of scenario generation by LLM.

    ``code`` is intentionally unbounded — generated Python programs vary widely
    in size. The manifest is capped so a single scenario can't balloon out.
    """

    code: str = Field(description="Complete Python code containing the bugs")
    manifest: list[ManifestEntry] = Field(
        max_length=30,
        description="List of bugs in the code; at most 30",
    )


class VerificationEntry(BaseModel):
    """Verification result for a single bug."""

    pattern_id: _PatternId = Field(description="Pattern ID being verified")
    line: int = Field(description="Claimed line number")
    correct: bool = Field(description="Whether the bug is actually present")
    actual_line: int | None = Field(default=None, description="Corrected line number if different")
    notes: str = Field(
        default="",
        max_length=300,
        description="Notes about the verification",
    )


class VerificationResult(BaseModel):
    """Result of manifest verification."""

    verified: list[VerificationEntry] = Field(
        max_length=30,
        description="Verification for each claimed bug (at most 30)",
    )
    quality: Literal["good", "needs_correction", "regenerate"] = Field(
        description="Overall quality verdict on the generated scenario",
    )
    corrected_manifest: list[ManifestEntry] | None = Field(
        default=None,
        max_length=30,
        description="Corrected manifest if needed",
    )


class JudgedFinding(BaseModel):
    """Judge evaluation of a single linter finding."""

    pattern_id: _PatternId = Field(description="Pattern ID from the finding")
    line: int = Field(description="Line number from the finding")
    category: Literal["tp_intended", "tp_bonus", "fp"] = Field(
        description="True positive intended, true positive bonus, or false positive",
    )
    reasoning: str = Field(
        max_length=400,
        description="Why this categorization (1-2 sentences)",
    )


class JudgedMiss(BaseModel):
    """Judge evaluation of a missed bug."""

    pattern_id: _PatternId = Field(description="Pattern ID that was missed")
    line: int = Field(description="Line number where bug should be")
    reasoning: str = Field(
        max_length=400,
        description="Why it was missed or if it's actually present",
    )


class JudgeResult(BaseModel):
    """Result of LLM judge evaluation."""

    findings: list[JudgedFinding] = Field(
        max_length=50,
        description="Evaluation of each linter finding (at most 50)",
    )
    misses: list[JudgedMiss] = Field(
        max_length=50,
        description="Evaluation of missed bugs (at most 50)",
    )
    # Note: summary accepts Any because judge sometimes returns dict instead of string
    summary: Any = Field(description="Overall assessment")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PatternInfo:
    """Pattern information from the catalog."""

    id: str
    description: str
    category: str


@dataclass
class ScenarioResult:
    """A generated scenario with evaluation results."""

    name: str
    code: str
    patterns: list[str]
    manifest: list[dict[str, Any]]
    verified: bool
    # Evaluation results
    bugs_intended: int = 0
    bugs_detected: int = 0
    tp_intended: int = 0
    tp_bonus: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    false_negatives: list[dict[str, Any]] = field(default_factory=list)


def categorize_judge_results(
    judge_result: JudgeResult | None,
    manifest: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    skip_judge: bool,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Categorize linter findings into TP/FP/FN.

    Args:
        judge_result: LLM judge result (None if skip_judge)
        manifest: Expected bugs from scenario
        findings: Linter findings
        skip_judge: If True, use deterministic comparison

    Returns:
        Tuple of (tp_intended, tp_bonus, false_positives, false_negatives, log_lines)
    """
    log_lines: list[str] = []

    if skip_judge:
        expected_ids = {m["pattern_id"] for m in manifest}
        found_ids = {f["pattern_id"] for f in findings}

        tp_intended = [f for f in findings if f["pattern_id"] in expected_ids]
        tp_bonus = [f for f in findings if f["pattern_id"] not in expected_ids]
        false_positives: list[dict[str, Any]] = []
        false_negatives = [m for m in manifest if m["pattern_id"] not in found_ids]
        log_lines.append("[JUDGE] skipped (deterministic comparison)\n")
    else:
        assert judge_result is not None
        log_lines.append(f"[JUDGE]\n{judge_result.model_dump_json(indent=2)}\n")

        tp_intended = [
            {"pattern_id": f.pattern_id, "line": f.line, "reasoning": f.reasoning}
            for f in judge_result.findings
            if f.category == "tp_intended"
        ]
        tp_bonus = [
            {"pattern_id": f.pattern_id, "line": f.line, "reasoning": f.reasoning}
            for f in judge_result.findings
            if f.category == "tp_bonus"
        ]
        false_positives = [
            {"pattern_id": f.pattern_id, "line": f.line, "reasoning": f.reasoning}
            for f in judge_result.findings
            if f.category == "fp"
        ]
        false_negatives = [
            {"pattern_id": m.pattern_id, "line": m.line, "reasoning": m.reasoning}
            for m in judge_result.misses
        ]

    return tp_intended, tp_bonus, false_positives, false_negatives, log_lines
