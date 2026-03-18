"""Data models for integration evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================


class PatternSelection(BaseModel):
    """Result of pattern selection by LLM."""

    patterns: list[str] = Field(description="List of selected pattern IDs (e.g., pt-001, ml-002)")
    scenario_type: str = Field(description="Brief description of the scenario type")
    reasoning: str = Field(description="Why these patterns work together")


class ManifestEntry(BaseModel):
    """Single bug entry in the manifest."""

    pattern_id: str = Field(description="Pattern ID (e.g., pt-001)")
    line: int = Field(description="Line number where the bug occurs")
    description: str = Field(description="Brief description of the bug instance")


class GeneratedScenario(BaseModel):
    """Result of scenario generation by LLM."""

    code: str = Field(description="Complete Python code containing the bugs")
    manifest: list[ManifestEntry] = Field(description="List of bugs in the code")


class VerificationEntry(BaseModel):
    """Verification result for a single bug."""

    pattern_id: str = Field(description="Pattern ID being verified")
    line: int = Field(description="Claimed line number")
    correct: bool = Field(description="Whether the bug is actually present")
    actual_line: int | None = Field(default=None, description="Corrected line number if different")
    notes: str = Field(default="", description="Notes about the verification")


class VerificationResult(BaseModel):
    """Result of manifest verification."""

    verified: list[VerificationEntry] = Field(description="Verification for each claimed bug")
    quality: str = Field(description="good, needs_correction, or regenerate")
    corrected_manifest: list[ManifestEntry] | None = Field(
        default=None, description="Corrected manifest if needed"
    )


class JudgedFinding(BaseModel):
    """Judge evaluation of a single linter finding."""

    pattern_id: str = Field(description="Pattern ID from the finding")
    line: int = Field(description="Line number from the finding")
    category: str = Field(description="tp_intended, tp_bonus, or fp")
    reasoning: str = Field(description="Why this categorization")


class JudgedMiss(BaseModel):
    """Judge evaluation of a missed bug."""

    pattern_id: str = Field(description="Pattern ID that was missed")
    line: int = Field(description="Line number where bug should be")
    reasoning: str = Field(description="Why it was missed or if it's actually present")


class JudgeResult(BaseModel):
    """Result of LLM judge evaluation."""

    findings: list[JudgedFinding] = Field(description="Evaluation of each linter finding")
    misses: list[JudgedMiss] = Field(description="Evaluation of missed bugs")
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
