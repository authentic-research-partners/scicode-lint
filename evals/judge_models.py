"""Pydantic models for LLM-as-judge evaluation."""

from typing import Literal

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """Judge's verdict on whether linter output matches expected behavior."""

    verdict: Literal["yes", "no", "partial"] = Field(
        description="Whether linter output matches test case intent"
    )
    reasoning: str = Field(description="One sentence explaining the verdict")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Judge's confidence in this verdict (0.0-1.0)"
    )
    thinking: str | None = Field(
        default=None,
        description="Model's internal reasoning/thinking (extracted from <think> tags). "
        "Not part of the structured output schema - populated automatically if present.",
    )


class TestCaseEvaluation(BaseModel):
    """Evaluation result for a single test case."""

    test_file: str
    test_type: Literal["positive", "negative", "context_dependent"]
    expected_behavior: str

    # Linter output format (name-based with reasoning)
    linter_detected: Literal["yes", "no", "context-dependent"]
    linter_name: str | None = None  # Detected function/class/method name
    linter_location_type: str | None = None  # "function", "method", "class", "module"
    linter_lines: list[int] = Field(default_factory=list)  # Resolved line numbers (context)
    linter_focus_line: int | None = None  # Specific line to look at (verified)
    linter_snippet: str = ""
    linter_reasoning: str = ""
    linter_issue: str | None = None
    linter_confidence: float = 0.0
    linter_thinking: str | None = None  # Model's thinking from <think> tags

    # Expected location from pattern.toml (for validation)
    expected_name: str | None = None  # Expected function/class/method name
    expected_location_type: str | None = None  # Expected type
    expected_lines: list[int] = Field(default_factory=list)

    # Name matching result (primary metric)
    name_match: bool = False  # Whether detected name matches expected name
    name_match_partial: bool = False  # Partial match (e.g., "train" vs "Trainer.train")

    # Focus line accuracy (whether linter's focus_line is in expected_lines)
    focus_line_match: bool = False  # True if linter_focus_line in expected_lines

    # Judge evaluation
    judge_verdict: Literal["yes", "no", "partial"]
    judge_reasoning: str
    judge_confidence: float
    judge_thinking: str | None = None  # Judge model's thinking from <think> tags

    # Direct metrics evaluation (computed from linter output)
    direct_passed: bool = False
    direct_reason: str = ""

    # Alignment between direct and judge
    alignment: Literal["both_pass", "both_fail", "quality_issue", "overly_strict"] = "both_fail"

    @property
    def aligned(self) -> bool:
        """Whether direct metrics and LLM judge agree."""
        return self.alignment in ("both_pass", "both_fail")

    @property
    def location_match_passed(self) -> bool:
        """Whether location matching passed based on name matching.

        Name is required in expected_location, so this always uses name matching.
        Returns True if no expected_name (negative test or legacy pattern).
        """
        if not self.expected_name:
            # No expected location = skip check (negative tests)
            return True
        return self.name_match or self.name_match_partial


class PatternJudgeMetrics(BaseModel):
    """LLM-as-judge metrics for a single pattern."""

    pattern_id: str
    total_tests: int

    # By test type
    positive_accuracy: float  # % where verdict = "yes"
    negative_accuracy: float
    context_accuracy: float  # % where verdict = "yes" or "partial"

    # Overall
    overall_accuracy: float  # (correct + 0.5*partial) / total

    # Detailed counts
    correct_count: int  # verdict = "yes"
    partial_count: int  # verdict = "partial"
    incorrect_count: int  # verdict = "no"

    # Alignment metrics (direct vs judge)
    aligned_count: int = 0
    both_pass_count: int = 0
    both_fail_count: int = 0
    quality_issue_count: int = 0  # direct pass, judge fail
    overly_strict_count: int = 0  # direct fail, judge pass

    # Focus line accuracy (positive tests with expected_lines where detection happened)
    focus_line_eligible: int = 0  # Positive tests with expected_lines where linter detected
    focus_line_matched: int = 0  # Where focus_line was in expected_lines
    focus_line_accuracy: float = 0.0  # focus_line_matched / focus_line_eligible

    # Evaluations
    evaluations: list[TestCaseEvaluation]

    @property
    def alignment_rate(self) -> float:
        """Percentage of tests where direct and judge agree."""
        return self.aligned_count / self.total_tests if self.total_tests > 0 else 0.0


class OverallJudgeMetrics(BaseModel):
    """Aggregated LLM-as-judge metrics across all patterns."""

    total_patterns: int
    total_tests: int

    # By test type
    positive_accuracy: float
    negative_accuracy: float
    context_accuracy: float

    # Overall
    overall_accuracy: float

    # Per-pattern metrics
    patterns: list[PatternJudgeMetrics]

    # Summary statistics
    avg_judge_confidence: float
    patterns_above_threshold: int  # accuracy >= 0.85

    # Alignment metrics (direct vs judge)
    semantic_alignment: float = 0.0  # % where both agree
    quality_issue_rate: float = 0.0  # % where direct pass, judge fail
    ground_truth_strictness_rate: float = 0.0  # % where direct fail, judge pass

    # Alignment counts
    aligned_count: int = 0
    both_pass_count: int = 0
    both_fail_count: int = 0
    quality_issue_count: int = 0
    overly_strict_count: int = 0

    # Focus line accuracy (aggregated across all patterns)
    focus_line_eligible: int = 0
    focus_line_matched: int = 0
    focus_line_accuracy: float = 0.0
