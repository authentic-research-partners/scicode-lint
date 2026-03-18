"""Shared data models for deterministic validation."""

from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str  # "error" or "warning"
    check: str  # which check found this
    message: str
    file: str = ""


@dataclass
class ValidationResult:
    """Result of validating a pattern."""

    pattern_id: str
    category: str
    issues: list[ValidationIssue] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)
