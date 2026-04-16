"""Tests for the Rich-polished text formatter in output/formatter.py.

Assert key substrings and structural properties — do NOT snapshot full
panel output (brittle across Rich versions and terminal widths).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scicode_lint.config import Severity
from scicode_lint.output.formatter import (
    Finding,
    LintError,
    LintResult,
    Location,
    format_findings,
)

_DEFAULT_SNIPPET = (
    "def train_model(data):\n"
    "    X, y = data.drop('target', axis=1), data['target']\n"
    "    scaler = StandardScaler()\n"
    "    X_scaled = scaler.fit_transform(X)\n"
    "    X_train, X_test = train_test_split(X_scaled, y)"
)


def _finding(
    focus: int | None = 12,
    snippet: str = _DEFAULT_SNIPPET,
    reasoning: str = "",
    detection_type: str = "yes",
) -> Finding:
    return Finding(
        id="ml-001",
        category="ai-training",
        severity=Severity.CRITICAL,
        location=Location(
            lines=[10, 11, 12, 13, 14],
            focus_line=focus,
            snippet=snippet,
            name="train_model",
            location_type="function",
        ),
        issue="Data leakage from scaler",
        explanation="Scaler fit on full X before split.",
        suggestion="Fit scaler on X_train only.",
        confidence=0.9,
        reasoning=reasoning,
        detection_type=detection_type,  # type: ignore[arg-type]
    )


def _result(tmp: Path, findings: list[Finding]) -> LintResult:
    return LintResult(file=tmp / "train.py", findings=findings)


def _format(results: list[LintResult], *, isatty: bool) -> str:
    # Tests always operate in non-TTY mode to guarantee no ANSI in captured
    # output. When isatty=True branch is exercised below we monkeypatch.
    return format_findings(results, output_format="text")


class TestCoreContent:
    def test_severity_label_appears(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([_result(tmp_path, [_finding()])], isatty=False)
        assert "CRITICAL" in out

    def test_pattern_id_appears(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([_result(tmp_path, [_finding()])], isatty=False)
        assert "ml-001" in out

    def test_location_name_and_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([_result(tmp_path, [_finding()])], isatty=False)
        assert "train_model" in out
        assert "10" in out and "14" in out

    def test_issue_and_explanation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([_result(tmp_path, [_finding()])], isatty=False)
        assert "Data leakage from scaler" in out
        assert "Scaler fit on full X before split" in out

    def test_reasoning_rendered_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format(
            [_result(tmp_path, [_finding(reasoning="fit_transform called on full X")])],
            isatty=False,
        )
        assert "Reasoning" in out
        assert "fit_transform" in out

    def test_context_dependent_qualifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format(
            [_result(tmp_path, [_finding(detection_type="context-dependent")])],
            isatty=False,
        )
        assert "CRITICAL?" in out


class TestColorHandling:
    def test_no_ansi_when_not_tty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Piped/redirected output must be ANSI-free."""
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([_result(tmp_path, [_finding()])], isatty=False)
        assert "\x1b[" not in out

    def test_ansi_present_when_tty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        out = _format([_result(tmp_path, [_finding()])], isatty=True)
        assert "\x1b[" in out


class TestErrorRendering:
    def test_error_line_shown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        err_result = LintResult(
            file=tmp_path / "broken.py",
            findings=[],
            error=LintError(
                file=tmp_path / "broken.py",
                error_type="LLMConnectionError",
                message="no vLLM",
                details=None,
            ),
        )
        out = _format([err_result], isatty=False)
        assert "Error during linting" in out
        assert "LLMConnectionError" in out
        assert "no vLLM" in out


class TestNoStdoutSideEffect:
    """Regression: format_findings must not write to stdout.

    Caller does ``print(format_findings(results))``. If the formatter's
    internal Console writes directly to stdout as well, output is duplicated.
    """

    def test_no_stdout_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        _format([_result(tmp_path, [_finding()])], isatty=False)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestSnippetRendering:
    """Code snippets must appear in the panel with original line numbers."""

    def test_snippet_lines_and_focus_marker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([_result(tmp_path, [_finding()])], isatty=False)
        # Snippet code must appear in the output
        assert "StandardScaler" in out
        # Line numbers from the original file must be rendered in the gutter
        assert "10" in out and "11" in out
        # Focus line (12) should be marked distinctly in the rendered text
        # (Rich uses "❱" in the gutter for highlight_lines)
        assert "❱" in out


class TestEmpty:
    def test_empty_results_returns_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([], isatty=False)
        assert out == ""

    def test_clean_result_renders_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)
        out = _format([_result(tmp_path, [])], isatty=False)
        assert out.strip() == ""
