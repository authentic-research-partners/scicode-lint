"""Tests for CLI exit-code semantics.

The linter follows standard linter convention (same as ruff, mypy, shellcheck):
    0 = clean (no findings, no errors)
    1 = findings detected
    2 = tool/runtime error

These tests mock `SciCodeLinter.check_file` so they run without vLLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scicode_lint.cli import main
from scicode_lint.config import Severity
from scicode_lint.exceptions import LLMConnectionError
from scicode_lint.output.formatter import (
    Finding,
    LintResult,
    Location,
)


def _make_clean_result(path: Path) -> LintResult:
    return LintResult(file=path, findings=[])


def _make_finding_result(path: Path) -> LintResult:
    finding = Finding(
        id="ml-001",
        category="ai-training",
        severity=Severity.CRITICAL,
        location=Location(lines=[1, 2], focus_line=1, name="f", location_type="function"),
        issue="Synthetic issue",
        explanation="For test",
        suggestion="Fix it",
        confidence=0.9,
    )
    return LintResult(file=path, findings=[finding])


@pytest.fixture
def fake_py_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.py"
    p.write_text("x = 1\n")
    return p


class TestLintExitCodes:
    def test_clean_file_returns_zero(
        self,
        fake_py_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fake_check(self: Any, path: Path) -> LintResult:
            return _make_clean_result(path)

        monkeypatch.setattr(
            "scicode_lint.linter.SciCodeLinter.check_file", fake_check, raising=True
        )

        exit_code = main(["lint", str(fake_py_file)])

        assert exit_code == 0

    def test_findings_return_one(
        self,
        fake_py_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fake_check(self: Any, path: Path) -> LintResult:
            return _make_finding_result(path)

        monkeypatch.setattr(
            "scicode_lint.linter.SciCodeLinter.check_file", fake_check, raising=True
        )

        exit_code = main(["lint", str(fake_py_file)])

        assert exit_code == 1

    def test_per_file_tool_error_returns_two(
        self,
        fake_py_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Per-file exceptions are captured into LintResult.error → exit 2."""

        def fake_check(self: Any, path: Path) -> LintResult:
            raise LLMConnectionError("no vLLM")

        monkeypatch.setattr(
            "scicode_lint.linter.SciCodeLinter.check_file", fake_check, raising=True
        )

        exit_code = main(["lint", str(fake_py_file)])

        assert exit_code == 2

    def test_top_level_scicode_lint_error_returns_two(
        self,
        fake_py_file: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A SciCodeLintError escaping the handler is caught by main() → exit 2."""

        def fake_run_lint(args: Any) -> int:
            raise LLMConnectionError("vLLM unreachable during setup")

        monkeypatch.setattr("scicode_lint.cli._run_lint", fake_run_lint, raising=True)

        exit_code = main(["lint", str(fake_py_file)])

        assert exit_code == 2
        captured = capsys.readouterr()
        assert "LLMConnectionError" in captured.err
        assert "vLLM unreachable during setup" in captured.err
