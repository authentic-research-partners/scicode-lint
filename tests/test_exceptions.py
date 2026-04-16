"""Tests for the user-facing exception hierarchy.

These tests run without any services (no vLLM, no network). Connection
behavior is exercised via monkeypatching `httpx.Client` in-place; no real
HTTP requests are made.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from scicode_lint.exceptions import LLMConnectionError, SciCodeLintError
from scicode_lint.linter import NotebookParseError
from scicode_lint.llm.client import MissingLocationError, _probe_base_url, detect_vllm
from scicode_lint.llm.exceptions import ContextLengthError


class TestHierarchy:
    """All user-facing exceptions descend from SciCodeLintError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            LLMConnectionError,
            ContextLengthError,
            MissingLocationError,
            NotebookParseError,
        ],
    )
    def test_inherits_from_base(self, exc_cls: type[Exception]) -> None:
        assert issubclass(exc_cls, SciCodeLintError)

    def test_base_is_exception(self) -> None:
        assert issubclass(SciCodeLintError, Exception)

    def test_missing_location_no_longer_value_error(self) -> None:
        """MissingLocationError dropped its ValueError base in 0.3.x.

        Callers should catch MissingLocationError or SciCodeLintError.
        """
        assert not issubclass(MissingLocationError, ValueError)


class TestLLMConnectionError:
    """detect_vllm() raises LLMConnectionError when no server is reachable."""

    def test_detect_vllm_raises_on_all_ports_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FailingClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _FailingClient:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def get(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "Client", _FailingClient)

        with pytest.raises(LLMConnectionError) as excinfo:
            detect_vllm()

        assert "scicode-lint vllm-server start" in str(excinfo.value)

    def test_caught_as_scicode_lint_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FailingClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _FailingClient:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def get(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.TimeoutException("slow")

        monkeypatch.setattr(httpx, "Client", _FailingClient)

        with pytest.raises(SciCodeLintError):
            detect_vllm()


class TestMissingLocationError:
    """MissingLocationError carries structured attributes for retry flows."""

    def test_attributes_preserved(self) -> None:
        err = MissingLocationError(detected="yes", reasoning="fit on full X", confidence=0.75)
        assert err.detected == "yes"
        assert err.reasoning == "fit on full X"
        assert err.confidence == 0.75
        assert "detected='yes'" in str(err)


class TestProbeBaseUrl:
    """_probe_base_url fast-fails with LLMConnectionError on unreachable URLs."""

    def test_connection_refused_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Refused:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _Refused:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def get(self, *args: Any, **kwargs: Any) -> Any:
                raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "Client", _Refused)

        with pytest.raises(LLMConnectionError) as excinfo:
            _probe_base_url("http://localhost:9999")
        assert "unreachable" in str(excinfo.value).lower()
        assert "scicode-lint vllm-server start" in str(excinfo.value)

    def test_non_200_status_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _NotVllm:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _NotVllm:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def get(self, *args: Any, **kwargs: Any) -> Any:
                resp = type("R", (), {"status_code": 404})()
                return resp

        monkeypatch.setattr(httpx, "Client", _NotVllm)

        with pytest.raises(LLMConnectionError) as excinfo:
            _probe_base_url("http://localhost:5001")
        assert "404" in str(excinfo.value)

    def test_success_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _OK:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _OK:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def get(self, *args: Any, **kwargs: Any) -> Any:
                return type("R", (), {"status_code": 200})()

        monkeypatch.setattr(httpx, "Client", _OK)

        # Probe returns None on success; just verify it doesn't raise
        _probe_base_url("http://localhost:5001")


class TestContextLengthError:
    """ContextLengthError still exposes its structured to_dict() for agents."""

    def test_to_dict_shape(self) -> None:
        err = ContextLengthError(
            file_path="big.py",
            estimated_tokens=500_000,
            max_tokens=32_768,
        )
        payload = err.to_dict()
        assert payload["error"] == "ContextLengthError"
        assert payload["file_path"] == "big.py"
        assert payload["overflow"] == 500_000 - 32_768
        assert any("smaller files" in s for s in payload["suggestions"])
