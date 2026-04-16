"""Tests for doc cache functionality in pattern verification."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import functions to test
from pattern_verification.deterministic.validate import (
    DOC_CACHE_MAX_AGE_DAYS,
    DocCutResponse,
    extract_doc_content_with_vllm,
    get_cache_filename,
    is_cache_valid,
)


class TestGetCacheFilename:
    """Tests for get_cache_filename function."""

    def test_basic_url(self) -> None:
        """Test basic URL generates consistent filename."""
        url = "https://pytorch.org/docs/stable/torch.html"
        filename = get_cache_filename(url)

        assert filename.endswith(".md")
        assert "pytorch_org" in filename
        assert len(filename) > 20  # Has hash component

    def test_same_url_same_filename(self) -> None:
        """Test same URL always generates same filename."""
        url = "https://example.com/page"
        assert get_cache_filename(url) == get_cache_filename(url)

    def test_different_urls_different_filenames(self) -> None:
        """Test different URLs generate different filenames."""
        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"
        assert get_cache_filename(url1) != get_cache_filename(url2)

    def test_with_pattern_id(self) -> None:
        """Test filename includes pattern ID when provided."""
        url = "https://pytorch.org/docs"
        filename_without = get_cache_filename(url)
        filename_with = get_cache_filename(url, pattern_id="pt-006")

        assert "pt-006" in filename_with
        assert "pt-006" not in filename_without


class TestIsCacheValid:
    """Tests for is_cache_valid function."""

    def test_nonexistent_file(self) -> None:
        """Test nonexistent file is not valid."""
        assert is_cache_valid(Path("/nonexistent/file.md")) is False

    def test_fresh_file_is_valid(self) -> None:
        """Test recently created file is valid."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            assert is_cache_valid(temp_path) is True
        finally:
            temp_path.unlink()

    def test_old_file_is_invalid(self) -> None:
        """Test file older than max age is invalid."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            # Set modification time to beyond max age
            old_time = time.time() - (DOC_CACHE_MAX_AGE_DAYS + 1) * 24 * 60 * 60
            import os

            os.utime(temp_path, (old_time, old_time))
            assert is_cache_valid(temp_path) is False
        finally:
            temp_path.unlink()


class TestExtractDocContentWithVllm:
    """Tests for extract_doc_content_with_vllm function."""

    def test_small_content_unchanged(self) -> None:
        """Test small content (< 50 lines) is returned unchanged with success=True."""
        small_content = "Line 1\nLine 2\nLine 3"
        result, success = extract_doc_content_with_vllm(small_content)
        assert result == small_content
        assert success is True

    def test_returns_original_on_vllm_error(self) -> None:
        """Test original content returned with success=False when vLLM fails."""
        large_content = "\n".join(f"Line {i}" for i in range(100))

        # Mock load_llm_config to raise an error (patch the source module)
        with patch(
            "scicode_lint.config.load_llm_config",
            side_effect=RuntimeError("No vLLM server"),
        ):
            result, success = extract_doc_content_with_vllm(large_content)

        # Should return original content on error with success=False
        assert result == large_content
        assert success is False

    def test_cuts_specified_lines(self) -> None:
        """Test lines are correctly removed based on vLLM response."""
        content = "\n".join(f"Line {i}" for i in range(100))

        # Mock config and client
        mock_config = MagicMock()
        mock_config.max_input_tokens = 16000
        mock_client = MagicMock()
        mock_response = DocCutResponse(cut=[[1, 5], [90, 100]])

        # Create async mock that returns our response
        async def mock_async_complete(*args: object, **kwargs: object) -> DocCutResponse:
            return mock_response

        mock_client.async_complete_structured = mock_async_complete

        with (
            patch(
                "scicode_lint.config.load_llm_config",
                return_value=mock_config,
            ),
            patch(
                "scicode_lint.llm.client.create_client",
                return_value=mock_client,
            ),
        ):
            result, success = extract_doc_content_with_vllm(content)

        # Lines 1-5 and 90-100 should be removed
        assert success is True
        result_lines = result.split("\n")
        assert "Line 0" not in result_lines  # Line 1 (1-indexed) = "Line 0"
        assert "Line 5" in result_lines  # Line 6 should still exist
        assert "Line 89" not in result_lines  # Line 90 = "Line 89"

    def test_empty_cut_returns_original(self) -> None:
        """Test empty cut response returns original content."""
        content = "\n".join(f"Line {i}" for i in range(100))

        # Mock config and client
        mock_config = MagicMock()
        mock_config.max_input_tokens = 16000
        mock_client = MagicMock()
        mock_response = DocCutResponse(cut=[])

        async def mock_async_complete(*args: object, **kwargs: object) -> DocCutResponse:
            return mock_response

        mock_client.async_complete_structured = mock_async_complete

        with (
            patch(
                "scicode_lint.config.load_llm_config",
                return_value=mock_config,
            ),
            patch(
                "scicode_lint.llm.client.create_client",
                return_value=mock_client,
            ),
        ):
            result, success = extract_doc_content_with_vllm(content)

        # All lines should be present
        assert success is True
        assert result == content


@pytest.mark.vllm
class TestVllmIntegration:
    """Integration tests for vLLM doc extraction (requires vLLM server)."""

    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "doc_with_nav.md"

    # Lines that should be kept (API documentation)
    KEEP_PATTERNS = [
        "torch.sigmoid",
        "Parameters",
        "input tensor",
        "Example",
        "torch.special.expit",
    ]

    # Lines that should be cut (navigation/boilerplate)
    CUT_PATTERNS = [
        "Skip to main content",
        "Rate this Page",
        "cookies on this site",
        "Privacy Policy",
        "Cookie Settings",
    ]

    def test_vllm_cuts_boilerplate(self) -> None:
        """Test that vLLM properly cuts navigation/boilerplate from docs.

        This test requires vLLM to be running. Skip if unavailable.
        """
        if not self.FIXTURE_PATH.exists():
            pytest.skip("Fixture file not found")

        content = self.FIXTURE_PATH.read_text()

        # Try to use real vLLM
        try:
            from scicode_lint.exceptions import LLMConnectionError
            from scicode_lint.llm.client import detect_vllm

            detect_vllm()
        except LLMConnectionError:
            pytest.skip("vLLM server not available")

        result, success = extract_doc_content_with_vllm(content)

        if not success:
            pytest.skip("vLLM call failed")

        # Check that API content is preserved
        for pattern in self.KEEP_PATTERNS:
            assert pattern in result, f"Expected to keep: {pattern}"

        # Check that boilerplate is removed
        for pattern in self.CUT_PATTERNS:
            assert pattern not in result, f"Expected to cut: {pattern}"
