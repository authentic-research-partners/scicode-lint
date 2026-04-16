"""Tests for LLM client reasoning parameter handling and retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scicode_lint.config import load_llm_config
from scicode_lint.llm.client import _TRANSIENT_RETRIES, VLLMClient
from scicode_lint.llm.models import DetectionResult


def _make_client() -> VLLMClient:
    """Create a VLLMClient with mock config for testing."""
    config = load_llm_config()
    # Override base_url to avoid auto-detection hitting real server
    config.base_url = "http://localhost:5001"
    return VLLMClient(config)


class TestBuildApiParamsThinking:
    """Tests for _build_api_params thinking/reasoning parameter handling."""

    def test_thinking_enabled_has_budget(self) -> None:
        """When thinking_budget > 0, extra_body should contain thinking.budget."""
        client = _make_client()
        params = client._build_api_params(
            "system",
            "user",
            {"type": "object"},
            "Test",
            4096,
            thinking_budget=3584,
        )
        assert "thinking" in params["extra_body"]
        assert params["extra_body"]["thinking"]["budget"] == 3584
        assert "chat_template_kwargs" not in params["extra_body"]

    def test_thinking_disabled_has_enable_thinking_false(self) -> None:
        """When thinking_budget=0, should set enable_thinking: False."""
        client = _make_client()
        params = client._build_api_params(
            "system",
            "user",
            {"type": "object"},
            "Test",
            4096,
            thinking_budget=0,
        )
        assert "thinking" not in params["extra_body"]
        assert params["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_reasoning_effort_top_level(self) -> None:
        """thinking_effort should be passed as top-level reasoning_effort."""
        client = _make_client()
        params = client._build_api_params(
            "system",
            "user",
            {"type": "object"},
            "Test",
            4096,
            thinking_budget=200,
            thinking_effort=0.3,
        )
        assert params["reasoning_effort"] == 0.3
        # Should NOT be inside extra_body.thinking
        assert "effort" not in params["extra_body"].get("thinking", {})

    def test_no_reasoning_effort_when_none(self) -> None:
        """Should not set reasoning_effort when thinking_effort is None."""
        client = _make_client()
        params = client._build_api_params(
            "system",
            "user",
            {"type": "object"},
            "Test",
            4096,
            thinking_budget=3584,
        )
        assert "reasoning_effort" not in params

    def test_no_reasoning_effort_when_budget_zero(self) -> None:
        """Should not set reasoning_effort when thinking is disabled."""
        client = _make_client()
        params = client._build_api_params(
            "system",
            "user",
            {"type": "object"},
            "Test",
            4096,
            thinking_budget=0,
            thinking_effort=0.5,
        )
        assert "reasoning_effort" not in params


# ---------------------------------------------------------------------------
# Helpers for retry tests
# ---------------------------------------------------------------------------


def _mock_completion(content: str | None, reasoning: str | None = None) -> MagicMock:
    """Build a mock ChatCompletion with given content and optional reasoning."""
    message = MagicMock()
    message.content = content
    message.reasoning = reasoning
    # Explicitly set reasoning_content=None to prevent MagicMock auto-creation.
    # The client uses `getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)`
    # and an auto-created MagicMock attribute is truthy, causing Pydantic validation failures.
    message.reasoning_content = None
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


_VALID_NO_ISSUE = '{"detected":"no","location":null,"confidence":0.9,"reasoning":"clean"}'
_VALID_YES_ISSUE = (
    '{"detected":"yes","location":{"name":"train","location_type":"function"},'
    '"confidence":0.95,"reasoning":"issue found"}'
)


def _patch_create(client: VLLMClient, mock_create: AsyncMock) -> None:
    """Patch the async OpenAI client's create method (mypy-safe)."""
    object.__setattr__(client._async_client.chat.completions, "create", mock_create)


# ---------------------------------------------------------------------------
# Transient retry: empty content
# ---------------------------------------------------------------------------


class TestTransientRetryEmptyContent:
    """Tests for retry when vLLM returns content=None (thinking exhausted budget)."""

    @pytest.mark.asyncio
    async def test_retries_on_empty_then_succeeds(self) -> None:
        """Should retry on empty content and succeed when next attempt returns content."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=None),
                _mock_completion(content=_VALID_NO_ISSUE),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "no"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_empty_retries_exhausted(self) -> None:
        """Should raise ValueError after all transient retries return empty content."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[_mock_completion(content=None)] * (_TRANSIENT_RETRIES + 1)
        )
        _patch_create(client, mock_create)

        with (
            patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ValueError, match="empty content"),
        ):
            await client.async_complete_structured("system", "user", DetectionResult)

        assert mock_create.call_count == _TRANSIENT_RETRIES + 1


# ---------------------------------------------------------------------------
# Transient retry: invalid JSON
# ---------------------------------------------------------------------------


class TestTransientRetryInvalidJSON:
    """Tests for retry when vLLM returns invalid JSON (rare network glitch)."""

    @pytest.mark.asyncio
    async def test_retries_on_invalid_json_then_succeeds(self) -> None:
        """Should retry on invalid JSON and succeed on next attempt."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content="not valid json{{{"),
                _mock_completion(content=_VALID_NO_ISSUE),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "no"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_json_retries_exhausted(self) -> None:
        """Should raise after all retries return invalid JSON."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[_mock_completion(content="bad json")] * (_TRANSIENT_RETRIES + 1)
        )
        _patch_create(client, mock_create)

        with (
            patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ValueError, match="JSON parse"),
        ):
            await client.async_complete_structured("system", "user", DetectionResult)


# ---------------------------------------------------------------------------
# Transient retry: mixed failures
# ---------------------------------------------------------------------------


class TestTransientRetryMixed:
    """Tests for mixed transient failure modes."""

    @pytest.mark.asyncio
    async def test_empty_then_bad_json_then_success(self) -> None:
        """Should handle empty → bad JSON → valid sequence."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=None),
                _mock_completion(content="truncated{"),
                _mock_completion(content=_VALID_NO_ISSUE),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "no"
        assert mock_create.call_count == 3

    @pytest.mark.asyncio
    async def test_valid_on_first_attempt(self) -> None:
        """No retries needed when first attempt succeeds."""
        client = _make_client()
        mock_create = AsyncMock(return_value=_mock_completion(content=_VALID_NO_ISSUE))
        _patch_create(client, mock_create)

        result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "no"
        assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_schema_validation_error_not_retried(self) -> None:
        """Schema validation errors (not JSON parse) should not be retried."""
        client = _make_client()
        # Valid JSON but invalid schema (confidence > 1.0)
        bad_schema = '{"detected":"no","location":null,"confidence":5.0,"reasoning":"x"}'
        mock_create = AsyncMock(return_value=_mock_completion(content=bad_schema))
        _patch_create(client, mock_create)

        with pytest.raises(ValueError, match="schema validation"):
            await client.async_complete_structured("system", "user", DetectionResult)

        # Should NOT retry — only 1 call
        assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# Missing-location correction retry (Layer 7)
# ---------------------------------------------------------------------------


_MISSING_LOCATION = '{"detected":"yes","location":null,"confidence":0.9,"reasoning":"issue found"}'


class TestMissingLocationRetry:
    """Tests for the correction-prompt retry when the model detects an issue
    but doesn't provide a location (business-rule violation, not transient).
    """

    @pytest.mark.asyncio
    async def test_correction_retry_recovers(self) -> None:
        """First call: detected=yes without location. Second call: valid yes with location."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=_MISSING_LOCATION),
                _mock_completion(content=_VALID_YES_ISSUE),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "yes"
        assert result.location is not None
        assert result.location.name == "train"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_flip_to_no_after_correction_fails(self) -> None:
        """Both attempts: detected=yes without location → final fallback flips to 'no'."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=_MISSING_LOCATION),
                _mock_completion(content=_MISSING_LOCATION),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        # Fallback converts detected='yes' with no location into detected='no'
        assert result.detected == "no"
        assert result.location is None
        assert "could not identify specific location" in result.reasoning.lower()
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_correction_prompt_sent_with_previous_response(self) -> None:
        """The correction prompt should be appended to the user prompt on retry."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=_MISSING_LOCATION),
                _mock_completion(content=_VALID_YES_ISSUE),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            await client.async_complete_structured(
                "system", "original-user-prompt", DetectionResult
            )

        # First call: just the original prompt
        first_call_kwargs = mock_create.call_args_list[0].kwargs
        assert "CORRECTION REQUIRED" not in first_call_kwargs["messages"][1]["content"]
        # Second call: original + correction prompt
        second_call_kwargs = mock_create.call_args_list[1].kwargs
        second_user_msg = second_call_kwargs["messages"][1]["content"]
        assert "original-user-prompt" in second_user_msg
        assert "CORRECTION REQUIRED" in second_user_msg
