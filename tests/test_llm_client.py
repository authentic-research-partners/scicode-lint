"""Tests for LLM client reasoning parameter handling and retry logic."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scicode_lint.config import load_llm_config
from scicode_lint.llm.client import (
    _THINKING_FLOOR,
    _THINKING_OFF,
    _TRANSIENT_RETRIES,
    StructuredOutputRetryError,
    VLLMClient,
    _step_down_thinking,
)
from scicode_lint.llm.models import DetectionResult


def _make_client() -> VLLMClient:
    """Create a VLLMClient with mock config for testing."""
    config = load_llm_config()
    # Override base_url to avoid auto-detection hitting real server
    config.base_url = "http://localhost:5001"
    client = VLLMClient(config)
    # Mark as already-probed so tests never hit the real reachability probe —
    # the zero-services rule. The LLM call itself is mocked per-test.
    client._probed = True
    return client


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


def _mock_completion(
    content: str | None,
    reasoning: str | None = None,
    finish_reason: str = "stop",
) -> MagicMock:
    """Build a mock ChatCompletion with given content, optional reasoning, and finish_reason."""
    message = MagicMock()
    message.content = content
    message.reasoning = reasoning
    # Explicitly set reasoning_content=None to prevent MagicMock auto-creation.
    # The client uses `getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)`
    # and an auto-created MagicMock attribute is truthy, causing Pydantic validation failures.
    message.reasoning_content = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
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
            pytest.raises(StructuredOutputRetryError, match="JSON parse"),
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


# ---------------------------------------------------------------------------
# Transient retry: schema-validation over-runs (unbounded wire)
# ---------------------------------------------------------------------------


class TestTransientRetrySchemaValidation:
    """Schema-validation over-runs are retried, not hard-failed or clipped.

    ``vllm_schema`` strips the length caps (``maxLength``/``maxItems``) from the wire,
    so a string/list field can exceed its Pydantic cap. We resample a fresh response —
    which at temp>0 almost always complies — and surface the failure loudly only if
    every retry over-runs. (These tests mock the vLLM response directly, so they can
    use any value that fails ``model_validate`` to exercise the resample path,
    including an out-of-range confidence that the real decoder would itself reject.)
    """

    _OUT_OF_RANGE = '{"detected":"no","location":null,"confidence":5.0,"reasoning":"x"}'

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_then_succeeds(self) -> None:
        """Synthetic over-run (out-of-range confidence) resamples and recovers.

        In production the decoder enforces the 0-1 range; this is a mocked value used
        only to drive a ValidationError through the resample path.
        """
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=self._OUT_OF_RANGE),
                _mock_completion(content=_VALID_NO_ISSUE),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "no"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_over_length_reasoning_then_succeeds(self) -> None:
        """Over-length reasoning resamples and recovers (no silent truncation)."""
        client = _make_client()
        over_length = (
            '{"detected":"no","location":null,"confidence":0.9,"reasoning":"' + "x" * 401 + '"}'
        )
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=over_length),
                _mock_completion(content=_VALID_NO_ISSUE),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "no"
        assert len(result.reasoning) <= 400
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_schema_retries_exhausted(self) -> None:
        """Every attempt over-runs → raise loudly (no silent clip) after the budget."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[_mock_completion(content=self._OUT_OF_RANGE)] * (_TRANSIENT_RETRIES + 1)
        )
        _patch_create(client, mock_create)

        with (
            patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(StructuredOutputRetryError, match="schema validation"),
        ):
            await client.async_complete_structured("system", "user", DetectionResult)

        assert mock_create.call_count == _TRANSIENT_RETRIES + 1


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


# ---------------------------------------------------------------------------
# Thinking-budget ladder
# ---------------------------------------------------------------------------


def _budget_of(call_args: Any) -> int:
    """Extract the thinking.budget (or 0 if disabled) from a mocked create() call."""
    extra_body = call_args.kwargs["extra_body"]
    if "thinking" in extra_body:
        budget: int = extra_body["thinking"]["budget"]
        return budget
    # enable_thinking=False → budget effectively 0
    assert extra_body["chat_template_kwargs"]["enable_thinking"] is False
    return 0


class TestStepDownThinkingContract:
    """Tests for _step_down_thinking pure helper."""

    def test_halves_when_above_floor(self) -> None:
        """Budget well above the floor halves on each step."""
        assert _step_down_thinking(3584) == 1792
        assert _step_down_thinking(1792) == 896
        assert _step_down_thinking(896) == 448

    def test_drops_to_off_below_floor(self) -> None:
        """When halving falls below the floor, drops straight to 0 (off)."""
        # Halved values below _THINKING_FLOOR must drop to 0
        below = _THINKING_FLOOR * 2 - 1  # halves to < floor
        assert _step_down_thinking(below) == _THINKING_OFF
        # A budget equal to the floor halves to below floor → drops to 0
        assert _step_down_thinking(_THINKING_FLOOR) == _THINKING_OFF

    def test_returns_none_when_already_off(self) -> None:
        """Budget 0 or negative means ladder exhausted."""
        assert _step_down_thinking(0) is None
        assert _step_down_thinking(-1) is None

    def test_small_budget_drops_to_off_immediately(self) -> None:
        """Classify-style small budgets (e.g. 200) step down sensibly."""
        # 200 // 2 = 100 — still >= floor (64), so one halving step first
        assert _step_down_thinking(200) == 100
        # 100 // 2 = 50 < floor → off
        assert _step_down_thinking(100) == _THINKING_OFF


class TestThinkingLadderOnLength:
    """Tests for ladder stepping on finish_reason=length (not content=None)."""

    @pytest.mark.asyncio
    async def test_finish_reason_length_triggers_ladder(self) -> None:
        """finish_reason='length' with truncated JSON should step thinking down."""
        client = _make_client()
        # Truncated JSON + finish_reason=length → ladder branch (not same-budget JSON retry)
        truncated_json = '{"detected":"no","location":null,"confidence":0.9,"reason'
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=truncated_json, finish_reason="length"),
                _mock_completion(content=_VALID_NO_ISSUE, finish_reason="stop"),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.async_complete_structured("system", "user", DetectionResult)

        assert result.detected == "no"
        assert mock_create.call_count == 2
        # Second call must use a smaller thinking budget than the first
        first_budget = _budget_of(mock_create.call_args_list[0])
        second_budget = _budget_of(mock_create.call_args_list[1])
        assert second_budget < first_budget

    @pytest.mark.asyncio
    async def test_empty_content_steps_down_budget(self) -> None:
        """content=None (any finish_reason) should step the thinking budget down."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=None, finish_reason="length"),
                _mock_completion(content=_VALID_NO_ISSUE, finish_reason="stop"),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            await client.async_complete_structured("system", "user", DetectionResult)

        # Budget must step down between call 0 and call 1
        first_budget = _budget_of(mock_create.call_args_list[0])
        second_budget = _budget_of(mock_create.call_args_list[1])
        assert second_budget == _step_down_thinking(first_budget)

    @pytest.mark.asyncio
    async def test_ladder_walks_three_steps(self) -> None:
        """Three consecutive length failures walk the ladder each step."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=None, finish_reason="length"),
                _mock_completion(content=None, finish_reason="length"),
                _mock_completion(content=_VALID_NO_ISSUE, finish_reason="stop"),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            await client.async_complete_structured("system", "user", DetectionResult)

        budgets = [_budget_of(c) for c in mock_create.call_args_list]
        assert len(budgets) == 3
        assert budgets[1] == _step_down_thinking(budgets[0])
        assert budgets[2] == _step_down_thinking(budgets[1])

    @pytest.mark.asyncio
    async def test_respects_per_call_budget_override(self) -> None:
        """Ladder starts from the per-call thinking_budget override, not the config default."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=None, finish_reason="length"),
                _mock_completion(content=_VALID_NO_ISSUE, finish_reason="stop"),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            await client.async_complete_structured(
                "system", "user", DetectionResult, thinking_budget=200
            )

        assert _budget_of(mock_create.call_args_list[0]) == 200
        # 200 // 2 = 100 → still >= floor → 100
        assert _budget_of(mock_create.call_args_list[1]) == 100


class TestThinkingLadderDoesNotTriggerOnJsonParse:
    """JSON parse errors should NOT step thinking down (same-budget backoff)."""

    @pytest.mark.asyncio
    async def test_json_parse_keeps_same_budget(self) -> None:
        """JSON parse error (finish_reason=stop) retries with the SAME thinking budget."""
        client = _make_client()
        # Invalid JSON but finish_reason=stop → same-budget backoff, not ladder
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content="not valid json{{{", finish_reason="stop"),
                _mock_completion(content=_VALID_NO_ISSUE, finish_reason="stop"),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            await client.async_complete_structured("system", "user", DetectionResult)

        first_budget = _budget_of(mock_create.call_args_list[0])
        second_budget = _budget_of(mock_create.call_args_list[1])
        assert first_budget == second_budget, "JSON parse error must not step down the ladder"


class TestThinkingLadderExhaustion:
    """Ladder-exhausted behavior: all retries fail → raise."""

    @pytest.mark.asyncio
    async def test_exhausts_ladder_after_max_retries(self) -> None:
        """After _TRANSIENT_RETRIES+1 length failures, raise with ladder context."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[_mock_completion(content=None, finish_reason="length")]
            * (_TRANSIENT_RETRIES + 1)
        )
        _patch_create(client, mock_create)

        with (
            patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ValueError, match="length-based failure"),
        ):
            await client.async_complete_structured("system", "user", DetectionResult)

        assert mock_create.call_count == _TRANSIENT_RETRIES + 1
        # All calls must have used progressively smaller budgets
        budgets = [_budget_of(c) for c in mock_create.call_args_list]
        for prev, curr in zip(budgets, budgets[1:], strict=False):
            assert curr < prev, f"Expected strict step-down, got {budgets}"

    @pytest.mark.asyncio
    async def test_ladder_exhausted_when_starting_at_off(self) -> None:
        """If a caller overrides thinking_budget=0, first length failure raises immediately."""
        client = _make_client()
        mock_create = AsyncMock(return_value=_mock_completion(content=None, finish_reason="length"))
        _patch_create(client, mock_create)

        with (
            patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ValueError, match="length-based failure"),
        ):
            await client.async_complete_structured(
                "system", "user", DetectionResult, thinking_budget=0
            )

        # Ladder is exhausted from the start → single call, no retry
        assert mock_create.call_count == 1


class TestThinkingLadderMixed:
    """Mixed branches: length failures and JSON parse errors compose correctly."""

    @pytest.mark.asyncio
    async def test_length_then_json_parse_then_success(self) -> None:
        """Length failure steps down, JSON parse keeps that stepped-down budget, then success."""
        client = _make_client()
        mock_create = AsyncMock(
            side_effect=[
                _mock_completion(content=None, finish_reason="length"),
                _mock_completion(content="bad json{", finish_reason="stop"),
                _mock_completion(content=_VALID_NO_ISSUE, finish_reason="stop"),
            ]
        )
        _patch_create(client, mock_create)

        with patch("scicode_lint.llm.client.asyncio.sleep", new_callable=AsyncMock):
            await client.async_complete_structured("system", "user", DetectionResult)

        budgets = [_budget_of(c) for c in mock_create.call_args_list]
        # Call 0 → Call 1: length failure steps down
        assert budgets[1] == _step_down_thinking(budgets[0])
        # Call 1 → Call 2: JSON parse does NOT step down
        assert budgets[2] == budgets[1]
