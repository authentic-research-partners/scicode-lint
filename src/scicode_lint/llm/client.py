"""LLM client for vLLM with structured output.

Structured Output with Qwen3
=============================

Qwen3 outputs reasoning in `<think>...</think>` blocks before the JSON answer.
This thinking phase is essential for accuracy.

Approach
--------
We use the OpenAI-standard `response_format: json_schema` for constrained decoding,
combined with vLLM's Qwen3 reasoning parser (`--reasoning-parser qwen3`).

The reasoning parser separates thinking from content server-side:
- `message.content` → clean JSON (guaranteed valid by XGrammar/Outlines)
- `message.reasoning` → thinking content (model's chain-of-thought)

No client-side `<think>` tag stripping needed.

Note: `--reasoning-parser qwen3` is Qwen3-specific. Other thinking models require
their own parser (e.g., `deepseek` for DeepSeek-R1). Non-thinking models don't
need any parser. (vLLM v0.18+ removed the separate `--enable-reasoning` flag;
setting `--reasoning-parser` alone now enables reasoning implicitly.)

Related vLLM features:
- `thinking.budget=N` → hard cap on thinking tokens (abruptly stops)
- `thinking.effort=F` → soft guide for thinking depth (0.0-1.0)
- Use both together: effort guides depth, budget prevents runaway
- `chat_template_kwargs.enable_thinking=False` → disables thinking entirely

Usage
-----
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": {
            "name": "MySchema",
            "schema": json_schema,
            "strict": True,
        }},
    )
    content = completion.choices[0].message.content      # clean JSON
    thinking = completion.choices[0].message.reasoning    # thinking (if any)
"""

import asyncio
import json
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx
from loguru import logger
from pydantic import BaseModel, ValidationError

from scicode_lint.config import LLMConfig
from scicode_lint.exceptions import LLMConnectionError, SciCodeLintError
from scicode_lint.llm.models import vllm_schema

# Suppress Pydantic serialization warning for OpenAI SDK's ParsedChatCompletionMessage.parsed field
# Root cause: When using OpenAI SDK's structured outputs with LangChain, the parsed field
# (TypeVar defaulting to None) receives our Pydantic model, triggering serialization warnings.
# This is a known upstream issue in openai-python v2.21.0+
# Reference: https://github.com/openai/openai-python/issues/2872
# The warning is harmless - structured output parsing works correctly despite the warning.
# This targeted suppression only affects the specific pydantic.main module warning.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings:", category=UserWarning, module="pydantic.main"
)

# Transient retry budget: 3 total attempts (initial + 2 retries).
# Covers content=None (thinking exhausted max_tokens) and rare network glitches.
# See CONSTRAINED_DECODING.md § "The three failure modes" for why content=None happens.
_TRANSIENT_RETRIES = 2


class StructuredOutputRetryError(ValueError):
    """A vLLM structured-output response that failed to parse or validate.

    Raised by ``_handle_response`` for the two recoverable response failures:
    ``JSONDecodeError`` (rare wire/streaming glitch) and Pydantic ``ValidationError``
    (a field exceeded its cap now that length bounds are stripped from the wire).
    The transient-retry loop catches this *type* — not a substring of the message —
    and resamples. Subclasses ``ValueError`` so existing value-error handling and the
    error message contract are preserved.

    Attributes:
        kind: ``"JSON parse"`` or ``"schema validation"`` — for logging only.
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


# Thinking-budget ladder: on finish_reason=length or content=None, halve the
# thinking budget and retry. Once halving would fall below _THINKING_FLOOR,
# drop straight to 0 (thinking disabled) as the final ladder step.
#
# Rationale: same-budget retries on length-based failures almost certainly fail
# again — if 3584 thinking tokens exhausted max_completion_tokens the first
# time, they will the second time too. Stepping down recovers on the retry
# instead of burning all attempts at the same broken budget.
_THINKING_FLOOR = 64
_THINKING_OFF = 0


def _step_down_thinking(current: int) -> int | None:
    """Return the next smaller thinking budget, or ``None`` if ladder exhausted.

    Halves the current budget. When the halved value falls below
    ``_THINKING_FLOOR``, returns ``0`` (thinking disabled) as the final
    ladder step — tiny budgets are rarely enough for useful reasoning, so
    we skip straight to "off" rather than crawl through sub-floor values.
    Returns ``None`` when called with ``current <= 0`` (already at floor).

    Args:
        current: Current thinking budget in tokens.

    Returns:
        Next budget to try, or ``None`` if the ladder has been exhausted.
    """
    if current <= 0:
        return None
    halved = current // 2
    return halved if halved >= _THINKING_FLOOR else _THINKING_OFF


T = TypeVar("T", bound=BaseModel)


class MissingLocationError(SciCodeLintError):
    """Raised when LLM returns detected='yes' but no location.

    This is a specific validation error that indicates the model understood
    there was an issue but failed to identify where it occurs in the code.
    The error message includes details for debugging and logging.
    """

    def __init__(self, detected: str, reasoning: str, confidence: float = 0.0):
        self.detected = detected
        self.reasoning = reasoning
        self.confidence = confidence
        super().__init__(
            f"LLM returned detected='{detected}' but no location. Reasoning: {reasoning[:100]}..."
        )


# Correction prompt added when retrying after missing location
# Includes previous response to help model understand the issue
MISSING_LOCATION_CORRECTION = """

CORRECTION REQUIRED - Your previous response was INVALID:
- You said: detected="{detected}"
- Your reasoning: "{reasoning}"
- But you provided: location=null

This is invalid. If you detect an issue, you MUST provide the location.
Identify the function, class, or method name where the issue occurs.

OPTIONS:
1. If you CAN identify the location: provide it as:
   "location": {{"name": "function_name", "location_type": "function", "near_line": 15}}
2. If you CANNOT identify a specific location: change your answer to detected="no"
   (An issue without identifiable location is not actionable)

Respond with valid JSON including proper location OR change to detected="no".
"""


class LLMClient(ABC):
    """Abstract base class for LLM clients with structured output."""

    @abstractmethod
    async def async_complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        **overrides: Any,
    ) -> T:
        """
        Get structured completion from LLM asynchronously.

        Args:
            system_prompt: System message
            user_prompt: User message
            schema: Pydantic model class for response structure
            **overrides: Per-call overrides merged into extra_body.
                Supported keys:
                - thinking_budget: int — hard cap on thinking tokens
                - thinking_effort: float — soft guide (0.0-1.0)
                - temperature: float — override sampling temperature

        Returns:
            Validated Pydantic model instance
        """
        pass

    @abstractmethod
    def get_max_model_len(self) -> int:
        """
        Get maximum context length in tokens.

        Returns:
            Maximum context length supported by the model
        """
        pass


class VLLMClient(LLMClient):
    """
    Client for vLLM servers with structured output support.

    Supports local and remote vLLM servers only.
    Does NOT support commercial APIs (OpenAI, Anthropic, etc.) to avoid accidental costs.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize vLLM client.

        Note: Only works with vLLM servers (local or remote).
        Uses OpenAI-compatible API format but is NOT for commercial APIs.

        The configured base_url is probed lazily on the first structured
        call (not at construction time) so that constructing a client does
        not require a live server — useful for unit tests and deferred
        setup. The probe fast-fails with ``LLMConnectionError`` if the URL
        is unreachable, avoiding the multi-minute openai retry loop on
        a refused connection.
        """
        self.config = config
        self._max_model_len: int = 0
        self._probed: bool = False

        # Use OpenAI SDK for vLLM's OpenAI-compatible API
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK is required for vLLM client. Install with: pip install openai"
            ) from e

        # Create async client for concurrent requests
        self._async_client = AsyncOpenAI(
            base_url=self.config.base_url + "/v1",
            api_key="dummy",  # vLLM doesn't require API keys
            timeout=float(self.config.timeout),
        )

        self._max_model_len = self.config.max_model_len

    async def _ensure_reachable(self) -> None:
        """Probe base_url once on first use; raise LLMConnectionError on failure.

        Awaited from the async call path — the probe uses ``httpx.AsyncClient`` so
        it never blocks the event loop (async-native rule).
        """
        if self._probed:
            return
        await _probe_base_url(self.config.base_url)
        self._probed = True

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """
        Strip markdown code fences from response.

        Models often wrap JSON in ```json ... ``` despite being told not to.
        """
        text = text.strip()

        # Remove ```json ... ``` fences
        if text.startswith("```json"):
            text = text[7:]  # Remove ```json
        elif text.startswith("```"):
            text = text[3:]  # Remove ```

        if text.endswith("```"):
            text = text[:-3]  # Remove trailing ```

        return text.strip()

    @staticmethod
    def _parse_and_validate(
        response_text: str, schema: type[T], thinking_content: str | None = None
    ) -> T:
        """
        Parse JSON response and validate against Pydantic schema.

        With --reasoning-parser set, vLLM separates thinking into a 'reasoning'
        field server-side. The content field contains clean JSON. Markdown
        fence stripping is kept as a safety net.

        Args:
            response_text: JSON string from LLM (already separated from thinking by vLLM)
            schema: Pydantic model class to validate against
            thinking_content: Thinking content from vLLM's reasoning field (if available)

        Returns:
            Validated Pydantic model instance

        Raises:
            json.JSONDecodeError: If JSON parsing fails
            MissingLocationError: If detected='yes'/'context-dependent' but location is None
            ValidationError: If other schema validation fails
        """
        cleaned_text = VLLMClient._strip_markdown_fences(response_text)

        if thinking_content:
            logger.debug(f"Received thinking content ({len(thinking_content)} chars)")

        try:
            response_data = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Original response: {response_text[:500]}")
            logger.error(f"Cleaned response: {cleaned_text[:500]}")
            raise

        # If schema has 'thinking' field, add extracted thinking content
        if thinking_content and "thinking" in schema.model_fields:
            response_data["thinking"] = thinking_content

        # Check for missing location before full validation (to raise specific error)
        detected = response_data.get("detected")
        location = response_data.get("location")
        if detected in ("yes", "context-dependent") and not location:
            reasoning = response_data.get("reasoning", "")
            confidence = response_data.get("confidence", 0.0)
            raise MissingLocationError(
                detected=detected, reasoning=reasoning, confidence=confidence
            )

        try:
            result = schema.model_validate(response_data)
        except ValidationError as e:
            logger.error(f"Failed to validate response against schema {schema.__name__}: {e}")
            logger.error(f"Response data: {response_data}")
            raise

        return result

    def _handle_response(
        self,
        response_text: str,
        schema: type[T],
        attempt: int,
        max_attempts: int,
        start_time: float,
        label: str,
        thinking_content: str | None = None,
    ) -> tuple[T | None, str | None]:
        """Handle LLM response: parse, validate, and manage retry logic.

        Args:
            response_text: Raw LLM response text
            schema: Pydantic model class to validate against
            attempt: Current attempt number (0-indexed)
            max_attempts: Maximum number of attempts
            start_time: Time when the call started (for elapsed logging)
            label: Log label ("vLLM call" or "Async vLLM call")
            thinking_content: Thinking content from vLLM's reasoning field

        Returns:
            Tuple of (result, correction_prompt). If result is not None, the call
            succeeded. If result is None, correction_prompt contains the retry prompt.

        Raises:
            StructuredOutputRetryError: If JSON parsing or schema validation fails
                (the transient-retry loop resamples on this type).
        """
        try:
            result = self._parse_and_validate(response_text, schema, thinking_content)
            elapsed = time.time() - start_time
            logger.info(f"{label} completed in {elapsed:.2f}s for {schema.__name__}")
            return result, None

        except MissingLocationError as e:
            if attempt < max_attempts - 1:
                correction = MISSING_LOCATION_CORRECTION.format(
                    detected=e.detected, reasoning=e.reasoning[:200]
                )
                logger.warning(
                    f"Missing location (detected='{e.detected}'), retrying with correction"
                )
                return None, correction
            else:
                # Final attempt failed - flip to "no" since we can't identify location
                logger.warning(
                    f"Missing location after retry - flipping to 'no': {e.reasoning[:100]}"
                )
                flipped_data = {
                    "detected": "no",
                    "location": None,
                    "confidence": 0.5,
                    "reasoning": (
                        f"Originally detected='{e.detected}' but could not identify "
                        f"specific location. Flipped to 'no' since unlocatable "
                        f"issues are not actionable. Original reasoning: {e.reasoning[:150]}"
                    ),
                }
                return schema.model_validate(flipped_data), None

        except (json.JSONDecodeError, ValidationError) as e:
            kind = "JSON parse" if isinstance(e, json.JSONDecodeError) else "schema validation"
            logger.error(f"LLM response {kind} failed")
            raise StructuredOutputRetryError(
                f"LLM returned invalid response ({kind} error). "
                f"Model not producing valid structured output. Error: {e}",
                kind=kind,
            ) from e

    def _build_api_params(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build parameters for the OpenAI API call.

        Args:
            **overrides: Per-call overrides. Supported keys:
                - thinking_budget: int — hard cap on thinking tokens
                - thinking_effort: float — soft guide for thinking depth (0.0-1.0)
                - temperature: float — override sampling temperature
        """
        thinking_budget = overrides.pop("thinking_budget", self.config.thinking_budget)
        thinking_effort = overrides.pop("thinking_effort", self.config.thinking_effort)
        temperature = overrides.pop("temperature", self.config.temperature)

        extra_body: dict[str, Any] = {"top_k": self.config.top_k}
        if thinking_budget > 0:
            # Active thinking: budget as hard cap
            extra_body["thinking"] = {"budget": thinking_budget}
        else:
            # Thinking disabled: explicitly tell model to skip <think> blocks
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        params: dict[str, Any] = {
            "model": self.config.model_served_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                },
            },
            "extra_body": extra_body,
            "temperature": temperature,
            "top_p": self.config.top_p,
            "max_tokens": max_tokens,
        }

        # reasoning_effort as top-level param (OpenAI standard, more portable)
        if thinking_budget > 0 and thinking_effort is not None:
            params["reasoning_effort"] = thinking_effort

        return params

    async def async_complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        **overrides: Any,
    ) -> T:
        """
        Get structured completion from vLLM asynchronously using OpenAI SDK.

        Uses response_format json_schema for schema-constrained output.
        vLLM's reasoning parser separates thinking from JSON server-side.

        Two retry layers:
        1. **Transient retry** — two branches depending on failure kind:
           - *Length-based failure* (``finish_reason='length'`` or ``content=None``)
             means thinking exhausted ``max_completion_tokens`` before JSON
             completed. Same-budget retries almost always fail again, so we
             step the thinking budget down the ladder (see ``_step_down_thinking``)
             and retry with a short 0.2s delay.
           - *JSON parse error* is a rare network/streaming glitch. We retry
             with the **same** budget using exponential backoff (0.5s, 1.0s),
             since dropping thinking wouldn't help here.
           Up to ``_TRANSIENT_RETRIES`` retries total across both branches.
        2. **Missing-location retry** — retries once with a correction prompt
           when model returns detected='yes' but no location.

        Args:
            **overrides: Per-call overrides (thinking_budget, temperature, etc.)

        Concurrent requests with shared prefixes benefit from automatic prefix caching.
        Uses OpenAI AsyncClient for true async concurrency.

        Raises:
            LLMConnectionError: If the configured vLLM server is unreachable
                (probed on first call, cached thereafter).
        """
        await self._ensure_reachable()
        start_time = time.time()
        logger.debug(f"Starting async vLLM call for {schema.__name__}")

        json_schema = vllm_schema(schema)
        max_tokens = self.config.max_completion_tokens or 2048
        correction_prompt: str | None = None
        max_location_attempts = 2

        # Track the current thinking budget across retries. Starts from per-call
        # override (if any) or the config default, and is stepped down by the
        # ladder on length-based failures. Persists across location-retry loops
        # too — if this prompt/model combo needs less thinking, the correction
        # prompt benefits from the same lower budget.
        current_thinking_budget: int = overrides.get("thinking_budget", self.config.thinking_budget)

        for location_attempt in range(max_location_attempts):
            current_prompt = user_prompt
            if correction_prompt:
                current_prompt = user_prompt + correction_prompt
                logger.info(f"Retrying with correction prompt (attempt {location_attempt + 1})")

            # Transient retry loop with two branches:
            # - Length-based failure (finish_reason=length or content=None)
            #   → step thinking budget down the ladder, short 0.2s delay.
            # - JSON parse error (rare network/streaming glitch)
            #   → same budget, exponential backoff.
            last_transient_error: Exception | None = None
            for transient_attempt in range(_TRANSIENT_RETRIES + 1):
                call_overrides = {**overrides, "thinking_budget": current_thinking_budget}
                try:
                    completion = await self._async_client.chat.completions.create(
                        **self._build_api_params(
                            system_prompt,
                            current_prompt,
                            json_schema,
                            schema.__name__,
                            max_tokens,
                            **call_overrides,
                        )
                    )
                except Exception as e:
                    # Map connection-class errors (vLLM died mid-call, network
                    # dropped) to the typed hierarchy so CLI exit codes and
                    # `except SciCodeLintError` consumers stay consistent.
                    # Schema/validation/structured-output failures remain
                    # RuntimeError — they're not domain errors.
                    from openai import APIConnectionError, APITimeoutError

                    if isinstance(e, (APIConnectionError, APITimeoutError)):
                        raise LLMConnectionError(
                            f"vLLM server at {self.config.base_url} became "
                            f"unreachable mid-request: {type(e).__name__}: {e}\n\n"
                            "Check server health: scicode-lint vllm-server status"
                        ) from e
                    raise RuntimeError(
                        f"vLLM structured output failed: {e}\n"
                        "response_format json_schema with XGrammar/Outlines is required."
                    ) from e

                message = completion.choices[0].message
                response_text = message.content
                finish_reason = completion.choices[0].finish_reason
                thinking_content = getattr(message, "reasoning", None) or getattr(
                    message, "reasoning_content", None
                )

                # Length-based failure: either content=None (thinking consumed
                # the entire budget before JSON started) or finish_reason=length
                # with truncated JSON. Step the thinking budget down and retry.
                length_failure = response_text is None or finish_reason == "length"
                if length_failure:
                    failure_desc = (
                        "empty content"
                        if response_text is None
                        else f"finish_reason={finish_reason}"
                    )
                    last_transient_error = ValueError(
                        f"vLLM length-based failure for {schema.__name__} "
                        f"({failure_desc}). Thinking budget {current_thinking_budget} "
                        f"too aggressive; max_completion_tokens={max_tokens}."
                    )
                    next_budget = _step_down_thinking(current_thinking_budget)
                    if next_budget is None or transient_attempt >= _TRANSIENT_RETRIES:
                        raise last_transient_error
                    logger.warning(
                        f"vLLM length-based failure ({failure_desc}) for {schema.__name__} "
                        f"(attempt {transient_attempt + 1}/{_TRANSIENT_RETRIES + 1}), "
                        f"stepping thinking budget {current_thinking_budget} → {next_budget}"
                    )
                    current_thinking_budget = next_budget
                    await asyncio.sleep(0.2)
                    continue

                # Try to parse. Two recoverable "bad response" modes share this
                # same-budget resample backoff:
                #  - JSONDecodeError: rare network/streaming glitch.
                #  - schema-validation over-run: vllm_schema strips length caps
                #    (maxLength/maxItems) from the wire, so a string/list field can
                #    exceed its Pydantic cap. A fresh sample at temp>0 almost always
                #    complies. We rerun rather than truncate, so the over-run is
                #    surfaced and re-attempted — never silently clipped.
                try:
                    result, correction_prompt = self._handle_response(
                        response_text,
                        schema,
                        location_attempt,
                        max_location_attempts,
                        start_time,
                        "Async vLLM call",
                        thinking_content,
                    )
                    break  # Parsed successfully (result may be None for correction retry)
                except StructuredOutputRetryError as e:
                    # Type-based: both parse and validation over-runs resample.
                    # Any other exception propagates (fail loudly).
                    last_transient_error = e
                    if transient_attempt < _TRANSIENT_RETRIES:
                        delay = 0.5 * (transient_attempt + 1)
                        logger.warning(
                            f"vLLM {e.kind} for {schema.__name__} "
                            f"(attempt {transient_attempt + 1}/{_TRANSIENT_RETRIES + 1}), "
                            f"resampling in {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise
            else:
                # Loop exhausted without break — all transient retries failed
                assert last_transient_error is not None
                raise last_transient_error

            if result is not None:
                return result

        raise RuntimeError("Unexpected: all attempts exhausted without result or exception")

    def get_max_model_len(self) -> int:
        """Get maximum context length in tokens.

        Returns:
            Maximum context length supported by the model

        Example:
            >>> client = VLLMClient(config)
            >>> max_len = client.get_max_model_len()
            >>> print(f"Max tokens: {max_len}")
        """
        return self._max_model_len


async def _probe_base_url(base_url: str, timeout: float = 3.0) -> None:
    """Verify a base URL responds at ``/v1/models``; raise fast on failure.

    Awaited from ``VLLMClient._ensure_reachable`` on the first structured call so
    callers get an immediate ``LLMConnectionError`` instead of a multi-minute openai
    retry loop. Uses ``httpx.AsyncClient`` so it never blocks the event loop.

    Args:
        base_url: Root URL of the vLLM server (no trailing ``/v1``).
        timeout: Per-probe timeout in seconds.

    Raises:
        LLMConnectionError: If the probe fails for any reason.
    """
    probe_url = f"{base_url.rstrip('/')}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(probe_url)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
        raise LLMConnectionError(
            f"vLLM server at {base_url} is unreachable: {type(e).__name__}: {e}\n\n"
            "Start a local server with:\n"
            "  scicode-lint vllm-server start\n\n"
            "Or pass --vllm-url pointing at a running server."
        ) from e

    if response.status_code != 200:
        raise LLMConnectionError(
            f"vLLM server at {base_url} returned HTTP {response.status_code} "
            f"from {probe_url}. The server is reachable but not serving the "
            "OpenAI-compatible API. Check that vLLM is running, not a "
            "different service on the same port."
        )


def detect_vllm() -> tuple[str, str | None]:
    """
    Auto-detect vLLM server and model.

    Tries vLLM on common ports (5001, 8000).

    Returns:
        Tuple of (base_url, model_name)
        model_name is None if it couldn't be detected

    Raises:
        LLMConnectionError: If no vLLM server is available
    """
    # Try vLLM on common ports
    vllm_urls = ["http://localhost:5001", "http://localhost:8000"]
    for url in vllm_urls:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{url}/v1/models")
                if response.status_code == 200:
                    data = response.json()
                    # Try to get the first available model
                    model_name = None
                    if "data" in data and len(data["data"]) > 0:
                        model_name = data["data"][0].get("id")
                    logger.info(f"Detected vLLM at {url} with model: {model_name}")
                    return (url, model_name)
        except (httpx.ConnectError, httpx.TimeoutException):
            continue

    raise LLMConnectionError(
        "No vLLM server detected. Please start vLLM:\n\n"
        "  scicode-lint vllm-server start\n\n"
        "Requires: podman or docker + nvidia-container-toolkit\n"
        "See INSTALLATION.md for setup instructions."
    )


def create_client(config: LLMConfig) -> LLMClient:
    """
    Create vLLM client.

    Auto-detects vLLM server if base_url is not specified.

    Args:
        config: LLM configuration

    Returns:
        vLLM client instance
    """
    # Auto-detect base_url if not specified
    if not config.base_url:
        detected_url, _ = detect_vllm()
        # Create a copy to avoid mutating the caller's config
        config = config.model_copy(update={"base_url": detected_url})

    return VLLMClient(config)
