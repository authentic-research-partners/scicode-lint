"""Pydantic models for structured LLM output.

Every string field has a `max_length` (chars) and every list field has a
`max_length` (item count), set at ~2x natural output length. **These bounds are
post-decode guards, not wire constraints.** `vllm_schema()` strips the length/count
keys (`maxLength`/`maxItems`/`minLength`/`minItems`) from the JSON schema sent to
vLLM, because they compile into XGrammar's grammar slow path, which collapses
structured-output throughput under concurrency
(see `evals/wire_bounds_throughput.py`). Numeric range keys (`minimum`/`maximum`)
are *not* stripped — an isolation measurement proved they don't trigger the slow
path, so `confidence`'s 0-1 range stays decoder-enforced.

Output size is controlled by three layers that never touch the decoder:

1. Prompt guidance ("1-2 sentences, under ~N words", "at most N items") — the
   model is asked to stay under the cap, so it writes short by default.
2. `max_completion_tokens` caps total output; over-runs surface as
   `finish_reason=length` and recover via the thinking-budget ladder (see
   `llm/CONSTRAINED_DECODING.md` § Transient retry).
3. The Pydantic `max_length` validates decoded output. An over-run raises
   `ValidationError`, which the client treats as a **retryable** failure and
   reruns (a fresh sample almost always complies) — over-runs are surfaced and
   re-attempted, never silently clipped.

`thinking` is intentionally unbounded: it's populated post-hoc by the client
from vLLM's server-side `message.reasoning` channel (via `--reasoning-parser
qwen3`), not decoded into the JSON response.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Length/count constraints compile into XGrammar's grammar slow path: shipping
# `maxLength` on the wire collapses structured-output throughput under concurrency
# (see evals/wire_bounds_throughput.py). vllm_schema strips this family;
# the Pydantic model still enforces the caps at construction, and an over-run from
# the now-unbounded wire is caught post-decode and rerun by the client.
#
# Only length/count keys are stripped. Numeric range keys (`minimum`/`maximum`) are
# NOT stripped: an isolation run with `maxLength` removed but `minimum`/`maximum`
# kept on the wire recovered full speed (100% / 7.9s), proving ranges don't trigger
# the slow path — so `confidence`'s 0-1 range stays decoder-enforced.
_WIRE_BANNED_KEYS = frozenset({"maxLength", "minLength", "maxItems", "minItems"})


def vllm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate the wire JSON schema dict for vLLM's constrained decoder.

    Two transforms make the schema safe and fast for vLLM's XGrammar backend:

    1. **Inline ``$ref``** — XGrammar doesn't resolve ``$defs``, so nested models
       (e.g. ``DetectionResult.location`` → ``NamedLocation``) are inlined.
    2. **Strip length/count keys** (``maxLength``, ``maxItems``, ``minLength``,
       ``minItems``) — these compile into XGrammar's grammar slow path and collapse
       throughput under concurrency (see ``evals/wire_bounds_throughput.py``). The
       Pydantic model keeps the caps for post-decode validation; an over-run from the
       unbounded wire is rerun by the client. Numeric ranges (``minimum``/``maximum``)
       and enums are *kept* — an isolation run proved they don't trigger the slow path.

    Also drops Pydantic ``title`` metadata to keep the schema compact.

    Args:
        model: Pydantic model class.

    Returns:
        Flat JSON schema dict with no ``$ref``, no ``$defs``, and no
        length/count constraint keys (numeric ranges and enums retained).

    Example:
        >>> schema = vllm_schema(DetectionResult)
        >>> assert "$ref" not in str(schema)
        >>> assert "maxLength" not in str(schema)
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].rsplit("/", 1)[-1]
                return _resolve(defs[ref_name])
            return {
                k: _resolve(v)
                for k, v in obj.items()
                if k != "title" and k not in _WIRE_BANNED_KEYS
            }
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    resolved: dict[str, Any] = _resolve(raw)
    return resolved


class NamedLocation(BaseModel):
    """Name-based location for detected issues.

    LLMs are good at identifying function/class names but unreliable at counting
    line numbers. This schema captures what LLMs do well, and we resolve to actual
    lines using AST parsing.
    """

    name: str = Field(
        max_length=200,
        description="Name of the function, class, or method where issue occurs. "
        "Use qualified names for methods (e.g., 'Trainer.train'). "
        "Use '<module>' for module-level code.",
    )
    location_type: Literal["function", "class", "method", "module"] = Field(
        description="Type of code construct: 'function' for standalone functions, "
        "'method' for class methods, 'class' for class definitions, "
        "'module' for module-level code."
    )
    near_line: int | None = Field(
        default=None,
        description="Approximate line number where issue occurs (optional hint). "
        "Used to disambiguate when multiple definitions have the same name.",
    )


class DetectionResult(BaseModel):
    """
    Three-way detection result: yes/no/context-dependent with reasoning.

    This format allows the LLM to express uncertainty when the answer
    depends on context, coding style, or interpretation.
    """

    detected: Literal["yes", "no", "context-dependent"] = Field(
        description="Whether the issue was detected: 'yes' (definite issue), "
        "'no' (no issue), or 'context-dependent' (depends on context/style)"
    )
    # Name-based location instead of line numbers. LLMs are better at names than lines.
    # For detected="no", use null. For detected="yes", provide name-based location.
    location: NamedLocation | None = Field(
        default=None,
        description="Location of the issue. REQUIRED when detected='yes' or 'context-dependent'. "
        "Use null when detected='no'.",
    )
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        max_length=400,
        description="Brief explanation (1-2 sentences, under ~50 words) of why this "
        "decision was made. Explain what pattern was detected or why it's not an issue.",
    )
    thinking: str | None = Field(
        default=None,
        description="Model's internal reasoning/thinking (extracted from <think> tags). "
        "Populated post-hoc from server-side reasoning channel — intentionally unbounded.",
    )

    @model_validator(mode="after")
    def validate_location_when_detected(self) -> "DetectionResult":
        """Require location when issue is detected.

        Raises ValueError if detected="yes" or "context-dependent" but location is None.
        This triggers retry logic in the client and prevents storing invalid findings.
        """
        if self.detected in ("yes", "context-dependent") and not self.location:
            raise ValueError(
                f"Location required when detected='{self.detected}'. "
                "Model must provide function/class name where issue occurs."
            )
        return self
