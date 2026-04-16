"""Pydantic models for structured LLM output.

Every string field has a `max_length` (chars) set at ~2x natural output length,
and every list field has `max_length` on both the list and each item. These tell
vLLM's constrained decoder how much space each value gets, so the full JSON
structure completes within `max_completion_tokens` regardless of what the model
generates. The limits are ceilings, not targets — the model writes naturally
below them.

Sizing math (at worst-case 3 chars/token):

    NamedLocation.name:          200 chars → ~67 tokens
    DetectionResult.reasoning:   400 chars → ~133 tokens
    Enums + floats + overhead:              ~25 tokens
                                          ────────────
    DetectionResult worst-case:            ~225 tokens

That leaves ~3870 of the 4096 `max_completion_tokens` budget for thinking
(current `thinking_budget=3584`). Without these caps, a verbose `reasoning`
field alone could consume the entire JSON budget, causing truncation mid-field
(`finish_reason=length`) or — if thinking runs long first — `content=None`
with no JSON at all. See `llm/CONSTRAINED_DECODING.md` for the full rationale
and empirical findings.

`thinking` is intentionally unbounded: it's populated post-hoc by the client
from vLLM's server-side `message.reasoning` channel (via `--reasoning-parser
qwen3`), not decoded into the JSON response. A Pydantic `max_length` cap here
would reject legitimate long reasoning content from the side channel.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def vllm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate a JSON schema dict for vLLM's constrained decoder.

    Inlines ``$ref`` references (vLLM's XGrammar backend doesn't resolve
    ``$defs``) and strips Pydantic metadata (``title``) to keep the schema
    compact. Use this instead of ``model.model_json_schema()`` when passing
    schemas to ``response_format: json_schema``.

    Args:
        model: Pydantic model class.

    Returns:
        JSON schema dict with all ``$ref`` entries resolved inline.

    Example:
        >>> schema = vllm_schema(DetectionResult)
        >>> assert "$ref" not in str(schema)
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].rsplit("/", 1)[-1]
                return _resolve(defs[ref_name])
            return {k: _resolve(v) for k, v in obj.items() if k != "title"}
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
        description="Brief explanation (1-2 sentences) of why this decision was made. "
        "Explain what pattern was detected or why it's not an issue.",
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
