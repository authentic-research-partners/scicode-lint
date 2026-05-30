"""Regression tests for vLLM wire-schema shaping and post-decode bounds.

Two-part contract (see ``src/scicode_lint/llm/CONSTRAINED_DECODING.md``):

1. The **wire** schema (``vllm_schema``) carries NO length/count constraint keys
   (``maxLength``/``maxItems``/``minLength``/``minItems``). Shipping them triggers
   XGrammar's grammar slow path, collapsing throughput under concurrency
   (see ``evals/wire_bounds_throughput.py``). ``$ref`` is inlined; enums
   and numeric ranges (``minimum``/``maximum``) ARE preserved — an isolation run
   proved ranges don't trigger the slow path.
2. The Pydantic **model** still enforces those bounds at construction (defense in
   depth). An over-run from the unbounded wire is caught post-decode and rerun by
   the client, never silently clipped.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evals.integration.models import (
    GeneratedScenario,
    JudgedFinding,
    JudgedMiss,
    JudgeResult,
    ManifestEntry,
    PatternSelection,
    VerificationEntry,
    VerificationResult,
)
from evals.judge_models import JudgeVerdict
from pattern_verification.deterministic.doc_cache import DocCutResponse
from real_world_demo.sources.papers_with_code.filter_abstracts import AbstractFilterResult
from scicode_lint.llm.models import DetectionResult, NamedLocation, vllm_schema
from scicode_lint.repo_filter.classify import FileClassification

# Length/count constraint keys that must NOT appear in any wire schema (they trigger
# XGrammar's slow path). Independent copy of the contract — NOT imported from models —
# so the test fails if the production set ever drifts. Numeric ranges (minimum/maximum)
# are deliberately absent here: they were measured not to trigger the slow path, so
# they stay on the wire.
_BANNED_WIRE_KEYS = {
    "maxLength",
    "minLength",
    "maxItems",
    "minItems",
}

# Every response schema sent to vLLM through vllm_schema().
_WIRE_MODELS = [
    DetectionResult,
    NamedLocation,
    FileClassification,
    JudgeVerdict,
    AbstractFilterResult,
    DocCutResponse,
    PatternSelection,
    ManifestEntry,
    GeneratedScenario,
    VerificationResult,
    VerificationEntry,
    JudgedFinding,
    JudgedMiss,
    JudgeResult,
]


def _all_keys(obj: Any) -> set[str]:
    """Recursively collect every dict key in a JSON-schema tree."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            keys.add(key)
            keys |= _all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            keys |= _all_keys(item)
    return keys


def _location_object_branch(detection_schema: dict[str, Any]) -> dict[str, Any]:
    """Pull the inlined NamedLocation object branch out of DetectionResult."""
    any_of = detection_schema["properties"]["location"]["anyOf"]
    return next(b for b in any_of if b.get("type") == "object")


# ---------------------------------------------------------------------------
# vllm_schema: $ref inlining
# ---------------------------------------------------------------------------


class TestVllmSchemaRefInlining:
    """vllm_schema() must inline $ref so vLLM's XGrammar can parse the schema."""

    def test_no_refs_in_detection_result(self) -> None:
        schema = vllm_schema(DetectionResult)
        assert "$ref" not in str(schema), (
            "vllm_schema(DetectionResult) still contains $ref. "
            "vLLM's XGrammar backend may not resolve $defs. "
            "See llm/CONSTRAINED_DECODING.md § '$ref inlining'."
        )

    def test_no_defs_in_detection_result(self) -> None:
        schema = vllm_schema(DetectionResult)
        assert "$defs" not in schema, (
            "vllm_schema(DetectionResult) still contains $defs after inlining."
        )

    def test_named_location_inlined(self) -> None:
        """NamedLocation fields should be inlined into DetectionResult schema."""
        obj_branch = _location_object_branch(vllm_schema(DetectionResult))
        assert "name" in obj_branch["properties"]
        assert "location_type" in obj_branch["properties"]

    def test_file_classification_has_no_refs(self) -> None:
        assert "$ref" not in str(vllm_schema(FileClassification))


# ---------------------------------------------------------------------------
# Wire schema must carry NO size/range constraint keys (the slow-path fix)
# ---------------------------------------------------------------------------


class TestWireSchemaHasNoBounds:
    """The wire schema must not ship length/count keys — they trigger XGrammar's
    grammar slow path under concurrency. Numeric ranges (minimum/maximum) ARE kept
    (measured not to trigger the slow path).
    """

    @pytest.mark.parametrize("model", _WIRE_MODELS, ids=lambda m: m.__name__)
    def test_no_banned_keys_anywhere(self, model: type) -> None:
        present = _all_keys(vllm_schema(model)) & _BANNED_WIRE_KEYS
        assert not present, (
            f"vllm_schema({model.__name__}) contains banned constraint keys "
            f"{sorted(present)} — these trigger XGrammar's slow path. vllm_schema "
            "must strip them. See llm/CONSTRAINED_DECODING.md."
        )

    def test_detection_reasoning_has_no_maxlength(self) -> None:
        schema = vllm_schema(DetectionResult)
        assert "maxLength" not in schema["properties"]["reasoning"]

    def test_detection_confidence_keeps_range(self) -> None:
        # Numeric ranges are NOT stripped — they don't trigger the slow path, so
        # confidence stays decoder-guaranteed in [0, 1].
        conf = vllm_schema(DetectionResult)["properties"]["confidence"]
        assert conf["minimum"] == 0.0
        assert conf["maximum"] == 1.0

    def test_named_location_name_has_no_maxlength(self) -> None:
        obj_branch = _location_object_branch(vllm_schema(DetectionResult))
        assert "maxLength" not in obj_branch["properties"]["name"]

    def test_file_classification_lists_have_no_caps(self) -> None:
        schema = vllm_schema(FileClassification)
        epi = schema["properties"]["entry_point_indicators"]
        assert "maxItems" not in epi
        assert "maxLength" not in epi["items"]


# ---------------------------------------------------------------------------
# Enums survive stripping (XGrammar handles them fine)
# ---------------------------------------------------------------------------


class TestWireSchemaPreservesEnums:
    """Literal/enum constraints are kept on the wire — only size/range keys go."""

    def test_detected_enum(self) -> None:
        schema = vllm_schema(DetectionResult)
        assert set(schema["properties"]["detected"]["enum"]) == {
            "yes",
            "no",
            "context-dependent",
        }

    def test_location_type_enum(self) -> None:
        obj_branch = _location_object_branch(vllm_schema(DetectionResult))
        assert set(obj_branch["properties"]["location_type"]["enum"]) == {
            "function",
            "class",
            "method",
            "module",
        }

    def test_judge_verdict_enum(self) -> None:
        schema = vllm_schema(JudgeVerdict)
        assert set(schema["properties"]["verdict"]["enum"]) == {"yes", "no", "partial"}

    def test_abstract_filter_literals(self) -> None:
        schema = vllm_schema(AbstractFilterResult)
        domains = set(schema["properties"]["science_domain"]["enum"])
        assert {"biology", "none", "engineering"} <= domains
        assert set(schema["properties"]["application_type"]["enum"]) == {
            "prediction",
            "analysis",
            "discovery",
            "simulation",
            "diagnosis",
            "methodology",
        }

    def test_verification_quality_enum(self) -> None:
        schema = vllm_schema(VerificationResult)
        assert set(schema["properties"]["quality"]["enum"]) == {
            "good",
            "needs_correction",
            "regenerate",
        }

    def test_judged_finding_category_enum(self) -> None:
        schema = vllm_schema(JudgedFinding)
        assert set(schema["properties"]["category"]["enum"]) == {
            "tp_intended",
            "tp_bonus",
            "fp",
        }


# ---------------------------------------------------------------------------
# Pydantic model still enforces bounds at construction (defense in depth)
# ---------------------------------------------------------------------------


class TestPydanticModelStillBounded:
    """The wire is unbounded, but the Pydantic model still validates over-runs —
    this is what the client's rerun path catches post-decode.
    """

    def test_detection_rejects_over_length_reasoning(self) -> None:
        with pytest.raises(ValidationError):
            DetectionResult(detected="no", location=None, confidence=0.5, reasoning="x" * 401)

    def test_detection_accepts_exact_length_reasoning(self) -> None:
        result = DetectionResult(detected="no", location=None, confidence=0.5, reasoning="x" * 400)
        assert len(result.reasoning) == 400

    def test_detection_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValidationError):
            DetectionResult(detected="no", location=None, confidence=1.5, reasoning="ok")

    def test_named_location_rejects_over_length_name(self) -> None:
        with pytest.raises(ValidationError):
            NamedLocation(name="x" * 201, location_type="function")

    def test_named_location_accepts_exact_length_name(self) -> None:
        loc = NamedLocation(name="x" * 200, location_type="function")
        assert len(loc.name) == 200

    def test_file_classification_rejects_too_many_indicators(self) -> None:
        with pytest.raises(ValidationError):
            FileClassification(
                classification="self_contained",
                confidence=0.9,
                entry_point_indicators=["item"] * 11,
                missing_components=[],
                reasoning="test",
            )

    def test_file_classification_accepts_exactly_10_indicators(self) -> None:
        result = FileClassification(
            classification="self_contained",
            confidence=0.9,
            entry_point_indicators=["item"] * 10,
            missing_components=[],
            reasoning="test",
        )
        assert len(result.entry_point_indicators) == 10

    def test_file_classification_rejects_over_length_item(self) -> None:
        with pytest.raises(ValidationError):
            FileClassification(
                classification="self_contained",
                confidence=0.9,
                entry_point_indicators=["x" * 81],
                missing_components=[],
                reasoning="test",
            )

    def test_judge_verdict_rejects_over_length_reasoning(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(verdict="yes", reasoning="x" * 401, confidence=0.9)

    def test_doc_cut_rejects_too_many_cuts(self) -> None:
        with pytest.raises(ValidationError):
            DocCutResponse(cut=[[i, i + 1] for i in range(51)])

    def test_abstract_filter_rejects_invalid_domain(self) -> None:
        with pytest.raises(ValidationError):
            AbstractFilterResult(
                is_ai_science=True,
                confidence=0.9,
                science_domain="made-up-domain",  # type: ignore[arg-type]
                application_type="prediction",
                explanation="x",
            )

    def test_verification_result_rejects_invalid_quality(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                verified=[],
                quality="unknown",  # type: ignore[arg-type]
            )
