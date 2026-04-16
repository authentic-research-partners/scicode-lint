"""Regression tests for output-bounding constraints in vLLM response schemas.

These tests lock in the ``max_length`` / ``maxItems`` constraints that
prevent ``finish_reason=length`` truncation in constrained decoding. If
any of these bounds are silently removed, vLLM's decoder loses its upper
bound on output size and long responses can truncate mid-JSON.

The bounds are documented in ``src/scicode_lint/llm/CONSTRAINED_DECODING.md``.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# vllm_schema: $ref inlining
# ---------------------------------------------------------------------------


class TestVllmSchemaRefInlining:
    """vllm_schema() must inline $ref so vLLM's XGrammar can parse the schema."""

    def test_no_refs_in_detection_result(self) -> None:
        """DetectionResult schema must not contain $ref after inlining."""
        schema = vllm_schema(DetectionResult)
        schema_str = str(schema)
        assert "$ref" not in schema_str, (
            "vllm_schema(DetectionResult) still contains $ref. "
            "vLLM's XGrammar backend may not resolve $defs. "
            "See llm/CONSTRAINED_DECODING.md § '$ref inlining'."
        )

    def test_no_defs_in_detection_result(self) -> None:
        """DetectionResult schema must not contain $defs after inlining."""
        schema = vllm_schema(DetectionResult)
        assert "$defs" not in schema, (
            "vllm_schema(DetectionResult) still contains $defs after inlining."
        )

    def test_named_location_inlined(self) -> None:
        """NamedLocation fields should be inlined into DetectionResult schema."""
        schema = vllm_schema(DetectionResult)
        # location is anyOf[NamedLocation, null] — find the object branch
        location = schema["properties"]["location"]
        any_of = location["anyOf"]
        obj_branch = next(b for b in any_of if b.get("type") == "object")
        assert "name" in obj_branch["properties"], (
            "NamedLocation.name not found in inlined location schema"
        )
        assert "location_type" in obj_branch["properties"], (
            "NamedLocation.location_type not found in inlined location schema"
        )

    def test_file_classification_has_no_refs(self) -> None:
        """FileClassification has no nested models, but verify no $ref anyway."""
        schema = vllm_schema(FileClassification)
        assert "$ref" not in str(schema)


# ---------------------------------------------------------------------------
# DetectionResult — per-field bounds
# ---------------------------------------------------------------------------


class TestDetectionResultBounds:
    """Lock in DetectionResult field bounds for vLLM constrained decoding."""

    def test_reasoning_max_length(self) -> None:
        schema = vllm_schema(DetectionResult)
        assert schema["properties"]["reasoning"]["maxLength"] == 400, (
            "DetectionResult.reasoning must have maxLength=400 (~133 tokens). "
            "See llm/CONSTRAINED_DECODING.md § 'Sizing for scicode-lint'."
        )

    def test_confidence_bounds(self) -> None:
        schema = vllm_schema(DetectionResult)
        conf = schema["properties"]["confidence"]
        assert conf["minimum"] == 0.0
        assert conf["maximum"] == 1.0

    def test_detected_enum_values(self) -> None:
        schema = vllm_schema(DetectionResult)
        assert set(schema["properties"]["detected"]["enum"]) == {
            "yes",
            "no",
            "context-dependent",
        }

    def test_pydantic_rejects_over_length_reasoning(self) -> None:
        with pytest.raises(ValidationError):
            DetectionResult(
                detected="no",
                location=None,
                confidence=0.5,
                reasoning="x" * 401,
            )

    def test_pydantic_accepts_exact_length_reasoning(self) -> None:
        result = DetectionResult(
            detected="no",
            location=None,
            confidence=0.5,
            reasoning="x" * 400,
        )
        assert len(result.reasoning) == 400


# ---------------------------------------------------------------------------
# NamedLocation — per-field bounds
# ---------------------------------------------------------------------------


class TestNamedLocationBounds:
    """Lock in NamedLocation field bounds."""

    def test_name_max_length(self) -> None:
        schema = vllm_schema(DetectionResult)
        location_any_of = schema["properties"]["location"]["anyOf"]
        obj_branch = next(b for b in location_any_of if b.get("type") == "object")
        assert obj_branch["properties"]["name"]["maxLength"] == 200, (
            "NamedLocation.name must have maxLength=200. See llm/CONSTRAINED_DECODING.md."
        )

    def test_location_type_enum(self) -> None:
        schema = vllm_schema(DetectionResult)
        location_any_of = schema["properties"]["location"]["anyOf"]
        obj_branch = next(b for b in location_any_of if b.get("type") == "object")
        assert set(obj_branch["properties"]["location_type"]["enum"]) == {
            "function",
            "class",
            "method",
            "module",
        }

    def test_pydantic_rejects_over_length_name(self) -> None:
        with pytest.raises(ValidationError):
            NamedLocation(name="x" * 201, location_type="function")

    def test_pydantic_accepts_exact_length_name(self) -> None:
        loc = NamedLocation(name="x" * 200, location_type="function")
        assert len(loc.name) == 200


# ---------------------------------------------------------------------------
# FileClassification — per-field and list bounds
# ---------------------------------------------------------------------------


class TestFileClassificationBounds:
    """Lock in FileClassification field bounds for vLLM constrained decoding."""

    def test_reasoning_max_length(self) -> None:
        schema = vllm_schema(FileClassification)
        assert schema["properties"]["reasoning"]["maxLength"] == 400

    def test_entry_point_indicators_max_items(self) -> None:
        schema = vllm_schema(FileClassification)
        epi = schema["properties"]["entry_point_indicators"]
        assert epi["type"] == "array"
        assert epi["maxItems"] == 10, (
            "FileClassification.entry_point_indicators must be capped at 10 "
            "to prevent mid-array truncation. "
            "See llm/CONSTRAINED_DECODING.md § 'Schema bounds'."
        )

    def test_entry_point_indicators_per_item_max_length(self) -> None:
        schema = vllm_schema(FileClassification)
        items = schema["properties"]["entry_point_indicators"]["items"]
        assert items["maxLength"] == 80, (
            "Per-item maxLength on entry_point_indicators must be 80 chars."
        )

    def test_missing_components_max_items(self) -> None:
        schema = vllm_schema(FileClassification)
        mc = schema["properties"]["missing_components"]
        assert mc["type"] == "array"
        assert mc["maxItems"] == 10

    def test_missing_components_per_item_max_length(self) -> None:
        schema = vllm_schema(FileClassification)
        items = schema["properties"]["missing_components"]["items"]
        assert items["maxLength"] == 80

    def test_confidence_bounds(self) -> None:
        schema = vllm_schema(FileClassification)
        conf = schema["properties"]["confidence"]
        assert conf["minimum"] == 0.0
        assert conf["maximum"] == 1.0

    def test_pydantic_rejects_too_many_indicators(self) -> None:
        with pytest.raises(ValidationError):
            FileClassification(
                classification="self_contained",
                confidence=0.9,
                entry_point_indicators=["item"] * 11,
                missing_components=[],
                reasoning="test",
            )

    def test_pydantic_accepts_exactly_10_indicators(self) -> None:
        result = FileClassification(
            classification="self_contained",
            confidence=0.9,
            entry_point_indicators=["item"] * 10,
            missing_components=[],
            reasoning="test",
        )
        assert len(result.entry_point_indicators) == 10

    def test_pydantic_rejects_over_length_indicator_item(self) -> None:
        with pytest.raises(ValidationError):
            FileClassification(
                classification="self_contained",
                confidence=0.9,
                entry_point_indicators=["x" * 81],
                missing_components=[],
                reasoning="test",
            )

    def test_pydantic_rejects_over_length_reasoning(self) -> None:
        with pytest.raises(ValidationError):
            FileClassification(
                classification="self_contained",
                confidence=0.9,
                entry_point_indicators=[],
                missing_components=[],
                reasoning="x" * 401,
            )


# ---------------------------------------------------------------------------
# JudgeVerdict — vLLM pattern-eval judge
# ---------------------------------------------------------------------------


class TestJudgeVerdictBounds:
    """Lock in JudgeVerdict field bounds."""

    def test_reasoning_max_length(self) -> None:
        schema = vllm_schema(JudgeVerdict)
        assert schema["properties"]["reasoning"]["maxLength"] == 400

    def test_verdict_enum(self) -> None:
        schema = vllm_schema(JudgeVerdict)
        assert set(schema["properties"]["verdict"]["enum"]) == {"yes", "no", "partial"}

    def test_pydantic_rejects_over_length_reasoning(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(verdict="yes", reasoning="x" * 401, confidence=0.9)


# ---------------------------------------------------------------------------
# AbstractFilterResult — vLLM paper-abstract classifier
# ---------------------------------------------------------------------------


class TestAbstractFilterResultBounds:
    """Lock AbstractFilterResult Literals and bounds."""

    def test_science_domain_is_literal(self) -> None:
        schema = vllm_schema(AbstractFilterResult)
        enum = set(schema["properties"]["science_domain"]["enum"])
        assert "biology" in enum
        assert "none" in enum
        assert "engineering" in enum  # post-fix addition

    def test_application_type_is_literal(self) -> None:
        schema = vllm_schema(AbstractFilterResult)
        enum = set(schema["properties"]["application_type"]["enum"])
        assert enum == {
            "prediction",
            "analysis",
            "discovery",
            "simulation",
            "diagnosis",
            "methodology",
        }

    def test_explanation_max_length(self) -> None:
        schema = vllm_schema(AbstractFilterResult)
        assert schema["properties"]["explanation"]["maxLength"] == 400

    def test_pydantic_rejects_invalid_domain(self) -> None:
        with pytest.raises(ValidationError):
            AbstractFilterResult(
                is_ai_science=True,
                confidence=0.9,
                science_domain="made-up-domain",  # type: ignore[arg-type]
                application_type="prediction",
                explanation="x",
            )


# ---------------------------------------------------------------------------
# DocCutResponse — vLLM doc-chunking response
# ---------------------------------------------------------------------------


class TestDocCutResponseBounds:
    """Lock DocCutResponse list cap."""

    def test_cut_max_items(self) -> None:
        schema = vllm_schema(DocCutResponse)
        assert schema["properties"]["cut"]["maxItems"] == 50

    def test_pydantic_rejects_too_many_cuts(self) -> None:
        with pytest.raises(ValidationError):
            DocCutResponse(cut=[[i, i + 1] for i in range(51)])


# ---------------------------------------------------------------------------
# Integration eval models — Claude CLI responses
# ---------------------------------------------------------------------------


class TestIntegrationEvalBounds:
    """Lock bounds on integration eval schemas (Claude CLI path)."""

    def test_pattern_selection_bounds(self) -> None:
        schema = vllm_schema(PatternSelection)
        assert schema["properties"]["patterns"]["maxItems"] == 20
        assert schema["properties"]["patterns"]["items"]["maxLength"] == 32
        assert schema["properties"]["scenario_type"]["maxLength"] == 200
        assert schema["properties"]["reasoning"]["maxLength"] == 500

    def test_manifest_entry_bounds(self) -> None:
        schema = vllm_schema(ManifestEntry)
        assert schema["properties"]["pattern_id"]["maxLength"] == 32
        assert schema["properties"]["description"]["maxLength"] == 300

    def test_generated_scenario_manifest_cap(self) -> None:
        schema = vllm_schema(GeneratedScenario)
        assert schema["properties"]["manifest"]["maxItems"] == 30
        # code is intentionally unbounded
        assert "maxLength" not in schema["properties"]["code"]

    def test_verification_result_quality_is_literal(self) -> None:
        schema = vllm_schema(VerificationResult)
        assert set(schema["properties"]["quality"]["enum"]) == {
            "good",
            "needs_correction",
            "regenerate",
        }
        assert schema["properties"]["verified"]["maxItems"] == 30

    def test_verification_entry_bounds(self) -> None:
        schema = vllm_schema(VerificationEntry)
        assert schema["properties"]["notes"]["maxLength"] == 300

    def test_judged_finding_category_is_literal(self) -> None:
        schema = vllm_schema(JudgedFinding)
        assert set(schema["properties"]["category"]["enum"]) == {
            "tp_intended",
            "tp_bonus",
            "fp",
        }
        assert schema["properties"]["reasoning"]["maxLength"] == 400

    def test_judged_miss_bounds(self) -> None:
        schema = vllm_schema(JudgedMiss)
        assert schema["properties"]["reasoning"]["maxLength"] == 400

    def test_judge_result_list_caps(self) -> None:
        schema = vllm_schema(JudgeResult)
        assert schema["properties"]["findings"]["maxItems"] == 50
        assert schema["properties"]["misses"]["maxItems"] == 50

    def test_pydantic_rejects_invalid_quality(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                verified=[],
                quality="unknown",  # type: ignore[arg-type]
            )
