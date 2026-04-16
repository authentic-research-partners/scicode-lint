"""Tests for repo_filter module (self-contained ML file detection)."""

from pathlib import Path

import pytest

from scicode_lint.repo_filter.classify import (
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_PROMPT,
    FileClassification,
)
from scicode_lint.repo_filter.scan import (
    RepoScanSummary,
    ScanResult,
    filter_scan_results,
    get_ml_import_keywords,
    has_ml_imports,
)


class TestHasMLImports:
    """Tests for has_ml_imports function."""

    def test_sklearn_import(self) -> None:
        """Code with sklearn import should have ML indicators."""
        code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data.csv")
model = RandomForestClassifier()
model.fit(data.drop("target", axis=1), data["target"])
"""
        assert has_ml_imports(code) is True

    def test_torch_import(self) -> None:
        """Code with torch import should have ML indicators."""
        code = """
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
"""
        assert has_ml_imports(code) is True

    def test_tensorflow_import(self) -> None:
        """Code with tensorflow import should have ML indicators."""
        code = """
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])
"""
        assert has_ml_imports(code) is True

    def test_no_ml_imports(self) -> None:
        """Code without ML imports should not have ML indicators."""
        code = """
print("hello world")

def add(a, b):
    return a + b

result = add(1, 2)
"""
        assert has_ml_imports(code) is False

    def test_pandas_only(self) -> None:
        """Code with pandas import should have ML indicators."""
        code = """
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
"""
        assert has_ml_imports(code) is True

    def test_numpy_only(self) -> None:
        """Code with numpy import should have ML indicators."""
        code = """
import numpy as np

arr = np.array([1, 2, 3])
print(arr.mean())
"""
        assert has_ml_imports(code) is True

    def test_case_insensitive(self) -> None:
        """ML indicator detection should be case-insensitive."""
        code = """
# Training a MODEL with FIT method
"""
        assert has_ml_imports(code) is True


class TestFileClassification:
    """Tests for FileClassification Pydantic model."""

    def test_valid_self_contained(self) -> None:
        """Valid self-contained classification should parse correctly."""
        result = FileClassification(
            classification="self_contained",
            confidence=0.95,
            entry_point_indicators=["if __name__ == '__main__'", "argparse"],
            missing_components=[],
            reasoning="Complete ML pipeline from data loading to model training.",
        )
        assert result.classification == "self_contained"
        assert result.confidence == 0.95
        assert len(result.entry_point_indicators) == 2

    def test_valid_fragment(self) -> None:
        """Valid fragment classification should parse correctly."""
        result = FileClassification(
            classification="fragment",
            confidence=0.88,
            entry_point_indicators=[],
            missing_components=["data loading", "model training"],
            reasoning="Only model definition, no training code.",
        )
        assert result.classification == "fragment"
        assert len(result.missing_components) == 2

    def test_valid_uncertain(self) -> None:
        """Valid uncertain classification should parse correctly."""
        result = FileClassification(
            classification="uncertain",
            confidence=0.5,
            entry_point_indicators=[],
            missing_components=[],
            reasoning="Cannot determine due to dynamic imports.",
        )
        assert result.classification == "uncertain"

    def test_confidence_bounds(self) -> None:
        """Confidence should be bounded between 0 and 1."""
        with pytest.raises(ValueError):
            FileClassification(
                classification="self_contained",
                confidence=1.5,  # Invalid
                entry_point_indicators=[],
                missing_components=[],
                reasoning="Test",
            )

        with pytest.raises(ValueError):
            FileClassification(
                classification="self_contained",
                confidence=-0.1,  # Invalid
                entry_point_indicators=[],
                missing_components=[],
                reasoning="Test",
            )

    def test_invalid_classification(self) -> None:
        """Invalid classification value should raise error."""
        with pytest.raises(ValueError):
            FileClassification(
                classification="invalid",  # type: ignore[arg-type]
                confidence=0.5,
                entry_point_indicators=[],
                missing_components=[],
                reasoning="Test",
            )


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_to_dict_basic(self) -> None:
        """Basic ScanResult should serialize correctly."""
        result = ScanResult(
            filepath=Path("/repo/train.py"),
            classification="self_contained",
        )
        data = result.to_dict()
        assert data["filepath"] == "/repo/train.py"
        assert data["classification"] == "self_contained"
        assert "skip_reason" not in data
        assert "details" not in data

    def test_to_dict_with_skip_reason(self) -> None:
        """ScanResult with skip reason should serialize correctly."""
        result = ScanResult(
            filepath=Path("/repo/utils.py"),
            classification="fragment",
            skip_reason="no_ml_imports_found",
        )
        data = result.to_dict()
        assert data["skip_reason"] == "no_ml_imports_found"

    def test_to_dict_with_too_large_skip_reason(self) -> None:
        """ScanResult skipped due to size should serialize correctly."""
        result = ScanResult(
            filepath=Path("/repo/huge_model.py"),
            classification="fragment",
            skip_reason="too_large_50000_tokens",
        )
        data = result.to_dict()
        assert data["skip_reason"] == "too_large_50000_tokens"
        assert data["classification"] == "fragment"

    def test_to_dict_with_details(self) -> None:
        """ScanResult with details should serialize correctly."""
        details = FileClassification(
            classification="self_contained",
            confidence=0.92,
            entry_point_indicators=["if __name__"],
            missing_components=[],
            reasoning="Complete workflow.",
        )
        result = ScanResult(
            filepath=Path("/repo/main.py"),
            classification="self_contained",
            details=details,
        )
        data = result.to_dict()
        assert "details" in data
        assert data["details"]["confidence"] == 0.92
        assert data["details"]["entry_point_indicators"] == ["if __name__"]


class TestRepoScanSummary:
    """Tests for RepoScanSummary dataclass."""

    def test_to_dict(self) -> None:
        """RepoScanSummary should serialize correctly."""
        summary = RepoScanSummary(
            total_files=10,
            passed_ml_import_filter=7,
            failed_ml_import_filter=2,
            skipped_too_large=1,
            self_contained=2,
            fragments=4,
            uncertain=1,
            results=[
                ScanResult(
                    filepath=Path("/repo/train.py"),
                    classification="self_contained",
                ),
                ScanResult(
                    filepath=Path("/repo/main.py"),
                    classification="self_contained",
                ),
            ],
        )
        data = summary.to_dict()
        assert data["summary"]["total_files"] == 10
        assert data["summary"]["passed_ml_import_filter"] == 7
        assert data["summary"]["failed_ml_import_filter"] == 2
        assert data["summary"]["skipped_too_large"] == 1
        assert data["summary"]["self_contained"] == 2
        assert data["summary"]["fragments"] == 4
        assert len(data["files"]) == 2


class TestFilterScanResults:
    """Tests for filter_scan_results function."""

    def test_filter_self_contained_only(self) -> None:
        """Default filter should return only self-contained files."""
        summary = RepoScanSummary(
            total_files=5,
            self_contained=2,
            fragments=2,
            uncertain=1,
            results=[
                ScanResult(filepath=Path("/repo/train.py"), classification="self_contained"),
                ScanResult(filepath=Path("/repo/model.py"), classification="fragment"),
                ScanResult(filepath=Path("/repo/main.py"), classification="self_contained"),
                ScanResult(filepath=Path("/repo/utils.py"), classification="fragment"),
                ScanResult(filepath=Path("/repo/test.py"), classification="uncertain"),
            ],
        )
        filtered = filter_scan_results(summary)
        assert len(filtered) == 2
        assert all(r.classification == "self_contained" for r in filtered)

    def test_filter_include_uncertain(self) -> None:
        """With include_uncertain=True, should return self-contained and uncertain."""
        summary = RepoScanSummary(
            total_files=5,
            self_contained=2,
            fragments=2,
            uncertain=1,
            results=[
                ScanResult(filepath=Path("/repo/train.py"), classification="self_contained"),
                ScanResult(filepath=Path("/repo/model.py"), classification="fragment"),
                ScanResult(filepath=Path("/repo/main.py"), classification="self_contained"),
                ScanResult(filepath=Path("/repo/utils.py"), classification="fragment"),
                ScanResult(filepath=Path("/repo/test.py"), classification="uncertain"),
            ],
        )
        filtered = filter_scan_results(summary, include_uncertain=True)
        assert len(filtered) == 3
        classifications = {r.classification for r in filtered}
        assert classifications == {"self_contained", "uncertain"}

    def test_filter_excludes_skipped(self) -> None:
        """Filter should exclude files with skip_reason."""
        summary = RepoScanSummary(
            total_files=3,
            self_contained=1,
            failed_ml_import_filter=1,
            skipped_too_large=1,
            results=[
                ScanResult(filepath=Path("/repo/train.py"), classification="self_contained"),
                ScanResult(
                    filepath=Path("/repo/utils.py"),
                    classification="fragment",
                    skip_reason="no_ml_imports_found",
                ),
                ScanResult(
                    filepath=Path("/repo/big.py"),
                    classification="fragment",
                    skip_reason="too_large_50000_tokens",
                ),
            ],
        )
        filtered = filter_scan_results(summary)
        assert len(filtered) == 1
        assert filtered[0].filepath == Path("/repo/train.py")


class TestPrompts:
    """Tests for classification prompts."""

    def test_system_prompt_exists(self) -> None:
        """System prompt should be non-empty."""
        assert CLASSIFY_SYSTEM_PROMPT
        assert "SELF-CONTAINED" in CLASSIFY_SYSTEM_PROMPT
        assert "FRAGMENT" in CLASSIFY_SYSTEM_PROMPT

    def test_user_prompt_template(self) -> None:
        """User prompt should have code placeholder and schema guidance."""
        assert "{code}" in CLASSIFY_USER_PROMPT
        assert "<CODE_TO_ANALYZE>" in CLASSIFY_USER_PROMPT
        assert "classification" in CLASSIFY_USER_PROMPT
        assert "confidence" in CLASSIFY_USER_PROMPT

    def test_user_prompt_formatting(self) -> None:
        """User prompt should format correctly."""
        code = "import torch\nmodel = torch.nn.Linear(10, 1)"
        prompt = CLASSIFY_USER_PROMPT.format(code=code)
        assert code in prompt
        assert "{code}" not in prompt


class TestMLImportKeywords:
    """Tests for ML import keyword loading from config.toml."""

    def test_essential_frameworks_present(self) -> None:
        """Essential ML frameworks should be in config-loaded indicators."""
        kws = get_ml_import_keywords()
        assert "sklearn" in kws
        assert "torch" in kws
        assert "tensorflow" in kws
        assert "keras" in kws

    def test_data_libraries_present(self) -> None:
        kws = get_ml_import_keywords()
        assert "pandas" in kws
        assert "numpy" in kws

    def test_ml_operations_present(self) -> None:
        kws = get_ml_import_keywords()
        assert "fit" in kws
        assert "predict" in kws
        assert "model" in kws
