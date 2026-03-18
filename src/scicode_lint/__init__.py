"""scicode-lint: AI-powered linter for scientific Python code.

Designed for both human users and GenAI coding agents.
Detects common bugs in scientific code: data leakage, missing seeds,
PyTorch training issues, numerical errors, and more.
"""

__version__ = "0.2.2"

from scicode_lint.detectors.catalog import DetectionCatalog, DetectionPattern
from scicode_lint.linter import SciCodeLinter

__all__ = ["SciCodeLinter", "DetectionCatalog", "DetectionPattern"]
