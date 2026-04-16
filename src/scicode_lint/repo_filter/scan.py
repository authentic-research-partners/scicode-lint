"""Repository scanner for self-contained ML files.

This module scans repositories to find files with complete ML workflows,
filtering out fragments and utility modules.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from scicode_lint.exceptions import LLMConnectionError
from scicode_lint.llm.client import LLMClient
from scicode_lint.llm.tokens import estimate_tokens
from scicode_lint.repo_filter.classify import FileClassification, classify_file


@dataclass
class ScanResult:
    """Result of classifying a single file.

    Attributes:
        filepath: Path to the classified file.
        classification: The classification result
            (self_contained, fragment, uncertain, unclassified).
        skip_reason: If skipped without LLM, why (e.g., "no_ml_imports_found").
        details: Full classification details from LLM (if LLM was used).
    """

    filepath: Path
    classification: Literal["self_contained", "fragment", "uncertain", "unclassified"]
    skip_reason: str | None = None
    details: FileClassification | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "filepath": str(self.filepath),
            "classification": self.classification,
        }
        if self.skip_reason:
            result["skip_reason"] = self.skip_reason
        if self.details:
            result["details"] = {
                "confidence": self.details.confidence,
                "entry_point_indicators": self.details.entry_point_indicators,
                "missing_components": self.details.missing_components,
                "reasoning": self.details.reasoning,
            }
        return result


@dataclass
class RepoScanSummary:
    """Summary of repository scan results.

    Attributes:
        total_files: Total Python files found in repository.
        passed_ml_import_filter: Files that contain ML imports (sent to LLM).
        failed_ml_import_filter: Files without ML imports (skipped, no LLM call).
        skipped_too_large: Files skipped due to exceeding token limit.
        self_contained: Files classified as complete ML workflows.
        fragments: Files classified as partial code.
        uncertain: Files with uncertain classification.
        results: All scan results.
    """

    total_files: int = 0
    passed_ml_import_filter: int = 0
    failed_ml_import_filter: int = 0
    skipped_too_large: int = 0
    self_contained: int = 0
    fragments: int = 0
    uncertain: int = 0
    results: list[ScanResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_files": self.total_files,
                "passed_ml_import_filter": self.passed_ml_import_filter,
                "failed_ml_import_filter": self.failed_ml_import_filter,
                "skipped_too_large": self.skipped_too_large,
                "self_contained": self.self_contained,
                "fragments": self.fragments,
                "uncertain": self.uncertain,
            },
            "files": [r.to_dict() for r in self.results],
        }


def get_ml_import_keywords() -> list[str]:
    """Get ML import keywords from config.

    config.toml's ``[preprocessing].ml_import_keywords`` is the single source of
    truth. Fails loudly if the config cannot be loaded — no hardcoded fallback.
    """
    from scicode_lint.config import get_ml_import_keywords as config_get_keywords

    keywords = config_get_keywords()
    if not keywords:
        raise RuntimeError(
            "No ML import keywords configured. "
            "Set [preprocessing].ml_import_keywords in config.toml."
        )
    return keywords


def has_ml_imports(code: str, keywords: list[str] | None = None) -> bool:
    """Quick check for ML-related imports (no LLM needed).

    This is a fast pre-filter to skip files that clearly aren't ML-related.

    Args:
        code: Python source code to check.
        keywords: Optional list of keywords to check for. If None, loads from config.

    Returns:
        True if code contains ML-related imports or keywords.

    Example:
        >>> has_ml_imports("import torch\\nmodel = torch.nn.Linear(10, 1)")
        True
        >>> has_ml_imports("print('hello world')")
        False
    """
    if keywords is None:
        keywords = get_ml_import_keywords()
    code_lower = code.lower()
    return any(keyword in code_lower for keyword in keywords)


def _extract_code_from_notebook(notebook_path: Path) -> str:
    """Extract Python code from Jupyter notebook.

    Args:
        notebook_path: Path to .ipynb file.

    Returns:
        Concatenated Python code from all code cells.

    Raises:
        ValueError: If notebook cannot be parsed.
    """
    try:
        with open(notebook_path, encoding="utf-8") as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Could not parse notebook {notebook_path}: {e}") from e

    code_cells = []
    cells = notebook.get("cells", [])

    for cell in cells:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                code_cells.append("".join(source))
            elif isinstance(source, str):
                code_cells.append(source)

    return "\n\n".join(code_cells)


async def scan_repo_for_ml_files(
    repo_path: Path,
    llm_client: LLMClient,
    max_concurrent: int = 10,
    max_file_tokens: int = 12000,
) -> RepoScanSummary:
    """Find self-contained ML files in a repository.

    Scans a repository for Python files and notebooks, classifying each as:
    - self_contained: Has complete ML workflow (data -> model -> train -> output)
    - fragment: Partial code requiring other files
    - uncertain: Classification couldn't be determined

    Files without ML indicators are skipped (not sent to LLM).
    Files exceeding max_file_tokens are skipped (too large for context).

    Returns ALL results in summary.results - use filter_scan_results() to get
    only self-contained files for display.

    Args:
        repo_path: Path to repository root.
        llm_client: LLM client for classification.
        max_concurrent: Maximum concurrent LLM requests.
        max_file_tokens: Maximum tokens per file (default: 12000, ~900 lines).

    Returns:
        RepoScanSummary with ALL classification results.

    Example:
        >>> from scicode_lint.config import load_llm_config
        >>> from scicode_lint.llm.client import create_client
        >>> llm_config = load_llm_config()
        >>> client = create_client(llm_config)
        >>> summary = await scan_repo_for_ml_files(Path("./my_ml_project"), client)
        >>> print(f"Found {summary.self_contained} self-contained ML files")
    """
    # Find Python files and notebooks
    python_files = list(repo_path.rglob("*.py"))
    notebooks = list(repo_path.rglob("*.ipynb"))
    all_files = python_files + notebooks

    logger.info(f"Found {len(python_files)} .py files and {len(notebooks)} .ipynb files")

    summary = RepoScanSummary(total_files=len(all_files))
    semaphore = asyncio.Semaphore(max_concurrent)

    async def classify_single_file(filepath: Path) -> ScanResult:
        """Classify a single file."""
        try:
            # Read file content
            if filepath.suffix == ".ipynb":
                code = _extract_code_from_notebook(filepath)
            else:
                code = filepath.read_text(encoding="utf-8")

            # Skip files that are too large (would exceed LLM context)
            tokens = estimate_tokens(code)
            if tokens > max_file_tokens:
                logger.debug(f"Skipping {filepath.name}: too large ({tokens} tokens)")
                return ScanResult(
                    filepath=filepath,
                    classification="unclassified",
                    skip_reason=f"too_large_{tokens}_tokens",
                )

            # Skip files without ML indicators
            if not has_ml_imports(code):
                logger.debug(f"Skipping {filepath.name}: no ML indicators")
                return ScanResult(
                    filepath=filepath,
                    classification="unclassified",
                    skip_reason="no_ml_imports_found",
                )

            # LLM classification with concurrency control
            async with semaphore:
                logger.debug(f"Classifying {filepath.name}")
                result = await classify_file(code, llm_client)

            return ScanResult(
                filepath=filepath,
                classification=result.classification,
                details=result,
            )

        except LLMConnectionError:
            # Systemic error — not a per-file classification failure.
            # Let it propagate so the CLI can exit 2 instead of silently
            # marking every file "uncertain" and returning exit 0.
            raise
        except Exception as e:
            logger.warning(f"Error classifying {filepath}: {e}")
            return ScanResult(
                filepath=filepath,
                classification="uncertain",
                skip_reason=f"error: {e}",
            )

    # Run all classifications concurrently
    tasks = [classify_single_file(f) for f in all_files]
    results = await asyncio.gather(*tasks)

    # Aggregate results
    for result in results:
        summary.results.append(result)
        if result.skip_reason == "no_ml_imports_found":
            summary.failed_ml_import_filter += 1
        elif result.skip_reason and result.skip_reason.startswith("too_large_"):
            summary.skipped_too_large += 1
        elif result.classification == "self_contained":
            summary.passed_ml_import_filter += 1
            summary.self_contained += 1
        elif result.classification == "fragment":
            summary.passed_ml_import_filter += 1
            summary.fragments += 1
        else:
            summary.passed_ml_import_filter += 1
            summary.uncertain += 1

    logger.info(
        f"Scan complete: {summary.total_files} files, "
        f"{summary.passed_ml_import_filter} passed ML import filter, "
        f"{summary.skipped_too_large} skipped (too large), "
        f"{summary.self_contained} self-contained, "
        f"{summary.fragments} fragments, {summary.uncertain} uncertain"
    )

    return summary


def filter_scan_results(
    summary: RepoScanSummary,
    include_uncertain: bool = False,
) -> list[ScanResult]:
    """Filter scan results to only self-contained files.

    Use this to get results for display or further processing.
    The original summary.results is not modified.

    Args:
        summary: RepoScanSummary from scan_repo_for_ml_files.
        include_uncertain: If True, include uncertain files too.

    Returns:
        List of ScanResult for self-contained (and optionally uncertain) files.

    Example:
        >>> summary = await scan_repo_for_ml_files(repo_path, client)
        >>> # For DB storage, use summary.results (all results)
        >>> # For display, use filtered results:
        >>> filtered = filter_scan_results(summary, include_uncertain=False)
    """
    if include_uncertain:
        return [
            r
            for r in summary.results
            if r.classification in ("self_contained", "uncertain") and not r.skip_reason
        ]
    return [r for r in summary.results if r.classification == "self_contained"]


def get_self_contained_files(summary: RepoScanSummary) -> list[Path]:
    """Get list of self-contained file paths from scan summary.

    Args:
        summary: RepoScanSummary from scan_repo_for_ml_files.

    Returns:
        List of paths to self-contained ML files.
    """
    return [r.filepath for r in summary.results if r.classification == "self_contained"]
