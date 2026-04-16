"""Repository filter module for self-contained ML file detection.

This module filters Python files in repositories, identifying self-contained
ML workflows vs fragments that need other files.

Three-stage filtering:
1. File extension filter (only .py and .ipynb files)
2. ML import presence check (deterministic, instant)
3. LLM classification (self-contained vs fragment)

Standalone usage (no database required):

    import asyncio
    from pathlib import Path
    from scicode_lint.config import load_llm_config
    from scicode_lint.llm.client import create_client
    from scicode_lint.repo_filter import scan_repo_for_ml_files, filter_scan_results

    llm_config = load_llm_config()
    client = create_client(llm_config)

    # Scan returns ALL results (self-contained, fragments, skipped)
    summary = asyncio.run(scan_repo_for_ml_files(
        repo_path=Path("./my_ml_project"),
        llm_client=client,
        max_file_tokens=12000,  # Skip files larger than ~48K chars
    ))

    # Access all results (for DB storage, full reporting)
    print(f"Total: {summary.total_files}, Self-contained: {summary.self_contained}")
    for result in summary.results:
        print(f"  {result.filepath}: {result.classification}")

    # Filter for display (only self-contained, or +uncertain)
    filtered = filter_scan_results(summary, include_uncertain=False)
    for r in filtered:
        print(f"  {r.filepath}: {r.details.confidence:.2f}")

    # Export to dict/JSON
    data = summary.to_dict()

CLI usage:

    scicode-lint filter-repo ./my_project --format json -o results.json
"""

from scicode_lint.repo_filter.classify import (
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_PROMPT,
    FileClassification,
    classify_file,
)
from scicode_lint.repo_filter.scan import (
    RepoScanSummary,
    ScanResult,
    filter_scan_results,
    get_ml_import_keywords,
    get_self_contained_files,
    has_ml_imports,
    scan_repo_for_ml_files,
)

__all__ = [
    # Classification
    "CLASSIFY_SYSTEM_PROMPT",
    "CLASSIFY_USER_PROMPT",
    "FileClassification",
    "classify_file",
    # Scanning
    "RepoScanSummary",
    "ScanResult",
    "filter_scan_results",
    "get_ml_import_keywords",
    "get_self_contained_files",
    "has_ml_imports",
    "scan_repo_for_ml_files",
]
