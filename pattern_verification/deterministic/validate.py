#!/usr/bin/env python3
"""Comprehensive pattern validation - deterministic quality checks.

Usage:
    python pattern_verification/deterministic/validate.py           # Check all patterns
    python pattern_verification/deterministic/validate.py --fix     # Auto-fix what's possible
    python pattern_verification/deterministic/validate.py ml-002    # Check specific pattern
    python pattern_verification/deterministic/validate.py --strict  # Fail on warnings too
    python pattern_verification/deterministic/validate.py --fetch-refs   # Fetch and cache reference docs
    python pattern_verification/deterministic/validate.py --clean-cache  # Remove orphaned cache files

Checks performed:
1. TOML/file sync - every test file has TOML entry and vice versa
2. Schema validation - pattern.toml matches Pydantic model
3. Intent hints - no docstrings/names that reveal expected answers (buggy_, correct_, etc.)
4. Test file count - minimum 3 positive, 3 negative (warning)
5. TODO markers - no unfinished placeholders in TOML
6. Detection question format - ends with YES/NO conditions
7. Test file syntax - all .py files are valid Python
8. Empty fields - required fields have content
9. Category mismatch - meta.category matches directory location
10. Snippet verification - expected_location snippets exist at specified lines
10b. Expected lines - positive/context-dependent tests MUST have expected_location.lines
10c. Expected name - expected_location.name exists in test file (AST check)
11. Related patterns - related_patterns references exist
12. Test file diversity - detect copy-paste (AST similarity)
13. No comments - test files must not contain # comments (stripped before LLM analysis)
14. Reference URLs - fetch and cache reference documentation (--fetch-refs)
15. Registry sync - _registry.toml matches actual pattern count (global check)
"""

import argparse
import sys
import tomllib
from pathlib import Path

# Add project root to sys.path so pattern_verification can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pattern_verification.deterministic.checks import (  # noqa: E402
    check_category_mismatch,
    check_detection_question,
    check_empty_fields,
    check_expected_lines,
    check_expected_location_snippets,
    check_expected_name_exists,
    check_intent_hints,
    check_no_comments,
    check_registry_sync,
    check_related_patterns_exist,
    check_schema,
    check_test_count,
    check_test_diversity,
    check_test_syntax,
    check_todo_markers,
    check_toml_file_sync,
)
from pattern_verification.deterministic.doc_cache import (  # noqa: E402
    DOC_CACHE_MAX_AGE_DAYS,
    DocCutResponse,
    check_reference_urls,
    clean_orphaned_cache,
    extract_doc_content_with_vllm,
    get_cache_filename,
    is_cache_valid,
)
from pattern_verification.deterministic.fix import fix_toml_sync  # noqa: E402
from pattern_verification.deterministic.models import (  # noqa: E402
    ValidationIssue,
    ValidationResult,
)

# Re-exports for backward compatibility (used by tests/test_doc_cache.py)
__all__ = [
    "ValidationIssue",
    "ValidationResult",
    "DOC_CACHE_MAX_AGE_DAYS",
    "DocCutResponse",
    "extract_doc_content_with_vllm",
    "get_cache_filename",
    "is_cache_valid",
]


def validate_pattern(
    pattern_dir: Path, fix: bool = False, fetch_refs: bool = False
) -> ValidationResult:
    """Run all validation checks on a pattern."""
    result = ValidationResult(
        pattern_id=pattern_dir.name,
        category=pattern_dir.parent.name,
    )

    toml_path = pattern_dir / "pattern.toml"
    if not toml_path.exists():
        result.issues.append(ValidationIssue("error", "missing", "pattern.toml not found"))
        return result

    # Load TOML
    try:
        with open(toml_path, "rb") as f:
            toml_data = tomllib.load(f)
    except Exception as e:
        result.issues.append(ValidationIssue("error", "toml_parse", f"Failed to parse TOML: {e}"))
        return result

    # Run checks
    sync_issues = check_toml_file_sync(pattern_dir, toml_data, result)
    check_schema(pattern_dir, toml_data, result)
    check_intent_hints(pattern_dir, result)
    check_test_count(pattern_dir, result)
    check_todo_markers(toml_path, result)
    check_detection_question(toml_data, result)
    check_test_syntax(pattern_dir, result)
    check_empty_fields(toml_data, result)
    check_category_mismatch(pattern_dir, toml_data, result)
    check_expected_location_snippets(pattern_dir, toml_data, result)
    check_expected_lines(pattern_dir, toml_data, result)
    check_expected_name_exists(pattern_dir, toml_data, result)
    check_related_patterns_exist(pattern_dir, toml_data, result)
    check_test_diversity(pattern_dir, result)
    check_no_comments(pattern_dir, result)
    check_reference_urls(toml_data, result, fetch=fetch_refs)

    # Apply fixes if requested
    if fix and (sync_issues.get("missing_in_toml") or sync_issues.get("missing_on_disk")):
        fix_toml_sync(pattern_dir, sync_issues, result)

    return result


def find_all_patterns(patterns_dir: Path) -> list[Path]:
    """Find all pattern directories.

    Delegates to shared utility in pattern_verification.utils.
    """
    from pattern_verification.utils import find_all_patterns as _find_all_patterns

    return _find_all_patterns(patterns_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pattern", nargs="?", help="Specific pattern ID (e.g., ml-002)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix what's possible")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show issues")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument(
        "--fetch-refs",
        action="store_true",
        help=f"Fetch and cache reference docs (expires after {DOC_CACHE_MAX_AGE_DAYS} days)",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove orphaned doc cache files (no longer referenced by any pattern)",
    )
    args = parser.parse_args()

    patterns_dir = Path("src/scicode_lint/patterns")
    if not patterns_dir.exists():
        print("Error: src/scicode_lint/patterns/ directory not found", file=sys.stderr)
        return 1

    # Clean orphaned cache if requested
    if args.clean_cache:
        removed = clean_orphaned_cache(patterns_dir)
        if removed:
            print(f"Removed {len(removed)} orphaned cache file(s):")
            for f in removed:
                print(f"  - {f}")
        else:
            print("No orphaned cache files found.")

    # Check registry sync (global check) - auto-fix if out of sync
    registry_ok, registry_msg = check_registry_sync(patterns_dir)
    if not registry_ok:
        # Auto-rebuild registry
        from scicode_lint.tools.rebuild_registry import RegistryBuilder

        builder = RegistryBuilder(patterns_dir)
        builder.write_registry()
        print(
            f"⚠ Registry was out of sync - rebuilt automatically ({builder.get_stats()['total']} patterns)"
        )
    elif not args.quiet:
        print(f"✓ Registry: {registry_msg}")

    # Find patterns
    if args.pattern:
        from pattern_verification.utils import resolve_pattern

        patterns = resolve_pattern(patterns_dir, args.pattern)
        if not patterns:
            print(f"Error: Pattern '{args.pattern}' not found", file=sys.stderr)
            return 1
    else:
        patterns = find_all_patterns(patterns_dir)

    # Validate
    error_count = 0
    warning_count = 0
    fixed_count = 0

    for pattern_dir in patterns:
        result = validate_pattern(pattern_dir, fix=args.fix, fetch_refs=args.fetch_refs)

        has_errors = result.has_errors
        has_warnings = result.has_warnings

        if has_errors:
            error_count += 1
        if has_warnings:
            warning_count += 1
        if result.fixed:
            fixed_count += 1

        # Print results
        if result.issues or result.fixed:
            prefix = "❌" if has_errors else ("⚠" if has_warnings else "✓")
            print(f"\n{prefix} {result.category}/{result.pattern_id}")

            for issue in result.issues:
                icon = "✗" if issue.level == "error" else "⚡"
                file_info = f" [{issue.file}]" if issue.file else ""
                print(f"   {icon} [{issue.check}]{file_info} {issue.message}")

            for fix_msg in result.fixed:
                print(f"   ✓ Fixed: {fix_msg}")

        elif not args.quiet:
            print(f"✓ {result.category}/{result.pattern_id}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Checked {len(patterns)} patterns")
    print(f"  Errors: {error_count}")
    print(f"  Warnings: {warning_count}")
    if args.fix:
        print(f"  Fixed: {fixed_count}")

    if args.strict:
        return 1 if (error_count > 0 or warning_count > 0) else 0
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
