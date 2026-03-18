"""Auto-fix functions for pattern validation issues.

Contains fix_toml_sync and its helpers for automatically correcting
TOML/file sync issues found by deterministic checks.
"""

import ast
import sys
import tomllib
from difflib import SequenceMatcher
from pathlib import Path

# Add project root to sys.path so pattern_verification can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pattern_verification.deterministic.checks import (
    get_test_files_in_toml,
    get_test_files_on_disk,
)
from pattern_verification.deterministic.models import ValidationResult


def find_best_match(wrong_name: str, available_names: set[str]) -> str | None:
    """Find the best matching filename from available names."""
    best_match = None
    best_ratio = 0.6

    for name in available_names:
        ratio = SequenceMatcher(None, wrong_name, name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = name

    return best_match


def extract_first_function_or_class(file_path: Path) -> tuple[str, str, str]:
    """Extract first function/class name and a code snippet."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                snippet = ""
                for stmt in node.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                        continue
                    if hasattr(stmt, "lineno"):
                        lines = source.split("\n")
                        if stmt.lineno <= len(lines):
                            snippet = lines[stmt.lineno - 1].strip()
                            break
                return "function", node.name, snippet or f"def {node.name}"
            elif isinstance(node, ast.ClassDef):
                return "class", node.name, f"class {node.name}"

        return "module", file_path.stem, ""
    except Exception:
        return "module", file_path.stem, ""


def fix_toml_sync(
    pattern_dir: Path, sync_issues: dict[str, dict[str, list[str]]], result: ValidationResult
) -> None:
    """Fix TOML/file sync issues."""
    toml_path = pattern_dir / "pattern.toml"
    content = toml_path.read_text()

    missing_on_disk = sync_issues.get("missing_on_disk", {})
    missing_in_toml = sync_issues.get("missing_in_toml", {})

    # Try to rename mismatched entries
    for test_type in ["positive", "negative", "context_dependent"]:
        wrong_names = list(missing_on_disk.get(test_type, []))
        available_names = set(missing_in_toml.get(test_type, []))

        for wrong_name in wrong_names:
            best_match = find_best_match(wrong_name, available_names)
            if best_match:
                old_path = f"test_{test_type}/{wrong_name}"
                new_path = f"test_{test_type}/{best_match}"
                content = content.replace(f'"{old_path}"', f'"{new_path}"')
                result.fixed.append(f"Renamed {wrong_name} -> {best_match}")
                available_names.discard(best_match)
            else:
                # Remove orphaned entry
                lines = content.split("\n")
                new_lines: list[str] = []
                skip_block = False
                old_path = f"test_{test_type}/{wrong_name}"

                for line in lines:
                    if f'"{old_path}"' in line:
                        j = len(new_lines) - 1
                        while j >= 0 and not new_lines[j].strip().startswith("[[tests."):
                            j -= 1
                        new_lines = new_lines[:j]
                        skip_block = True
                        result.fixed.append(f"Removed orphaned entry: {wrong_name}")
                    elif skip_block:
                        starts_new = line.strip().startswith(("[[tests.", "[tests."))
                        if starts_new:
                            skip_block = False
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                content = "\n".join(new_lines)

    # Re-read to get updated missing_in_toml
    toml_path.write_text(content)

    # Re-check what's still missing
    with open(toml_path, "rb") as f:
        updated_toml = tomllib.load(f)
    on_disk = get_test_files_on_disk(pattern_dir)
    in_toml = get_test_files_in_toml(updated_toml)

    # Add missing entries
    new_entries = []
    for test_type in ["positive", "negative", "context_dependent"]:
        still_missing = on_disk[test_type] - in_toml[test_type]
        for filename in sorted(still_missing):
            if test_type == "positive":
                file_path = pattern_dir / "test_positive" / filename
                loc_type, loc_name, snippet = extract_first_function_or_class(file_path)
                snippet = snippet.replace('"', '\\"')
                new_entries.append(f'''
[[tests.positive]]
file = "test_positive/{filename}"
description = "TODO: Add description"
expected_issue = "TODO: Add expected issue"
min_confidence = 0.85

[tests.positive.expected_location]
type = "{loc_type}"
name = "{loc_name}"
snippet = "{snippet}"
''')
            elif test_type == "negative":
                new_entries.append(f"""
[[tests.negative]]
file = "test_negative/{filename}"
description = "TODO: Add description"
""")
            else:
                new_entries.append(f"""
[[tests.context_dependent]]
file = "test_context_dependent/{filename}"
description = "TODO: Add description"
context_notes = "TODO: Add context notes"
allow_detection = true
allow_skip = true
""")
            result.fixed.append(f"Added entry for {test_type}/{filename}")

    if new_entries:
        content = toml_path.read_text()
        with open(toml_path, "a") as f:
            f.write("\n# === Auto-generated entries (review and update) ===")
            for entry in new_entries:
                f.write(entry)
