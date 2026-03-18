"""Deterministic validation checks for pattern quality.

Contains CHECK 1-13 and CHECK 15 (all deterministic checks that don't require
network access or LLM calls).
"""

import ast
import hashlib
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

# Add project root to sys.path so pattern_verification can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pattern_verification.deterministic.models import ValidationIssue, ValidationResult

# Intent hint patterns to detect in test files
# These reveal expected answers, preventing fair LLM evaluation
# Note: Comments are caught by CHECK 13 (no_comments), so we focus on:
# - Docstrings with hints
# - Function/variable names that reveal correctness
INTENT_HINT_PATTERNS = [
    # Docstring hints - bug indicators (evaluative language about code quality)
    (
        r'"""[^"]*\b(bug|buggy|issue|problem|incorrect|wrong|broken|bad|leaky|leak|flaw|defect)\b[^"]*"""',
        "Docstring hints at bug",
    ),
    (
        r"'''[^']*\b(bug|buggy|issue|problem|incorrect|wrong|broken|bad|leaky|leak|flaw|defect)\b[^']*'''",
        "Docstring hints at bug",
    ),
    # Docstring hints - correctness indicators (explicit "this is correct/fixed")
    (
        r'"""[^"]*(correct|fixed|proper)\s+\w*\s*(implementation|version|approach|code|way|method)[^"]*"""',
        "Docstring hints at correctness",
    ),
    (
        r"'''[^']*(correct|fixed|proper)\s+\w*\s*(implementation|version|approach|code|way|method)[^']*'''",
        "Docstring hints at correctness",
    ),
    # Naming hints - function/class/variable names that reveal intent
    (r"\bdef\s+(buggy_|incorrect_|wrong_|broken_|bad_|leaky_)", "Function name hints at bug"),
    (r"\bdef\s+(correct_|fixed_|proper_|safe_|good_)", "Function name hints at correctness"),
    (r"\bclass\s+(Buggy|Incorrect|Wrong|Broken|Bad|Leaky)", "Class name hints at bug"),
    (r"\bclass\s+(Correct|Fixed|Proper|Safe|Good)", "Class name hints at correctness"),
    (
        r"\b(buggy|incorrect|wrong|broken|bad|leaky)_(function|method|class|data|model|scaler)\s*=",
        "Variable name hints at bug",
    ),
    (
        r"\b(correct|fixed|proper|safe|good)_(function|method|class|data|model|scaler)\s*=",
        "Variable name hints at correctness",
    ),
]

# Required YES/NO pattern at end of detection question
YES_NO_PATTERN = re.compile(r"YES\s*=.*\n\s*NO\s*=", re.IGNORECASE | re.MULTILINE)


def get_test_files_on_disk(pattern_dir: Path) -> dict[str, set[str]]:
    """Get all test files on disk for a pattern."""
    result: dict[str, set[str]] = {"positive": set(), "negative": set(), "context_dependent": set()}

    for test_type in result.keys():
        test_dir = pattern_dir / f"test_{test_type}"
        if test_dir.exists():
            result[test_type] = {
                f.name for f in test_dir.glob("*.py") if not f.name.startswith("_")
            }

    return result


def get_test_files_in_toml(toml_data: dict[str, Any]) -> dict[str, set[str]]:
    """Get all test files referenced in pattern.toml."""
    result: dict[str, set[str]] = {"positive": set(), "negative": set(), "context_dependent": set()}

    tests = toml_data.get("tests", {})
    for test_type in result.keys():
        for entry in tests.get(test_type, []):
            file_path = entry.get("file", "")
            result[test_type].add(Path(file_path).name)

    return result


# =============================================================================
# CHECK 1: TOML/File Sync
# =============================================================================


def check_toml_file_sync(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> dict[str, dict[str, list[str]]]:
    """Check that test files on disk match TOML entries."""
    on_disk = get_test_files_on_disk(pattern_dir)
    in_toml = get_test_files_in_toml(toml_data)

    issues: dict[str, dict[str, list[str]]] = {"missing_in_toml": {}, "missing_on_disk": {}}

    for test_type in ["positive", "negative", "context_dependent"]:
        missing_in_toml = on_disk[test_type] - in_toml[test_type]
        missing_on_disk = in_toml[test_type] - on_disk[test_type]

        if missing_in_toml:
            issues["missing_in_toml"][test_type] = sorted(missing_in_toml)
            for f in missing_in_toml:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "toml_sync",
                        f"File on disk not in TOML: test_{test_type}/{f}",
                    )
                )

        if missing_on_disk:
            issues["missing_on_disk"][test_type] = sorted(missing_on_disk)
            for f in missing_on_disk:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "toml_sync",
                        f"TOML entry but file missing: test_{test_type}/{f}",
                    )
                )

    return issues


# =============================================================================
# CHECK 2: Schema Validation
# =============================================================================


def check_schema(pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult) -> bool:
    """Validate pattern.toml against Pydantic schema."""
    try:
        from scicode_lint.detectors.pattern_models import PatternTOML

        PatternTOML(**toml_data)
        return True
    except Exception as e:
        result.issues.append(ValidationIssue("error", "schema", f"Schema validation failed: {e}"))
        return False


# =============================================================================
# CHECK 3: Intent Hints Detection (docstrings and naming)
# =============================================================================


def check_intent_hints(pattern_dir: Path, result: ValidationResult) -> None:
    """Check test files for hints that reveal expected answers.

    Detects:
    - Docstrings containing words like "bug", "incorrect", "wrong"
    - Function/class/variable names like "buggy_function", "correct_approach"

    Note: Comments are handled by CHECK 13 (no_comments).
    """
    for test_type in ["positive", "negative", "context_dependent"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        for py_file in test_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text()
            except Exception:
                continue

            for pattern, desc in INTENT_HINT_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    result.issues.append(
                        ValidationIssue(
                            "error",
                            "intent_hints",
                            f"{desc}",
                            file=f"test_{test_type}/{py_file.name}",
                        )
                    )


# =============================================================================
# CHECK 4: Test File Count
# =============================================================================


def check_test_count(pattern_dir: Path, result: ValidationResult) -> None:
    """Check minimum test file counts (3 positive, 3 negative recommended)."""
    on_disk = get_test_files_on_disk(pattern_dir)

    pos_count = len(on_disk["positive"])
    neg_count = len(on_disk["negative"])

    if pos_count < 3:
        result.issues.append(
            ValidationIssue(
                "warning",
                "test_count",
                f"Only {pos_count} positive tests (recommend 3+)",
            )
        )

    if neg_count < 3:
        result.issues.append(
            ValidationIssue(
                "warning",
                "test_count",
                f"Only {neg_count} negative tests (recommend 3+)",
            )
        )


# =============================================================================
# CHECK 5: TODO Markers
# =============================================================================


def check_todo_markers(toml_path: Path, result: ValidationResult) -> None:
    """Check for unfinished TODO placeholders in TOML."""
    content = toml_path.read_text()

    # Find TODO patterns
    todo_matches = re.findall(r"TODO[:\s].*", content, re.IGNORECASE)
    for match in todo_matches:
        result.issues.append(
            ValidationIssue("error", "todo_marker", f"Unfinished: {match.strip()}")
        )


# =============================================================================
# CHECK 6: Detection Question Format
# =============================================================================


def check_detection_question(toml_data: dict[str, Any], result: ValidationResult) -> None:
    """Check detection question ends with YES/NO conditions."""
    detection = toml_data.get("detection", {})
    question = detection.get("question", "")

    if not question:
        result.issues.append(
            ValidationIssue("error", "detection_format", "Missing detection question")
        )
        return

    if not YES_NO_PATTERN.search(question):
        result.issues.append(
            ValidationIssue(
                "warning",
                "detection_format",
                "Detection question should end with 'YES = ...' and 'NO = ...' conditions",
            )
        )


# =============================================================================
# CHECK 7: Test File Syntax
# =============================================================================


def check_test_syntax(pattern_dir: Path, result: ValidationResult) -> None:
    """Check all test files are valid Python."""
    for test_type in ["positive", "negative", "context_dependent"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        for py_file in test_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                source = py_file.read_text()
                ast.parse(source)
            except SyntaxError as e:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "syntax",
                        f"Invalid Python syntax: {e}",
                        file=f"test_{test_type}/{py_file.name}",
                    )
                )


# =============================================================================
# CHECK 8: Empty Fields
# =============================================================================


def check_empty_fields(toml_data: dict[str, Any], result: ValidationResult) -> None:
    """Check required fields have content."""
    # Check meta fields
    meta = toml_data.get("meta", {})
    for field_name in ["description", "explanation"]:
        value = meta.get(field_name, "")
        if not value or not value.strip():
            result.issues.append(
                ValidationIssue("error", "empty_field", f"meta.{field_name} is empty")
            )

    # Check detection fields
    detection = toml_data.get("detection", {})
    for field_name in ["question", "warning_message"]:
        value = detection.get(field_name, "")
        if not value or not value.strip():
            result.issues.append(
                ValidationIssue("error", "empty_field", f"detection.{field_name} is empty")
            )

    # Check test entries
    tests = toml_data.get("tests", {})
    for test_type in ["positive", "negative", "context_dependent"]:
        for i, entry in enumerate(tests.get(test_type, [])):
            if not entry.get("description", "").strip():
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "empty_field",
                        f"tests.{test_type}[{i}].description is empty",
                    )
                )
            if test_type == "positive" and not entry.get("expected_issue", "").strip():
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "empty_field",
                        f"tests.{test_type}[{i}].expected_issue is empty",
                    )
                )


# =============================================================================
# CHECK 9: Category Mismatch
# =============================================================================


def check_category_mismatch(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that meta.category matches the pattern's directory location."""
    meta = toml_data.get("meta", {})
    toml_category = meta.get("category", "")
    dir_category = pattern_dir.parent.name

    if toml_category and toml_category != dir_category:
        result.issues.append(
            ValidationIssue(
                "error",
                "category_mismatch",
                f"meta.category='{toml_category}' but pattern is in '{dir_category}/' directory",
            )
        )


# =============================================================================
# CHECK 10: Expected Location Snippet Verification
# =============================================================================


def check_expected_location_snippets(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that expected_location snippets exist at specified lines.

    Two checks:
    1. Snippet must exist somewhere in the file (error if not)
    2. If lines are specified, snippet must appear within those lines (error if not)
    """
    tests = toml_data.get("tests", {})

    for test_type in ["positive", "context_dependent"]:
        for entry in tests.get(test_type, []):
            expected_loc = entry.get("expected_location", {})
            snippet = expected_loc.get("snippet", "")
            lines = expected_loc.get("lines", [])
            file_path_str = entry.get("file", "")

            if not snippet or not file_path_str:
                continue

            file_path = pattern_dir / file_path_str
            if not file_path.exists():
                continue  # Already caught by toml_sync check

            try:
                content = file_path.read_text()
                file_lines = content.splitlines()

                # Normalize whitespace for comparison
                normalized_content = " ".join(content.split())
                normalized_snippet = " ".join(snippet.split())

                # Check 1: Snippet exists somewhere in file
                if normalized_snippet not in normalized_content:
                    result.issues.append(
                        ValidationIssue(
                            "error",
                            "snippet_not_in_file",
                            f"Snippet not found in file: '{snippet[:50]}...'",
                            file=file_path_str,
                        )
                    )
                    continue  # Skip line check if snippet not in file at all

                # Check 2: If lines specified, snippet must be at those lines
                if lines:
                    # Extract content at specified lines (1-indexed)
                    lines_content = []
                    for line_num in lines:
                        if 1 <= line_num <= len(file_lines):
                            lines_content.append(file_lines[line_num - 1])

                    lines_text = " ".join(" ".join(lines_content).split())

                    if normalized_snippet not in lines_text:
                        result.issues.append(
                            ValidationIssue(
                                "error",
                                "snippet_not_at_lines",
                                f"Snippet '{snippet[:30]}...' not at lines {lines}",
                                file=file_path_str,
                            )
                        )
            except Exception:
                pass


# =============================================================================
# CHECK 10b: Expected Lines Required
# =============================================================================


def check_expected_lines(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that expected_lines is defined for positive/context_dependent tests.

    Expected lines are required for:
    - positive tests: MUST have expected_lines (bug location is known)
    - context_dependent tests: MUST have expected_lines (potential issue location)
    - negative tests: should NOT have expected_lines (no bug to locate)
    """
    tests = toml_data.get("tests", {})

    # Check positive tests - MUST have expected_lines
    for entry in tests.get("positive", []):
        file_path_str = entry.get("file", "")
        expected_loc = entry.get("expected_location", {})
        lines = expected_loc.get("lines", [])

        if not lines:
            result.issues.append(
                ValidationIssue(
                    "error",
                    "missing_expected_lines",
                    "Positive test must have expected_location.lines (add lines = [n, m, ...])",
                    file=file_path_str,
                )
            )

    # Check context_dependent tests - MUST have expected_lines
    for entry in tests.get("context_dependent", []):
        file_path_str = entry.get("file", "")
        expected_loc = entry.get("expected_location", {})
        lines = expected_loc.get("lines", [])

        if not lines:
            result.issues.append(
                ValidationIssue(
                    "error",
                    "missing_expected_lines",
                    "Context-dependent test must have expected_location.lines",
                    file=file_path_str,
                )
            )


# =============================================================================
# CHECK 10c: Expected Name Exists in Test File
# =============================================================================


def _get_defined_names(tree: ast.AST) -> dict[str, str]:
    """Extract all function, class, and method names from AST.

    Returns:
        Dict mapping name to location_type ("function", "class", "method").
    """
    names: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check if it's a method (inside a class)
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for item in parent.body:
                        if item is node:
                            names[node.name] = "method"
                            break
            if node.name not in names:
                names[node.name] = "function"
        elif isinstance(node, ast.ClassDef):
            names[node.name] = "class"

    return names


def check_expected_name_exists(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that expected_location.name exists in test file via AST.

    With name-based detection, the linter matches on function/class names.
    This check ensures the expected_location.name actually exists in the test file,
    catching typos early during validation instead of waiting for evals to fail.
    """
    tests = toml_data.get("tests", {})

    for test_type in ["positive", "context_dependent"]:
        for entry in tests.get(test_type, []):
            file_path_str = entry.get("file", "")
            expected_loc = entry.get("expected_location", {})
            expected_name = expected_loc.get("name", "")
            expected_type = expected_loc.get("type", "")

            if not expected_name:
                continue  # Missing name is caught by schema validation

            # Skip module-level (no specific name to check)
            if expected_type == "module":
                continue

            file_path = pattern_dir / file_path_str
            if not file_path.exists():
                continue  # File existence checked elsewhere

            try:
                source = file_path.read_text()
                tree = ast.parse(source)
            except (SyntaxError, OSError):
                continue  # Syntax errors caught elsewhere

            defined_names = _get_defined_names(tree)

            if expected_name not in defined_names:
                result.issues.append(
                    ValidationIssue(
                        "error",
                        "expected_name_not_found",
                        f"expected_location.name '{expected_name}' not found in file. "
                        f"Defined names: {list(defined_names.keys())[:5]}",
                        file=file_path_str,
                    )
                )
            elif expected_type and defined_names[expected_name] != expected_type:
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "expected_type_mismatch",
                        f"expected_location.type '{expected_type}' but "
                        f"'{expected_name}' is a {defined_names[expected_name]}",
                        file=file_path_str,
                    )
                )


# =============================================================================
# CHECK 11: Related Patterns Exist
# =============================================================================


def check_related_patterns_exist(
    pattern_dir: Path, toml_data: dict[str, Any], result: ValidationResult
) -> None:
    """Check that related_patterns references exist."""
    meta = toml_data.get("meta", {})
    related = meta.get("related_patterns", [])
    patterns_root = pattern_dir.parent.parent

    for related_id in related:
        # Search for the pattern in all categories
        # Patterns can be referenced by:
        # - Full name: "pt-007-inference-without-eval"
        # - ID prefix: "pt-007" (matches "pt-007-*")
        found = False
        for category_dir in patterns_root.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith((".", "_")):
                continue
            for pdir in category_dir.iterdir():
                # Match full name or ID prefix (e.g., "pt-007" matches "pt-007-...")
                if pdir.name == related_id or pdir.name.startswith(f"{related_id}-"):
                    found = True
                    break
            if found:
                break

        if not found:
            result.issues.append(
                ValidationIssue(
                    "error",
                    "related_pattern",
                    f"Related pattern '{related_id}' does not exist",
                )
            )


# =============================================================================
# CHECK 12: Test File Diversity (AST similarity)
# =============================================================================


def get_ast_hash(file_path: Path) -> str | None:
    """Get a normalized AST hash for a Python file."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        # Normalize: remove docstrings, normalize names
        for node in ast.walk(tree):
            # Remove docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    node.body = node.body[1:]

        # Convert to string and hash
        ast_str = ast.dump(tree, annotate_fields=False)
        return hashlib.md5(ast_str.encode()).hexdigest()[:8]
    except Exception:
        return None


def check_test_diversity(pattern_dir: Path, result: ValidationResult) -> None:
    """Check test files aren't too similar (copy-paste detection)."""
    for test_type in ["positive", "negative"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        files = [f for f in test_dir.glob("*.py") if not f.name.startswith("_")]
        if len(files) < 2:
            continue

        # Get AST hashes
        hashes: dict[str, list[str]] = {}
        for f in files:
            h = get_ast_hash(f)
            if h:
                hashes.setdefault(h, []).append(f.name)

        # Check for duplicates
        for h, filenames in hashes.items():
            if len(filenames) > 1:
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "diversity",
                        f"Nearly identical files in test_{test_type}: {', '.join(filenames)}",
                    )
                )


# =============================================================================
# CHECK 13: No Comments in Test Files
# =============================================================================


def check_no_comments(pattern_dir: Path, result: ValidationResult) -> None:
    """Check test files contain no # comments.

    Why forbid comments if they're stripped before LLM evaluation?
    1. Consistency - developers see exactly what LLM sees when reviewing
    2. No fossilized interpretations - old/wrong comments don't mislead maintainers
    3. Defense in depth - if stripping fails (syntax error), no comments leak through

    Uses Python's tokenize module to correctly identify comments
    (handles # inside strings).
    """
    import io
    import tokenize

    for test_type in ["positive", "negative", "context_dependent"]:
        test_dir = pattern_dir / f"test_{test_type}"
        if not test_dir.exists():
            continue

        for py_file in test_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text()
                for tok in tokenize.generate_tokens(io.StringIO(content).readline):
                    if tok.type == tokenize.COMMENT:
                        comment_text = tok.string.strip()
                        if len(comment_text) > 30:
                            comment_text = comment_text[:30] + "..."
                        result.issues.append(
                            ValidationIssue(
                                "error",
                                "no_comments",
                                f"Comment found at line {tok.start[0]}: {comment_text}",
                                file=f"test_{test_type}/{py_file.name}",
                            )
                        )
                        break
            except tokenize.TokenError:
                pass
            except Exception:
                pass


# =============================================================================
# CHECK 15: Registry Sync (global check)
# =============================================================================


def check_registry_sync(patterns_dir: Path) -> tuple[bool, str]:
    """Check that _registry.toml is in sync with actual patterns.

    Returns:
        Tuple of (is_valid, message). is_valid is True if registry is up to date.
    """
    registry_path = patterns_dir / "_registry.toml"

    if not registry_path.exists():
        return False, "Registry file _registry.toml not found"

    # Count actual patterns
    actual_count = 0
    for category in patterns_dir.iterdir():
        if not category.is_dir() or category.name.startswith((".", "_")):
            continue
        for pattern_dir in category.iterdir():
            if pattern_dir.is_dir() and (pattern_dir / "pattern.toml").exists():
                actual_count += 1

    # Read registry count
    try:
        with open(registry_path, "rb") as f:
            registry_data = tomllib.load(f)
        registry_count = registry_data.get("total_patterns", 0)
    except Exception as e:
        return False, f"Failed to read registry: {e}"

    if registry_count != actual_count:
        return (
            False,
            f"Registry out of sync: has {registry_count} patterns but {actual_count} exist. "
            f"Run: python src/scicode_lint/tools/rebuild_registry.py --patterns-dir {patterns_dir}",
        )

    return True, f"Registry in sync ({actual_count} patterns)"
