# API Reference

Complete reference for all public classes and methods exposed by scicode-lint.

---

## Exported Classes

```python
from scicode_lint import SciCodeLinter, DetectionCatalog, DetectionPattern
```

---

## SciCodeLinter

Main linter class for checking scientific Python code.

### Class Documentation

```python
class SciCodeLinter:
    """Main linter class for checking scientific Python code.

    Designed for both human users and GenAI coding agents.
    Detects 66 common patterns of bugs in scientific code including
    data leakage, PyTorch training issues, numerical errors, and more.
    """
```

### Constructor

```python
def __init__(self, config: LinterConfig | None = None)
```

**Args:**
- `config`: Linter configuration (uses defaults if None)

**Example:**
```python
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig, Severity

# Default configuration
linter = SciCodeLinter()

# Custom configuration
config = LinterConfig(
    enabled_severities={Severity.CRITICAL, Severity.HIGH}
)
linter = SciCodeLinter(config)
```

### Methods

#### `check_file()`

```python
def check_file(self, file_path: Path) -> LintResult
```

Check a single file for issues.

**Args:**
- `file_path`: Path to Python file to check

**Returns:**
- `LintResult` object with findings

**Example:**
```python
from pathlib import Path
result = linter.check_file(Path("myfile.py"))
for finding in result.findings:
    print(f"{finding.id}: {finding.explanation}")
```

#### `get_pattern()`

```python
def get_pattern(self, pattern_id: str) -> DetectionPattern | None
```

Get pattern details by ID. Useful for GenAI agents to understand what a pattern checks for.

**Args:**
- `pattern_id`: Pattern ID (e.g., "ml-001", "pt-001")

**Returns:**
- `DetectionPattern` object with id, category, severity, detection_question, and warning_message fields
- Returns `None` if pattern not found

**Example:**
```python
pattern = linter.get_pattern("ml-001")
if pattern:
    print(f"Severity: {pattern.severity}")
    print(f"What it checks: {pattern.detection_question}")
    print(f"How to fix: {pattern.warning_message}")
```

#### `list_patterns()`

```python
def list_patterns(self) -> list[DetectionPattern]
```

List all available detection patterns.

**Returns:**
- List of `DetectionPattern` objects (66 total)

**Example:**
```python
for pattern in linter.list_patterns():
    print(f"{pattern.id}: {pattern.warning_message[:50]}...")
```

---

## DetectionCatalog

Loads and manages detection patterns from YAML catalog.

### Class Documentation

```python
class DetectionCatalog:
    """Loads and manages detection patterns from YAML catalog.

    Provides access to all 66 detection patterns with methods to filter
    by ID, severity, or category.
    """
```

### Constructor

```python
def __init__(self, catalog_path: Path)
```

**Args:**
- `catalog_path`: Path to detection_catalog.yaml file

**Example:**
```python
from scicode_lint import DetectionCatalog
from pathlib import Path

catalog = DetectionCatalog(Path("src/scicode_lint/detection_catalog.yaml"))
```

### Methods

#### `get_pattern()`

```python
def get_pattern(self, pattern_id: str) -> DetectionPattern | None
```

Get pattern by ID.

**Args:**
- `pattern_id`: Pattern identifier (e.g., "ml-001", "pt-001")

**Returns:**
- `DetectionPattern` object if found, `None` otherwise

**Example:**
```python
pattern = catalog.get_pattern("ml-001")
if pattern:
    print(pattern.warning_message)
```

#### `get_patterns_by_severity()`

```python
def get_patterns_by_severity(self, severity: Severity) -> list[DetectionPattern]
```

Get all patterns matching a severity level.

**Args:**
- `severity`: Severity level (Severity.CRITICAL, Severity.HIGH, or Severity.MEDIUM)

**Returns:**
- List of `DetectionPattern` objects matching the severity

**Example:**
```python
from scicode_lint.config import Severity

critical = catalog.get_patterns_by_severity(Severity.CRITICAL)
print(f"Found {len(critical)} critical patterns")
```

#### `get_patterns_by_category()`

```python
def get_patterns_by_category(self, category: str) -> list[DetectionPattern]
```

Get all patterns in a category.

**Args:**
- `category`: Category name (e.g., "ai-training", "ai-inference", "scientific-numerical")

**Returns:**
- List of `DetectionPattern` objects in the category

**Example:**
```python
ml_patterns = catalog.get_patterns_by_category("ai-training")
for p in ml_patterns:
    print(f"{p.id}: {p.warning_message}")
```

---

## DetectionPattern

A single detection pattern from the catalog.

### Class Documentation

```python
@dataclass
class DetectionPattern:
    """A single detection pattern from the catalog.

    Attributes:
        id: Pattern identifier (e.g., "ml-001", "pt-001")
        category: Pattern category (e.g., "ai-training", "ai-inference")
        severity: Severity level (CRITICAL, HIGH, or MEDIUM)
        detection_question: What the pattern checks for (used in prompts)
        warning_message: Explanation of the issue and how to fix it
    """
```

### Attributes

- **`id: str`** - Pattern identifier (e.g., "ml-001", "pt-001")
- **`category: str`** - Pattern category (e.g., "ai-training", "ai-inference")
- **`severity: Severity`** - Severity level (CRITICAL, HIGH, or MEDIUM)
- **`detection_question: str`** - What the pattern checks for (used in prompts)
- **`warning_message: str`** - Explanation of the issue and how to fix it

### Example

```python
pattern = linter.get_pattern("ml-001")

# Access fields
print(f"ID: {pattern.id}")                      # "ml-001"
print(f"Category: {pattern.category}")          # "ai-training"
print(f"Severity: {pattern.severity}")          # Severity.CRITICAL
print(f"Checks: {pattern.detection_question}")  # What it detects
print(f"Fix: {pattern.warning_message}")        # How to fix it
```

---

## Supporting Types

### LintResult

Result of checking a file.

**Attributes:**
- `file: Path` - File that was checked
- `findings: list[Finding]` - List of issues found
- `error: LintError | None` - Error that occurred (if any)
- `summary: dict` - Summary statistics

**Example:**
```python
result = linter.check_file(Path("myfile.py"))
print(f"File: {result.file}")
print(f"Issues: {len(result.findings)}")

# Check for errors (e.g., file too large)
if result.error:
    print(f"Error: {result.error.error_type}")
    if result.error.details:
        print(f"Details: {result.error.details}")
```

**JSON Output:**
```python
result_dict = result.to_dict()
# Includes both findings and errors in structured format
```

### Finding

A single issue found in the code.

**Attributes:**
- `id: str` - Pattern ID (e.g., "ml-001")
- `category: str` - Category name
- `severity: str` - "critical", "high", or "medium"
- `location: Location` - Where the issue was found
- `issue: str` - Short description
- `explanation: str` - Detailed explanation and fix instructions
- `suggestion: str` - How to fix
- `confidence: float` - Detection confidence (0.0-1.0)

**Example:**
```python
for finding in result.findings:
    print(f"{finding.id}: {finding.explanation}")
    print(f"Location: {finding.location.name}")
    print(f"Code: {finding.location.snippet}")
```

### Location

Where an issue was found in the code. Uses name-based identification with AST-verified line numbers.

**Attributes:**
- `name: str | None` - Function/class/method name where issue occurs
- `location_type: str | None` - "function", "class", "method", or "module"
- `lines: list[int]` - Full line range of the function/method (for context)
- `focus_line: int | None` - Specific line to look at (most actionable)
- `snippet: str` - Code snippet from the function/method

**Example:**
```python
location = finding.location
print(f"Found in {location.location_type}: {location.name}")
print(f"Lines: {location.lines[0]}-{location.lines[-1]}")
if location.focus_line:
    print(f"Focus on line: {location.focus_line}")
print(f"Code: {location.snippet}")
```

### LintError

An error that occurred during linting. Designed for both human and GenAI agent consumption.

**Attributes:**
- `file: Path` - File where error occurred
- `error_type: str` - Type of error (e.g., "ContextLengthError")
- `message: str` - Human-readable error message
- `details: dict[str, Any] | None` - Structured error details (optional)

**Example:**
```python
result = linter.check_file(Path("large_file.py"))
if result.error:
    print(f"Error type: {result.error.error_type}")
    print(f"Message: {result.error.message}")

    # For programmatic handling (GenAI agents)
    if result.error.details:
        if result.error.details.get("overflow"):
            print(f"File exceeds limit by {result.error.details['overflow']} tokens")
```

**JSON Output:**
```python
error_dict = result.error.to_dict()
# Returns structured dictionary for programmatic parsing
```

---

## Exceptions

All documented linter failures inherit from `SciCodeLintError`. Catch the base
class to cover every case in one handler; catch specific subclasses when you
need branch-specific logic.

```python
from scicode_lint.exceptions import SciCodeLintError, LLMConnectionError
from scicode_lint.linter import NotebookParseError
from scicode_lint.llm.client import MissingLocationError
from scicode_lint.llm.exceptions import ContextLengthError
```

| Exception               | Raised when                                                    |
|-------------------------|----------------------------------------------------------------|
| `SciCodeLintError`      | Base class — catch this to handle any of the below uniformly   |
| `LLMConnectionError`    | vLLM server unreachable on any expected port                   |
| `ContextLengthError`    | Input exceeds model context window (has structured `to_dict()`) |
| `NotebookParseError`    | Jupyter notebook cannot be parsed                              |
| `MissingLocationError`  | LLM returned `detected=yes` but no location (surfaces only if retries exhaust) |

### ContextLengthError

Raised when input exceeds model's context length.

**Attributes:**
- `file_path: str` - Path to the file being checked
- `estimated_tokens: int` - Estimated input size in tokens
- `max_tokens: int` - Maximum context length supported
- `message: str` - Error message with helpful suggestions

**Methods:**

#### `.to_dict()`

Convert exception to structured dictionary for GenAI agent consumption.

**Returns:**
- `dict[str, Any]` - Structured error data

**Example:**
```python
from scicode_lint.llm.exceptions import ContextLengthError

try:
    result = linter.check_file(Path("very_large_file.py"))
except ContextLengthError as e:
    # Human-readable
    print(str(e))

    # For GenAI agents - structured parsing
    error_data = e.to_dict()
    print(f"Overflow: {error_data['overflow']} tokens")
    print(f"Suggestions: {error_data['suggestions']}")

    # Programmatic decision making
    if error_data['overflow'] > 1000:
        # File too large, split it
        split_file(error_data['file_path'])
    else:
        # Close to limit, retry with larger model
        use_larger_context_window()
```

**Structured Output:**
```python
{
    "error": "ContextLengthError",
    "file_path": "/path/to/file.py",
    "estimated_tokens": 12000,
    "max_tokens": 8000,
    "overflow": 4000,
    "suggestions": [
        "Split into smaller files (< 8,000 tokens)",
        "Focus linting on specific functions/classes",
        "Increase max_model_len when starting vLLM server",
        "Use a model with larger context window"
    ]
}
```

---

## Configuration

### LinterConfig

Configuration for the linter.

**Key Parameters:**
- `enabled_severities: set[Severity]` - Which severity levels to check
- `enabled_patterns: set[str] | None` - Specific patterns to check
- `enabled_categories: set[str] | None` - Specific categories to check
- `min_confidence: float` - Minimum confidence threshold (0.0-1.0)

**Example:**
```python
from scicode_lint.config import LinterConfig, Severity

config = LinterConfig(
    enabled_severities={Severity.CRITICAL, Severity.HIGH},
    enabled_patterns={"ml-001", "pt-001"},  # Check specific patterns
    min_confidence=0.8,
)
```

### Severity

Severity enum.

**Values:**
- `Severity.CRITICAL` - Must fix (wrong results)
- `Severity.HIGH` - Should fix (likely problems)
- `Severity.MEDIUM` - Consider fixing (potential issues)

---

## Complete Example

```python
from pathlib import Path
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig, Severity

# Initialize linter
config = LinterConfig(
    enabled_severities={Severity.CRITICAL},
    min_confidence=0.75,
)
linter = SciCodeLinter(config)

# Check file
result = linter.check_file(Path("ml_pipeline.py"))

# Process findings
for finding in result.findings:
    print(f"\n[{finding.severity}] {finding.id}")
    print(f"Location: {finding.location.location_type} '{finding.location.name}'")
    print(f"Code: {finding.location.snippet}")
    print(f"Issue: {finding.explanation}")

    # Look up full pattern details
    pattern = linter.get_pattern(finding.id)
    if pattern:
        print(f"Category: {pattern.category}")
        print(f"What it checks: {pattern.detection_question}")

# List all available patterns
print("\nAvailable patterns:")
for pattern in linter.list_patterns():
    print(f"  {pattern.id} [{pattern.severity.value}]")
```

---

## Documentation Status

✅ **All public methods have:**
- Docstrings with clear descriptions
- Args documentation
- Returns documentation
- Type annotations
- Usage examples

✅ **All classes have:**
- Class-level docstrings
- Attribute documentation (for dataclasses)
- Usage examples

✅ **Verified with:**
- mypy type checking
- Automated docstring tests
- help() introspection
