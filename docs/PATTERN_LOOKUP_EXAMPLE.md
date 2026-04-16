# Pattern Lookup API - Complete Example

This example shows how GenAI agents can look up pattern details based on issue IDs.

## Simple Pattern Lookup

```python
from scicode_lint import SciCodeLinter

# Initialize linter
linter = SciCodeLinter()

# Look up a specific pattern by ID
pattern = linter.get_pattern("ml-001")

if pattern:
    print(f"ID: {pattern.id}")
    print(f"Category: {pattern.category}")
    print(f"Severity: {pattern.severity}")
    print(f"What it checks: {pattern.detection_question}")
    print(f"Warning: {pattern.warning_message}")
else:
    print("Pattern not found")
```

**Output:**
```
ID: ml-001
Category: ai-training
Severity: critical
What it checks: Is there a scaler, normalizer, or encoder (e.g., StandardScaler, MinMaxScaler, LabelEncoder) that is fit or fit_transform'd on the full dataset before a train/test split occurs?
Warning: Data leakage: scaler/encoder is fit on full data including test set. Model performance will be inflated. Use sklearn.pipeline.Pipeline so fitting happens inside each fold.
```

## List All Available Patterns

```python
from scicode_lint import SciCodeLinter

linter = SciCodeLinter()

# List all patterns
print("Available patterns:")
for pattern in linter.list_patterns():
    print(f"  {pattern.id} [{pattern.severity}] - {pattern.warning_message[:60]}...")
```

**Output:**
```
Available patterns:
  ml-001 [critical] - Data leakage: scaler/encoder is fit on full data includi...
  ml-002 [critical] - Data leakage: features are selected using information fr...
  pt-001 [critical] - Missing optimizer.zero_grad(): gradients accumulate acro...
  ...
```

## Complete Workflow: Check → Lookup → Fix

```python
from pathlib import Path
from scicode_lint import SciCodeLinter

def check_and_explain(file_path: str):
    """Check code and provide detailed explanations for each issue."""

    linter = SciCodeLinter()
    result = linter.check_file(Path(file_path))

    if not result.findings:
        print(f"✓ {file_path}: No issues found")
        return

    print(f"\n{file_path}: {len(result.findings)} issues found\n")

    for finding in result.findings:
        # Basic info from finding
        print(f"Issue: {finding.id}")
        print(f"  Severity: {finding.severity}")
        print(f"  Location: {finding.location.location_type} '{finding.location.name}'")
        print(f"  Code: {finding.location.snippet}")
        print(f"  Explanation: {finding.explanation}")

        # Look up full pattern details
        pattern = linter.get_pattern(finding.id)
        if pattern:
            print(f"  Category: {pattern.category}")
            print(f"  Detection criteria: {pattern.detection_question}")

        print()

# Usage
check_and_explain("ml_pipeline.py")
```

**Output:**
```
ml_pipeline.py: 2 issues found

Issue: ml-001
  Severity: critical
  Location: function 'preprocess_data'
  Code: scaler.fit_transform(X)
  Explanation: Data leakage: scaler/encoder is fit on full data including test set. Model performance will be inflated. Use sklearn.pipeline.Pipeline so fitting happens inside each fold.
  Category: ai-training
  Detection criteria: Is there a scaler, normalizer, or encoder (e.g., StandardScaler, MinMaxScaler, LabelEncoder) that is fit or fit_transform'd on the full dataset before a train/test split occurs?

Issue: rep-001
  Severity: high
  Location: module 'ml_pipeline.py'
  Code: np.random.seed(42)
  Explanation: Incomplete random seeds: NumPy, PyTorch, and Python random each use separate RNGs. Set all three for reproducibility.
  Category: reproducibility
  Detection criteria: Are random seeds set incompletely (missing numpy/torch/python random)?
```

## Use Case: Build Pattern Reference for AI Context

**Note:** scicode-lint does NOT include a pre-built database of fixes. However, GenAI agents can build their own reference from the pattern catalog:

```python
from scicode_lint import SciCodeLinter

def build_pattern_reference():
    """
    Build a pattern reference dictionary for AI agent context.

    This is NOT a built-in feature - just an example of how an AI agent
    could organize pattern information for its own use.
    """

    linter = SciCodeLinter()

    # Organize by category for easy lookup
    by_category = {}
    for pattern in linter.list_patterns():
        if pattern.category not in by_category:
            by_category[pattern.category] = []

        by_category[pattern.category].append({
            "id": pattern.id,
            "severity": pattern.severity.value,
            "what_to_check": pattern.detection_question,
            "how_to_fix": pattern.warning_message,
        })

    return by_category

# Usage: AI agent builds this once and keeps in context
pattern_reference = build_pattern_reference()

# When agent encounters ml-001 in results, look it up
if "ai-training" in pattern_reference:
    ml_patterns = pattern_reference["ai-training"]
    ml_001 = next(p for p in ml_patterns if p["id"] == "ml-001")
    print(f"Fix for ml-001: {ml_001['how_to_fix']}")
```

**Alternative (simpler):** Just use `linter.get_pattern()` directly:
```python
# Instead of building a database, just look up when needed
pattern = linter.get_pattern("ml-001")
print(f"Fix: {pattern.warning_message}")
```

## Advanced: Filter and Search Patterns

```python
from scicode_lint import SciCodeLinter, DetectionCatalog
from scicode_lint.config import Severity

linter = SciCodeLinter()

# Find all critical patterns
critical_patterns = [
    p for p in linter.list_patterns()
    if p.severity == Severity.CRITICAL
]
print(f"Critical patterns: {len(critical_patterns)}")

# Find patterns in specific category
training_patterns = [
    p for p in linter.list_patterns()
    if p.category == "ai-training"
]
print(f"AI Training patterns: {len(training_patterns)}")

# Search pattern descriptions
search_term = "data leakage"
matching = [
    p for p in linter.list_patterns()
    if search_term.lower() in p.warning_message.lower()
]
print(f"Patterns related to '{search_term}': {len(matching)}")
for p in matching:
    print(f"  {p.id}: {p.warning_message[:60]}...")
```

## Key Takeaways

1. **Simple lookup**: `linter.get_pattern(pattern_id)` → returns full pattern details
2. **List all**: `linter.list_patterns()` → returns all 66 patterns
3. **Workflow**: Check file → Get findings → Look up pattern details → Apply fix
4. **What you get**:
   - `pattern.id` - Pattern identifier
   - `pattern.category` - Category (ai-training, ai-inference, etc.)
   - `pattern.severity` - Severity level (critical, high, medium)
   - `pattern.detection_question` - What the pattern checks for
   - `pattern.warning_message` - What's wrong and how to fix it

**All info needed to understand and fix issues is available via the API!**
