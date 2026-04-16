# Detection Architecture: Name-Based Location Resolution

This document describes the detection flow in scicode-lint, specifically how we identify and verify the location of detected issues.

## Design Philosophy

**Core insight: LLMs are good at semantic understanding, bad at counting lines.**

When we ask an LLM "is there data leakage in this code?", it excels at:
- Understanding code structure and intent
- Recognizing function/class boundaries
- Identifying which function contains the issue

But LLMs struggle with:
- Accurately counting line numbers (±1-2 variance is common)
- Maintaining consistency across runs
- Handling line number changes in modified files

**Solution: Name-based location with AST verification.**

The LLM identifies WHAT has the issue (function/class/method name), and we use deterministic AST parsing to find WHERE it is (exact line numbers).

## Detection Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DETECTION FLOW                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Code Input                                                      │
│     ┌─────────────────────────────────────────────────────┐         │
│     │ def train_model(data):        # line 10            │         │
│     │     scaler = StandardScaler()  # line 11           │         │
│     │     X = scaler.fit_transform(data)  # line 12  ◄── issue     │
│     │     X_train, X_test = split(X)      # line 13      │         │
│     │     return X_train, X_test          # line 14      │         │
│     └─────────────────────────────────────────────────────┘         │
│                              │                                      │
│                              ▼                                      │
│  2. LLM Detection (Semantic Understanding)                          │
│     ┌─────────────────────────────────────────────────────┐         │
│     │ {                                                   │         │
│     │   "detected": "yes",                                │         │
│     │   "location": {                                     │         │
│     │     "name": "train_model",      ◄── identifies WHAT │         │
│     │     "location_type": "function",                    │         │
│     │     "near_line": 12             ◄── approximate hint│         │
│     │   },                                                │         │
│     │   "confidence": 0.95,                               │         │
│     │   "reasoning": "Scaler fit on full data..."         │         │
│     │ }                                                   │         │
│     └─────────────────────────────────────────────────────┘         │
│                              │                                      │
│                              ▼                                      │
│  3. AST Resolution (Deterministic Verification)                     │
│     ┌─────────────────────────────────────────────────────┐         │
│     │ resolve_name(code, "train_model", "function")       │         │
│     │                                                     │         │
│     │ Returns:                                            │         │
│     │   name: "train_model"                               │         │
│     │   location_type: "function"                         │         │
│     │   start_line: 10          ◄── verified boundaries   │         │
│     │   end_line: 14                                      │         │
│     │   snippet: "def train_model(data):..."              │         │
│     └─────────────────────────────────────────────────────┘         │
│                              │                                      │
│                              ▼                                      │
│  4. Output (Verified Location)                                      │
│     ┌─────────────────────────────────────────────────────┐         │
│     │ Location:                                           │         │
│     │   name: "train_model"                               │         │
│     │   location_type: "function"                         │         │
│     │   lines: [10, 11, 12, 13, 14]  ◄── full context     │         │
│     │   focus_line: 12               ◄── specific issue   │         │
│     │   snippet: "def train_model..."                     │         │
│     └─────────────────────────────────────────────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Benefits of Name-Based Detection

### 1. Forces Better Code Understanding

By requiring the LLM to identify the function/method name, we force it to:
- Parse the code structure before making a judgment
- Understand scope boundaries (what's inside which function)
- Think about the semantic meaning of code blocks

This is harder than just pointing at a line number, but produces more reliable results.

### 2. Eliminates Line Number Oscillation

Previous approach with direct line numbers:
```
Run 1: detected on lines [24, 25]
Run 2: detected on lines [23, 24, 25]
Run 3: detected on lines [24]
```

Name-based approach:
```
Run 1: detected in "train_model" (near_line: 24) → AST resolves to lines 20-30
Run 2: detected in "train_model" (near_line: 23) → AST resolves to lines 20-30
Run 3: detected in "train_model" (near_line: 24) → AST resolves to lines 20-30
```

The variance in `near_line` doesn't matter because AST resolution is deterministic.

### 3. Enables Verification

We can verify that:
- The named function/class/method actually exists
- The issue location is valid code
- The context makes sense

If AST resolution fails, we know something is wrong with the detection.

### 4. Better User Output

Instead of just "issue on line 24", users see:
```
🔴 CRITICAL [in train_model (lines 20-30, focus: line 24)] ml-001: Issue detected
```

This gives:
- The function name (for quick identification)
- The function boundaries (for context)
- The specific line to look at (for fixing)

## Schema Details

### LLM Output Schema (NamedLocation)

```python
class NamedLocation(BaseModel):
    name: str           # Function/class/method name, e.g., "train_model", "Trainer.fit"
    location_type: str  # "function", "method", "class", or "module"
    near_line: int | None  # Approximate line (optional hint for disambiguation)
```

### Resolved Location (from AST)

```python
class ResolvedLocation:
    name: str           # Verified name
    location_type: str  # Verified type
    start_line: int     # First line of definition
    end_line: int       # Last line of definition
    snippet: str        # Full code snippet
```

### Output Location (to user)

```python
class Location(BaseModel):
    name: str | None           # Function/class/method name
    location_type: str | None  # Type of code construct
    lines: list[int]           # Full line range (context)
    focus_line: int | None     # Specific line to look at
    snippet: str               # Code snippet
```

## One Finding Per Pattern

**Design decision: Each pattern check produces at most ONE finding per file.**

If the same bug pattern appears multiple times in a file (e.g., data leakage in both `train_model()` and `evaluate_model()`), the linter detects only the most clear instance.

**Rationale:**
1. **LLM reliability** - Asking for ONE clear instance produces more reliable results than asking for all instances
2. **Actionable output** - Users fix one issue, re-run, fix next - iterative improvement
3. **Simpler validation** - Pattern tests expect ONE location per positive test file
4. **Reduced false positives** - Forcing "find the clearest example" filters out marginal cases

**User workflow for multiple issues:**
```
1. Run linter → finds issue in train_model()
2. Fix train_model()
3. Re-run linter → now finds issue in evaluate_model()
4. Fix evaluate_model()
5. Re-run linter → clean
```

The LLM prompt explicitly instructs: "If multiple instances exist, report the MOST CLEAR example."

## Single-Location Reporting (Known Limitation)

**The linter reports exactly one location per finding: one function name + one approximate line.**

This means:

1. **Multi-line bugs within a function** (e.g., `start = time.time()` on line 10 and `end = time.time()` on line 13 without `synchronize()` between them) — the linter reports the function name. The user sees the whole function and can identify all relevant lines. **This works well enough.**

2. **Cross-function bugs** (e.g., a scaler fitted on train+test data in `preprocess()` then used in `evaluate()`) — the linter can only point to one function. **This is not supported.**

**Why this is acceptable today:**
- All 66 current patterns are function-local — no pattern requires cross-function detection
- The constrained local LLM (Qwen3) already struggles with single-location accuracy; multiple locations would reduce reliability
- Function-level granularity is sufficient for the user to find and fix the issue

**Why we are not implementing multi-location reporting now:**
- No current pattern needs it (YAGNI)
- Would require schema changes (`NamedLocation` → list), prompt changes, output formatter changes, and eval framework changes
- Higher complexity with no proven benefit for current patterns

**When to revisit:** If we add patterns that detect cross-function bugs (e.g., data leakage across pipeline stages), this limitation becomes real and should be addressed.

**Related:** The `lines` field in `pattern.toml` test entries lists all buggy lines within the named function. The linter does not consume them at runtime, but evals use them for the focus line accuracy metric (see below).

## Focus Line Accuracy (Eval Metric)

**What it measures:** Does the linter's `focus_line` (derived from the LLM's `near_line` hint) land on one of the expected buggy lines from `pattern.toml`?

**How it works:**
1. Pattern.toml lists `lines = [10, 11, 12, 13]` — all buggy lines within the named function
2. The linter's LLM returns `near_line` (approximate), which is verified against the AST-resolved function range to produce `focus_line`
3. Evals check: `focus_line ∈ expected_lines`

**Scope:** Only positive tests where the linter detected an issue and the pattern has `expected_location.lines`.

**Current baseline:** ~42% accuracy. The LLM finds the right function reliably but points at the exact buggy line less than half the time.

**Not a gate in the improvement loop.** This is an observational metric, not actionable:
- It doesn't affect detection correctness (right function is found via name match)
- It doesn't affect user experience (users see the whole function context)
- Tuning prompts for line precision could hurt detection accuracy
- `near_line` is documented as an optional, approximate hint

**When to revisit:** If we add output features that depend on precise line targeting (e.g., inline IDE annotations), focus line accuracy becomes important.

## Edge Cases

### Module-Level Code

When issues occur outside any function/class:
```python
# Module-level code
import numpy as np
data = load_data()  # Issue here
X = preprocess(data)
```

LLM returns:
```json
{"name": "<module>", "location_type": "module", "near_line": 3}
```

AST resolution returns context around `near_line` (±3 lines).

### Duplicate Names

When multiple definitions have the same name:
```python
def process(x):  # line 5
    return x * 2

def process(x):  # line 10 (redefinition)
    return x * 3
```

LLM provides `near_line` to disambiguate. AST resolution picks the closest match.

### Nested Definitions

For nested classes/methods:
```python
class Trainer:
    class Config:
        def validate(self):  # Issue here
            pass
```

LLM returns qualified name: `"Trainer.Config.validate"`.

### AST Resolution Failure

If the named function doesn't exist (LLM hallucinated):
1. Fall back to `near_line` context (±3 lines)
2. Log warning for debugging
3. Still provide useful output to user

## Implementation Files

| File | Purpose |
|------|---------|
| `src/scicode_lint/ast_utils.py` | AST parsing and name resolution |
| `src/scicode_lint/llm/models.py` | `NamedLocation` schema for LLM output |
| `src/scicode_lint/detectors/prompts.py` | Prompts that ask for name-based location |
| `src/scicode_lint/linter.py` | Orchestration: LLM → AST → Output |
| `src/scicode_lint/output/formatter.py` | `Location` model with verified lines |

## Relationship to Pattern Tests

Pattern test files in `pattern.toml` use the same location schema:

```toml
[tests.positive.expected_location]
type = "function"
name = "train_model"
snippet = "X = scaler.fit_transform(data)"
lines = [12]  # Expected focus line(s)
```

This enables eval to verify:
1. **Name match**: Did the LLM identify the right function?
2. **Type match**: Is it the right kind of construct?
3. **Line match**: Is the focus in the expected area?

Name matching is the primary validation metric.
