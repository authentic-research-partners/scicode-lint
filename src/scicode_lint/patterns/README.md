# Patterns - Detection Pattern Definitions

Pattern specifications and test data for scicode-lint.

## Core Principle: Design for Thinking Models

**scicode-lint uses a small local thinking model (Qwen3, fits in 16GB VRAM).** This is the middle ground between grep-style matching and expensive cloud APIs.

Detection questions should leverage the model's reasoning:
- **Explain the WHY** - Describe why the bug matters, not just what syntax to find
- **Trust the reasoning** - Don't add "LITERALLY" or "do not assume" directives
- **Binary answer** - End with clear YES/NO conditions

## Runtime Context: System Prompt

Detection questions don't run in isolation. They run within a system prompt that frames how the LLM approaches the task. Understanding this context helps write better detection questions.

**Source:** `src/scicode_lint/detectors/prompts.py`

### What the System Prompt Tells the LLM

**1. Scientific correctness framing:**
> "The question examines scientific correctness - the perspective of a domain researcher checking if analysis code produces valid, reproducible results."

Your detection question should align with this framing. The LLM is primed to think like a scientist checking research code, not a general code reviewer.

**2. Narrow focus instruction:**
> "This is NOT a general code review. Do not look for: Style issues, Performance problems, General bugs or errors, Other scientific issues beyond the specific question."

The LLM is explicitly told to ignore everything except your specific question. This means your question must be self-contained - don't assume the LLM will notice related issues.

**3. Analysis approach (before answering):**
> 1. First understand the overall code structure and what it's trying to accomplish
> 2. Identify the intent and purpose of key operations
> 3. Trace the flow of data and operations relevant to the detection question
> 4. THEN answer the specific detection question

The LLM understands context first, then answers. Your question can reference concepts like "training loop" or "preprocessing pipeline" - the LLM will identify these structures before evaluating.

**4. YES/NO answer rules:**
> "Read the detection question - it defines what YES and NO mean. YES typically means bug found, NO means code is correct."

Your YES/NO conditions at the end of the question are the definitive criteria. Make them unambiguous.

**5. Confidence scale:**
| Score | Meaning |
|-------|---------|
| 0.95-1.0 | Issue definitely present with clear evidence |
| 0.85-0.95 | Very likely an issue based on pattern matching |
| 0.7-0.85 | Probable bug but context might justify it |
| <0.7 | Uncertain - low confidence |

The `min_confidence` field in test cases should align with this scale.

**6. Few-shot examples:**
The system prompt includes examples showing the expected reasoning pattern. These set expectations for concise, evidence-based responses.

### Implications for Detection Questions

| System Prompt Says | So Your Question Should |
|-------------------|------------------------|
| "Scientific correctness perspective" | Frame bugs in terms of research validity, not code style |
| "Stay narrowly focused" | Be self-contained - include all context needed |
| "Understand structure first" | Reference high-level concepts (LLM will find them) |
| "YES/NO conditions are definitive" | Make YES/NO conditions unambiguous |
| "Context-dependent is valid" | Allow edge cases where both answers are reasonable |

### Example: Question Aligned with System Context

```toml
question = """
Analyze data preprocessing in this code.

Data leakage occurs when statistics (mean, std, min, max) are computed
on the full dataset before splitting. This leaks test set information
into the training process, causing overly optimistic evaluation metrics.

Correct code computes statistics on training data only, after the split.

Does this code have data leakage from preprocessing?

Look for:
- Scaler/normalizer fit on combined train+test data
- Statistics computed before train_test_split()

YES = Bug found: preprocessing uses test data information
NO = Correct: statistics computed on training data only
"""
```

**Why this works with the system context:**
- "overly optimistic evaluation metrics" → scientific correctness framing
- "Analyze data preprocessing" → LLM will identify preprocessing steps first
- "Look for" section → specific symptoms to check
- Clear YES/NO → unambiguous decision criteria

## Overview

This directory contains the **definitions** of what issues to detect and **test data** to validate detection quality.

**Not to be confused with:**
- `evals/` - Evaluation frameworks (code that runs quality checks)
- `tests/` - Unit tests (infrastructure validation)
- `benchmarks/` - Performance measurements

## Directory Structure

```
patterns/
├── ai-training/           # ML training correctness patterns
│   ├── ml-001-scaler-leakage/
│   │   ├── pattern.toml               # Complete pattern definition (detection + tests)
│   │   ├── test_positive/             # Code WITH issues (must detect)
│   │   │   └── scaler_before_split.py
│   │   ├── test_negative/             # Code WITHOUT issues (must NOT detect)
│   │   │   └── scaler_after_split.py
│   │   └── test_context_dependent/    # Ambiguous edge cases
│   │       └── scaler_on_train_val.py
│   ├── ml-002-target-leakage/
│   └── ...
├── ai-inference/          # Inference correctness patterns
├── scientific-numerical/  # Numerical precision patterns
├── scientific-performance/ # Performance patterns
└── scientific-reproducibility/ # Reproducibility patterns

Total: 66 patterns
```

## Pattern ID Format

Pattern directories use long-form IDs: `{prefix}-{number}-{descriptive-name}` (e.g., `ml-001-scaler-leakage`).

**Canonical form (long):** `ml-001-scaler-leakage` - used in directory names and pattern.toml
**Short form:** `ml-001` - works for convenience in CLI and scripts

The short form works because tools match by substring (e.g., `ml-001` matches `ml-001-scaler-leakage`).

**Examples:**
```bash
# Both work:
python pattern_verification/semantic/semantic_validate.py ml-001-scaler-leakage  # canonical
python pattern_verification/semantic/semantic_validate.py ml-001                  # short form

# Both work:
scicode-lint lint myfile.py --pattern ml-001-scaler-leakage  # canonical
scicode-lint lint myfile.py --pattern ml-001                  # short form
```

## File Roles

Each pattern directory contains exactly 2 things:

### 1. `pattern.toml` - Pattern Definition & Test Cases (Required)

**Purpose**: Complete pattern specification - both detection rules and test cases

**What it contains**:
- **[meta]** - Pattern ID, name, category, severity
- **[detection]** - Detection question (what the LLM looks for), warning message, explanation
- **[tests]** - Test case definitions (positive, negative, context_dependent)
- **references** - URLs to official documentation (optional but recommended)

```toml
[meta]
id = "ml-001"
name = "scaler-leakage"
category = "ai-training"
severity = "critical"

[detection]
question = """
Analyze data preprocessing in this code.

Data leakage occurs when statistics (mean, std, min, max) are computed
on the full dataset before splitting. This leaks test set information
into the training process, causing overly optimistic evaluation metrics.

Correct code computes statistics on training data only, after the split.

Does this code have data leakage from preprocessing?

Look for:
- Scaler/normalizer fit on combined train+test data
- Statistics computed before train_test_split()

YES = Bug found: preprocessing uses test data information
NO = Correct: statistics computed on training data only
"""
warning_message = "Data leakage: scaler is fit on full data including test set..."

[[tests.positive]]
file = "test_positive/scaler_before_split.py"
description = "Scaler fit on combined train+test data"
expected_issue = "Data leakage: scaler statistics include test set"
min_confidence = 0.85

[tests.positive.expected_location]
type = "function"
name = "normalize_with_test_stats"
snippet = "all_data = torch.cat"
lines = [6, 7, 8]  # Lines where leakage occurs

[[tests.negative]]
file = "test_negative/scaler_after_split.py"
description = "Correct scaler usage after split"

# Optional: Official documentation URLs (can have multiple)
references = [
    "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
    "https://scikit-learn.org/stable/modules/cross_validation.html"
]
```

**References field** (recommended):
- URLs to official documentation that explain the **concept** behind the bug
- **Limit: 5 URLs max** - prioritize the most relevant

**Choosing good reference URLs:**

1. **Page must focus on this specific issue** - not just mention it
   - Good: "Why you need model.eval() for inference"
   - Good: API docs for `torch.inference_mode()`
   - Bad: "PyTorch basics" tutorial that mentions eval() once

2. **Explains the consequence** - shows what goes wrong without the fix

3. **Keep it focused** - if page is huge, find more specific page or use anchor link

4. **Any page type works** (API docs, tutorials, GitHub issues) as long as it's focused

**Bad references** are thin or unfocused:
- `torch.sigmoid.html` → just says "Alias for expit()" - useless
- `numpy.mean.html` → just function signature - doesn't explain pitfalls
- Giant tutorial page that briefly mentions the issue

Cached locally for semantic review (see `pattern_verification/deterministic/doc_cache/`)

**What each component sees**:
- **Linter** sees: `[meta]` and `[detection]` sections only
- **Evaluator** sees: `[tests]` section only

### 2. Test Code Files (Required)


**Purpose**: Pure Python code to test detection on (NO hints, NO documentation)

**⚠️ CRITICAL - NO INTENT HINTS IN TEST FILES:**

Test files must contain **ONLY code** - no comments, no hint docstrings, no revealing names.

**Absolutely forbidden**:
- **ANY `#` comments** - ALL comments are stripped before LLM evaluation
- **Docstrings with hints** - words like "bug", "issue", "incorrect", "wrong", "broken"
- **Naming that reveals intent**:
  - Functions: `buggy_function`, `correct_approach`, `leaky_scaler`
  - Classes: `BrokenModel`, `CorrectImplementation`
  - Variables: `bad_data`, `fixed_result`
- **ANY text that tells the LLM what the answer is**

**Why no comments (even though they're stripped)?**
1. **Consistency** - Developers see exactly what LLM sees when reviewing test files
2. **No fossilized interpretations** - Old/wrong comments don't mislead pattern maintainers
3. **Defense in depth** - If stripping fails (syntax error), no comments leak through
4. **Clean test data** - Test files are pure code samples, no annotations

**Why no docstrings/names with hints?**
1. Docstrings and names are NOT stripped - they're part of the code structure
2. If test files contain hints, the LLM sees the answer
3. This invalidates evaluation - we're testing detection ability, not reading comprehension
4. The deterministic validator (`validate.py`) enforces these rules

All context belongs in `pattern.toml` (detection question and test descriptions), which the linter partially sees (only [detection] section, not [tests]).

#### Positive Tests (Code WITH Issues)

```python
import torch


def normalize_with_test_stats(train_data, test_data):
    all_data = torch.cat([train_data, test_data])
    mean = all_data.mean()
    std = all_data.std()

    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std

    return train_normalized, test_normalized
```

**Requirements:**
- Pure code, no documentation
- Realistic, runnable code
- Contains the issue pattern
- 10-50 lines typical

#### Negative Tests (Correct Code)

```python
def normalize_correctly(train_data, test_data):
    mean = train_data.mean()
    std = train_data.std()

    train_normalized = (train_data - mean) / std
    test_normalized = (test_data - mean) / std

    return train_normalized, test_normalized
```

**Requirements:**
- Pure code, no documentation
- Similar structure to positive (tests pattern detection, not code quality)
- Does NOT contain the issue pattern
- 2-3 different correct implementations

#### Context-Dependent Tests (Ambiguous Cases)

```python
import torch


def fit_on_train_and_val(train_data, val_data, test_data):
    train_val_combined = torch.cat([train_data, val_data])
    mean = train_val_combined.mean()
    std = train_val_combined.std()

    test_normalized = (test_data - mean) / std
    return test_normalized
```

**Requirements:**
- Pure code, no documentation
- Edge case where detection is debatable
- Both "yes" and "no" can be acceptable

## Why No Documentation in Test Files?

**Problem**: If test files have comments like `# BUG: this causes data leakage`, the LLM sees the answer in the code itself. This is data leakage in our evaluation.

**Solution**: Test files are pure code. All context is in pattern.toml:
- **Linter sees**: Code file + `pattern.toml` [detection] section
- **Evaluator sees**: Code file + `pattern.toml` [tests] section
- **Test files see**: Nothing (just pure code)

## Pattern Categories

### ai-training (19 patterns)
Critical ML training correctness issues:
- Data leakage (scaler, target, temporal)
- PyTorch training modes
- Gradient management
- Loss function selection

### ai-inference (12 patterns)
Model inference correctness:
- Missing eval mode / inference_mode
- Missing no_grad context
- Device mismatches
- CUDA timing without sync
- Half precision on CPU
- JIT tracing control flow
- ONNX export without dynamic axes
- CuDNN benchmark with variable shapes
- Softmax before cross entropy
- Benchmark without warmup

### scientific-numerical (10 patterns)
Numerical precision and stability:
- Float equality
- NumPy array mutations
- Catastrophic cancellation
- Division by zero

### scientific-performance (11 patterns)
Performance anti-patterns:
- Python loops over NumPy
- Threading on CPU-bound work
- Missing vectorization
- Memory inefficiency

### scientific-reproducibility (14 patterns)
Reproducibility issues:
- Missing random seeds
- CUDA non-determinism
- Hardcoded hyperparameters
- Unsorted file iteration
- DataLoader missing worker_init_fn
- Unstable sort with ties
- Pandas sample without random_state
- CV splitter without random_state
- Naive datetime without timezone
- Set iteration order dependency
- Model pickle without version
- RandomState instance reuse

## Creating New Patterns

### 1. Create Directory Structure

```bash
mkdir -p patterns/ai-training/ml-999-my-pattern/{test_positive,test_negative,test_context_dependent}
```

### 2. Create `pattern.toml`

**Detection question design:**
- Narrow enough for one specific bug type
- General enough to catch syntactic variations

Use this template:

```toml
[meta]
id = "ml-999"
name = "my-pattern"
category = "ai-training"
severity = "critical"
description = "Brief description of the bug"
explanation = "Why this matters and how to fix it"
tags = ["relevant", "tags"]

[detection]
question = """
Analyze [concept/structure] in this code.

[1-2 sentences: What the bug is and why it matters]

[What correct code looks like]

Does this code have [the bug]?

Look for:
- [Specific symptom 1]
- [Specific symptom 2]

YES = Bug found: [condition]
NO = Correct: [condition]
"""
warning_message = "Brief explanation of why this is an issue and how to fix it."
```

**Example detection question:**

```toml
question = """
Analyze the PyTorch training loop in this code.

PyTorch models start in training mode by default, but after calling
model.eval() for validation, you must call model.train() to re-enable
dropout and batch norm training behavior.

Correct training code calls model.train() before or inside the training loop.

Does this code have missing train mode?

Look for:
- Training loop (backward + optimizer.step) without model.train() call
- Training resuming after model.eval() without model.train()

YES = Bug found: training code runs without explicit model.train() call
NO = Correct: model.train() is called before training begins
"""
```

```toml
[[tests.positive]]
file = "test_positive/example.py"
description = "Description of the issue in this file"
expected_issue = "What the issue is"
min_confidence = 0.85

[tests.positive.expected_location]
type = "function"           # "function", "class", "method", or "module"
name = "train_model"        # Name of the function/class/method (or "module" for module-level)
snippet = "optimizer.step"  # Code snippet that identifies the bug location
lines = [15, 16, 17]        # REQUIRED: Line numbers where the bug occurs

[[tests.negative]]
file = "test_negative/correct.py"
description = "Correct code that should NOT trigger"

[[tests.context_dependent]]
file = "test_context_dependent/edge_case.py"
description = "Ambiguous case"
allow_detection = true
allow_skip = true

[tests.context_dependent.expected_location]
type = "function"
name = "process_data"
snippet = "scaler.fit"
lines = [8, 9]              # REQUIRED for context-dependent tests too
```

**`expected_location` fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Location type: "function", "class", "method", or "module" |
| `name` | Yes | Name of the containing function/class/method |
| `snippet` | Yes | Code snippet that identifies the bug |
| `lines` | **Yes (positive & context-dependent)** | Line numbers where issue occurs |

**How location validation works:**

Evals use **name-based matching** as the primary validation:
1. **Type match**: `type` must match exactly (function, class, method, or module)
2. **Name match**: Function/class/method name must match (exact or partial)
3. **Snippet match**: Code snippet similarity (Jaccard threshold)

The `lines` field serves as:
- **Focus line eval metric**: Checks whether the linter's `focus_line` lands on one of the expected buggy lines
- **Deterministic validation**: Snippet must exist at these lines (checked by `validate.py`)
- **Debugging aid**: When evals fail, shows expected vs detected lines

**How to write `lines`:**
- List **all lines** that are part of the bug, not just key lines
- For contiguous bug spans, include every line (e.g., `[10, 11, 12, 13]` not `[10, 13]`)
- For non-contiguous buggy spots in the same function (e.g., two separate calls), list each spot's lines
- Only list lines within the named function — never across function boundaries
- The linter reports one `focus_line`; evals check if it falls within these expected lines

**Note:** The linter's LLM outputs a name-based location (function name + approximate line), which is then resolved to exact lines via AST. See [DETECTION_ARCHITECTURE.md](../../../docs/dev/DETECTION_ARCHITECTURE.md) for details.

Negative tests should NOT have `expected_location` (no bug to locate).

**One finding per file:**

Each pattern produces at most **ONE finding per file**. If the same bug appears multiple times, only the clearest instance is reported. Users fix one issue, re-run, and catch the next.

### 3. Create Test Files

Write pure Python code (no docstrings, no comments):
- `test_positive/*.py` - 3 files with the issue (required)
- `test_negative/*.py` - 3 files without the issue (required)
- `test_context_dependent/*.py` - 1 file with edge case (optional)

**⚠️ DIVERSITY REQUIREMENT:**

Test files must be **structurally diverse** - not similar copies with minor changes:

- **Within positive tests**: Each should demonstrate the bug in a different context (different function signatures, class vs function, different libraries, different surrounding code)
- **Within negative tests**: Each should show correct code using different approaches (not just the same fix repeated with different variable names)
- **Between positive and negative**: Avoid making negatives just "fixed versions" of positives - use different code structures entirely

**Why**: Similar test files don't challenge the model. If all 3 positive tests look alike with only variable names changed, the model learns shallow pattern matching instead of understanding the concept.

**Good diversity example** (scaler leakage):
```
test_positive/
  scaler_before_split.py      # StandardScaler in a function
  pipeline_combined_data.py   # MinMaxScaler on concatenated arrays in a class
  robust_scaler_leakage.py    # RobustScaler with QuantileTransformer

test_negative/
  scaler_after_split.py       # Fit on train, transform both
  sklearn_pipeline.py         # Using Pipeline with proper column transformer
  manual_normalization.py     # Computing stats manually on train only
```

### 4. Validate

```bash
# Sync test files with pattern.toml (run first!)
python pattern_verification/deterministic/validate.py --fix

# Semantic review (catches consistency issues before eval)
python pattern_verification/semantic/semantic_validate.py ml-999-my-pattern

# Run evaluation (reads test cases from pattern.toml [tests] section)
python evals/run_eval.py --pattern ml-999-my-pattern
```

**Why run validate.py?** Ensures every test file on disk has a corresponding entry in pattern.toml. The `--fix` flag auto-generates stub entries.

**Why run semantic_validate.py?** The script checks things evals can't:
- Detection question clarity and focus
- Test file diversity (not just copies with different names)
- Test files match their `[tests]` metadata descriptions
- No intent hints in test code (comments, docstrings, naming)

## Two Evaluation Modes

### 1. Simple Rule-Based (`run_eval.py`)
- Fast: Just checks "did linter detect anything?"
- Good for: Quick iteration, CI/CD

### 2. LLM-as-a-Judge (`run_eval.py`)
- Thorough: Semantic evaluation of correctness
- Checks: Are lines correct? Is reasoning sound?
- Good for: Deep quality assessment

## File Separation Summary

**pattern.toml has two audiences:**
- **[meta] + [detection]** sections → Read by the linter at runtime
- **[tests]** section → Read by the evaluator only, linter never sees this

This provides logical separation while keeping everything in one file for simplicity.

## Validation Tools

### `pattern_verification/deterministic/validate.py` - Comprehensive Validation

Runs 18 deterministic quality checks on pattern definitions:

```bash
# Check all patterns
python pattern_verification/deterministic/validate.py

# Auto-fix what's possible
python pattern_verification/deterministic/validate.py --fix

# Check specific pattern
python pattern_verification/deterministic/validate.py ml-002

# Fail on warnings too
python pattern_verification/deterministic/validate.py --strict
```

**Checks performed:**

1. **TOML/file sync** - Every test file has TOML entry and vice versa
2. **Schema validation** - pattern.toml matches Pydantic model
3. **Intent hints** - No docstrings/names revealing answers (`buggy_function`, etc.)
4. **Test file count** - Minimum 3 positive, 3 negative (warning)
5. **TODO markers** - No unfinished placeholders in TOML
6. **Detection question format** - Ends with YES/NO conditions
7. **Test file syntax** - All .py files are valid Python
8. **Empty fields** - Required fields have content
9. **Test file diversity** - Detect copy-paste (AST similarity)
10. **Snippet at lines** - Snippet must exist at specified `expected_location.lines`
11. **Expected lines** - Positive/context-dependent tests must have `expected_location.lines`
12. **Expected name** - `expected_location.name` must exist in test file (AST check)
13. **No comments** - All `#` comments forbidden in test files

**Auto-fix capabilities:**
- **Add**: Creates TOML entries for test files on disk that aren't referenced
- **Remove**: Deletes TOML entries for files that don't exist
- **Rename**: Auto-corrects filename typos if a close match exists

### Semantic Validation

Deep review using LLM reasoning (catches issues deterministic checks miss):

```bash
# Single pattern
python pattern_verification/semantic/semantic_validate.py ml-001-scaler-leakage

# All patterns
python pattern_verification/semantic/semantic_validate.py --all
```

**What it checks:**
- Description/expected_issue consistency with actual test file content
- Detection question clarity and focus
- Test file relevance to the bug being detected
- Snippet existence in test files

**📖 Full verification guide:** [pattern_verification/README.md](../../../pattern_verification/README.md)

## See Also

- [pattern_verification/](../../../pattern_verification/) - Complete verification guide (deterministic + semantic)
- [evals/](../../../evals/) - Evaluation frameworks
- [CONTRIBUTING.md](../../../CONTRIBUTING.md) - How to contribute patterns
- [ARCHITECTURE.md](../../../docs/dev/ARCHITECTURE.md) - Design principles
