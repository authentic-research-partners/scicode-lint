# Evals - Quality Evaluation Framework

Quality validation for scicode-lint pattern detection.

## Two Evaluation Types (TL;DR)

| Type | What it tests | Test data | Use for |
|------|---------------|-----------|---------|
| **Pattern-Specific** | 1 pattern in isolation | Static files per pattern | Iteration on individual patterns |
| **Integration** | Multiple patterns together | LLM-generated scenarios | Holdout - test generalization |

```bash
# Pattern-specific
python evals/run_eval.py

# Integration (full pipeline: Generate → Verify → Lint → Judge)
python evals/integration/integration_eval.py --generate-count 10
```

---

## Why Two Types?

Pattern-specific tests are for iteration. Integration tests are the **holdout set** - they test if patterns generalize beyond the specific test files we tuned on.

| Evaluation | Role |
|------------|------|
| Pattern-specific | Iterate on individual patterns |
| **Integration** | Holdout - fresh LLM-generated code |

---

## Overview

This directory contains tools for evaluating whether pattern detections are **correct** and **helpful**.

**Not to be confused with:**
- `tests/` - Deterministic unit tests (mocked, fast)
- `benchmarks/` - Performance measurements (timing only)
- `patterns/` - Pattern definitions and test data

## Detailed Evaluation Types

### 1. Pattern-Specific Evaluations (this directory)

Test individual patterns in isolation using:
- **`run_eval.py`** - Comprehensive evaluation (LLM judge + direct metrics + alignment)
- **`run_eval.py --skip-judge`** - Fast deterministic evaluation (no judge LLM)

### 2. Integration Evaluations (`integration/`)

Full pipeline using LLM-generated scenarios:
- **`integration_eval.py`** - Generate → Verify → Lint → Judge

**Pipeline:**
1. **GENERATE (Sonnet)** - Select patterns, generate code with bugs
2. **VERIFY (Sonnet)** - Confirm manifest is accurate
3. **LINT (vLLM)** - Run scicode-lint on generated code
4. **JUDGE (Sonnet)** - Categorize findings

**Categories (Sonnet-judged):**
- TP-intended: Finding matches manifest bug
- TP-bonus: Verified extra bug (not in manifest)
- FP: Rejected (not a real bug)
- FN: Manifest bug missed

## What's in This Directory

### Core Evaluation Scripts

**`run_eval.py`** - Comprehensive evaluation (recommended) ✅
- Uses LLM to evaluate semantic correctness
- Also computes direct metrics and alignment in one pass
- Compares against `pattern.toml` descriptions (not docstrings - avoids data leakage)
- Usage: `python evals/run_eval.py --pattern ml-001-scaler-leakage`

**`run_eval.py --skip-judge`** - Fast deterministic evaluation
- Compares linter output against exact location/snippet expectations
- No judge LLM calls - fast and deterministic
- Good for regression testing
- Usage: `python evals/run_eval.py --skip-judge --pattern ml-001-scaler-leakage`

### Supporting Modules

**`judge_models.py`** - Pydantic models for judge verdicts
- `PatternJudgeMetrics` - Per-pattern evaluation results
- `OverallJudgeMetrics` - Aggregated metrics across all patterns
- `TestEvaluation` - Individual test case evaluation

**`prompts/judge_system_prompt.py`** - LLM judge system prompts
- Separate prompt templates for positive, negative, and context-dependent tests

### Configuration

**`test_definitions.yaml`** - Evaluation configuration
- Pattern registry
- Execution settings

## Running Evaluations

### Single Pattern

```bash
# Comprehensive evaluation (recommended)
python evals/run_eval.py --pattern ml-001-scaler-leakage

# Quick evaluation (no judge LLM)
python evals/run_eval.py --skip-judge --pattern ml-001-scaler-leakage
```

### All Patterns

```bash
# Comprehensive evaluation - all patterns
python evals/run_eval.py

# Quick evaluation - all patterns
python evals/run_eval.py --skip-judge

# Generate reports with custom output directory
python evals/run_eval.py --format all --output-dir evals/reports
```

### Via Pytest

```bash
# Run pattern evaluations as tests
pytest evals/test_all_patterns.py -v

# Run specific pattern
pytest evals/test_all_patterns.py -k ml-001
```

## Evaluation Types

### Hardcoded Ground Truth

**Test data location:** `patterns/{category}/{pattern}/`

Each pattern has test files with explicit expectations in `pattern.toml`:

```toml
[[tests.positive]]
file = "positive/bug_example.py"
expected_issue = "Data leakage from scaler"
min_confidence = 0.85

[tests.positive.expected_location]
type = "function"
name = "normalize_with_test_stats"
snippet = "all_data = torch.cat([train_data, test_data])"
```

**Validation:**
1. Run linter on test file
2. Check detection matches expected location
3. Check confidence ≥ threshold
4. Count TP, FP, FN
5. Calculate metrics

**Pros:**
- Fast and deterministic
- Exact location validation
- Good for regression testing

**Cons:**
- Brittle to prompt changes
- Can't evaluate explanation quality
- Requires exact matches

### LLM-as-Judge ✅

**Test data location:** Same as above, uses `pattern.toml` descriptions

Expected behavior is extracted from `pattern.toml` test entries (NOT from file docstrings - this avoids data leakage since the linter sees docstrings):

```toml
[[tests.positive]]
file = "test_positive/missing_zero_grad.py"
description = "Training loop that forgets optimizer.zero_grad()"
expected_issue = "Gradients accumulate incorrectly across batches"
```

**What the LLM judge receives:**
```
<TEST_CASE>
File: patterns/ai-training/pt-004/test_positive/missing_zero_grad.py
Type: positive
Expected behavior: Training loop that forgets optimizer.zero_grad()
                   Expected issue: Gradients accumulate incorrectly across batches
</TEST_CASE>

<LINTER_OUTPUT>
Detected: yes
Lines: line 42
Code snippet: loss.backward()  # no zero_grad before this
Linter's reasoning: [full reasoning from linter]
Confidence: 0.92
</LINTER_OUTPUT>
```

**Validation:**
1. Run linter on test file
2. Extract expected behavior from `pattern.toml` (NOT docstring)
3. Ask LLM judge: "Does output match expected?"
4. Judge returns: yes/no/partial
5. Calculate accuracy

**Pros:**
- Semantic correctness (not just exact matches)
- Can evaluate explanation quality
- More flexible with prompt changes

**Cons:**
- Non-deterministic
- Slower (requires extra LLM calls)
- Needs validation against ground truth

## Metrics

### Pattern-Level Metrics

For each pattern:
- **Precision** = TP / (TP + FP) - minimize false alarms
- **Recall** = TP / (TP + FN) - catch most bugs
- **F1 Score** = harmonic mean of precision and recall

### Overall Metrics

Aggregated across all patterns:
- Critical severity precision ≥ 0.95

### By Category

- ai-training
- ai-inference
- scientific-numerical
- scientific-performance
- scientific-reproducibility

## Data Flow Summary (No Leakage)

Understanding what each component sees is critical for evaluation integrity:

| Evaluation | Linter Sees | Judge/Validator Sees | Ground Truth Source |
|------------|-------------|---------------------|---------------------|
| **Direct Metrics** | Code only | Linter output + expected from TOML | `pattern.toml` |
| **LLM Judge** | Code only | Linter output + expected from TOML | `pattern.toml` |
| **Integration** | Generated code only | Code + manifest + detections | Generated manifest |

**Key principle:** The linter NEVER sees expected outputs or ground truth. It analyzes code blind.

## Alignment Metrics (Comparing Both Approaches)

The `run_eval.py` script computes both direct metrics and LLM judge verdicts in a single pass, calculating:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Semantic Alignment** | (both_pass + both_fail) / total | % where both approaches agree |
| **Quality Issue Rate** | quality_issues / total | % where direct passes but judge fails |
| **Ground Truth Strictness Rate** | overly_strict / total | % where direct fails but judge passes |

### Interpretation

| Direct | Judge | Category | Action |
|--------|-------|----------|--------|
| ✅ Pass | ✅ Yes | `both_pass` | Pattern working correctly |
| ❌ Fail | ❌ No | `both_fail` | Pattern has detection issues |
| ✅ Pass | ❌ No | `quality_issue` | Right location, wrong explanation - improve prompts |
| ❌ Fail | ✅ Yes | `overly_strict` | Ground truth too rigid - relax expectations |

### Usage

```bash
# Run evaluation with alignment metrics
python evals/run_eval.py --pattern ml-001-scaler-leakage

# Run all patterns - output includes divergent cases
python evals/run_eval.py
```

### Example Output

```
ALIGNMENT METRICS
----------------------------------------------------------------------
Semantic Alignment:           92.5%
  - Both Pass:                85 (70.8%)
  - Both Fail:                26 (21.7%)

DIVERGENCE METRICS
----------------------------------------------------------------------
Quality Issue Rate:           4.2% (5 cases)
  (Direct passes, Judge fails - right location, wrong explanation)

Ground Truth Strictness Rate: 3.3% (4 cases)
  (Direct fails, Judge passes - ground truth too rigid)
```

## Output Files

**`reports/`** directory contains:
- `evaluation_report.json` - Full metrics (machine-readable)
- `evaluation_report.md` - Summary report (human-readable)
- Pattern-specific reports

## Integration with Development

### Before Committing Pattern Changes

```bash
# 1. Validate pattern structure
python src/scicode_lint/tools/validate_pattern.py --pattern patterns/ai-training/ml-001

# 2. Run evaluations
python evals/run_eval.py --pattern ml-001

# 3. Check metrics meet thresholds
# - F1 ≥ 0.85
```

## See Also

- [integration/](integration/) - Multi-pattern integration tests
- [patterns/](../src/scicode_lint/patterns/) - Pattern definitions and test data
- [tests/](../tests/) - Deterministic unit tests
- [benchmarks/](../benchmarks/) - Performance benchmarks
- [ARCHITECTURE.md](../docs/dev/ARCHITECTURE.md) - System design

## Choosing the Right Evaluation

| Script | Use Case | Speed | Judge LLM |
|--------|----------|-------|-----------|
| `run_eval.py` | Development, quality assessment | Slower | Yes |
| `run_eval.py --skip-judge` | Regression testing | Fast | No |

*Note: Both scripts require vLLM for the linter. "Judge LLM" refers to the additional semantic evaluation.*

**Recommended workflow:**
1. Run `run_eval.py` during development (comprehensive feedback)
2. Review alignment metrics to identify quality issues or overly strict ground truth
3. Use `run_eval.py --skip-judge` for fast regression checks
4. Run integration eval before release: `python evals/integration/integration_eval.py --generate-count 10`
