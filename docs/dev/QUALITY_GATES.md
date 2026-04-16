# Quality Gates & Validation Forces

This document explains the different validation layers, what each checks, and the tensions between them.

**Related:** [CONTINUOUS_IMPROVEMENT.md](CONTINUOUS_IMPROVEMENT.md) for the workflow.

---

## Overview

Pattern development has multiple validation layers that catch different types of issues:

```
┌─────────────────────────────────────────────────────────────┐
│                    Pattern Creation                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. DETERMINISTIC CHECKS (free, fast)                        │
│     Structure, syntax, file sync, data leakage               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. DIVERSITY CHECK (Claude CLI)                               │
│     Redundant tests, non-diverse negatives                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. SEMANTIC VALIDATION (Claude CLI)                         │
│     Docs ↔ tests ↔ question consistency                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. PATTERN EVALS (vLLM)                                     │
│     Detection accuracy on pattern test files                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. INTEGRATION TESTS (vLLM + Claude)                        │
│     Generalization to realistic multi-pattern code           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. REAL-WORLD VALIDATION (vLLM + Claude)                    │
│     External validation on scientific ML papers              │
└─────────────────────────────────────────────────────────────┘
```

Each layer filters different problems. Passing earlier checks doesn't guarantee passing later ones.

---

## Layer 1: Deterministic Checks

**Location:** `pattern_verification/deterministic/validate.py`

**Cost:** Free, fast (~seconds)

**13 checks:**

| Check | Severity | What It Catches |
|-------|----------|-----------------|
| TOML/file sync | error | Missing or orphaned test files |
| Schema validation | error | Invalid `pattern.toml` structure |
| Data leakage | error | Hint comments (`# BUG:`, `# CORRECT:`) in tests |
| Test file count | warning | < 3 positive or negative tests |
| TODO markers | error | Unfinished placeholders |
| Detection question format | warning | Missing YES/NO conditions |
| Test file syntax | error | Python parse errors |
| Empty fields | error | Required fields without content |
| Category mismatch | error | `meta.category` doesn't match directory |
| Snippet verification | error | Snippet not in file or not at specified lines |
| Expected lines | error | Positive/context-dependent tests missing `expected_location.lines` |
| Related patterns exist | error | Invalid `related_patterns` references |
| Test diversity | warning | Copy-paste tests (AST similarity) |
| Reference URL validation | warning | Unreachable URLs |

**Auto-fix:** `--fix` flag can repair TOML sync issues.

---

## Layer 2: Diversity Check

**Location:** `pattern_verification/deterministic/diversity_check.py`

**Cost:** Claude CLI tokens (Sonnet with low effort, one call per pattern via `dev_lib.ClaudeCLI`)

**What it checks:**

| Check | What It Catches |
|-------|-----------------|
| Redundant positive pairs | Two positive tests that demonstrate the same concept |
| Redundant negative pairs | Two negative tests that are too similar |
| Non-diverse negatives | Negative test is just positive with bug removed (same structure) |

**Why this matters:** Redundant tests waste evaluation time and don't improve detector robustness. Diverse tests catch edge cases. The AST-based similarity check (Layer 1) only catches copy-paste; this catches semantic similarity.

---

## Layer 3: Semantic Validation

**Location:** `pattern_verification/semantic/semantic_validate.py`

**Cost:** Token cost (Claude CLI via `dev_lib.ClaudeCLI`)

**What it checks:**

- Description accurately describes test file code
- `expected_issue` aligns with detection question's YES condition
- Snippets in `expected_location` exist in test files
- Positive tests contain the described bug
- Negative tests avoid the bug
- Pattern aligns with official documentation (if cached)

**Why separate from deterministic:** These require understanding code semantics, not just structure.

---

## Layer 4: Pattern Evals

**Location:** `evals/run_eval.py`

**Cost:** vLLM calls (local)

**Two modes:**

| Mode | What It Measures | Use Case |
|------|------------------|----------|
| Direct metrics | Name-based location matching | Fast regression testing |
| LLM judge | Semantic correctness | Comprehensive evaluation |

**Metrics:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 Score
- Critical severity precision ≥ 0.95 threshold
- **Name match** = detected function/class name matches expected (primary validation metric)

**Location validation:** LLM outputs function/class name + `near_line` hint. AST resolves to actual lines.
Evals validate by comparing detected name vs `expected_location.name` in pattern.toml.

**Why separate from semantic validation:** Semantic validation checks internal consistency. Evals check whether the pattern actually **detects** bugs correctly at runtime.

---

## Layer 5: Integration Tests

**Location:** `evals/integration/`

**Cost:** vLLM + Claude calls

**Pipeline:** Generate (Sonnet) → Verify (Sonnet) → Lint (vLLM) → Judge (Sonnet)

```bash
# Full pipeline
python evals/integration/integration_eval.py --generate-count 10

# Save with ID for regression
python evals/integration/integration_eval.py --generate-count 10 --save --id baseline_v1

# Re-evaluate saved run
python evals/integration/integration_eval.py --id baseline_v1
```

**Why integration tests exist:** Pattern tests are narrow by design. Integration tests catch:
- Overfitting to specific test file patterns
- Cross-pattern interactions
- Behavior on realistic multi-bug code

---

## Layer 6: Real-World Validation

**Location:** `real_world_demo/`

**Cost:** vLLM + Claude calls

**What it does:**
- Collects Python files from scientific ML papers (via PapersWithCode)
- Runs scicode-lint on real research code
- Claude verifies whether findings are real issues or false positives

**Metrics:**
- Precision on real-world code
- Finding rate by scientific domain
- Category distribution of issues found

**Why separate from integration tests:** Integration tests use controlled scenarios (synthetic or LLM-generated). Real-world validation uses actual research code from published papers - the ultimate test of whether patterns catch bugs scientists actually make.

See [real_world_demo/README.md](../../real_world_demo/README.md) for pipeline details.

---

## Tensions Between Forces

These forces pull in different directions. Understanding the tensions helps when fixing issues.

### Speed vs Depth

| Force | Tradeoff |
|-------|----------|
| Deterministic checks | Fast, free, but can't verify semantics |
| Semantic validation | Thorough, but costs tokens and time |

**Resolution:** Run deterministic first (catch cheap errors early), semantic only when structure is valid.

### Strictness vs Flexibility

| Force | Tradeoff |
|-------|----------|
| Direct metrics | Exact location match required |
| LLM judge | Semantic match accepted |

**Resolution:** Use direct metrics for regression, LLM judge for comprehensive evaluation.

### Test Clarity vs Data Leakage

| Force | Tradeoff |
|-------|----------|
| Clear test descriptions | Help understand what test validates |
| No hints in test files | Prevent leaking ground truth to linter |

**Resolution:** Put descriptions in `pattern.toml`, keep test files as pure code.

### Precision vs Recall

| Force | Tradeoff |
|-------|----------|
| Minimize false positives | A noisy tool gets ignored |
| Catch all bugs | Missing bugs defeats the purpose |

**Resolution:** Project philosophy prioritizes precision. Better to miss some bugs than flood users with false positives.

### Simple Prompts vs Accuracy

| Force | Tradeoff |
|-------|----------|
| Local LLM constraints | Small model needs simple questions |
| Detection accuracy | Complex bugs need detailed context |

**Resolution:** Include "why it matters" in detection questions. Qwen3 is a thinking model - context helps reasoning without making questions complex.

### Doc Grounding vs Focus

| Force | Tradeoff |
|-------|----------|
| Reference URL validation | Ensures patterns cite real docs |
| Cached docs may be verbose | Lots of boilerplate in official docs |

**Resolution:** Two-stage cache cleaning (HTML stripping + AI boilerplate removal).

### Detection Question vs System Prompt

| Force | Tradeoff |
|-------|----------|
| Detection question flexibility | Pattern authors write questions their way |
| System prompt framing | LLM is primed for "scientific correctness" perspective |

**Resolution:** Document the system prompt context in `patterns/README.md` ("Runtime Context: System Prompt" section). Pattern-reviewer checks alignment.

**Key system prompt constraints:**
- LLM is told: "scientific correctness perspective" - questions should frame bugs as research validity issues
- LLM is told: "do NOT look for style/performance/general bugs" - questions must be self-contained
- LLM is told: "understand structure first, then answer" - questions can reference high-level concepts

### Pattern Development Effort vs Generalization Confidence

| Force | Tradeoff |
|-------|----------|
| Minimal tests | Faster pattern development |
| Diverse, uncorrelated tests | Proves detection question captures the concept |

**Resolution:** Require ≥3 positive and ≥3 negative tests that are diverse (different code structures, contexts, variable names) - not copy-paste variations. If a detection question works on multiple uncorrelated examples, it captures the underlying bug concept rather than overfitting to surface features. This is the same principle as ML generalization: diverse training data prevents overfitting.

---

## What Each Layer Catches (Examples)

| Issue | Caught By |
|-------|-----------|
| Missing test file entry in TOML | Deterministic |
| `# BUG: this is wrong` in test file | Deterministic (data leakage) |
| Positive/context-dependent test missing `expected_location.lines` | Deterministic |
| Two positive tests are conceptually identical | Diversity check |
| Negative test is just positive with bug removed | Diversity check |
| Description says X but test does Y | Semantic validation |
| Expected_location snippet not in test | Deterministic (but semantic validates alignment) |
| Detection question is ambiguous | Semantic validation |
| Question misaligned with system prompt framing | Semantic validation (pattern-reviewer) |
| Pattern detects wrong location | Pattern evals |
| Detected function/class doesn't match expected | Pattern evals (name matching) |
| Pattern works on tests but fails on real code | Integration tests |
| Pattern conflicts with another pattern | Integration tests |
| False positives on real research code | Real-world validation |
| Pattern detects unrealistic bugs only | Real-world validation |

---

## Failure Patterns

| Symptom | Likely Cause | Check First |
|---------|--------------|-------------|
| Deterministic fails, semantic passes | Impossible - deterministic runs first | - |
| Diversity check finds redundant pairs | Test files too similar | Delete one or rewrite with different approach |
| Diversity check finds non-diverse negatives | Negatives are just fixed copies of positives | Rewrite negatives with different code structure |
| Semantic passes, evals fail | Detection question unclear or tests too easy | Review detection question phrasing |
| Evals pass, integration fails | Overfitting to test patterns | Check test diversity, add varied tests |
| High precision, low recall | Detection question too narrow | Broaden YES conditions |
| Low precision, high recall | Detection question too broad | Narrow YES conditions or add NO exceptions |
| LLM ignores question, flags style issues | Question not self-contained | Check system prompt alignment (see `patterns/README.md`) |
| Integration passes, real-world fails | Tests don't reflect actual code | Review real-world findings, add similar tests |

---

## Summary Table

| Layer | Type | Location | Catches | Cost |
|-------|------|----------|---------|------|
| 1. Deterministic | Automated | `pattern_verification/deterministic/validate.py` | Structure, syntax, leakage | Free |
| 2. Diversity | LLM | `pattern_verification/deterministic/diversity_check.py` | Redundant tests, non-diverse negatives | Claude tokens |
| 3. Semantic | LLM review | `pattern_verification/semantic/` | Consistency, alignment | Claude tokens |
| 4. Pattern evals | Automated + LLM | `evals/run_eval.py` | Detection accuracy | vLLM calls |
| 5. Integration | Automated + LLM | `evals/integration/` | Generalization | vLLM + Claude |
| 6. Real-world | Automated + LLM | `real_world_demo/` | Precision on actual papers | vLLM + Claude |
| Unit tests | Automated | `tests/` | Framework correctness | Free |
| Style/type checks | Manual | `ruff check && ruff format && mypy .` | Code style, types | Free |
