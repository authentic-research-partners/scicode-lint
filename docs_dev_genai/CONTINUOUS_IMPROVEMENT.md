# Continuous Improvement Loop

**Key principle:** Don't fine-tune to specific tests. Extract generalizable principles and apply to ALL patterns.

**Requires:** vLLM server running (runs actual detection against test files)

## Quick Reference

```bash
# 0. Refresh reference docs and clean orphaned cache
python pattern_verification/deterministic/validate.py --fetch-refs --clean-cache

# Structural validation (no LLM needed)
python pattern_verification/deterministic/validate.py           # all patterns
python pattern_verification/deterministic/validate.py --fix     # auto-fix

# Diversity check (uses Claude CLI)
python pattern_verification/deterministic/diversity_check.py              # all patterns
python pattern_verification/deterministic/diversity_check.py ml-001       # single pattern
python pattern_verification/deterministic/diversity_check.py --category X # category

# Semantic validation (uses Claude CLI)
python pattern_verification/semantic/semantic_validate.py <id>  # single pattern
python pattern_verification/semantic/semantic_validate.py --all # all patterns

# Evals (uses vLLM)
python evals/run_eval.py                                        # all patterns
python evals/run_eval.py -p <pattern-name>                      # single pattern
python evals/run_eval.py -p <name1> -p <name2>                  # multiple patterns
```

## Per-Pattern Fix Cycle

When a pattern has issues, iterate on **that specific pattern** until it passes:

```mermaid
flowchart TD
    A[1. Deterministic: validate.py] --> B{Issues?}
    B -->|Yes| C[Fix: validate.py --fix]
    B -->|No| D[2. Diversity: diversity_check.py]
    C --> A
    D --> E{Issues?}
    E -->|Yes| F[Fix test files]
    E -->|No| G[3. Semantic: semantic_validate.py]
    F --> A
    G --> H{Issues?}
    H -->|Yes| I[Fix in Claude Code]
    H -->|No| J[4. Evals: run_eval.py -p NAME]
    I --> A
    J --> K{Pass?}
    K -->|Yes| L[Done - next pattern]
    K -->|No| M[5. Debug & Fix]
    M --> A
```

### Commands (for specific pattern)

```bash
# 0. Refresh reference docs and clean orphaned cache
python pattern_verification/deterministic/validate.py --fetch-refs --clean-cache

# 1. Deterministic validation
python pattern_verification/deterministic/validate.py
python pattern_verification/deterministic/validate.py --fix  # auto-fix if issues

# 2. Diversity check (uses Claude CLI)
python pattern_verification/deterministic/diversity_check.py pt-007
# If issues: redundant pairs → delete or diversify one file
#            non-diverse negatives → rewrite to use different approach

# 3. Semantic validation (uses pattern-reviewer agent - read-only)
python pattern_verification/semantic/semantic_validate.py pt-007
# If issues found → fix directly in Claude Code session before proceeding to evals

# 4. Evals (requires vLLM) - note: uses full pattern name
python evals/run_eval.py -p pt-007-inference-without-eval

# 5. If evals fail: debug and fix
python -m scicode_lint lint <test-file> --pattern pt-007 --verbose
# Fix issues directly in Claude Code session

# After ANY pattern changes, run deterministic validation immediately:
python pattern_verification/deterministic/validate.py pt-007

# Repeat from step 1 until all pass
```

**Eval validation:** Evals use **name-based matching** as the primary metric. The detected function/class name
must match `expected_location.name` in pattern.toml. LLM outputs names (not line numbers), AST resolves to lines.

**Focus line accuracy** is also tracked: does the linter's `focus_line` land on one of the expected buggy `lines` from pattern.toml? This is an observational metric (~42% baseline), not a gate — don't optimize for it at the expense of detection accuracy.

**Note:** `semantic_validate.py` uses the `pattern-reviewer` agent (read-only) to identify issues. Fix issues directly in your Claude Code session.

### Multiple Patterns

When fixing a group of patterns, use the same cycle:

```bash
# Validate all (or use --fix for auto-fixes)
python pattern_verification/deterministic/validate.py

# Semantic validation for specific patterns
python pattern_verification/semantic/semantic_validate.py pt-007 pt-013 ml-005

# Run evals on the group
python evals/run_eval.py -p pt-007-inference-without-eval \
                         -p pt-013-missing-inference-mode \
                         -p ml-005-cv-temporal-shuffle

# Fix any failures, repeat until all pass
```

## Category-by-Category Workflow (Recommended)

Work through categories sequentially - complete one fully before moving to next:

```mermaid
flowchart TD
    A[Pick category] --> B[1. Diversity: --category X]
    B --> C{Issues?}
    C -->|Yes| D[Fix test files]
    C -->|No| E[2. Semantic: --category X]
    D --> B
    E --> F{Issues?}
    F -->|Yes| G[Fix in Claude Code]
    F -->|No| H[3. Evals: -c X]
    G --> B
    H --> I{Pass?}
    I -->|Yes| J{More categories?}
    I -->|No| K[Per-pattern fix cycle]
    K --> B
    J -->|Yes| A
    J -->|No| L[Full evals + integration]
```

**Why this approach:**
- Smaller batches (12-15 patterns) = easier to track
- Issues in same category often share patterns
- Generalizable fixes apply to whole category at once
- Clear completion milestones

**Parallel execution (recommended):** Run 5 Claude Code terminals simultaneously, one per category. Categories are independent, so all can run in parallel:
```bash
# Terminal 1: claude (work on ai-inference)
# Terminal 2: claude (work on ai-training)
# Terminal 3: claude (work on scientific-numerical)
# Terminal 4: claude (work on scientific-performance)
# Terminal 5: claude (work on scientific-reproducibility)
```

**Important:** When a step finds issues, fix them before proceeding to the next step. Don't accumulate issues across steps.

```bash
# Categories in order
# ai-inference, ai-training, scientific-numerical, scientific-performance, scientific-reproducibility

# Refresh reference docs and clean orphaned cache
python pattern_verification/deterministic/validate.py --fetch-refs --clean-cache

# Example: complete ai-inference category
python pattern_verification/deterministic/validate.py
python pattern_verification/deterministic/diversity_check.py --category ai-inference  # ~2-3 min
# Fix diversity issues (redundant pairs, non-diverse negatives)

python pattern_verification/semantic/semantic_validate.py --category ai-inference
# Fix issues in Claude Code session...

python evals/run_eval.py -c ai-inference
# Fix any failures, repeat until category passes

# Move to next category
python pattern_verification/deterministic/diversity_check.py --category ai-training
python pattern_verification/semantic/semantic_validate.py --category ai-training
# ... continue
```

## Final Verification

After all categories pass individually, run full validation:

```bash
# Full validation
python pattern_verification/deterministic/validate.py
python pattern_verification/deterministic/diversity_check.py  # ~5-10 min
python pattern_verification/semantic/semantic_validate.py --all

# Full evals - get overall accuracy
python evals/run_eval.py
# Report: evals/reports/judge/llm_judge_report.md

# Update README.md with accuracy stats from report

# Integration tests (holdout)
python evals/integration/integration_eval.py --generate-count 10
```

## Generalizable Fixes

When fixing a pattern, look for principles that apply broadly:

1. **Found a generalizable principle?**
   - Add to `patterns/README.md`
   - Apply to ALL relevant patterns, not just the failing one

2. **Example generalizations:**
   - "Snippets should point to bug location, not class definition"
   - "NO criteria must cover all valid negative cases explicitly"
   - "Detection questions need clear YES/NO decision boundaries"

## Integration Tests (holdout set)

Tests generalization - did we overfit to pattern-specific tests?

```bash
# Full pipeline: Generate (Sonnet) → Verify (Sonnet) → Lint (vLLM) → Judge (Sonnet)
python evals/integration/integration_eval.py --generate-count 10

# Save with ID for regression
python evals/integration/integration_eval.py --generate-count 10 --save --id baseline_v1

# Re-evaluate saved run
python evals/integration/integration_eval.py --id baseline_v1
```

## Evaluation Types

| Type | Purpose |
|------|---------|
| Pattern-specific | Iterate on individual patterns |
| Integration | Holdout - test on fresh LLM-generated code |

## Diagnosing Non-Determinism

If a pattern produces inconsistent results across eval runs (passes sometimes, fails other times), this is a signal that the detection instructions are ambiguous. The model interprets unclear instructions differently depending on random state (temperature, sampling). Fix by making the detection question more precise — clear instructions produce consistent results regardless of randomness.

## Critical Constraints

- **No hints in test files** - pure code only, no comments about bugs (data leakage in evaluation)
- **No shell loops** - Do NOT use `for p in ...; do ... done` or similar bash loops to run commands on multiple patterns. Shell loops require per-iteration approval in Claude Code, blocking automatic execution. Instead, use the tool's built-in multi-pattern support (e.g., `validate.py pt-007 pt-013`, `run_eval.py -p name1 -p name2`, `--category X`)

---

## See Also

- **[META_IMPROVEMENT_LOOP.md](META_IMPROVEMENT_LOOP.md)** - Real-world validation using Papers with Code corpus + Sonnet verification (~3-4 hours, ~1.5M tokens)
