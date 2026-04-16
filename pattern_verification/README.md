# Pattern Verification

Tools for verifying pattern quality. Two complementary approaches:

## Deterministic Checks (no LLM needed)

Fast, automated checks that catch structural issues:

```bash
# Check all patterns
python pattern_verification/deterministic/validate.py

# Check specific pattern
python pattern_verification/deterministic/validate.py ml-001-scaler-leakage

# Auto-fix what's possible
python pattern_verification/deterministic/validate.py --fix

# Fail on warnings too
python pattern_verification/deterministic/validate.py --strict
```

### Semantic Diversity Check (uses Claude CLI)

Detects conceptually similar test files **within each pattern** that AST hash check misses. Requires Claude CLI (`claude login`).

Uses Sonnet with low thinking effort — one Claude call per pattern (all pairs batched).

```bash
# Check all patterns (66 Claude calls)
python pattern_verification/deterministic/diversity_check.py

# Check specific pattern (1 Claude call)
python pattern_verification/deterministic/diversity_check.py ml-001

# Check category
python pattern_verification/deterministic/diversity_check.py --category ai-training

# Override model or parallelism
python pattern_verification/deterministic/diversity_check.py --model opus
```

**What it detects:**
- **Redundant pairs**: Two positive (or two negative) tests that demonstrate the same concept
- **Non-diverse negatives**: Negative test is just positive test with bug removed (same structure)

**Output example:**
```
[diversity] ai-training/ml-004-imbalanced-accuracy-metric

   Redundant positive pairs:
   - src/scicode_lint/patterns/.../test_positive/accuracy_on_imbalanced.py
     src/scicode_lint/patterns/.../test_positive/binary_accuracy.py

[diversity] ai-training/ml-007-test-set-preprocessing

   Non-diverse negatives (same code as positive, with bug removed):
   - src/scicode_lint/patterns/.../test_negative/transform_train_first.py
     (same structure as src/scicode_lint/patterns/.../test_positive/transform_test_first.py)
```

**Configuration:** `diversity_model` in `config.toml` under `[pattern_verification]`. Rate limiting: `[claude_cli]` section.

**18 checks performed (key checks shown):**

| Check | Type | Description |
|-------|------|-------------|
| TOML/file sync | error | Every test file has TOML entry and vice versa |
| Schema validation | error | pattern.toml matches Pydantic model |
| Intent hints | error | No docstrings/names revealing answers (`buggy_function`, etc.) |
| No comments | error | All `#` comments forbidden in test files |
| Test file count | warning | Minimum 3 positive, 3 negative recommended |
| TODO markers | error | No unfinished placeholders in TOML |
| Detection format | error | Question ends with YES/NO conditions |
| Test file syntax | error | All .py files are valid Python |
| Empty fields | error | Required fields have content |
| Test diversity | warning | Detect copy-paste via AST similarity |

**Auto-fix capabilities:**
- Add missing TOML entries for test files on disk
- Remove orphaned TOML entries (files that don't exist)
- Rename entries when close filename match exists

## Semantic Checks (uses LLM)

Deep review that catches consistency issues scripts can't detect.

**Requires:** Claude CLI (`claude login`) with Claude Code subscription. No API keys needed. Uses `dev_lib.ClaudeCLI` wrapper.

### ⚠️ COST WARNING

**THIS USES CLAUDE CODE AGENTS AND CONSUMES YOUR TOKENS QUICKLY!**

Each pattern review spawns a Claude Code agent (Sonnet model) that reads and analyzes files. This costs tokens from your Claude Code subscription.

Before running semantic validation, CONFIRM with user:
- **Single pattern?** OK to proceed.
- **Selected few patterns?** OK if deterministic checks pass first.
- **ALL 66 patterns?** CONFIRM with user first - this is expensive!

When asked to review patterns:
1. Always run deterministic checks first (free, no tokens)
2. Ask user: "Run semantic validation on selected patterns or all 66?"
3. Only proceed with bulk review after explicit user confirmation

**Token costs:**
- Each pattern review reads pattern.toml + all test files
- Reviewing all 66 patterns in parallel multiplies token usage significantly
- Always run deterministic checks first - they're free and catch most issues

### Usage

**Script-based (recommended for batch validation):**

```bash
# Single pattern
python pattern_verification/semantic/semantic_validate.py pt-001

# All patterns in a category
python pattern_verification/semantic/semantic_validate.py --category ai-training

# All 66 patterns (auto-generates timestamped output directory)
python pattern_verification/semantic/semantic_validate.py --all
# Creates: reports/YYYYMMDD_HHMMSS_all/
#   ├── summary.md       # Overall summary
#   ├── progress.log     # One line per pattern
#   └── patterns/*.log   # Raw Claude output per pattern

# Increase parallelism (default: 16)
python pattern_verification/semantic/semantic_validate.py --all

# Background execution with real-time monitoring
python pattern_verification/semantic/semantic_validate.py --all &
tail -f pattern_verification/semantic/reports/*/progress.log
```

**Note:** Uses Sonnet model via Claude CLI. Agent definition at `pattern_verification/pattern-reviewer/`.

### What it checks

- Does `description` accurately describe the test file code?
- Does `expected_issue` align with detection question's YES condition?
- Does `snippet` actually exist in the test file?
- Do positive tests actually contain the bug described?
- Do negative tests actually avoid the bug?
- Does the pattern align with official documentation? (if cached docs available)

### When to use

- After auto-generating descriptions (especially if generated with cheaper models)
- When evals fail unexpectedly
- Before merging new patterns
- **NOT for routine checks** - use deterministic validation instead

## Recommended Workflow

```bash
# 1. Run deterministic checks first (fast, catches obvious issues)
python pattern_verification/deterministic/validate.py

# 2. Fix any errors
python pattern_verification/deterministic/validate.py --fix

# 3. Run semantic review (uses pattern-reviewer agent - read-only)
python pattern_verification/semantic/semantic_validate.py <pattern-id>  # Single
python pattern_verification/semantic/semantic_validate.py --all        # All (16 concurrent)

# 4. Fix issues directly in Claude Code session
# Claude Code has write permissions - no need for separate agent

# 5. Run evals (requires vLLM server)
python evals/run_eval.py --pattern <pattern-id>
```

## Pattern Reviewer Agent

| Agent | Role | Tools | Location |
|-------|------|-------|----------|
| `pattern-reviewer` | Identifies issues (read-only) | Read, Glob, Grep | `pattern_verification/pattern-reviewer/` |

Semantic validation (step 3) runs `pattern-reviewer` to identify issues. Fix issues directly in your Claude Code session.

## ⚠️ Memory Safety: No Sub-Agent Spawning

**CRITICAL:** The `pattern-reviewer` agent MUST NOT use the Task tool to spawn sub-agents.

When running batch validation (66 patterns), each Claude process may try to spawn sub-agents via the Task tool. This causes uncontrolled and unpredictable token usage.

**How it's prevented:**
1. `semantic_validate.py` passes `--disallowed-tools Task,WebSearch,WebFetch,Bash,Write,Edit,NotebookEdit` to Claude CLI
2. Agent system prompt explicitly prohibits Task tool usage

If you modify the agent or script, ensure Task tool remains blocked.

**RAM requirements (same for sonnet/opus):**
- ~450MB per Claude process (model choice doesn't affect RAM)
- 32 parallel processes ≈ 16GB RAM
- Formula: `parallel_count × 500MB` (use 500MB for safety margin)
- Example: 16GB RAM system → safe with `--parallel 28` max

## Reference Documentation Cache

For documentation alignment checks, fetch and cache official docs:

```bash
python pattern_verification/deterministic/validate.py --fetch-refs
```

Cached docs are stored as `<pattern-id>_<domain>_<hash>.md` in `doc_cache/clean/`. The semantic reviewer uses these to verify patterns align with official documentation.

### Doc Processing Pipeline

Fetched HTML goes through two-stage cleaning:

1. **HTML stripping** (deterministic) - Removes `<nav>`, `<footer>`, `<aside>`, `<header>` tags and elements with nav-related CSS classes (navbar, sidebar, breadcrumb, toc, etc.)

2. **vLLM cleaning** (AI-based) - Removes remaining boilerplate (cookie notices, "Rate this page", etc.) while preserving technical content

**Requirements:**
- vLLM server must be running for stage 2
- Content must be <64K chars after HTML stripping (configurable via `max_input_tokens`)
- If vLLM unavailable or content too large: warns and skips clean cache

### Warnings

The validator warns about:
- **Doc too large** (>1000 lines after cleanup) - find more specific page or use anchor link
- **Useless docs** flagged by semantic reviewer - thin API pages that don't explain the concept

See "Choosing good reference URLs" in [patterns/README.md](../src/scicode_lint/patterns/README.md) for URL selection guidance.

## Directory Structure

```
pattern_verification/
├── deterministic/
│   ├── validate.py          # Structural checks (no LLM)
│   ├── diversity_check.py   # Semantic diversity check (Claude CLI)
│   ├── doc_cache/           # Cached reference documentation
│   │   ├── raw/             # Unprocessed fetched docs
│   │   └── clean/           # Cleaned docs (nav stripped)
│   └── reports/             # Diversity check output (gitignored)
│       └── YYYYMMDD_HHMMSS_<scope>/
│           ├── summary.md
│           ├── progress.log
│           └── patterns/*.log
├── semantic/
│   ├── semantic_validate.py # Batch semantic validation script
│   └── reports/             # Semantic validation output (gitignored)
│       └── YYYYMMDD_HHMMSS_<scope>/
│           ├── summary.md   # Overall summary
│           ├── progress.log # One line per pattern completion
│           └── patterns/    # Raw Claude output per pattern
│               ├── pt-001.log
│               └── ...
└── README.md

pattern-reviewer/                # Read-only analysis agent
├── agent.json
└── system_prompt.md
```
