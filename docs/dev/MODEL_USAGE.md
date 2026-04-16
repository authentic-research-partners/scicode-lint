# Model Usage Guide

Which AI models are used throughout scicode-lint and why.

## Overview

| Class | Model | Purpose |
|-------|-------|---------|
| **SOTA** | Claude Sonnet 4.6 (via CLI) | Automated tasks: pattern verification, FP analysis, integration evals, pattern review |
| **SOTA** | Claude Opus 4.6 (via CLI) | Interactive development only (Claude Code) |
| **Local** | Qwen3-8B-FP8 (via vLLM) | Runtime detection, pattern eval verdict review, data source preprocessing |

**Principle:** Opus for development (reliability matters, fewer iterations). Sonnet for all automated/batch tasks (fast, cost-effective, capable enough with appropriate thinking effort).

---

## Claude CLI Wrapper

All Claude CLI invocations go through the shared `dev_lib.ClaudeCLI` wrapper:

```python
from dev_lib.claude_cli import ClaudeCLI, ClaudeCLIError, DISALLOWED_TOOLS_ALL

cli = ClaudeCLI(model="sonnet", effort="low", timeout=120)
result = await cli.arun(prompt, disallowed_tools=DISALLOWED_TOOLS_ALL)
data = await cli.arun_json(prompt, effort="high")  # --output-format json with double-parse
```

**Error hierarchy:** `ClaudeCLIError` → `ClaudeCLINotFoundError`, `ClaudeCLITimeoutError`, `ClaudeCLIProcessError`, `ClaudeCLIParseError`

**Logging:** Every call logs via loguru — model, label, elapsed time, estimated input/output tokens.

**Location:** `dev_lib/claude_cli.py` (dev-only, not part of installed package)

**Disk streaming:** All batch verification tools stream results to disk via `dev_lib.RunOutput` + `write_worker`. See `dev_lib/run_output.py`.

---

## Thinking Effort

All Claude CLI invocations pass `--effort` to control thinking depth:

| Level | Description | Use Case |
|-------|-------------|----------|
| `max` | Unconstrained thinking | Not used in automated tasks (Opus-only feature) |
| `high` | Deep reasoning, always thinks | Judgment tasks: verification, judging, finding verification |
| `medium` | Moderate thinking | Structured tasks: pattern review, code generation |
| `low` | Minimal thinking | Structured pairwise comparisons: diversity check |

Effort is configured per-component, not globally.

---

## Usage by Component

### 1. Development (Claude Code)

**Model:** Opus 4.6

Interactive development: writing code, patterns, debugging, refactoring. Opus produces code that works correctly on the first try more often — fewer iterations means faster overall despite slower per-response speed.

---

### 2. Pattern Semantic Validation

**File:** `pattern_verification/semantic/semantic_validate.py`

**Model:** Sonnet (configurable in `config.toml`)
**Effort:** `medium`

```toml
# config.toml
[pattern_verification]
semantic_model = "sonnet"
```

Override: `python pattern_verification/semantic/semantic_validate.py --model opus --all`

---

### 3. Diversity Check

**File:** `pattern_verification/deterministic/diversity_check.py`

**Model:** Sonnet (configurable in `config.toml`)
**Effort:** `low`

Batches all pairwise comparisons per pattern into a single Claude call. Returns JSON verdicts.

```toml
# config.toml
[pattern_verification]
diversity_model = "sonnet"
```

Rate limiting (parallel processes + RPM) is controlled globally via `[claude_cli]` in config.toml.

Override model: `python pattern_verification/deterministic/diversity_check.py --model opus`

---

### 4. Pattern Evals (LLM-as-Judge)

**File:** `evals/run_eval.py`

**Model:** Qwen3-8B-FP8 via vLLM (local LLM — same model used for runtime detection)

Two-step process, both using the local LLM:
1. **Detection** — runs each pattern's detection question against test files (positive, negative, context-dependent)
2. **Verdict review** — reviews the linter's detection output (detected/not, reasoning, confidence) against the test file's expected behavior, and returns a `yes`/`no`/`partial` verdict

The verdict reviewer assesses reasoning quality, not just detection correctness. For context-dependent tests, it judges whether the linter's reasoning is sound regardless of the detection outcome.

**Not Sonnet.** The local LLM reviews via `async_complete_structured()` with a `JudgeVerdict` Pydantic schema. This is intentional: the reviewer uses the same model that runs in production, so eval results reflect real-world accuracy.

---

### 5. Integration Evals

**File:** `evals/integration/integration_eval.py`

**Model:** Sonnet (via Claude CLI)

Unlike pattern evals (which use the local LLM for judging), integration evals use **Sonnet for all steps** including judging. This is because integration evals test end-to-end scenarios with generated code, requiring stronger reasoning than the local model provides.

| Task | Model | Effort | Rationale |
|------|-------|--------|-----------|
| Pattern selection | Sonnet | `medium` | Structured: pick compatible patterns |
| Code generation | Sonnet | `medium` | Structured: generate test code |
| Manifest verification | Sonnet | `high` | Judge if generated code correctly embeds bugs |
| Result judging | Sonnet | `high` | Compare findings against ground truth |

---

### 6. Real-World Verification

**File:** `real_world_demo/verify_findings.py`

**Model:** Sonnet (default)
**Effort:** `high`

Override: `python real_world_demo/verify_findings.py --model opus`

---

### 8. Data Source Preprocessing (PapersWithCode)

**Files:**
- `real_world_demo/sources/papers_with_code/filter_abstracts.py` — classifies paper abstracts as AI-for-science or not
- `real_world_demo/sources/papers_with_code/prefilter_files.py` — classifies Python files as self-contained vs fragments

**Model:** Qwen3-8B-FP8 via vLLM (local LLM)

Both scripts use the local model for structured classification via `async_complete_structured()` with Pydantic schemas. These are preprocessing/filtering steps that run before the main scan pipeline, not judgment tasks — the local model is sufficient for binary/categorical classification.

---

### 9. False Positive Analysis

**File:** `real_world_demo/analyze_errors.py`

**Model:** Sonnet (default)
**Effort:** `high`

Analyzes verification reasoning to extract common FP themes, root causes, and specific recommendations for improving detection questions. Part of the meta improvement loop (Stage 3).

Override: `python -m real_world_demo.analyze_errors --model opus`

---

## Local LLM (Runtime Detection + Eval Verdict Review)

**Model:** Qwen3-8B-FP8 via vLLM

- Local execution (privacy, no API costs)
- Reproducibility (open-source model)
- Fits in 16GB VRAM
- Prefix caching for speed

**Used in three contexts:**
1. **Runtime detection** — `python -m scicode_lint lint` runs patterns against user code
2. **Pattern eval verdict review** — `evals/run_eval.py` uses the same local model to review whether the linter's detection reasoning is correct against expected behavior
3. **Data source preprocessing** — `filter_abstracts.py` and `prefilter_files.py` classify papers and files before scanning

Configuration: `config.toml` `[llm]` section

---

## Decision Framework

```
Interactive development (Claude Code)?
  → Opus with max effort

Judgment task (verification, judging, finding review)?
  → Sonnet with high effort

Structured/mechanical task (generation, selection, pattern review)?
  → Sonnet with medium effort

Runtime detection?
  → Local LLM (Qwen3-8B)

Pattern eval verdict review (evals/run_eval.py)?
  → Local LLM (Qwen3-8B) — reviews linter reasoning, same model as runtime

Integration eval judging (evals/integration/)?
  → Sonnet — judges end-to-end generated scenarios, needs stronger reasoning
```

---

## Changing Defaults

Semantic validation model:
```toml
# config.toml
[pattern_verification]
semantic_model = "sonnet"  # or "opus"
```

Other components: modify defaults in the source files or pass `--model` CLI flag.

All Claude CLI wrapper defaults (model, effort, timeout) can be overridden per-call via `ClaudeCLI` constructor or method arguments.
