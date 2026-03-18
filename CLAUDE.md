# scicode-lint

AI-powered linter for scientific Python code using local LLM (Qwen3 via vLLM).

> **Keep this file focused.** Rules and critical info only. No directory structures, no redundant examples. Claude can explore the codebase itself.

---

## Table of Contents

- [Key Files](#key-files)
- [Project Statistics](#project-statistics)
- [Architecture Overview](#architecture-overview)
- [Local LLM Setup](#local-llm-setup)
- [Claude CLI for Development](#claude-cli-for-development)
- [Pattern Development](#pattern-development)
- [Critical Development Rules](#critical-development-rules)
- [AI Coding Agent Workflow](#ai-coding-agent-workflow)
- [Continuous Improvement Loop](#continuous-improvement-loop)

---

## Key Files

**Main files (root):**
- [README.md](./README.md) - Main file for humans
- [INSTALLATION.md](./INSTALLATION.md) - Detailed installation guide
- [DOCUMENTATION_MAP.md](./DOCUMENTATION_MAP.md) - Navigation guide to all documentation
- [CLAUDE.md](./CLAUDE.md) - This file, main instructions for AI agents working on the codebase
- [src/scicode_lint/patterns/](./src/scicode_lint/patterns/) - Detection patterns (66 patterns across 5 categories)
- [src/scicode_lint/patterns/ROADMAP.md](./src/scicode_lint/patterns/ROADMAP.md) - Pattern coverage roadmap and future priorities

**For humans (docs_use_human/):**
- [README.md](docs_use_human/README.md) - Documentation index
- [USAGE.md](docs_use_human/USAGE.md) - CLI usage guide
- [VRAM_REQUIREMENTS.md](docs_use_human/VRAM_REQUIREMENTS.md) - Hardware requirements
- [performance/](docs_use_human/performance/) - Performance & benchmarking guides

**For AI agents USING scicode-lint (docs_use_genai/):**
- [GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md) - Complete guide for AI agents using scicode-lint
- [INTERFACE_ANALYSIS.md](docs_use_genai/INTERFACE_ANALYSIS.md) - Package interface analysis

**For AI agents WORKING ON scicode-lint (docs_dev_genai/):**
- [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) - Core design principles (READ THIS FIRST)
- [CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md) - Fast evaluation → improvement workflow
- [META_IMPROVEMENT_LOOP.md](docs_dev_genai/META_IMPROVEMENT_LOOP.md) - Real-world validation with Claude verification
- [IMPLEMENTATION.md](docs_dev_genai/IMPLEMENTATION.md) - Technical implementation details
- [MODEL_USAGE.md](docs_dev_genai/MODEL_USAGE.md) - SOTA model usage (Opus 4.6 for dev, Sonnet 4.6 for automation, local LLM for runtime)

**Dev-only shared utilities (dev_lib/):**
- [dev_lib/claude_cli.py](dev_lib/claude_cli.py) - Unified async Claude CLI wrapper (ClaudeCLI class)
- [dev_lib/config.py](dev_lib/config.py) - Shared config.toml loading for dev tools

**Pattern verification (pattern_verification/):**
- [pattern_verification/](pattern_verification/) - Deterministic + semantic pattern quality checks

**Repository filter (src/scicode_lint/repo_filter/):**
- `classify.py` - LLM-based file classification (self-contained vs fragment)
- `scan.py` - Repository scanner with two-stage filtering
- ML import keywords configurable in `config.toml` → `[preprocessing]` section

**Consolidated results (consolidated_results/):**
- [consolidated_results/](consolidated_results/) - Unified performance report across all eval layers
- `generate_consolidated_report.py` - Reads JSON, DB, markdown → single `CONSOLIDATED_REPORT.md`

**Real-world validation (real_world_demo/):**
- [real_world_demo/README.md](real_world_demo/README.md) - Run scicode-lint on real scientific ML code
- Data sources: `sources/papers_with_code/` (PapersWithCode repos), `sources/leakage_paper/` (Yang et al. ASE'22)
- Paper set references: `real_world_demo/paper_sets/` (git-tracked `.md` + `.json` for each data pool)
- Uses Claude CLI for finding verification (`verify_findings.py`)
- Database stores scan results: `repo_scans` table, `files.self_contained_class`

---

## Project Statistics

**To get project stats, run the stats script — do NOT gather stats manually:**

```bash
python scripts/project_stats_generate.py
```

Collects: git stats (commits, branches, age), tech stack (from pyproject.toml), code line counts, pattern counts, documentation stats. Outputs to `PROJECT_STATS.md`.

---

## Architecture Overview

**Core principle: Constrained-capacity local LLM by design.**

scicode-lint uses a small local model (fits in 16GB VRAM) - a deliberate choice between grep-style matching and expensive SOTA models. This enables local execution, privacy, no API costs, and reproducibility (open-source models remain available), but requires detection questions to be simple and direct.

**Two-phase design:**
- **Build time:** Design detection questions that constrained models can reliably answer
- **Runtime:** Local model (FP8 via vLLM) runs focused checks against code

**Philosophy:** Detection only, no automatic fixes.

**⚠️ CRITICAL ARCHITECTURAL RULES** (see [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) for complete rationale):
- Code MUST come before detection instructions in prompts (enables vLLM prefix caching)
- Detection questions must be focused (one issue per question) with self-contained context
- Include "why it matters" directly in the question (Qwen3 is a thinking model - context helps reasoning)

**📖 Complete architecture:** [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md)

---

## Local LLM Setup

**Required:** vLLM server with FP8 model (16GB+ VRAM, native FP8 support required)

```bash
pip install scicode-lint[vllm-server]
scicode-lint vllm-server start              # Auto-detects GPU, validates FP8 support (default port: 5001)
scicode-lint lint myfile.py
```

**📖 Complete setup:** [INSTALLATION.md](INSTALLATION.md)

### vLLM Client Usage in Code

When writing code that calls vLLM, **always use the shared infrastructure**:

```python
from scicode_lint.config import load_llm_config
from scicode_lint.llm.client import create_client

# Create client (auto-detects vLLM server)
llm_config = load_llm_config()
llm_client = create_client(llm_config)

# Use async with semaphore for concurrency control
semaphore = asyncio.Semaphore(max_concurrent)
async with semaphore:
    result = await llm_client.async_complete_structured(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        schema=MyPydanticSchema,
    )
```

**Rules:**
- **Use `VLLMClient`** from `scicode_lint.llm.client` — NOT raw `httpx` calls
- **Use Pydantic schemas** for structured output — client uses vLLM's `guided_json` for guaranteed valid JSON
- **Use `asyncio.Semaphore`** for concurrency control — prevents overwhelming the server
- **Define response schemas** with `pydantic.BaseModel` — enables type-safe responses

**Example Pydantic schema:**
```python
from pydantic import BaseModel, Field

class FilterResult(BaseModel):
    is_valid: bool = Field(description="Whether the item passes the filter")
    explanation: str = Field(description="Brief explanation of the decision")
```

---

## Claude CLI for Development

**Pattern verification and improvement uses Claude CLI (`claude` command), NOT the Anthropic Python SDK.**

- **Auth:** `claude login` (OAuth) — uses Claude Code subscription (Pro/Team)
- **No API keys needed** — if you have Claude Code subscription, you're set
- **NOT compatible with:** `anthropic.Client()` or `ANTHROPIC_API_KEY` (separate billing)

All Claude CLI calls go through the shared **`dev_lib.ClaudeCLI`** wrapper (`dev_lib/claude_cli.py`):
- Async-only API: `await cli.arun(prompt)` (text) or `await cli.arun_json(prompt)` (parsed JSON)
- **Global rate limiting**: Built-in semaphore + RPM limiter shared across ALL instances (configured in `[claude_cli]` section of config.toml). Callers do NOT need their own semaphores.
- Typed errors: `ClaudeCLITimeoutError`, `ClaudeCLIProcessError`, `ClaudeCLIParseError`
- Tool blocking constants: `DEFAULT_DISALLOWED_TOOLS`, `DISALLOWED_TOOLS_ALL`
- Shared config loading: `dev_lib.config.load_project_config()`
- **Logging**: Every call logs via loguru — model, label, elapsed time, estimated input/output tokens

**Disk streaming** via `dev_lib.run_output`:
- `RunOutput.create(reports_dir, scope, items_dirname)` — timestamped output directory
- `write_worker(queue)` — async queue worker serializing disk writes (prevents I/O contention)
- Used by: `semantic_validate.py`, `diversity_check.py`, `integration_eval.py`

**`dev_lib/` is dev-only** — NOT part of the installed package. Used by `pattern_verification/`, `evals/`, `real_world_demo/`.

---

## Pattern Development

**📖 Pattern guide:** [src/scicode_lint/patterns/README.md](src/scicode_lint/patterns/README.md) - Structure, detection question template, test file rules

### Pattern Verification

**📖 Full guide:** [pattern_verification/README.md](pattern_verification/README.md)

Three-step verification (does NOT run evals):

```bash
# 1. Deterministic checks (fast, no LLM)
python pattern_verification/deterministic/validate.py
python pattern_verification/deterministic/validate.py --fix  # auto-fix

# 2. Diversity check (Claude CLI) - detects redundant/non-diverse test files
python pattern_verification/deterministic/diversity_check.py  # all patterns
python pattern_verification/deterministic/diversity_check.py ml-001  # single pattern

# 3. Semantic review (uses pattern-reviewer agent - read-only)
python pattern_verification/semantic/semantic_validate.py --all  # All patterns (recommended)
python pattern_verification/semantic/semantic_validate.py pt-001  # Single pattern
python pattern_verification/semantic/semantic_validate.py --category ai-training  # Category

# 4. Fix issues directly in Claude Code session
# Claude Code has write permissions - no separate agent needed
```

**Agent:** `pattern-reviewer` identifies issues (read-only). Fix issues directly in your Claude Code session.

**Output:** Auto-generates timestamped directory with summary, progress log, and per-pattern logs:
```
pattern_verification/semantic/reports/YYYYMMDD_HHMMSS_<scope>/
├── summary.md       # Overall results
├── progress.log     # Real-time progress
└── patterns/*.log   # Raw Claude output per pattern
```

### Improvement Loop (runs evals, requires vLLM)

Evaluates detection accuracy and iteratively improves patterns. See [Continuous Improvement Loop](#continuous-improvement-loop) below.

---

## Critical Development Rules

### Code Quality
- **DRY (Don't Repeat Yourself)** - Define constants, configs, and model names once; reference them everywhere
- **YAGNI (You Aren't Gonna Need It)** - Don't add features "just in case"; build for current requirements only
- **Fail fast** - Raise errors early with clear messages; don't let invalid state propagate
- **No silent fallbacks** - If config cannot be loaded, FAIL. Never silently use default values when config loading fails
- **Explicit over implicit** - Make intent clear; avoid magic numbers, use named constants
- **Small functions** - Each function does one thing; if a block needs a comment, extract it to a named function
- **Reasonable file size** - Split files > 1000 lines; each file has one clear responsibility
- **Meaningful names** - Names describe purpose, not type (`max_retries` not `n`, `user_ids` not `list1`)
- **NEVER hide issues** - Investigate root cause, don't suppress without understanding
- **Known issues require references** - Cite specific GitHub issue/docs (verify it exists, no made-up citations)
- **NO LEGACY CODE** - Never create "legacy/", "old/", "deprecated/" directories - delete unused code completely
- **Dual audience** - Detection output must be clear for BOTH humans and GenAI agents
- Every detection pattern must have eval coverage

### Prompt Files & Linting
- **NEVER modify prompt content to fix lint errors** - Prompt wording affects model behavior
- If ruff reports E501 (line too long) in a prompt file, add per-file-ignores in pyproject.toml:
  ```toml
  [tool.ruff.lint.per-file-ignores]
  "path/to/prompts.py" = ["E501"]
  ```

### Git Commits
- **Do NOT include Claude as co-author** in commit messages (no `Co-Authored-By: Claude` lines)

### Release Checklist (before any build/release)
Before building or releasing, verify version consistency:
```bash
# Check all version references match
grep -rn "0\.[0-9]\.[0-9]" --include="*.py" --include="*.toml" --include="*.md" . | grep -v __pycache__ | grep -v cloned_repos | grep -v output_examples | grep -v CHANGELOG
```
Files that must have matching versions:
- `pyproject.toml` - `version = "X.Y.Z"`
- `src/scicode_lint/__init__.py` - `__version__ = "X.Y.Z"`
- `evals/__init__.py` - `__version__ = "X.Y.Z"`
- `README.md` - version in Project Status section

Also check:
- `CHANGELOG.md` has entry for new version
- No secrets/credentials in code (`grep -rE "sk-|apikey|password"`)

### Initial Setup (once per clone)
```bash
# Activate dedicated environment (see INSTALLATION.md for setup)
conda activate scicode  # or: source ~/.scicode-venv/bin/activate

# Enable pre-commit hooks
git config core.hooksPath scripts
```

### Mandatory Checks (RUN AFTER EVERY CODE CHANGE)
```bash
ruff check . && ruff format .  # Code style
mypy .                         # Type checking (patterns/ excluded via pyproject.toml)
python -m py_compile path/to/new_or_changed_file.py  # Catches circular imports that pytest misses
pytest                         # Tests (last — slowest)
```

### Type Hints & Documentation
- **All functions** must have type annotations (args + return)
- **All exported classes/methods** (in `__all__`) must have docstrings (Args, Returns, Examples)
- **Update API_REFERENCE.md** when adding public APIs
- Private methods (_prefixed) need type hints only

### Documentation Rules
- **Docs are snapshots** - Describe current state, not history (history lives in git)
- **No "improved from" language** - Just state what IS, not what changed (e.g., "Precision: 34%" not "Precision: 34% (improved from 24%)")
- **src/scicode_lint/patterns/README.md** - Pattern guide with detection question template
- **Check after every change** - Fix documentation drift immediately
- **Human docs:** Concise, link to external docs, scicode-lint-specific only
- **GenAI docs:** Comprehensive, self-contained, complete examples, explain rationale

### Project Principles
- **NO backward compatibility** - Clean breaks over cruft (young project in active development)
- **Minimize false positives** - A noisy tool gets ignored
- **Keep it simple** - Modern Python packaging (pyproject.toml, ruff, pytest)

**📖 Complete guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md)

**License:** MIT

---

## AI Coding Agent Workflow

After writing code, **always** run:

1. **Document** - Add docstrings + type hints to public APIs, update API_REFERENCE.md if in `__all__`
2. **Quality** - `ruff check . && ruff format .` then `mypy .`
3. **Test** - `pytest`
4. **Docs** - Update docs to match current state (no "updated"/"changed" language)
5. **Commit** - Only commit clean, type-safe, documented code

---

## Continuous Improvement Loop

**Also called:** improvement loop, eval loop, pattern improvement, improve patterns, run evals, do evals

**What it is:** A structured workflow for iteratively evaluating and improving detection pattern quality. Requires vLLM server running.

**Different from pattern review:** Pattern review (above) is static analysis of pattern files. Improvement loop runs actual detection against test files.

**📖 Complete instructions:** [docs_dev_genai/CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md)

**Quick start:**
```bash
# 1. Full validation (structural + diversity + semantic)
python pattern_verification/deterministic/validate.py
python pattern_verification/deterministic/diversity_check.py  # ~5-10 min
python pattern_verification/semantic/semantic_validate.py --all

# 2. Full evals (detection accuracy)
python evals/run_eval.py

# 3. If patterns fail: per-pattern fix cycle
python pattern_verification/deterministic/diversity_check.py <pattern-id>
python pattern_verification/semantic/semantic_validate.py <pattern-id>
python evals/run_eval.py -p <pattern-name>
# Fix, repeat until pass

# 4. When all pass: update README stats, run integration tests
python evals/integration/integration_eval.py --generate-count 10
# Save baseline for regression
python evals/integration/integration_eval.py --generate-count 50 --save --id baseline_v1
```
