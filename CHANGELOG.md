# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-16

### Breaking
- **CLI exit code semantics changed.** Now `0` = clean, `1` = findings
  detected, `2` = tool/runtime error (vLLM unreachable, file parse failure,
  etc.). Previously `1` was returned for both findings and tool errors. CI
  scripts that treat any non-zero exit as a tool error must branch on `2`
  instead. See [USAGE.md § `lint`](docs/USAGE.md#lint).
- **`--json-errors` flag removed** from both `lint` and `analyze` commands.
  Per-file errors are always included in both text and JSON output now, and
  any script still passing `--json-errors` will fail argparse with
  "unrecognized arguments". The structured `error` object in JSON results is
  unchanged.
- **Text-output format restructured.** Findings now render as Rich panels with
  syntax-highlighted code snippets, line numbers, and focus-line gutter
  instead of flat indented text. Scripts grepping raw text output (e.g.
  `grep "🔴 CRITICAL \["`) must be rewritten — use `--format json` for
  stable, machine-readable output. Color is emitted only when stdout is a
  TTY; piped output stays ANSI-free and fixed-width.
- **`LintError.error_type` strings changed** for vLLM connection failures:
  `"RuntimeError"` → `"LLMConnectionError"`. Consumers parsing this JSON
  field must update their matchers.
- **`MissingLocationError` no longer inherits from `ValueError`.** Callers
  catching it via `except ValueError` must switch to the exception name or
  `SciCodeLintError`. Verified in-tree: only raised and handled inside
  `_handle_response`; no external `except ValueError` caught it.

### Added
- **Typed exception hierarchy** — new `scicode_lint.exceptions` module with
  `SciCodeLintError` base class. `ContextLengthError`, `MissingLocationError`,
  and `NotebookParseError` now inherit from it, and a new `LLMConnectionError`
  replaces the generic `RuntimeError` raised by `detect_vllm()`. Callers can
  catch a single base class to handle every documented failure mode. See
  [USAGE.md § Error Handling](docs/USAGE.md#error-handling).
- **CLI exit code `2`** for tool/runtime errors (see Breaking above).
- **Rich-polished text output** — findings render as syntax-highlighted panels
  with line numbers and focus-line gutter (see Breaking above for migration).
- **"For GenAI Agents" README section** — documents the `--format json`
  path, exit-code conventions, and exception hierarchy for programmatic
  consumers.
- **Fast-fail on unreachable vLLM URL.** `VLLMClient` lazily probes
  `{base_url}/v1/models` on the first request (3s timeout). A bad or
  dead `--vllm-url` now raises `LLMConnectionError` in seconds instead
  of burning the full openai retry budget (previously up to ~9 min on
  a refused connection).
- **Consistent `LLMConnectionError` propagation to exit code `2`** across
  every stage of the pipeline:
  - **Per-file lint** (`linter.py`): systemic connection failures re-raise
    through the pattern loop instead of being demoted to `patterns_failed`
    entries that left `LintResult.error = None`.
  - **Filter / classify** (`repo_filter/scan.py`): `scicode-lint analyze`
    used to swallow connection failures and report "no self-contained ML
    files found" with exit `0`; now fails fast with exit `2`.
  - **Mid-call drops** (`llm/client.py`): `openai.APIConnectionError` and
    `openai.APITimeoutError` raised during a live request now map to
    `LLMConnectionError` (previously wrapped as generic `RuntimeError`).
- **Container-based vLLM** — `scicode-lint vllm-server start` now launches vLLM
  in a podman/docker container. New commands: `restart`, `logs`, `rm`, `monitor`.
  Requires `podman` or `docker` + `nvidia-container-toolkit`.
- **FP8 KV cache** — `--kv-cache-dtype fp8` roughly doubles concurrent-request
  budget at a given VRAM level. Requires FP8 hardware (compute capability ≥ 8.9).
- **Thinking budget controls** — `thinking_budget` (hard cap) and
  `thinking_effort` (soft guide) in `config.toml` tune model reasoning per call.
  Default: uncapped.
- **Transient retry** on `async_complete_structured()` (empty content or JSON
  glitches) and `ClaudeCLI.arun_json()` (parse errors).
  See [CONSTRAINED_DECODING.md](./src/scicode_lint/llm/CONSTRAINED_DECODING.md#transient-retry).
- **Rich-based `vllm-server monitor`** — live panel with VRAM/GPU bars, request
  counts, KV cache + prefix-cache hit rate, latency, throughput.
- **Paper citation** — `CITATION.cff`, README section, and PyPI sidebar point to
  [arXiv:2603.17893](https://arxiv.org/abs/2603.17893).
- **Reproducible install** via `requirements-pinned.txt`.

### Changed
- **`response_format` replaces `guided_json`** (deprecated in vLLM v0.12.0).
  Reasoning enabled server-side via `--reasoning-parser qwen3`.
- **Bounded LLM response schemas** — all structured-output schemas now set
  `max_length` / `maxItems` / `Literal` constraints to prevent JSON truncation
  under tight thinking budgets.
  See [CONSTRAINED_DECODING.md](./src/scicode_lint/llm/CONSTRAINED_DECODING.md#schema-bounds-max_length-and-max_items-are-mandatory).
- **Qwen3 reasoning alignment** — `thinking_effort` now uses top-level
  `reasoning_effort` (OpenAI standard); `enable_thinking: False` passed
  explicitly when `thinking_budget=0`.
- **Dependency minimums raised** to match tested dev env. Adds `rich>=14.0.0`.
- **License metadata** modernized to PEP 639 SPDX form.

### Fixed
- **vLLM v0.18+ compatibility** — no longer passes removed `--enable-reasoning`
  flag (now implicit when `--reasoning-parser` is set).
- **`vllm-server monitor`** stays open during model load instead of exiting on
  first failed health check.

### Removed
- **Native-subprocess vLLM launcher** — container launcher is now the only
  supported path.
- **Google Colab setup instructions** — Colab lacks a container runtime.
- **Hardcoded config fallbacks** — `config.toml` is the single source of truth;
  missing values raise `RuntimeError`.
- **`[vllm-server]` pip extra** — vLLM runs inside the container, not as a pip
  dependency.

### Packaging
- **Production-only sdist** — `pip download scicode-lint` returns just runtime
  + user-facing metadata. Tests, evals, and pattern verification stay in the
  git repo.

## [0.2.3] - 2026-03-18

### Added
- **File size benchmark** — scan time vs code size, sub-linear scaling (31x lines ~ 3x time) on RTX 4000 Ada

### Changed
- **Kaggle notebook eval re-run** with updated patterns — `pre` label: precision 65%, recall 100%, F1 79%
- Corrected max processable file size in docs: ~1,400 lines (was ~1,500)

### Fixed
- `run_analysis` file existence validation now correctly uses `base_dir` for path resolution

## [0.2.2] - 2026-03-17

### Added
- **`vllm-server` CLI subcommand** — `scicode-lint vllm-server start/stop/status`
- **Consolidated performance report** across all eval layers (`consolidated_results/`)

### Changed
- **3 meta-improvement passes** across all 66 patterns — real-world precision improved from 20.2% to 62.0% on feedback set, 37.9% to 54.1% on holdout set
- 8 large files (>1000 lines) split into focused modules

## [0.2.1] - 2026-03-16

### Added
- **`analyze` command** - Lint any git repository in one step: clones the repo, finds self-contained ML files, runs all detection patterns
- **`filter-repo` command** - Find self-contained ML files in a repository without running detection
- **Comment stripping** - Strips `#` comments before LLM analysis (prevents intention leakage, reduces tokens)
- **Name-based location detection** - LLM identifies function/class names, AST resolves to lines (stable across runs)
- **One finding per pattern** - Each pattern produces at most one finding per file
- **Meta improvement loop** - Real-world validation workflow (`docs_dev_genai/META_IMPROVEMENT_LOOP.md`)
- **Diversity check** - Semantic similarity check for pattern test files, detects redundant or non-diverse tests
- **Concurrency flags** - `--filter-concurrency`, `--lint-concurrency` for tuning parallel LLM calls
- **`--save-to-db`** - Optional database storage for scan results

### Changed
- **Renamed `check` → `lint`**, **`scan-repo` → `filter-repo`**
- **SOTA model policy** - Opus 4.6 for interactive dev, Sonnet 4.6 for automated tasks
- **All 66 patterns reviewed** through full improvement loop
- Unified Claude CLI wrapper, disk streaming for verification tools (dev tooling)

### Removed
- **Line overlap validation** - Replaced by name-based location matching

### Fixed
- Line numbers correctly stored in findings DB
- LLM retry with correction when detection has no location
- Pattern accuracy fixes across all 5 categories

### Other
- Unified Claude CLI wrapper (`dev_lib/claude_cli.py`) with rate limiting and logging
- Intent hints validation — detects test files that accidentally reveal the expected answer via names or docstrings

## [0.2.0] - 2026-03-13

**First PyPI release** (alpha)

### Changed
- **Patterns bundled with package** - moved from `patterns/` to `src/scicode_lint/patterns/` so patterns are included when installing via `pip install scicode-lint`
- Updated all documentation and scripts to reflect new patterns location

### Fixed
- Package now works correctly when installed from PyPI (patterns were previously not included in the wheel)

## [0.1.6] - 2026-03-12

**Theme: Real-world validation on scientific ML code**

### Real-World Validation (`real_world_demo/`)

End-to-end pipeline for validating scicode-lint on real scientific ML code:

**Pipeline:** PapersWithCode API → filter by scientific domain (biology, medical, physics) → clone repos → filter files with ML imports → LLM prefilter for scientific code → run scicode-lint → verify findings with Claude Opus

**Two data sources tested:**

- **PapersWithCode repos** - AI/ML + Science papers with code
  - Every reviewed paper had at least one verified bug
  - ~19% precision (5/26 verified findings are real issues)

- **Yang et al. ASE'22 dataset** - 99 Jupyter notebooks with ground truth labels
  - Academic benchmark for data leakage detection
  - 67-90% precision on leakage patterns

### Jupyter Notebook Support
- **Python extraction** - extracts code cells before LLM analysis (99% token reduction)
- Proper error handling with `NotebookParseError`

### New Patterns (64 → 66)
- `ml-009-overlap-leakage` - train/test data overlap (SMOTE before split, duplicates)
- `ml-010-multi-test-leakage` - test set used for tuning/model selection

### Pipeline Features
- **Incremental saving** - analysis and verification results saved as each file completes
- **Resume support** (`--resume`) - continue interrupted runs
- **Pattern filtering** (`--patterns`) - analyze specific patterns only
- **Report filtering** (`--verified-only`) - show only verified findings
- **Valid findings sample** - curated export of verified critical findings

### Infrastructure
- **Python 3.13 type syntax** - `X | None`, `StrEnum`, `UTC` constant
- **Direct async API** - no subprocess overhead, shared semaphore for vLLM
- **Timeout fix** - per-pattern timeout inside semaphore (queue wait excluded)
- Renamed `--pilot` to `--papers`

### Tests
- 49 tests for real_world_demo modules

## [0.1.5] - 2026-03-10

### Added
- **Semantic validation script** (`semantic_validate.py`) - batch pattern review using Claude Code CLI
- **Documentation grounding** - new `references` field in pattern.toml linking to official docs, with URL fetching/caching in deterministic validator
- **Doc caching** with two-stage cleaning pipeline for fetched reference documentation
- Extended test coverage

### Changed
- **Architecture principle**: Patterns must be grounded in official documentation
- **Improvement loop restructured**: Per-pattern fix cycle, `--fetch-refs` and `--clean-cache` options
- **Semantic reviewer**: Added documentation alignment check
- **Concurrency control**: Added semaphore to linter (150 limit)
- **Eval runner refactored**: Direct async linter API instead of subprocess calls

### Fixed
- Memory explosion in semantic validation from sub-agent spawning

## [0.1.4] - 2026-03-09

### Changed
- **Performance optimization**: max_completion_tokens to 4096, concurrency to 150
- **GPU memory utilization**: reduced to 0.85 for stability
- **Faster execution**: single pattern ~35s (was ~50s), full scan ~75s (was ~90s)

### Added
- Development Approach section in README (GenAI-native development methodology)
- Speed benchmark documentation with vLLM dashboard screenshot

## [0.1.3] - 2026-03-08

### Added
- **Deterministic pattern validation** (`pattern_verification/deterministic/validate.py`) - 9 structural checks for pattern quality
- **Dependency Health Checker** (`scripts/check_dependencies.py`) - security auditing with pip-audit, safety, bandit, and deprecation warning capture
- **vLLM Monitoring Dashboard** (`tools/vllm_dashboard.py`) - Streamlit real-time metrics (requests, throughput, KV cache, GPU utilization) with time-series charts
- **Alignment Metrics** in evaluation - tracks agreement between direct metrics and LLM judge, highlights divergent cases
- **Dynamic integration evaluation** (`evals/integration/dynamic_eval.py`) - generates fresh test code via Claude to avoid overfitting
- **Improvement Insights notepad** (`docs_dev_genai/IMPROVEMENT_INSIGHTS.md`) - persistent learnings across sessions
- **Release automation** - automates package build and GitHub release
- LLM reasoning output in verbose mode (`--verbose`)
- Three-tier evaluation methodology docs (pattern-specific → static integration → dynamic integration)

### Changed
- **Expanded pattern library to 64 patterns** (from 44 in v0.1.0)
- **Increased minimum test samples** to 3 positive and 3 negative per pattern for better eval coverage
- **Switched to thinking model** (Gemma 3 27B) for improved detection accuracy
- Consolidated evaluation scripts: merged `run_eval_llm_judge.py` into `run_eval.py` (use `--skip-judge` for fast eval)
- Enhanced improvement loop with insights step and debug-with-reasoning step
- Updated CONTRIBUTING.md with dependency groups and security checks

### Dependencies
- Added to `[dev]`: pip-audit, bandit, safety
- Added to `[dashboard]`: nvidia-ml-py, altair

### Fixed
- **pt-007-inference-without-eval**: counting approach for reliable detection (R: 0→1.0)
- **ml-007-test-set-preprocessing**: explicit test data variable identification
- **np-002-broadcasting-shape-mismatch**: semantic dimension mismatch focus (P: 0.33→1.0)

## [0.1.2] - 2026-03-06

### Added
- Continuous Improvement Loop documentation (`docs_dev_genai/CONTINUOUS_IMPROVEMENT.md`)
  - Structured workflow: evaluate → identify failures → analyze with agent → validate → re-evaluate
  - Clear criteria for pattern quality (precision ≥ 0.90, recall ≥ 0.80)
  - Mandatory rule: every pattern change requires running pattern-reviewer agent
  - Documentation for improving the agent itself when needed
  - Critical rules that must never be broken (data leakage detection, etc.)

### Changed
- Added brief reference to continuous improvement workflow in contributor docs

### Fixed
- **rep-001-incomplete-random-seeds**: Added proper test case for `train_test_split()` missing `random_state` (previous test cases were checking unrelated incomplete seeding issue)
- **pt-001-missing-train-mode**: Simplified detection question to improve recall (P=0→1.0, R=0→0.5)
- **pt-007-inference-without-eval**: Focused detection question on missing `model.eval()` in inference code
- **ml-007-test-set-preprocessing**: Made test data variable identification more explicit in detection question

## [0.1.1] - 2026-03-06

### Added
- New pattern: `pt-011-unscaled-gradient-accumulation` - detects gradient accumulation without proper loss scaling
- New pattern: `perf-002-array-allocation-in-loop` - detects inefficient array allocation inside loops
- New pattern: `perf-005-unnecessary-array-copy` - detects redundant array copies
- Project statistics generation script (`scripts/project_stats_generate.py`)
- Mandatory data leakage check in pattern reviewer agent

### Changed
- Improved code quality: ruff and mypy compliance across codebase
- Completed pattern descriptions (filled in missing documentation)
- Clarified architecture documentation for constrained-capacity model approach
- Documentation consistency improvements across the project
- Refined pattern reviewer agent prompts and workflow
- Reorganized pattern test file structure for consistency
- Updated evaluation framework with improved metrics handling

### Removed
- Removed pattern: `pt-002-missing-eval-mode` (merged functionality into `pt-007-inference-without-eval`)

## [0.1.0] - 2026-03-04

Initial public release.

### Features
- AI-powered detection of scientific Python code issues
- 44 detection patterns across 6 categories:
  - ML correctness and data leakage
  - PyTorch training bugs
  - Numerical precision issues
  - Reproducibility problems
  - Performance anti-patterns
  - Parallelization issues
- Local LLM integration via vLLM (Gemma 3 12B)
- Command-line interface with JSON and text output
- Support for Python scripts and Jupyter notebooks
- Evaluation framework with precision/recall metrics
- Designed for both human developers and AI coding agents

[0.3.0]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.3.0
[0.2.1]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.2.1
[0.2.0]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.2.0
[0.1.6]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.1.6
[0.1.5]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.1.5
[0.1.4]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.1.4
[0.1.3]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.1.3
[0.1.2]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.1.2
[0.1.1]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.1.1
[0.1.0]: https://github.com/authentic-research-partners/scicode-lint/releases/tag/v0.1.0
