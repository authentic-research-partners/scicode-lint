# scicode-lint

[![arXiv](https://img.shields.io/badge/arXiv-2603.17893-b31b1b.svg)](https://arxiv.org/abs/2603.17893)

**AI-powered linter for ML code written for scientific applications. Catches methodology bugs that traditional linters miss: the kind that quietly invalidate your results.**

Runs locally on your GPU or institutional cluster. No cloud APIs, your code stays private, no unexpected GenAI bills.

---

## TL;DR

Local LLM linter for scientific ML code. Catches data leakage, missing seeds, numerical bugs. Runs on your GPU (16GB+ VRAM), code stays private.

```bash
uv tool install scicode-lint --python 3.13   # Recommended; uv installs Python 3.13 if needed

# Local vLLM also needs podman (or docker) + nvidia-container-toolkit
sudo apt install podman nvidia-container-toolkit
scicode-lint vllm-server start               # Starts vLLM in a container
scicode-lint lint train.py                   # Check a file for issues
```

---

## For GenAI Agents

scicode-lint is designed for programmatic use by AI coding agents. Two entry points, same structured data:

- **CLI, machine-readable:** `scicode-lint lint file.py --format json` emits a list of `LintResult` objects (findings + per-file errors). Exit codes follow linter convention — `0` clean, `1` findings, `2` tool/runtime error — so CI can branch on outcome. See [Exit codes](docs/USAGE.md#lint) and [Errors in JSON output](docs/USAGE.md#errors-in-json-output).
- **Python API:** `from scicode_lint import SciCodeLinter`, then `linter.check_file(path)` returns a typed `LintResult` (no subprocess). Catch `scicode_lint.exceptions.SciCodeLintError` for all documented failure modes (`LLMConnectionError`, `ContextLengthError`, `NotebookParseError`, …). See [Error Handling](docs/USAGE.md#error-handling).

Full reference: [USAGE.md](docs/USAGE.md) (CLI + Python API in one guide) and [API_REFERENCE.md](docs/API_REFERENCE.md).

---

## The Problem

Traditional linters catch syntax errors and style issues. They can't catch methodology bugs: the kind where code runs, tests pass, and results are still wrong. Data leakage. Missing random seeds. Silent numerical errors.

And it's getting harder with AI coding tools. These tools are trained on public repositories full of methodology mistakes: Kaggle notebooks with data leakage, tutorials that skip random seeds, Stack Overflow answers with broken cross-validation. The bugs transfer seamlessly into your codebase.

**Built for scientists applying ML** - biology, chemistry, physics, neuroscience, engineering, and beyond. You're an expert in your domain; catching ML methodology bugs isn't your day job. This tool fills that gap.

---

## What It Does

Scans Python scripts and Jupyter notebooks for 66 patterns across five categories:

- **ai-training** (19 patterns): data leakage, PyTorch training modes, gradient management, DataLoader configuration
- **ai-inference** (12 patterns): missing eval mode, missing no_grad, device mismatches, CUDA timing, JIT tracing
- **scientific-numerical** (10 patterns): float comparison, dtype overflow, catastrophic cancellation
- **scientific-performance** (11 patterns): loops vs vectorization, memory inefficiency
- **scientific-reproducibility** (14 patterns): missing seeds, CUDA non-determinism, unsorted iteration, pickle versioning

It tells you what's wrong and why it matters. No auto-fixes: you stay in control of your code.

### Example Output

```
test_scaler.py — 1 issue found

🔴 CRITICAL [method fit_transform] ml-001: Issue detected
   Data leakage: scaler is fit on full data including test set.
   Model performance will be inflated. Use sklearn.pipeline.Pipeline
   so fitting happens inside each fold.

   Code: X_scaled = scaler.fit_transform(X)
```

---

## Quick Start

### Prerequisites

- GPU with 16GB+ VRAM and native FP8 support (RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, L4, etc.)
- **Container runtime**: `podman` (or `docker`) + `nvidia-container-toolkit` — vLLM runs in a container with GPU passthrough. One-time install: `sudo apt install podman nvidia-container-toolkit`.
- Default model: `RedHatAI/Qwen3-8B-FP8-dynamic` (first run auto-downloads ~8GB into the container image).

Skip the container runtime if you're pointing at a remote vLLM server instead.

See [INSTALLATION.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/INSTALLATION.md) for detailed setup.

### Installation

```bash
# Recommended: uv tool install (fast, isolated, manages Python itself)
uv tool install scicode-lint --python 3.13

# Alternative (if you prefer pip or already have Python 3.13 set up)
pip install scicode-lint

# For local use: install container runtime (one-time, see Prerequisites above)
sudo apt install podman nvidia-container-toolkit

# For remote vLLM (e.g., university/institutional server):
scicode-lint lint my_code.py --vllm-url https://vllm.your-institution.edu
```

For development install or exact version reproducibility, see
[INSTALLATION.md → Development install](https://github.com/authentic-research-partners/scicode-lint/blob/main/INSTALLATION.md#development-install) and
[INSTALLATION.md → Reproducible environment](https://github.com/authentic-research-partners/scicode-lint/blob/main/INSTALLATION.md#reproducible-environment-requirements-pinnedtxt).

### Start vLLM Server

Before running scicode-lint, start the vLLM server (skip if using remote server):

```bash
# Start vLLM container (auto-detects podman or docker; first run downloads ~8GB)
scicode-lint vllm-server start

# Watch it come up, then live metrics once the model is loaded
scicode-lint vllm-server monitor
```

The container is persistent (`--restart unless-stopped`), so it survives reboots. Other lifecycle commands: `stop`, `restart`, `status`, `logs`, `rm`.

### Usage

```bash
# Check a file
scicode-lint lint train.py

# Check with specific pattern
scicode-lint lint my_pipeline.py --pattern ml-001

# Check Jupyter notebooks
scicode-lint lint analysis.ipynb

# Check by category
scicode-lint lint train.py --category ai-training

# Filter by severity
scicode-lint lint train.py --severity critical,high
```

### Analyze a Repository

For entire repos, the `analyze` command clones, filters, and lints automatically:

```bash
# Analyze a GitHub/GitLab repo
scicode-lint analyze https://github.com/user/ml-project

# Analyze a local repo
scicode-lint analyze ./my_ml_project
```

Two-stage filter (runs automatically):
1. **ML import presence** (instant) - skips files without sklearn/torch/tensorflow/etc.
2. **LLM classification** - identifies complete workflows vs code fragments

---

## Current Limitations

- **Single-file analysis only.** Issues that span multiple files (like preprocessing done differently in train.py and test.py) are out of scope for now.
- **Requires a GPU with 16GB+ VRAM.** Not practical for laptops or CPU-only setups. RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, or L4 are the target hardware.

---

## Project Status

**Work in Progress** (v0.3.1 alpha)

| Test Type | Precision | Recall | Description |
|-----------|-----------|--------|-------------|
| Controlled tests | 97.7% | 97.0% | Curated positive/negative test files per pattern (452 tests, 66 patterns) |
| Integration (LLM-generated) | 58.0% | 85.1% | 50 Sonnet-generated scenarios with 148 planted bugs; 27 bonus TPs found |
| Labeled Kaggle notebooks | 65% | 100% | Yang et al. ASE'22 dataset (`pre` label), human-labeled ground truth |
| Published papers (iteration) | 62.0% | - | 32 repos analyzed (120 files); used for pattern refinement |
| Published papers (holdout) | 54.1% | - | 17 repos analyzed (45 files); unseen during development |

Example reports: [`real_world_demo/output_examples/`](https://github.com/authentic-research-partners/scicode-lint/tree/main/real_world_demo/output_examples)

---

## How It Works

**Design philosophy: Middle ground between grep and SOTA cloud reasoning.**

Traditional linters use grep-style pattern matching - fast but misses context. Cloud AI APIs offer deep reasoning but cost money, raise privacy concerns, and models get deprecated. scicode-lint uses a **local LLM with thinking mode** (RedHatAI/Qwen3-8B-FP8-dynamic, fits in 16GB VRAM via vLLM) that sits between these extremes:

- **Smarter than grep**: Understands code structure, follows data flow, catches semantic issues
- **Reasoning capability**: Uses thinking mode to analyze code behavior and intent, not just literal text
- **Local and private**: Your code never leaves your machine, no API costs
- **Reproducible**: Open-source models remain available; results stay consistent over time

**How patterns run:** Each pattern is a focused detection question in a TOML file. All 66 patterns run concurrently - vLLM's prefix caching means your code is processed once and shared across all checks. Processing N patterns takes approximately the time of 1 pattern.

**Design goal:** Patterns are grounded in official documentation (PyTorch docs, scikit-learn guides, NumPy API references). See [ARCHITECTURE.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/docs/dev/ARCHITECTURE.md) for technical details.

---

## Documentation

- 📖 [DOCUMENTATION_MAP.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/DOCUMENTATION_MAP.md): find all documentation
- 🚀 [INSTALLATION.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/INSTALLATION.md): detailed setup guide
- 📘 [docs/USAGE.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/docs/USAGE.md): CLI + Python API usage guide (for humans and AI coding agents)
- 🏗️ [CONTRIBUTING.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/CONTRIBUTING.md): adding new patterns
- 📊 [evals/README.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/evals/README.md): pattern evaluation framework
- 🔬 [evals/integration/README.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/evals/integration/README.md): integration tests
- 📈 [docs/dev/CONTINUOUS_IMPROVEMENT.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/docs/dev/CONTINUOUS_IMPROVEMENT.md): pattern quality improvement workflow
- 📊 [docs/performance/VLLM_MONITORING.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/docs/performance/VLLM_MONITORING.md): vLLM monitoring dashboard
- 🌐 [real_world_demo/README.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/real_world_demo/README.md): real-world validation on scientific ML code from PapersWithCode repos and Yang et al. ASE'22 leakage paper (79% F1 on preprocessing leakage detection)

---

## Feedback Wanted

Early release. If you're a researcher applying ML to your domain:

- Which patterns are missing from your field?
- Which detections are too noisy to be useful?
- What would make this fit your actual workflow?

Open an issue or start a discussion on GitHub.

---

## Contributing

Each pattern lives in `src/scicode_lint/patterns/{category}/{id}/` and needs:
- `pattern.toml`: the detection question and warning message
- Test files: examples of buggy code (positive), correct code (negative), and edge cases (context-dependent)

See [CONTRIBUTING.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/CONTRIBUTING.md) for the full guide.

---

## Development Approach

GenAI-native development with Claude Code. Patterns are generated and iterated by AI agents within human-designed evaluation frameworks and quality gates.

- **Fast loop:** [CONTINUOUS_IMPROVEMENT.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/docs/dev/CONTINUOUS_IMPROVEMENT.md) - Iterate on synthetic test files
- **Meta loop:** [META_IMPROVEMENT_LOOP.md](https://github.com/authentic-research-partners/scicode-lint/blob/main/docs/dev/META_IMPROVEMENT_LOOP.md) - Validate on real Papers with Code + Claude verification

---

## Citation

If you use scicode-lint in your research, please cite the companion paper:

```bibtex
@misc{samsonau2026scicodelint,
  title        = {scicode-lint: Detecting Methodology Bugs in Scientific Python Code with LLM-Generated Patterns},
  author       = {Samsonau, Sergey V.},
  year         = {2026},
  eprint       = {2603.17893},
  archivePrefix= {arXiv},
  primaryClass = {cs.SE},
  url          = {https://arxiv.org/abs/2603.17893},
}
```

---

## License

MIT