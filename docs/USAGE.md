# scicode-lint Usage Guide

Single guide for both human users and AI coding agents. CLI and Python API are covered side-by-side.

## Quick Start

```bash
# 1. Install
pip install scicode-lint

# 2. Start vLLM (local GPU)
sudo apt install podman nvidia-container-toolkit   # one-time
scicode-lint vllm-server start                     # first run downloads ~8GB
scicode-lint vllm-server monitor                   # live metrics

# 3. Check a file
scicode-lint lint myfile.py
```

```python
# Python API equivalent
from pathlib import Path
from scicode_lint import SciCodeLinter

linter = SciCodeLinter()
result = linter.check_file(Path("myfile.py"))
for f in result.findings:
    print(f"{f.id} @ {f.location.name}: {f.explanation}")
```

See [INSTALLATION.md](../INSTALLATION.md) for detailed setup and [VRAM_REQUIREMENTS.md](VRAM_REQUIREMENTS.md) for hardware requirements. For programmatic server control (auto-start/stop in workflows), see [VLLM_UTILITIES.md](VLLM_UTILITIES.md).

---

## vLLM Server Lifecycle

```bash
scicode-lint vllm-server start     # start container with GPU passthrough
scicode-lint vllm-server stop      # stop (preserves container)
scicode-lint vllm-server restart   # recreate and start
scicode-lint vllm-server status    # container + API + VRAM state
scicode-lint vllm-server monitor   # live rich UI (Ctrl+C to exit)
scicode-lint vllm-server logs      # tail container logs
scicode-lint vllm-server rm        # remove container
```

### Remote vLLM (institutional server)

```bash
scicode-lint lint myfile.py --vllm-url https://vllm.your-institution.edu
```

---

## Commands

| Command | Purpose |
|---|---|
| `lint` | Check specific files or directories (most common) |
| `filter-repo` | Find self-contained ML files in a repo (no linting) |
| `analyze` | Full pipeline for a repo: clone → filter → lint |

### `lint`

```bash
scicode-lint lint myfile.py                      # single file
scicode-lint lint file1.py file2.py notebook.ipynb
scicode-lint lint src/                           # recursive directory
scicode-lint lint myfile.py --severity critical,high
scicode-lint lint myfile.py --min-confidence 0.85
scicode-lint lint myfile.py --category ai-training,ai-inference
scicode-lint lint myfile.py --pattern ml-001,pt-001
scicode-lint lint myfile.py --format json        # machine-readable (for agents/CI)
```

**Exit codes** (linter convention — same as ruff, mypy, bandit, shellcheck):

| Code | Meaning              | When                                     |
|------|----------------------|------------------------------------------|
| `0`  | Clean, no findings   | All checked files passed all patterns    |
| `1`  | Findings detected    | One or more patterns matched             |
| `2`  | Runtime/tool error   | vLLM unreachable, file parse failure, etc. |

The convention treats scicode-lint as a **gate, not a search tool**. In CI,
exit `1` means "fix the code"; exit `2` means "tool/infra is broken — retry or
fix config". Exit `1` and `2` both fail the build; distinguishing them lets
pipelines branch.

```bash
# Typical CI usage — fail on any finding or tool error:
scicode-lint lint src/

# Allow findings but fail only on tool errors:
scicode-lint lint src/
case $? in
  0|1) echo "lint ran successfully" ;;
  2)   echo "tool error — check vLLM / config"; exit 1 ;;
esac
```

The Python API doesn't exit — it returns `LintResult` objects. Inspect
`result.error` (a `LintError` or `None`) to distinguish the three states
programmatically; see the [Python API](#python-api) section.

### `analyze`

Clone (or use local) → classify files → lint self-contained ones:

```bash
scicode-lint analyze https://github.com/user/ml-project
scicode-lint analyze ./my_ml_project
scicode-lint analyze https://github.com/user/repo --keep-clone --clone-dir ./repos/my-repo
scicode-lint analyze https://github.com/user/repo --format json > results.json
```

Concurrency:

| Phase | Flag | Default | Description |
|---|---|---|---|
| 1. Filter | `--filter-concurrency` | 50 | Concurrent LLM calls for file classification |
| 2. Lint | `--lint-concurrency` | 100 | Concurrent pattern checks per file |

### `filter-repo`

Find self-contained ML files without linting:

```bash
scicode-lint filter-repo ./my_project
scicode-lint filter-repo ./my_project -o scan.json --format json
scicode-lint filter-repo ./repo --save-to-db                # SQLite cache
scicode-lint filter-repo ./repo --include-uncertain
```

Classification tiers: `self_contained` (complete workflow), `fragment` (partial, needs other files), `uncertain` (can't determine).

Two-stage workflow:

```bash
scicode-lint filter-repo ./repo -o ml_files.json --format json
cat ml_files.json | jq -r '.files[].filepath' | xargs scicode-lint lint
```

---

## Python API

### Basic

```python
from pathlib import Path
from scicode_lint import SciCodeLinter

linter = SciCodeLinter()
result = linter.check_file(Path("myfile.py"))

for f in result.findings:
    print(f"{f.severity} | {f.id} | {f.location.name}")
    print(f"  Problem: {f.explanation}")
    print(f"  Code:    {f.location.snippet}")
```

### Configuration

```python
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig, LLMConfig, Severity

config = LinterConfig(
    llm_config=LLMConfig(
        base_url="http://localhost:5001",             # auto-detects if empty
        model="RedHatAI/Qwen3-8B-FP8-dynamic",        # auto-detects if empty
    ),
    min_confidence=0.7,
    enabled_severities={Severity.CRITICAL, Severity.HIGH},
    enabled_categories={"ai-training", "ai-inference"},
    enabled_patterns={"ml-001", "pt-001"},            # single-pattern check is ~4-5s
)
linter = SciCodeLinter(config)
```

### Result structure

```python
result.file:     Path                # file that was checked
result.findings: list[Finding]       # issues found
result.summary:  dict                # {"total_findings", "by_severity", "by_category"}

# Finding tells you WHICH pattern failed, WHAT it means, WHERE, and EXACT CODE
finding.id:                     str    # pattern ID, e.g. "ml-001"
finding.category:               str    # "ai-training", "ai-inference", ...
finding.severity:               str    # "critical", "high", "medium"
finding.explanation:            str    # what's wrong + how to fix
finding.confidence:             float  # 0.0-1.0
finding.location.location_type: str    # "function" | "method" | "class" | "module"
finding.location.name:          str    # qualified name, e.g. "Trainer.train"
finding.location.lines:         list[int]  # full range of the function/method
finding.location.focus_line:    int    # single line to focus on
finding.location.snippet:       str    # exact code snippet
```

### Pattern lookup (what does this ID mean?)

```python
linter = SciCodeLinter()

# By ID
pattern = linter.get_pattern("ml-001")
print(pattern.category, pattern.severity)
print(pattern.detection_question)
print(pattern.warning_message)

# List all
for p in linter.list_patterns():
    print(f"{p.id} ({p.severity}): {p.warning_message[:60]}")
```

More advanced catalog queries (by category, by severity): see [API_REFERENCE.md](API_REFERENCE.md).

---

## Targeted Checking (Recommended During Development)

When fixing a specific issue, restrict to the relevant patterns so each check is seconds, not minutes:

```python
config = LinterConfig(enabled_patterns={"ml-001"})          # ~4-5s per small file
config = LinterConfig(enabled_patterns={"pt-001", "pt-002"})
config = LinterConfig(enabled_categories={"ai-training"})   # 19 patterns
```

CLI equivalent:

```bash
scicode-lint lint pipeline.py --pattern ml-001
scicode-lint lint train.py --pattern pt-001,pt-002,pt-003
scicode-lint lint pipeline.py --category ai-training
```

Run a full scan before final commit.

---

## AI Agent Workflow

```python
from pathlib import Path
from scicode_lint import SciCodeLinter
from scicode_lint.config import LinterConfig

def fix_ml_pipeline(file_path: str) -> None:
    # 1. Check ONLY the category you're working on
    linter = SciCodeLinter(LinterConfig(enabled_categories={"ai-training"}))
    result = linter.check_file(Path(file_path))
    if not result.findings:
        print(f"✓ {file_path}: no ML correctness issues")
        return

    # 2. Read each finding and fix
    for f in result.findings:
        print(f"\nFixing {f.id} in {f.location.name}: {f.explanation}")
        if f.id == "ml-001":
            fix_scaler_leakage(file_path, f)
        elif f.id == "ml-004":
            fix_metric(file_path, f)

    # 3. Verify
    result = linter.check_file(Path(file_path))
    print("✓ all fixed" if not result.findings else f"⚠ {len(result.findings)} remaining")
```

For auto-started vLLM inside a workflow:

```python
from scicode_lint.vllm import VLLMServer

with VLLMServer():                               # auto-start
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
    for f in result.findings:
        apply_fix(f)
# server auto-stopped
```

Programmatic server management details: [VLLM_UTILITIES.md](VLLM_UTILITIES.md).

---

## Detection Categories (66 patterns)

| Category | Patterns | Examples |
|---|---|---|
| **ai-training** | 19 | Data leakage, missing `zero_grad`, gradient issues |
| **ai-inference** | 12 | Missing `eval`, missing `no_grad`, device mismatch |
| **scientific-numerical** | 10 | Float equality, int overflow, catastrophic cancellation |
| **scientific-performance** | 11 | Loops vs vectorization, memory waste |
| **scientific-reproducibility** | 14 | Missing seeds, CUDA non-determinism |

Full list and detection questions: `src/scicode_lint/patterns/`.

---

## Common Patterns & Fixes

### `ml-001` — Data Leakage (Scaler on Full Data)

```python
# Bad: scaler fit on full data before split
scaler.fit_transform(X)
X_train, X_test = train_test_split(X)

# Good: fit only on training
X_train, X_test = train_test_split(X)
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)       # transform, not fit_transform
```

### `pt-001` — Missing `zero_grad()`

```python
# Bad: gradients accumulate
for batch in dataloader:
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()

# Good: clear before backward
for batch in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(batch), targets)
    loss.backward()
    optimizer.step()
```

### `pt-003` — Missing `torch.no_grad()`

```python
# Bad: autograd still active in eval
model.eval()
predictions = model(X_test)

# Good: disable autograd for inference
model.eval()
with torch.no_grad():
    predictions = model(X_test)
```

More fixes: each pattern's `pattern.toml` in `src/scicode_lint/patterns/` has a `warning_message` with domain-specific guidance.

---

## Output Formats

### Text (default)

Human-readable with severity indicators.

### JSON

```bash
scicode-lint lint myfile.py --format json
```

```json
[
  {
    "file": "myfile.py",
    "findings": [
      {
        "id": "ml-001",
        "category": "ai-training",
        "severity": "critical",
        "location": {
          "type": "function",
          "name": "preprocess_data",
          "snippet": "scaler.fit_transform(X)"
        },
        "explanation": "Data leakage: scaler fit on full data ...",
        "confidence": 0.92
      }
    ],
    "summary": {
      "total_findings": 1,
      "by_severity": {"critical": 1},
      "by_category": {"ai-training": 1}
    }
  }
]
```

### Errors in JSON output

Per-file errors always surface as structured `error` objects alongside findings — no flag needed. The runtime still exits `2` when any file errors (see [Exit codes](#lint)).

```json
[
  {
    "file": "large_file.py",
    "findings": [],
    "error": {
      "error_type": "ContextLengthError",
      "message": "File too large for context window\n...",
      "details": {
        "estimated_tokens": 12000,
        "max_tokens": 8000,
        "suggestions": ["Split into smaller files", "..."]
      }
    }
  }
]
```

---

## Error Handling

scicode-lint exposes a typed exception hierarchy rooted at `SciCodeLintError`. Catching the base class covers every documented failure mode.

```python
from pathlib import Path

from scicode_lint import SciCodeLinter
from scicode_lint.exceptions import LLMConnectionError, SciCodeLintError
from scicode_lint.linter import NotebookParseError
from scicode_lint.llm.client import MissingLocationError
from scicode_lint.llm.exceptions import ContextLengthError

try:
    linter = SciCodeLinter()
    result = linter.check_file(Path("myfile.py"))
except LLMConnectionError:
    print("vLLM not running — run: scicode-lint vllm-server start")
except ContextLengthError as e:
    print(f"File too large: see e.to_dict() for structured suggestions: {e.to_dict()}")
except NotebookParseError:
    print("Notebook is malformed — check it opens in Jupyter")
except SciCodeLintError as e:
    # Catch-all for any other documented linter failure
    print(f"{type(e).__name__}: {e}")
```

| Exception                | Raised when                                       |
|--------------------------|---------------------------------------------------|
| `LLMConnectionError`     | vLLM server unreachable — at startup, during filter/classify, or mid-request (server dies between calls). Triggers CLI exit code `2`. |
| `ContextLengthError`     | Input exceeds model context window (structured `to_dict()`) |
| `NotebookParseError`     | Jupyter notebook cannot be parsed                 |
| `MissingLocationError`   | LLM returned `detected=yes` but no location (internal; surfaces only if retries exhaust) |
| `SciCodeLintError`       | Base class — catch this to cover all of the above |

| Error | Solution |
|---|---|
| `Connection refused` | Start vLLM: `scicode-lint vllm-server start` |
| `Container runtime missing` | `sudo apt install podman nvidia-container-toolkit` |
| `Model not found` | Downloads on first run; retry after wait |
| `Timeout` | Increase timeout, or reduce concurrent patterns |
| Out of memory | Lower `gpu_memory_utilization` in `config.toml` |
| `File too large for context window` | See "File too large" below |

### "File too large for context window"

```
File too large for context window
  File: large_module.py
  Estimated tokens: 10,000
  Context limit:   8,000
  Overflow:        2,000 tokens

Suggestions:
  • Split into smaller files (< 8,000 tokens)
  • Focus linting on specific functions/classes
  • Increase max_model_len when starting vLLM server
  • Use a model with larger context window
```

Options:

1. **Split the file** into focused modules (~2K–3K tokens each).
2. **Raise `max_model_len`** on the server: already 40K by default; can go higher at a VRAM cost.
3. **Env override:** `export SCICODE_LINT_MAX_MODEL_LEN=40000`.

Estimate tokens before checking:

```python
from scicode_lint.llm import estimate_tokens
print(estimate_tokens(Path("myfile.py").read_text()))
```

---

## Configuration

### Environment variables

- `OPENAI_BASE_URL` — override vLLM URL
- `SCICODE_LINT_TEMPERATURE` — sampling temperature
- `SCICODE_LINT_TIMEOUT` — per-call timeout in seconds
- `SCICODE_LINT_MAX_MODEL_LEN` — context-window override

### Config file

`~/.config/scicode-lint/config.toml`:

```toml
[llm]
# base_url = "http://localhost:5001"   # optional, auto-detects
temperature = 0.3

[linter]
min_confidence = 0.7
enabled_severities = ["critical", "high"]
```

---

## Performance

### Full scan (all 66 patterns)

| File size | Typical time | Notes |
|---|---|---|
| Small (<100 lines) | ~50-60s | ~15% of input budget |
| Medium (~200 lines) | ~90s | ~30% |
| Large (~500 lines) | ~115s | ~50% |
| Max (~1000 lines) | ~170s | ~90% |

### Single pattern

| File size | Typical time |
|---|---|
| Small (<100 lines) | ~15s |
| Medium (~200 lines) | ~24s |
| Large (~500 lines) | ~30s |
| Max (~1000 lines) | ~44s |

Full scan is **not** 66× single-pattern: vLLM prefix caching means the code is cached after the first pattern, so subsequent patterns are cheap. Scaling is sub-linear (31× more lines ≈ 3× more time).

Tuning:
- `--severity critical` for faster checks
- `--category` / `--pattern` for targeted analysis
- Lower `--lint-concurrency` if KV cache thrashes (default 100)

Hardware: benchmarked on NVIDIA RTX 4000 Ada (20GB VRAM) with Qwen3-8B-FP8 @ 40K context. Minimum: 16GB VRAM with native FP8 (compute capability ≥ 8.9).

More: [performance/BENCHMARKING.md](performance/BENCHMARKING.md), [performance/CONCURRENCY_GUIDE.md](performance/CONCURRENCY_GUIDE.md).

---

## CI/CD

```bash
# Exit code is 1 if issues found — fails the build
scicode-lint lint src/ --severity critical --format json > findings.json
```

Full scan per commit is expensive; typical patterns:
- **PR gate:** `--severity critical` only (fast)
- **Nightly:** full scan, archive `findings.json` as artifact
- **Release:** full scan + `--min-confidence 0.85`

---

## Limitations

1. **Single-file analysis** — cross-file issues not detected
2. **Name-based locations** — linter reports function/class names; AST resolves to line ranges post-hoc (more reliable than direct LLM line numbers, but `focus_line` is only a best guess within the function)
3. **False positives possible** — always review findings
4. **Requires running vLLM** — local container or remote server
5. **Speed** — ~50s (small files) to ~170s (1000-line files) for full scan
6. **File size limit** — default ~1,400 lines max at 16K input budget (covers 90–95th percentile of Python files in the wild). Thinking tokens are output, not input, so they don't reduce code space.

---

## Further Reading

- [API_REFERENCE.md](API_REFERENCE.md) — complete class and method reference
- [VLLM_UTILITIES.md](VLLM_UTILITIES.md) — programmatic vLLM lifecycle control
- [PATTERN_LOOKUP_EXAMPLE.md](PATTERN_LOOKUP_EXAMPLE.md) — catalog query patterns
- [MODEL_SELECTION.md](MODEL_SELECTION.md) — which model/quantization to run
- [dev/ARCHITECTURE.md](dev/ARCHITECTURE.md) — core design principles
