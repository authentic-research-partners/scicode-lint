# scicode-lint Usage Guide

## Installation

```bash
pip install -e .
```

## Prerequisites

The linter requires a vLLM server. You can run locally or connect to a remote server.

### Start vLLM Server (Local)

```bash
# Install vLLM server
pip install scicode-lint[vllm-server]

# Start vLLM server (auto-detects GPU, validates FP8 support)
scicode-lint vllm-server start

# Or run in background
nohup scicode-lint vllm-server start > /tmp/vllm.log 2>&1 &

# Restart with different settings
scicode-lint vllm-server start --restart
```

The script auto-detects GPU capabilities and configures optimal settings. First run downloads the model (~8GB).

### Use Remote vLLM Server

```bash
scicode-lint lint my_code.py --vllm-url https://vllm.your-institution.edu
```

### Optional: Monitoring Dashboard

```bash
bash tools/start_dashboard.sh
# Opens at http://localhost:8501
```

See [INSTALLATION.md](../INSTALLATION.md) for detailed setup and [VRAM_REQUIREMENTS.md](VRAM_REQUIREMENTS.md) for hardware requirements.

## Commands

scicode-lint provides three main commands:

- **`lint`** - Lint specific files for issues (most common)
- **`filter-repo`** - Find self-contained ML files in a repository (filter only)
- **`analyze`** - Full pipeline for repositories: clone → filter → lint

## Analyze a Repository

The `analyze` command runs the full pipeline: clone → filter → lint.

### Analyze a GitHub repo

```bash
scicode-lint analyze https://github.com/user/ml-project
```

### Analyze a local repo

```bash
scicode-lint analyze ./my_ml_project
```

### Output example

```
Repository: https://github.com/user/ml-project

=== Scan Results ===
Total files: 45
Passed ML import filter: 16
Self-contained: 3
Fragments: 13

=== Analysis Results ===
Files analyzed: 3
Total findings: 5

[findings listed here]

Scan time: 8.23s
Total time: 45.67s
```

### Keep the cloned repo

```bash
scicode-lint analyze https://github.com/user/repo --keep-clone --clone-dir ./repos/my-repo
```

### JSON output

```bash
scicode-lint analyze https://github.com/user/repo --format json > results.json
```

JSON includes scan summary, findings, and timing:

```json
{
  "repo": "https://github.com/user/repo",
  "scan": {"summary": {...}, "files": [...]},
  "findings": [...],
  "summary": {
    "total_files_scanned": 45,
    "self_contained_files": 3,
    "files_analyzed": 3,
    "total_findings": 5,
    "scan_time_seconds": 8.23,
    "total_time_seconds": 45.67
  }
}
```

### Concurrency control

The `analyze` command runs two phases, each with configurable concurrency:

| Phase | Flag | Default | Description |
|-------|------|---------|-------------|
| 1. Filter | `--filter-concurrency` | 50 | Concurrent LLM calls for file classification |
| 2. Lint | `--lint-concurrency` | 150 | Concurrent pattern checks per file |

```bash
# Reduce concurrency if vLLM server is under load
scicode-lint analyze ./repo --filter-concurrency 20 --lint-concurrency 50
```

**Note:** Files are processed sequentially in Phase 2, but patterns within each file are checked concurrently.

## Lint Specific Files

### Check a single file

```bash
scicode-lint lint myfile.py
```

or

```bash
python -m scicode_lint lint myfile.py
```

### Check multiple files

```bash
scicode-lint lint file1.py file2.py notebook.ipynb
```

### Check a directory (recursive)

```bash
scicode-lint lint src/
```

## Output Formats

### Text output (default)

Human-readable output with severity indicators.

### JSON output

Use `--format json` for machine-parseable output.

### JSON output with error reporting

For GenAI agents and automated workflows, use `--json-errors` to include errors in structured format:

```bash
scicode-lint lint myfile.py --format json --json-errors
```

When errors occur (e.g., file too large), they're included in the output:

```json
[
  {
    "file": "large_file.py",
    "findings": [],
    "error": {
      "file": "large_file.py",
      "error_type": "ContextLengthError",
      "message": "File too large for context window\n  File: large_file.py\n  ...",
      "details": {
        "error": "ContextLengthError",
        "file_path": "large_file.py",
        "estimated_tokens": 12000,
        "max_tokens": 8000,
        "overflow": 4000,
        "suggestions": [
          "Split into smaller files (< 8,000 tokens)",
          "Focus linting on specific functions/classes",
          "Increase max_model_len when starting vLLM server",
          "Use a model with larger context window"
        ]
      }
    },
    "summary": {
      "total_findings": 0,
      "by_severity": {},
      "by_category": {}
    }
  }
]
```

**Use cases:**
- **Automated pipelines**: Parse errors programmatically
- **GenAI agents**: Structured error handling in AI workflows
- **Logging systems**: Forward structured errors to monitoring tools

## Filtering

### By severity

```bash
# Only critical issues
scicode-lint lint myfile.py --severity critical

# Critical and high
scicode-lint lint myfile.py --severity critical,high

# All levels (default)
scicode-lint lint myfile.py --severity critical,high,medium
```

### By confidence

```bash
# Only high-confidence findings (default: 0.7)
scicode-lint lint myfile.py --min-confidence 0.85
```

## Repository Filtering (Pre-filter)

For large repositories, use `filter-repo` to find self-contained ML files before running detailed linting. This avoids false positives from partial code fragments.

### Basic scan

```bash
scicode-lint filter-repo ./my_ml_project
```

Output:
```
Scan completed in 12.34s

Total files: 45
Passed ML import filter: 16
Failed ML import filter: 29

After LLM classification:
  Self-contained: 3
  Fragments: 12
  Uncertain: 1

Self-contained ML files:
  train.py (confidence: 0.95)
  experiments/run_baseline.py (confidence: 0.88)
  notebooks/full_pipeline.ipynb (confidence: 0.92)
```

### Save results to JSON

```bash
scicode-lint filter-repo ./my_project -o scan_results.json --format json
```

### Two-stage workflow

```bash
# 1. Find self-contained files
scicode-lint filter-repo ./repo -o ml_files.json --format json

# 2. Run detailed analysis only on those files
cat ml_files.json | jq -r '.files[].filepath' | xargs scicode-lint lint
```

### How it works

1. **File extension filter**: Only `.py` and `.ipynb` files are scanned (README.md, requirements.txt, etc. are skipped)
2. **ML import filter** (deterministic, instant): Files without ML imports (`sklearn`, `torch`, `tensorflow`, `pandas`, etc.) are skipped - no LLM call needed
3. **LLM classification** (files that passed ML import filter): Classified as:
   - `self_contained`: Complete ML workflow (data → model → train → output)
   - `fragment`: Partial code (model definition only, utility functions, etc.)
   - `uncertain`: Cannot determine (dynamic imports, etc.)

### Options

| Option | Description |
|--------|-------------|
| `--format {text,json}` | Output format |
| `-o, --output FILE` | Save JSON results to file |
| `--include-uncertain` | Include uncertain files in results |
| `--filter-concurrency N` | Max concurrent LLM requests for filtering (default: 50) |
| `--save-to-db` | Store results to SQLite database |
| `--db-path PATH` | Path to SQLite database (implies `--save-to-db`) |
| `-v, --verbose` | Increase verbosity |

### Programmatic usage

Use the module directly in Python:

```python
import asyncio
from pathlib import Path
from scicode_lint.config import load_llm_config
from scicode_lint.llm.client import create_client
from scicode_lint.repo_filter import scan_repo_for_ml_files, filter_scan_results

# Create LLM client
llm_config = load_llm_config()
client = create_client(llm_config)

# Scan repository (returns ALL results)
summary = asyncio.run(scan_repo_for_ml_files(
    repo_path=Path("./my_ml_project"),
    llm_client=client,
    max_concurrent=50,
))

# Get stats
print(f"Total files: {summary.total_files}")
print(f"Passed ML import filter: {summary.passed_ml_import_filter}")
print(f"Skipped (too large): {summary.skipped_too_large}")
print(f"Self-contained: {summary.self_contained}")

# Filter for self-contained files (for display/further processing)
filtered = filter_scan_results(summary)  # or include_uncertain=True
for r in filtered:
    print(f"  {r.filepath}: {r.details.confidence:.2f}")

# Export to dict/JSON (contains all results)
import json
print(json.dumps(summary.to_dict(), indent=2))
```

### Database integration (optional)

For `real_world_demo` workflows, store results in SQLite:

```bash
# Store to default database (real_world_demo/data/analysis.db)
scicode-lint filter-repo ./repo --save-to-db

# Store to custom database
scicode-lint filter-repo ./repo --db-path ./my_analysis.db
```

Results are always returned via stdout (text or JSON). Database storage is additional.

## Server Configuration

### Default (auto-detect)

```bash
# Auto-detects vLLM on ports 5001 or 8000
scicode-lint lint myfile.py
```

### Custom vLLM server

```bash
# Specify custom URL and model
scicode-lint lint myfile.py \
  --vllm-url http://localhost:8000 \
  --model RedHatAI/Qwen3-8B-FP8-dynamic
```

## Exit Codes

- `0`: No issues found
- `1`: Issues found or error occurred

## Performance Tips

1. **vLLM with prefix caching** enables fast checking of multiple patterns on the same file
2. **Reduce severity levels** to check only critical/high issues if needed
3. **Increase confidence threshold** to reduce false positives

## Detection Categories

The linter checks 66 patterns across these categories:

- **ai-training** (19 patterns) - Data leakage, PyTorch training modes, gradient management
- **ai-inference** (12 patterns) - Missing eval mode, missing no_grad, device mismatches
- **scientific-numerical** (10 patterns) - Float comparison, dtype overflow, catastrophic cancellation
- **scientific-performance** (11 patterns) - Loops vs vectorization, memory inefficiency
- **scientific-reproducibility** (14 patterns) - Missing seeds, CUDA non-determinism

See the `patterns/` directory for complete list.

## Limitations

1. **Requires LLM**: vLLM must be running
2. **Speed**: Checking 66 patterns takes time (varies by file size and GPU)
3. **False positives possible**: Review all findings - the linter is conservative but not perfect
4. **Function/class level only**: Findings report function/class names, not exact line numbers
5. **Context length limits**: Files must fit within model's input context (16K tokens = ~1,500 lines)
   - Files that are too large will be skipped with a clear error message
   - See "File too large" error below for solutions

## Troubleshooting

### "Connection refused" error

Make sure vLLM is running:
```bash
vllm serve RedHatAI/Qwen3-8B-FP8-dynamic
```

### "Model not found" error

The model should download automatically on first run. If issues persist, pre-download using HuggingFace's `snapshot_download`.

### Slow performance

- Filter to only critical/high severity
- Use smaller context window if memory is limited

### "File too large for context window" error

The linter validates file size before sending to vLLM. If a file is too large, you'll see:

```
File too large for context window
  File: large_module.py
  Estimated tokens: 10,000
  Context limit: 8,000
  Overflow: 2,000 tokens

Suggestions:
  • Split into smaller files (< 8,000 tokens)
  • Focus linting on specific functions/classes
  • Increase max_model_len when starting vLLM server
  • Use a model with larger context window
```

**Solutions:**

1. **Split the file**: Break large modules into smaller, focused files
   ```bash
   # Good: Multiple focused files
   src/preprocessing.py   (2000 tokens)
   src/training.py       (3000 tokens)
   src/evaluation.py     (2000 tokens)
   ```

2. **Adjust context limit**: The default 20K context (16K input + 4K response) supports ~1,500 line files (90-95th percentile):
   ```bash
   # Standard: 20K total tokens (16K input + 4K response)
   vllm serve RedHatAI/Qwen3-8B-FP8-dynamic --max-model-len 20000
   ```

3. **Use environment variable**: Override context limit via config
   ```bash
   export SCICODE_LINT_MAX_MODEL_LEN=20000
   scicode-lint lint large_file.py
   ```

**Context window:**
- Standard: 20K tokens total (16K input + 4K response, 16GB VRAM, covers ~1,500 lines)

**Estimate tokens before checking:**
```python
from scicode_lint.llm import estimate_tokens
from pathlib import Path

tokens = estimate_tokens(Path("myfile.py").read_text())
print(f"Estimated tokens: {tokens:,}")
```

## Integration with CI/CD

```bash
# In GitHub Actions, GitLab CI, etc.
scicode-lint lint src/ --severity critical --format json > findings.json

# Exit code is 1 if issues found, so it will fail the build
```
