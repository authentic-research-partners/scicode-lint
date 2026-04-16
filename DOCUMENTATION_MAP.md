# Documentation Map

Quick guide to find the right documentation for your needs.

---

## For Human Users

**Getting Started:**
- [README.md](README.md) - Overview, quick start, features
- [INSTALLATION.md](INSTALLATION.md) - Detailed installation instructions
- [docs/USAGE.md](docs/USAGE.md) - Usage guide

**Setup Guides:**
- [INSTALLATION.md](INSTALLATION.md) - Installation and vLLM setup
- [docs/VRAM_REQUIREMENTS.md](docs/VRAM_REQUIREMENTS.md) - VRAM requirements and model selection
- [docs/MODEL_SELECTION.md](docs/MODEL_SELECTION.md) - Model selection guide

**Performance:**
- [docs/performance/BENCHMARKING.md](docs/performance/BENCHMARKING.md) - Benchmarking guide
- [docs/performance/CONCURRENCY_GUIDE.md](docs/performance/CONCURRENCY_GUIDE.md) - Concurrency optimization
- [docs/performance/VLLM_MONITORING.md](docs/performance/VLLM_MONITORING.md) - vLLM monitoring and dashboard

**Contributing:**
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## For GenAI Agents USING scicode-lint

**Use Case:** AI coding assistant helping scientist write code → uses scicode-lint to check/fix bugs

**Documentation:**
- **[docs/USAGE.md](docs/USAGE.md)** ⭐ START HERE
  - Complete guide for integrating scicode-lint
  - Python API and CLI usage
  - Understanding results
  - Targeted checking
  - Workflow examples
  - Common fixes

---

## For GenAI Agents WORKING ON scicode-lint

**Use Case:** AI assistant modifying/contributing to the scicode-lint codebase

**Documentation:**
- **[docs/dev/ARCHITECTURE.md](docs/dev/ARCHITECTURE.md)** - Design principles
- **[docs/dev/IMPLEMENTATION.md](docs/dev/IMPLEMENTATION.md)** - Technical implementation
- **[docs/dev/QUALITY_GATES.md](docs/dev/QUALITY_GATES.md)** - Validation layers and tensions

**Pattern verification:**
- **[pattern_verification/](pattern_verification/)** - Deterministic and semantic quality checks
  - `deterministic/validate.py` - 18 automated checks (no LLM needed)
  - `semantic/semantic_validate.py` - LLM-based consistency checking

**Pattern reviewer agent:**
- **[pattern_verification/pattern-reviewer/](pattern_verification/pattern-reviewer/)** - Read-only analysis agent (identifies issues)

---

## Directory Structure

```
/
├── README.md                       # Main readme (humans)
├── INSTALLATION.md                 # Setup guide (humans)
├── DOCUMENTATION_MAP.md            # This file
│
├── docs/                    # All public documentation (shipped to public repo)
│   ├── README.md                   # Public-facing README (→ root README in public repo)
│   ├── USAGE.md                    ⭐ Unified guide (CLI + Python API, humans + AI agents)
│   ├── API_REFERENCE.md            # Class/method reference
│   ├── VRAM_REQUIREMENTS.md        # Hardware requirements
│   ├── MODEL_SELECTION.md          # GPU / quantization choice
│   ├── VLLM_UTILITIES.md           # Programmatic vLLM control
│   ├── PATTERN_LOOKUP_EXAMPLE.md   # Pattern lookup API example
│   ├── performance/
│   │   ├── BENCHMARKING.md
│   │   ├── CONCURRENCY_GUIDE.md
│   │   └── VLLM_MONITORING.md
│   └── dev/                        # Contributor-facing (public per paper)
│       ├── ARCHITECTURE.md         # Core design principles (READ FIRST)
│       ├── DETECTION_ARCHITECTURE.md
│       ├── IMPLEMENTATION.md
│       ├── CONTINUOUS_IMPROVEMENT.md
│       ├── META_IMPROVEMENT_LOOP.md
│       ├── MODEL_USAGE.md          # Which LLM per pipeline step
│       └── QUALITY_GATES.md
│
├── pattern_verification/           # Pattern quality verification
│   ├── deterministic/validate.py   # 18 automated checks
│   ├── semantic/semantic_validate.py  # Batch validation script
│   └── pattern-reviewer/           # Read-only analysis agent
│
├── src/scicode_lint/patterns/      # Pattern definitions (bundled with package)
│   ├── README.md                  # Pattern guide (structure, format, detection question template)
│   └── {category}/{pattern}/       # Individual pattern directories
│
├── consolidated_results/            # Unified performance report
│   ├── README.md
│   ├── generate_consolidated_report.py  # Reads JSON + DB → single report
│   └── CONSOLIDATED_REPORT.md       # Generated output
│
├── evals/                          # Evaluation framework
│   ├── README.md                   # Pattern-specific evaluations
│   ├── run_eval.py                 # Eval runner (use --skip-judge for fast mode)
│   └── integration/                # Multi-pattern integration tests
│       ├── README.md
│       └── integration_eval.py     # Full pipeline (Generate → Verify → Lint → Judge)
│
├── benchmarks/                     # Performance benchmarks
│   └── max_tokens_experiment.py    # Token limit tuning
│
├── real_world_demo/                # Real-world validation demo
│   ├── README.md                   # Pipeline documentation
│   ├── sources/                    # Data source implementations
│   │   ├── papers_with_code/       # PapersWithCode repos
│   │   └── leakage_paper/          # Yang et al. ASE'22 notebooks
│   └── reports/                    # Generated findings reports
│
├── tools/                          # User-facing tools
│   ├── vllm_dashboard.py           # Streamlit monitoring dashboard
│   └── start_dashboard.sh          # Dashboard launcher script
│
└── scripts/                        # Maintenance and dev scripts
    ├── check_dependencies.py       # Security audit (pip-audit + safety + bandit)
    ├── project_stats_generate.py   # Project statistics generator (--help for usage)
    ├── regenerate_pinned_requirements.py  # Regenerate requirements-pinned.txt
    └── review_patterns.sh          # Pattern review helper
```

---

## Quick Links by Task

### "I want to use scicode-lint to check my code"
→ [README.md](README.md) → [INSTALLATION.md](INSTALLATION.md) → [docs/USAGE.md](docs/USAGE.md)

### "I'm a GenAI agent helping a scientist write code"
→ [docs/USAGE.md](docs/USAGE.md)

### "I want to contribute a new detection pattern"
→ [CONTRIBUTING.md](CONTRIBUTING.md) → [docs/dev/ARCHITECTURE.md](docs/dev/ARCHITECTURE.md)

### "I want to understand the architecture"
→ [docs/dev/ARCHITECTURE.md](docs/dev/ARCHITECTURE.md) → [docs/dev/IMPLEMENTATION.md](docs/dev/IMPLEMENTATION.md)

### "I want to optimize performance"
→ [docs/performance/CONCURRENCY_GUIDE.md](docs/performance/CONCURRENCY_GUIDE.md)

### "I want to monitor vLLM during evals"
→ [docs/performance/VLLM_MONITORING.md](docs/performance/VLLM_MONITORING.md)

### "I want to benchmark scicode-lint"
→ [docs/performance/BENCHMARKING.md](docs/performance/BENCHMARKING.md)

### "I want to review or improve pattern definitions"
→ [pattern_verification/README.md](pattern_verification/README.md) → [docs/dev/CONTINUOUS_IMPROVEMENT.md](docs/dev/CONTINUOUS_IMPROVEMENT.md)

### "I want to validate patterns on real-world code"
→ [docs/dev/META_IMPROVEMENT_LOOP.md](docs/dev/META_IMPROVEMENT_LOOP.md) → [real_world_demo/README.md](real_world_demo/README.md)

### "I want to understand validation layers and their tensions"
→ [docs/dev/QUALITY_GATES.md](docs/dev/QUALITY_GATES.md)

### "I want to run evaluations"
→ [evals/README.md](evals/README.md) (pattern-specific) → [evals/integration/README.md](evals/integration/README.md) (multi-pattern)

### "I want a consolidated performance report"
→ Run `python consolidated_results/generate_consolidated_report.py`

### "I need project statistics"
→ Run `python scripts/project_stats_generate.py --help`

### "I want to validate on real-world scientific ML papers"
→ [real_world_demo/README.md](real_world_demo/README.md)
