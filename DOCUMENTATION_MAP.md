# Documentation Map

Quick guide to find the right documentation for your needs.

---

## For Human Users

**Getting Started:**
- [README.md](README.md) - Overview, quick start, features
- [INSTALLATION.md](INSTALLATION.md) - Detailed installation instructions
- [docs_use_human/USAGE.md](docs_use_human/USAGE.md) - Usage guide

**Setup Guides:**
- [INSTALLATION.md](INSTALLATION.md) - Installation and vLLM setup
- [docs_use_human/VRAM_REQUIREMENTS.md](docs_use_human/VRAM_REQUIREMENTS.md) - VRAM requirements and model selection
- [docs_use_human/MODEL_SELECTION.md](docs_use_human/MODEL_SELECTION.md) - Model selection guide

**Performance:**
- [docs_use_human/performance/BENCHMARKING.md](docs_use_human/performance/BENCHMARKING.md) - Benchmarking guide
- [docs_use_human/performance/CONCURRENCY_GUIDE.md](docs_use_human/performance/CONCURRENCY_GUIDE.md) - Concurrency optimization
- [docs_use_human/performance/VLLM_MONITORING.md](docs_use_human/performance/VLLM_MONITORING.md) - vLLM monitoring and dashboard

**Contributing:**
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CHANGELOG.md](CHANGELOG.md) - Version history

---

## For GenAI Agents USING scicode-lint

**Use Case:** AI coding assistant helping scientist write code → uses scicode-lint to check/fix bugs

**Documentation:**
- **[docs_use_genai/GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md)** ⭐ START HERE
  - Complete guide for integrating scicode-lint
  - Python API and CLI usage
  - Understanding results
  - Targeted checking
  - Workflow examples
  - Common fixes

- [docs_use_genai/INTERFACE_ANALYSIS.md](docs_use_genai/INTERFACE_ANALYSIS.md) - Package interface analysis

---

## For GenAI Agents WORKING ON scicode-lint

**Use Case:** AI assistant modifying/contributing to the scicode-lint codebase

**Documentation:**
- **[docs_dev_genai/ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md)** - Design principles
- **[docs_dev_genai/IMPLEMENTATION.md](docs_dev_genai/IMPLEMENTATION.md)** - Technical implementation
- **[docs_dev_genai/QUALITY_GATES.md](docs_dev_genai/QUALITY_GATES.md)** - Validation layers and tensions
- [CLAUDE.md](CLAUDE.md) - Instructions for Claude Code CLI

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
├── docs_use_human/                 # Human user documentation
│   ├── USAGE.md
│   ├── VRAM_REQUIREMENTS.md
│   ├── MODEL_SELECTION.md
│   └── performance/
│       ├── BENCHMARKING.md
│       ├── CONCURRENCY_GUIDE.md
│       └── VLLM_MONITORING.md
│
├── docs_use_genai/               # GenAI agents USING scicode-lint
│   ├── README.md
│   ├── GENAI_AGENT_GUIDE.md       ⭐ Main guide
│   ├── API_REFERENCE.md
│   ├── INTERFACE_ANALYSIS.md
│   ├── VLLM_UTILITIES.md
│   └── PATTERN_LOOKUP_EXAMPLE.md
│
├── docs_dev_genai/                 # GenAI agents WORKING ON scicode-lint
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── DETECTION_ARCHITECTURE.md
│   ├── IMPLEMENTATION.md
│   ├── CONTINUOUS_IMPROVEMENT.md
│   ├── META_IMPROVEMENT_LOOP.md
│   ├── MODEL_USAGE.md
│   └── QUALITY_GATES.md
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
├── tools/                          # Development and monitoring tools
│   ├── vllm_dashboard.py           # Streamlit monitoring dashboard
│   └── start_dashboard.sh          # Dashboard launcher script
│
└── scripts/                        # Utility scripts
    └── project_stats_generate.py   # Project statistics generator (--help for usage)
```

---

## Quick Links by Task

### "I want to use scicode-lint to check my code"
→ [README.md](README.md) → [INSTALLATION.md](INSTALLATION.md) → [docs_use_human/USAGE.md](docs_use_human/USAGE.md)

### "I'm a GenAI agent helping a scientist write code"
→ [docs_use_genai/GENAI_AGENT_GUIDE.md](docs_use_genai/GENAI_AGENT_GUIDE.md)

### "I want to contribute a new detection pattern"
→ [CONTRIBUTING.md](CONTRIBUTING.md) → [docs_dev_genai/ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md)

### "I want to understand the architecture"
→ [docs_dev_genai/ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md) → [docs_dev_genai/IMPLEMENTATION.md](docs_dev_genai/IMPLEMENTATION.md)

### "I want to optimize performance"
→ [docs_use_human/performance/CONCURRENCY_GUIDE.md](docs_use_human/performance/CONCURRENCY_GUIDE.md)

### "I want to monitor vLLM during evals"
→ [docs_use_human/performance/VLLM_MONITORING.md](docs_use_human/performance/VLLM_MONITORING.md)

### "I want to benchmark scicode-lint"
→ [docs_use_human/performance/BENCHMARKING.md](docs_use_human/performance/BENCHMARKING.md)

### "I want to review or improve pattern definitions"
→ [pattern_verification/README.md](pattern_verification/README.md) → [docs_dev_genai/CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md)

### "I want to validate patterns on real-world code"
→ [docs_dev_genai/META_IMPROVEMENT_LOOP.md](docs_dev_genai/META_IMPROVEMENT_LOOP.md) → [real_world_demo/README.md](real_world_demo/README.md)

### "I want to understand validation layers and their tensions"
→ [docs_dev_genai/QUALITY_GATES.md](docs_dev_genai/QUALITY_GATES.md)

### "I want to run evaluations"
→ [evals/README.md](evals/README.md) (pattern-specific) → [evals/integration/README.md](evals/integration/README.md) (multi-pattern)

### "I want a consolidated performance report"
→ Run `python consolidated_results/generate_consolidated_report.py`

### "I need project statistics"
→ Run `python scripts/project_stats_generate.py --help`

### "I want to validate on real-world scientific ML papers"
→ [real_world_demo/README.md](real_world_demo/README.md)
