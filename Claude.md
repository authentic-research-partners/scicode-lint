# scicode-lint

AI-powered linter for scientific Python code. Designed for both human users and GenAI coding agents.

Detects pitfalls across ML correctness, PyTorch, numerical precision, reproducibility, performance, and parallelization.

---

## Key Files

**Main files (root):**
- [README.md](./README.md) - Main file for humans
- [INSTALLATION.md](./INSTALLATION.md) - Detailed installation guide
- [DOCUMENTATION_MAP.md](./DOCUMENTATION_MAP.md) - Navigation guide to all documentation
- [Claude.md](./Claude.md) - This file, main instructions for AI agents working on the codebase
- [detection_catalog.yaml](./src/scicode_lint/detection_catalog.yaml) - Detection patterns catalog (44 patterns)

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
- [CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md) - Evaluation → improvement workflow
- [IMPLEMENTATION.md](docs_dev_genai/IMPLEMENTATION.md) - Technical implementation details
- [TOOLS.md](docs_dev_genai/TOOLS.md) - Development tools and agents

**Development tools (.claude/agents/):**
- [pattern-reviewer](.claude/agents/pattern-reviewer/) - Specialized agent for reviewing and improving pattern definitions

---

## Architecture Overview

**Core principle: Constrained-capacity local LLM by design.**

scicode-lint uses Gemma 3 12B - a deliberate choice between grep-style matching and expensive SOTA models. This enables local execution, privacy, and no API costs, but requires detection questions to be simple and direct.

**Two-phase design:**
- **Build time:** Design detection questions that constrained models can reliably answer
- **Runtime:** Local model (Gemma 3 12B via vLLM) runs focused checks against code

**Philosophy:** Detection only, no automatic fixes.

**⚠️ CRITICAL ARCHITECTURAL RULES:**
- Code MUST come before detection instructions in prompts (enables vLLM prefix caching)
- Detection questions must be ultra-simple (one search, one check, binary answer)
- No complex reasoning chains - if the model needs to "think," simplify the question

**📖 Complete architecture:** [ARCHITECTURE.md](docs_dev_genai/ARCHITECTURE.md)

---

## Local LLM Setup

**Required:** vLLM server with Gemma 3 12B FP8 model (20GB+ VRAM, native FP8 support required)

```bash
pip install scicode-lint[vllm-server]
vllm serve --model RedHatAI/gemma-3-12b-it-FP8-dynamic --trust-remote-code --max-model-len 16000
python -m scicode_lint check myfile.py
```

**📖 Complete setup:** [INSTALLATION.md](INSTALLATION.md)

---

## 🤖 Pattern Reviewer Agent

**Specialized Claude Code agent for reviewing and improving detection patterns**

**Location:** `.claude/agents/pattern-reviewer/`

**Usage:**
```bash
# Review single pattern
claude --agent pattern-reviewer "Review ml-001-scaler-leakage"

# Batch review (parallel processing)
claude --agent pattern-reviewer "Review ml-001, ml-002, ml-003 in parallel"

# Helper script
./scripts/review_patterns.sh review ml-001
./scripts/review_patterns.sh batch-review "ml-001 ml-002 ml-003"
```

**Capabilities:**
- Reviews pattern definitions for completeness and clarity
- Suggests improvements to detection questions and warnings
- Creates test cases (positive/negative/ambiguous)
- Supports batch operations (3-10 patterns in parallel)
- Iteratively improves patterns based on evaluation metrics

**Quality targets:**
- Precision ≥ 0.90 (minimize false positives)
- Recall ≥ 0.80 (catch most bugs)
- Critical severity precision ≥ 0.95 (very high confidence)

**Workflow:**
1. Review pattern: `claude --agent pattern-reviewer "Review <pattern-id>"`
2. Implement suggested fixes (update TOML, create test cases)
3. Evaluate: `python evals/run_eval.py --pattern <pattern-id>`
4. If metrics below target, iterate with agent

**📖 Complete documentation:**
- [Quick Start](.claude/agents/pattern-reviewer/QUICK_START.md) - 30 second start
- [README](.claude/agents/pattern-reviewer/README.md) - Full guide
- [Examples](.claude/agents/pattern-reviewer/examples.md) - Usage examples
- [Batch Operations](.claude/agents/pattern-reviewer/BATCH_OPERATIONS.md) - Parallel processing

---

## Critical Development Rules

### Code Quality
- **NEVER hide issues** - Investigate root cause, don't suppress without understanding
- **Known issues require references** - Cite specific GitHub issue/docs (verify it exists, no made-up citations)
- **NO LEGACY CODE** - Never create "legacy/", "old/", "deprecated/" directories - delete unused code completely
- **Dual audience** - Detection output must be clear for BOTH humans and GenAI agents
- Every detection pattern must have eval coverage

### Mandatory Checks (RUN AFTER EVERY CODE CHANGE)
```bash
ruff check . && ruff format .  # Code style
mypy src/                      # Type checking
pytest                         # Tests
```

### Type Hints & Documentation
- **All functions** must have type annotations (args + return)
- **All exported classes/methods** (in `__all__`) must have docstrings (Args, Returns, Examples)
- **Update API_REFERENCE.md** when adding public APIs
- Private methods (_prefixed) need type hints only

### Documentation Rules
- **Docs are snapshots** - Describe current state, not history (history lives in git)
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

## Project Structure & Module References

```
scicode-lint/
├── src/scicode_lint/              # Main linter code
│   ├── cli.py, linter.py         # CLI and orchestration
│   ├── detectors/                # Pattern loading and prompt generation
│   │   └── 📖 See: IMPLEMENTATION.md#detection-patterns
│   ├── llm/                      # vLLM client with structured output
│   │   └── 📖 See: IMPLEMENTATION.md#llm-integration, docs_use_genai/VLLM_UTILITIES.md
│   └── output/                   # Output formatting
│       └── 📖 See: IMPLEMENTATION.md#output-formatting
│
├── patterns/                      # Pattern definitions and test cases
│   ├── {category}/{pattern}/     # Test cases: test_positive/, test_negative/, test_context_dependent/
│   │   └── 📖 See: patterns/README.md, CONTRIBUTING.md#adding-patterns
│   └── _registry.toml            # Pattern registry
│
├── evals/                         # Evaluation framework (precision/recall)
│   ├── run_eval.py               # Hardcoded ground truth evaluation
│   ├── run_eval_llm_judge.py     # LLM-as-judge evaluation
│   └── integration/              # Multi-pattern integration tests
│       └── 📖 See: evals/README.md, evals/integration/README.md
│
├── docs_use_genai/               # AI agents USING scicode-lint
│   ├── GENAI_AGENT_GUIDE.md      # Complete usage guide
│   ├── API_REFERENCE.md          # Full API reference
│   └── VLLM_UTILITIES.md         # vLLM integration utilities
│
├── docs_dev_genai/               # AI agents WORKING ON scicode-lint
│   ├── ARCHITECTURE.md           # Design principles (READ FIRST)
│   ├── IMPLEMENTATION.md         # Technical implementation details
│   └── TOOLS.md                  # Development tools and workflows
│
└── .claude/agents/pattern-reviewer/  # Pattern review agent (see section above)
```

---

## Development Commands

```bash
# Run linter
python -m scicode_lint check myfile.py

# Tests & quality (RUN AFTER EVERY CODE CHANGE)
pytest
ruff check . && ruff format .
mypy src/

# Pattern evaluation
python evals/run_eval.py --pattern <pattern-id>
```

---

## AI Coding Agent Workflow

After writing code, **always** run:

1. **Document** - Add docstrings + type hints to public APIs, update API_REFERENCE.md if in `__all__`
2. **Quality** - `ruff check . && ruff format .` then `mypy src/`
3. **Test** - `pytest`
4. **Docs** - Update docs to match current state (no "updated"/"changed" language)
5. **Commit** - Only commit clean, type-safe, documented code

---

## Continuous Improvement Loop

**Core principle: Quality improvement is directly proportional to effort invested.**

Each cycle of evaluation → analysis → improvement advances pattern quality. The more cycles run, the better the results.

**⚠️ MANDATORY: Every pattern change requires running the pattern-reviewer agent.**

```bash
# After ANY pattern modification
claude --agent pattern-reviewer "Review <pattern-id>"
python evals/run_eval.py --pattern <pattern-id>   # Evaluate
ruff check . && mypy src/ && pytest               # Validate
```

**Targets:** Precision ≥ 0.90, Recall ≥ 0.80 (Critical: Precision ≥ 0.95)

**📖 Complete workflow:** [docs_dev_genai/CONTINUOUS_IMPROVEMENT.md](docs_dev_genai/CONTINUOUS_IMPROVEMENT.md)
