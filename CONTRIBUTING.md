# Contributing to scicode-lint

Thank you for your interest in contributing to scicode-lint! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Adding New Detection Rules](#adding-new-detection-rules)
- [Running Tests](#running-tests)
- [Code Quality](#code-quality)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

Be respectful, constructive, and professional in all interactions.

## Getting Started

Fork and clone the repository.

## Development Setup

### Install in editable mode

```bash
cd scicode-lint

# Install with dev dependencies only
pip install -e ".[dev]"

# Install with ALL optional dependencies (recommended for full development)
pip install -e ".[dev,eval,dashboard]"

# Verify installation
scicode-lint --help

# Enable pre-commit hooks (ruff + mypy)
git config core.hooksPath scripts
```

### Optional Dependency Groups

| Group | Purpose | Packages |
|-------|---------|----------|
| `dev` | Development tools | ruff, mypy, pytest, pip-audit, bandit, safety |
| `eval` | Evaluation framework | pytest, pytest-xdist |
| `dashboard` | vLLM monitoring dashboard | streamlit, pandas, altair, nvidia-ml-py |
| `vllm-server` | Local LLM server | vllm |

### Dependencies

All project dependencies are managed in `pyproject.toml`. When adding dependencies, update `pyproject.toml`.

### Security Checks

Run security and dependency health checks:

```bash
# Full check (vulnerabilities, code security, outdated packages, deprecation warnings)
python scripts/check_dependencies.py

# Quick check (skip slow fresh-venv deprecation check)
python scripts/check_dependencies.py --skip-warnings

# Check specific dependency groups
python scripts/check_dependencies.py --groups dev,dashboard
```

This runs:
- **pip-audit** - Dependency vulnerability scanning (PyPI advisory DB)
- **safety** - Dependency vulnerability scanning (Safety DB)
- **bandit** - Static code security analysis
- **Fresh venv install** - Capture deprecation warnings

### Install LLM backend

**vLLM Server** (runs in a container)
```bash
# One-time: install container runtime
sudo apt install podman nvidia-container-toolkit

# Start the vLLM container
scicode-lint vllm-server start
```

See [INSTALLATION.md](INSTALLATION.md) for detailed setup.

## How to Contribute

### Types of Contributions

- **Bug reports**: Open an issue with details about the problem
- **Feature requests**: Propose new detection patterns or features
- **Documentation**: Improve READMEs, docstrings, or guides
- **Code**: Fix bugs, add features, or improve existing code
- **Detection patterns**: Add new scientific code anti-patterns

## Adding New Detection Patterns

**📖 Complete guide:** [src/scicode_lint/patterns/README.md](src/scicode_lint/patterns/README.md) - Structure, detection question template, test file rules

### 1. Create scaffold

```bash
python -m scicode_lint.tools.new_pattern \
    --id ml-050 \
    --name temporal-split-leakage \
    --category ai-training \
    --severity critical
```

### 2. Edit pattern.toml

See [src/scicode_lint/patterns/README.md](src/scicode_lint/patterns/README.md) for detection question template.

### 3. Add test files

See [src/scicode_lint/patterns/README.md](src/scicode_lint/patterns/README.md) for test file requirements (pure code, no hints).

### 4. Validate and evaluate

```bash
# Run comprehensive validation (9 checks - see pattern_verification/README.md)
python pattern_verification/deterministic/validate.py ml-050-temporal-split-leakage

# Semantic review for consistency
python pattern_verification/semantic/semantic_validate.py ml-050-temporal-split-leakage

# Rebuild registry
python -m scicode_lint.tools.rebuild_registry

# Evaluate
python evals/run_eval.py --pattern ml-050
```

### 5. Submit

1. **Commit your changes:**
```bash
git add src/scicode_lint/patterns/ai-training/ml-050-temporal-split-leakage/
git add src/scicode_lint/patterns/_registry.toml
git commit -m "Add pattern ml-050: temporal split leakage detection

- Detects shuffle=True on time-series data
- Precision: 0.95, Recall: 0.85, F1: 0.90
- 3 positive tests, 2 negative tests, 1 context-dependent test"
```

2. **Push and create PR:**
```bash
git push origin feature/ml-050-temporal-leakage
```

3. **In your PR description, include:**
   - Pattern ID and description
   - Why this pattern is important (research impact)
   - Evaluation metrics
   - Any edge cases or limitations

### Tips and Best Practices

#### Choosing Pattern IDs

Pattern IDs follow the format: `{category-prefix}-{number}`. Check `src/scicode_lint/patterns/_registry.toml` for next available IDs.

#### Common Pitfalls to Avoid

❌ **Don't**: Write detection questions that require deep domain knowledge
```toml
question = "Will this cause overfitting in production?"
```

✅ **Do**: Focus on observable code patterns
```toml
question = "Is the validation set used during hyperparameter tuning?"
```

❌ **Don't**: Create overlapping patterns
- Check if a similar pattern already exists
- Combine closely related checks into one pattern

✅ **Do**: Make patterns orthogonal and focused
- Each pattern detects one specific bug type
- Use `related_patterns` to link similar issues

❌ **Don't**: Use placeholder test files
```python
# TODO: Add real test case
def placeholder():
    pass
```

✅ **Do**: Use realistic, representative code
```python
# Realistic example from scikit-learn documentation
from sklearn.preprocessing import StandardScaler
X_train, X_test = train_test_split(X)
scaler = StandardScaler().fit(X)  # BUG: fit on all X
```

#### Getting Help

If you're stuck or unsure:
1. Look at existing patterns for inspiration (especially `ml-001-scaler-leakage`)
2. Open a draft PR early for feedback
3. Ask in GitHub issues if you have questions about pattern design

## Running Tests

### Unit Tests

```bash
pytest tests/ -v
```

### Evaluation Framework

The project has a **two-tier evaluation system**:

**1. Pattern-Specific Evaluations** (`evals/`)
- Tests individual patterns in isolation
- Both hardcoded ground truth and LLM-as-judge approaches

```bash
# Hardcoded ground truth (fast)
python evals/run_eval.py --pattern ml-001

# LLM-as-judge (semantic)
python evals/run_eval.py --pattern ml-001

# All patterns
python evals/run_eval.py
```

**2. Integration Evaluations** (`evals/integration/`)
- Tests multi-pattern detection on realistic code
- Validates patterns work together without interference

```bash
# Full pipeline: Generate → Verify → Lint → Judge
python evals/integration/integration_eval.py --generate-count 10

# Skip verification/judge for faster evaluation
python evals/integration/integration_eval.py --generate-count 10 --skip-verification --skip-judge
```

**See:** [evals/README.md](evals/README.md) and [evals/integration/README.md](evals/integration/README.md) for details.

### Monitoring vLLM Performance

When running evaluations, monitor vLLM server performance to ensure efficient GPU utilization:

**Streamlit Dashboard (recommended for interactive development):**
```bash
./tools/start_dashboard.sh
# Open http://localhost:8501
```

**CLI metrics during evals:**
```bash
# Default: logs metrics every 5s to console and evals/reports/vllm_metrics.log
python evals/run_eval.py

# Custom interval
python evals/run_eval.py --monitor-interval 2

# Disable
python evals/run_eval.py --monitor-interval 0
```

**See:** [docs/performance/VLLM_MONITORING.md](docs/performance/VLLM_MONITORING.md) for key metrics and tuning guidance.

## Code Quality

Run type checking (`mypy src/`) and linting (`ruff check . && ruff format .`) before committing.

Ensure all checks pass and documentation is updated.

## Submitting Changes

1. Create a feature branch
2. Make your changes following the guidelines above
3. Commit with clear messages
4. Push to your fork
5. Open a Pull Request with:
   - Description of changes
   - Related issues
   - Test results for detection patterns

### PR Checklist

- [ ] Code follows project style (ruff, mypy pass)
- [ ] Tests added/updated and passing
- [ ] Documentation updated (README, docstrings, CHANGELOG)
- [ ] New detection patterns have test cases
- [ ] Integration tests pass (if modifying core linter code)

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be specific and include examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
