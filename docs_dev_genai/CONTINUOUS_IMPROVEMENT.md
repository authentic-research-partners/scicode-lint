# Continuous Improvement Loop

**Core principle: Quality improvement is directly proportional to effort invested.**

This project has a structured improvement workflow. Each cycle of evaluation → analysis → improvement advances pattern quality. The more cycles run, the better the results.

---

## What Can Be Improved

### 1. Detection Patterns (Primary Focus)
- Detection questions, warning messages, test cases
- **Reviewed** via **pattern-reviewer agent**: `claude --agent pattern-reviewer "Review <pattern-id>"`
- Agent analyzes and suggests improvements → you implement the changes

### 2. The Agent Itself (When Needed)
- Agent instructions, system prompts, capabilities
- Location: `.claude/agents/pattern-reviewer/`
- Improve when agent consistently gives poor suggestions or misses issues

### 3. Evaluation Framework
- Test scenarios, metrics calculation, reporting
- Location: `evals/`

### Critical Rules (NEVER Break)

**Pattern Design Rules** (how detection prompts work at runtime):

| Rule | Rationale |
|------|-----------|
| Data leakage patterns must detect leakage | Core scientific integrity - false negatives here invalidate research |
| Critical severity requires precision ≥ 0.95 | High-confidence warnings only for critical issues |
| Detection questions must stay simple | Constrained-capacity LLM (Gemma 3 12B) cannot handle complex reasoning |
| Code-first prompt structure | User code comes before detection instructions - required for vLLM prefix caching |

These rules apply to **pattern TOML files** and **prompt generation** - not to the pattern-reviewer agent.

**If an "improvement" would violate these rules, reject it.**

---

## Current Quality Metrics

| Metric | Pattern-Specific | Integration | Target |
|--------|------------------|-------------|--------|
| Precision | 64.8% | 19.1% | ≥ 90% |
| Recall | 84.0% ✓ | 60% | ≥ 80% |
| Pass rate | 9/44 patterns | 0/4 scenarios | 100% |

**Source:** [README.md](../README.md) - update after each evaluation run.

---

## The Improvement Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   1. EVALUATE                                               │
│   ├── Pattern-specific: python evals/run_eval.py           │
│   └── Integration: python evals/integration/run_integration_eval.py │
│                         │                                   │
│                         ▼                                   │
│   2. IDENTIFY FAILURES                                      │
│   ├── Which patterns fail precision/recall targets?         │
│   └── What are the false positive/negative cases?           │
│                         │                                   │
│                         ▼                                   │
│   3. ANALYZE & IMPROVE                                      │
│   ├── Use pattern-reviewer agent for analysis               │
│   ├── Improve detection questions (simpler, more focused)   │
│   ├── Add/fix test cases (positive/negative/context)        │
│   └── Update warnings for clarity                           │
│                         │                                   │
│                         ▼                                   │
│   4. VALIDATE                                               │
│   ├── ruff check . && ruff format .                         │
│   ├── mypy src/                                             │
│   └── pytest                                                │
│                         │                                   │
│                         ▼                                   │
│   5. RE-EVALUATE → Loop back to step 1                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Commands

### Step 1: Run Evaluations

```bash
# All patterns (isolated)
python evals/run_eval.py

# Specific pattern
python evals/run_eval.py --pattern ml-001-scaler-leakage

# Integration tests
python evals/integration/run_integration_eval.py
```

### Step 2: Identify Failing Patterns

- Check eval output for patterns below targets (precision < 0.90, recall < 0.80)
- Review false positives (detected but shouldn't be) and false negatives (missed)

### Step 3: Run Pattern-Reviewer Agent

The **pattern-reviewer agent** analyzes patterns and suggests improvements (it reviews, not writes):

```bash
# Review single pattern
claude --agent pattern-reviewer "Review ml-001-scaler-leakage"

# Batch review failing patterns
claude --agent pattern-reviewer "Review all patterns with precision < 0.90"

# Get specific suggestions
claude --agent pattern-reviewer "ml-001 has 3 false positives. Analyze and suggest fixes."
```

**What the agent does:**
- Analyzes pattern and identifies issues
- Suggests specific improvements
- **You implement the suggested changes** (edit TOML, add test cases)

**What the agent checks:**
- Detection question clarity and simplicity
- Warning message actionability
- Test case coverage (positive/negative/context-dependent)
- Alignment with constrained-capacity LLM principles

**Agent location:** `.claude/agents/pattern-reviewer/`

### Step 4: Validate Code Quality

```bash
ruff check . && ruff format .
mypy src/
pytest
```

### Step 5: Re-evaluate and Iterate

- Run evals again
- If metrics improved but still below target, continue loop
- If metrics meet targets, move to next failing pattern

---

## When to Run This Loop

**MANDATORY: Every pattern change requires running the pattern-reviewer agent.**

```bash
# After ANY pattern modification
claude --agent pattern-reviewer "Review <pattern-id>"
```

| Trigger | Action |
|---------|--------|
| After implementing new pattern | Run pattern-reviewer agent + full loop |
| After modifying detection question | Run pattern-reviewer agent + re-evaluate |
| After changing test cases | Run pattern-reviewer agent + re-evaluate |
| Before any release | Run full eval suite, document results in README |
| Pattern precision/recall below target | Run pattern-reviewer agent for analysis |
| Integration tests failing | Focus on cross-pattern interference |

---

## Success Criteria

### Pattern Passes When

- Precision ≥ 0.90 (≥ 0.95 for critical severity)
- Recall ≥ 0.80
- All test cases (positive/negative/context-dependent) behave as expected

### Project Milestone Achieved When

- All 44 patterns meet individual thresholds
- Integration scenarios pass with acceptable precision
- README metrics updated to reflect current state

---

## Improving the Agent Itself

When the pattern-reviewer agent consistently gives poor suggestions:

1. **Identify the issue** - What types of suggestions are wrong?
2. **Update agent instructions** - Edit `.claude/agents/pattern-reviewer/system_prompt.md`
3. **Add examples** - Improve `.claude/agents/pattern-reviewer/examples.md`
4. **Test the changes** - Run the agent on known patterns and verify better output

**Agent files:**
```
.claude/agents/pattern-reviewer/
├── agent.json          # Agent configuration
├── system_prompt.md    # Main instructions (edit this)
├── README.md           # Documentation
├── QUICK_START.md      # Quick reference
├── BATCH_OPERATIONS.md # Batch processing guide
└── examples.md         # Usage examples
```

**Remember:** Agent improvements must respect Critical Rules (see above).

---

## Resources

- **Pattern-reviewer agent:** [.claude/agents/pattern-reviewer/](../.claude/agents/pattern-reviewer/)
- **Evaluation framework:** [evals/README.md](../evals/README.md)
- **Integration tests:** [evals/integration/README.md](../evals/integration/README.md)
- **Pattern structure:** [patterns/README.md](../patterns/README.md)
- **Architecture (detection question design):** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## See Also

- [TOOLS.md](TOOLS.md) - Development tools including pattern-reviewer agent
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design principles for detection questions
- [../Claude.md](../Claude.md) - Main AI agent instructions
