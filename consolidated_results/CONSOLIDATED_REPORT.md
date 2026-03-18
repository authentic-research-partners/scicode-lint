# Consolidated Performance Report

- **Generated:** 2026-03-18 02:55 UTC
- **scicode-lint version:** 0.2.2
- **Git commit:** `aa3f173`

Every number in the paper should trace to a row in this report.

## Data Sources

| Source | File | Status |
|--------|------|--------|
| Controlled tests | `src/scicode_lint/evals/reports/judge/20260317_173544_all/llm_judge_report.json` | OK |
| Integration eval | `evals/integration/reports/20260316_172513_generate_50/report.json` | OK |
| Integration logs | `evals/integration/reports/20260316_172513_generate_50/scenarios` | OK |
| Analysis DB | `real_world_demo/data/analysis.db` | OK |
| Feedback paper set | `real_world_demo/paper_sets/meta_loop_set.md` | OK |
| Holdout paper set | `real_world_demo/paper_sets/holdout_set.md` | OK |

### Source Data Git Commits

**WARNING: DB runs were generated at different commits:**

- Kaggle (run 57): `3a93f57`
- Feedback (run 56): `f2a15fc`
- Holdout (run 60): `f2a15fc`

## 1. Controlled Tests (LLM-as-Judge)

Source: `src/scicode_lint/evals/reports/judge/20260317_173544_all/llm_judge_report.json`

| Metric | Value |
|--------|-------|
| Patterns | 66 |
| Total tests | 452 |
| **Overall accuracy** | **97.68%** |
| Positive accuracy | 99.01% |
| Negative accuracy | 97.86% |
| Context-dependent accuracy | 81.2% |
| Semantic alignment | 98.9% |
| Quality issue rate | 0.9% |
| Patterns at 100% | 56/66 |
| Focus line accuracy | 40.8% (82/201) |

## 2. Integration Evaluation (Generated Scenarios)

Source: `evals/integration/reports/20260316_172513_generate_50/report.json` + `evals/integration/reports/20260316_172513_generate_50/scenarios/*.log`

| Metric | Value |
|--------|-------|
| Scenarios | 50 |
| Bugs intended | 148 |
| TP-intended (expected bugs found) | 126 |
| TP-bonus (verified extra bugs) | 27 |
| False positives | 111 |
| False negatives | 19 |
| Total TP | 153 |
| **Precision** | **58.0%** |
| **Recall** | **85.1%** |
| **F1** | **69.0%** |

## 3. Kaggle Labeled Notebooks (Yang et al. ASE'22)

Source: `analysis.db run_id=57`

- Files analyzed: 82
- Files with findings: 11
- Excluded (timeouts): 12

| Label | TP | FP | FN | TN | Precision | Recall | F1 |
|-------|---:|---:|---:|---:|----------:|-------:|---:|
| pre | 6 | 2 | 5 | 25 | 75.0% | 54.5% | 63.2% |
| overlap | 0 | 1 | 2 | 44 | 0.0% | 0.0% | 0.0% |
| multi | 0 | 0 | 20 | 30 | 0.0% | 0.0% | 0.0% |

## 4. PapersWithCode — Feedback Set

Sources:
- `analysis.db run_id=65`
- `real_world_demo/paper_sets/meta_loop_set.md`

### Data Funnel

| Metric | Value |
|--------|-------|
| Papers sampled | 38 |
| Papers with self-contained files | 32 |
| Self-contained files | 119 |
| Papers with verified real bugs | 24 |
| Total findings | 137 |
| Valid | 85 |
| Invalid | 45 |
| Uncertain | 7 |
| **Precision** | **62.0%** |

### By Severity

| Severity | Findings | Valid | Invalid | Precision |
|----------|----------|-------|---------|-----------|
| Critical | 21 | 5 | 16 | 24% |
| High | 77 | 52 | 18 | 68% |
| Medium | 39 | 28 | 11 | 72% |

### By Category

| Category | Valid | Invalid | Uncertain | Precision |
|----------|-------|---------|-----------|-----------|
| ai-inference | 28 | 5 | 4 | 76% |
| scientific-reproducibility | 40 | 12 | 2 | 74% |
| scientific-numerical | 5 | 3 | 0 | 62% |
| ai-training | 8 | 17 | 0 | 32% |
| scientific-performance | 4 | 8 | 1 | 31% |

## 5. PapersWithCode — Holdout Set

Sources:
- `analysis.db run_id=66`
- `real_world_demo/paper_sets/holdout_set.md`

### Data Funnel

| Metric | Value |
|--------|-------|
| Papers sampled | 35 |
| Papers with self-contained files | 17 |
| Self-contained files | 45 |
| Papers with verified real bugs | 12 |
| Total findings | 74 |
| Valid | 40 |
| Invalid | 28 |
| Uncertain | 6 |
| **Precision** | **54.1%** |

### By Severity

| Severity | Findings | Valid | Invalid | Precision |
|----------|----------|-------|---------|-----------|
| Critical | 6 | 1 | 2 | 17% |
| High | 37 | 24 | 12 | 65% |
| Medium | 31 | 15 | 14 | 48% |

### By Category

| Category | Valid | Invalid | Uncertain | Precision |
|----------|-------|---------|-----------|-----------|
| scientific-numerical | 4 | 2 | 0 | 67% |
| scientific-reproducibility | 18 | 9 | 0 | 67% |
| ai-inference | 15 | 7 | 2 | 62% |
| scientific-performance | 2 | 5 | 1 | 25% |
| ai-training | 1 | 5 | 3 | 11% |

### Generalization Gap (Feedback vs Holdout)

- Feedback precision: 62.0%
- Holdout precision: 54.1%
- Gap: 8.0 percentage points

## Summary Table (for paper)

| Layer | Precision | Recall | Key Detail |
|-------|-----------|--------|------------|
| Controlled tests | — | — | 452 tests, 66 patterns, 97.7% overall accuracy |
| Integration (n=50) | 58.0% | 85.1% | 148 intended bugs, 27 bonus TPs |
| Kaggle labeled (`pre`) | 75.0% | 54.5% | Human ground truth (Yang et al. ASE'22) |
| PapersWithCode (feedback) | 62.0% | — | 38 papers, 119 files |
| PapersWithCode (holdout) | 54.1% | — | 35 papers, 45 files |

## Warnings

- DB runs were generated at different git commits (3a93f57, f2a15fc)

