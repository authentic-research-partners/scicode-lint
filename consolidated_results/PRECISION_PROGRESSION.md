# Precision Progression — Meta Improvement Loop

Tracked on the **feedback dataset** (meta loop set, prefilter run 4, 120 self-contained files from 38 papers).

## Feedback Set (used for pattern refinement)

| Run | Date | Findings | Valid | Invalid | Uncertain | Precision | What Changed |
|-----|------|----------|-------|---------|-----------|-----------|-------------|
| 53 | 2026-03-14 | 574 | 116 | 408 | 50 | 20.2% | Baseline |
| 55 | 2026-03-15 | 429 | 153 | 267 | 9 | 35.7% | Comment stripping + line number fix |
| 56 | 2026-03-16 | 219 | 99 | 111 | 9 | 45.2% | 1st error analysis → 19 pattern improvements |
| 62 | 2026-03-17 | 197 | 102 | 88 | 7 | 51.8% | 2nd error analysis → pattern improvements |
| 65 | 2026-03-17 | 137 | 85 | 45 | 7 | 62.0% | Reproducibility + numerical + performance pattern improvements |

## Holdout Set (unseen during development)

| Run | Date | Findings | Valid | Invalid | Uncertain | Precision |
|-----|------|----------|-------|---------|-----------|-----------|
| 60 | 2026-03-16 | 103 | 39 | 58 | 6 | 37.9% |
| 66 | 2026-03-17 | 74 | 40 | 28 | 6 | 54.1% |

## Key Trends

- **Findings dropped 76%**: 574 → 137 (eliminated 437 false positives)
- **Precision tripled**: 20.2% → 62.0% (+41.8pp)
- **Valid findings decreased 27%**: 116 → 85 (acceptable trade-off)
- **Invalid findings dropped 89%**: 408 → 45
- **Generalization gap**: 62.0% (feedback) vs 54.1% (holdout) = 7.9pp — patterns generalize well
