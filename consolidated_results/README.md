# Consolidated Results

Unified performance report across all evaluation layers.

```bash
conda run -n scicode python consolidated_results/generate_consolidated_report.py
# → consolidated_results/CONSOLIDATED_REPORT.md
```

## What It Produces

`CONSOLIDATED_REPORT.md` — one file with every key metric:
controlled test accuracy, integration P/R/F1, Kaggle ground truth,
PapersWithCode feedback/holdout precision, severity and category breakdowns,
generalization gap. Includes scicode-lint version and git commit.

## Data Sources

The script reads structured data, not markdown:

| Section | Source | Type |
|---------|--------|------|
| 1. Controlled tests | `llm_judge_report.json` | JSON |
| 2. Integration eval | `report.json` + `scenarios/*.log` | JSON + logs |
| 3. Kaggle labeled | `analysis.db` (run 57) | SQLite |
| 4. Feedback set | `analysis.db` (run 56) + `meta_loop_set.md` | SQLite + markdown |
| 5. Holdout set | `analysis.db` (run 60) + `holdout_set.md` | SQLite + markdown |

Missing sources produce warnings in the report and zeros in the tables.
Source data git commits are validated — a warning appears if runs were generated at different commits.

## How to Regenerate Source Data

| Section | Script | Requires |
|---------|--------|----------|
| 1 | `python evals/run_eval.py` | vLLM server |
| 2 | `python evals/integration/integration_eval.py --generate-count 50` | vLLM + Claude CLI |
| 3 | `python real_world_demo/run_analysis.py --source leakage_paper` | vLLM + collected notebooks |
| 4 | `python real_world_demo/run_analysis.py` then `verify_findings.py` | vLLM + Claude CLI + cloned repos |
| 5 | Same as 4 with holdout paper set | Same |
