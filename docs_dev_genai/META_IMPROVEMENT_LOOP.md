# Meta Improvement Loop (Real-World Validation)

**Different from the fast loop in [CONTINUOUS_IMPROVEMENT.md](CONTINUOUS_IMPROVEMENT.md).** This is a resource-intensive, statistical approach for discovering systematic pattern weaknesses that synthetic test files miss.

## Purpose

Discovers systematic pattern weaknesses that synthetic test files miss by running on real scientific code and getting structured feedback from Claude on why detections fail.

## Requirements

| Resource | Estimate | Purpose |
|----------|----------|---------|
| Real code corpus | ~120 files | Papers with Code self-contained files |
| vLLM detection | ~1-2 hours | Run all patterns on all files |
| Sonnet verification | ~1-2 hours | Verify each finding |
| Sonnet tokens | ~1.5M tokens | ~3K input + ~300 output per finding × 400-500 findings |
| Total time | ~3-4 hours | Full run + verification |

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    META IMPROVEMENT LOOP                        │
│         (Resource-intensive, run periodically)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. RUN ON REAL CODE                                             │
│    python -m real_world_demo.run_analysis --from-prefilter-run N│
│    Output: Findings with line numbers, snippets                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. VERIFY WITH SONNET                                            │
│    python -m real_world_demo.verify_findings --run-id N         │
│    Output: Each finding marked valid/invalid/uncertain          │
│            + reasoning explaining WHY                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ANALYZE FALSE POSITIVES (automated)                           │
│    python -m real_world_demo.analyze_errors --run-id N          │
│    Output: Per-pattern themes, root causes, fix recommendations │
│            reports/error_analysis/ERROR_ANALYSIS_<date>.md       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. REFINE DETECTION QUESTIONS                                   │
│    - Add context checks verification identified                 │
│    - Update test files to cover new edge cases                  │
│    - Run fast loop (deterministic → semantic → evals)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. MEASURE IMPROVEMENT                                          │
│    - Re-run on same corpus                                      │
│    - Compare precision: before vs after                         │
│    - Track per-pattern precision trends                         │
└─────────────────────────────────────────────────────────────────┘
```

## Commands

```bash
# 1. Run analysis on real code (requires vLLM, ~1-2 hours for 120 files)
python -m real_world_demo.run_analysis --from-prefilter-run 4 --source papers_with_code

# 2. Verify findings with Sonnet (~1-2 hours)
python -m real_world_demo.verify_findings --run-id <RUN_ID>

# 3. Analyze false positives (Claude Opus — extracts themes, recommends fixes)
python -m real_world_demo.analyze_errors --run-id <RUN_ID>
# Options: --min-invalid 3, --patterns perf-004 par-005, --model sonnet
# Output: reports/error_analysis/ERROR_ANALYSIS_<date>.md + per-pattern .md files

# 4. Generate reports
python -m real_world_demo.generate_report --run-id <RUN_ID>
python -m real_world_demo.generate_report --run-id <RUN_ID> --verified-only
```

## Key Insight: Structured Feedback

The verifier doesn't just say "wrong" - it explains *why* each detection failed:

| Pattern | Common False Positive Theme |
|---------|----------------------------|
| rep-002 | "This is inference, not training" |
| pt-010 | "Data already pre-loaded to GPU" |
| pt-014 | "High-level timing, not GPU kernel benchmark" |
| pt-021 | "Timing for monitoring, not benchmarking" |
| num-005 | "Using library that handles this internally" |
| rep-005 | "Glob results used for dict lookup, order doesn't matter" |
| ml-008 | "Pre-defined splits loaded from file" |
| ml-010 | "This is inference/prediction script, not training" |

These themes become **specific context checks** to add to detection questions.

## Example: Pattern Refinement from Verification Feedback

**Before (rep-002):** "Does code use CUDA without deterministic settings?"

**Verification feedback:** "This is inference only (model.eval(), torch.no_grad()), non-determinism only matters during training"

**After (rep-002):** "Does code use CUDA without deterministic settings DURING TRAINING? Only flag if code has model.train(), optimizer.step(), or loss.backward(). Inference-only code is NOT a bug."

## Tracking Progress

| Run | Date | Files | Findings | Precision | Notes |
|-----|------|-------|----------|-----------|-------|
| 49 | 2026-03-13 | 884 | 2661 | 12.4% | Before self-contained filter |
| 53 | 2026-03-14 | 120 | 574 | N/A | Self-contained filter added |
| 55 | 2026-03-15 | 120 | 429 | 36.4% | Comment stripping + line number fix |
| 56 | 2026-03-16 | 120 | 219 | 45.2% | 1st error analysis → 19 pattern improvements |
| 62 | 2026-03-17 | 120 | 197 | 51.8% | 2nd error analysis → pattern improvements |
| 65 | 2026-03-17 | 120 | 137 | 62.0% | 3rd round: reproducibility + numerical + performance |

**Run 55 precision jump (12% → 36%)** came from two changes together:
1. **Comment stripping** - Remove comments before LLM analysis (fewer distractions)
2. **Line number fix** - Store line numbers correctly in findings (better verification context)

## Cost

- **Time:** ~3-4 hours (vLLM detection + Sonnet verification)
- **Tokens:** ~1.5M Sonnet tokens per run (~400-500 findings × ~3.5K tokens each)

## Data Separation Rules

### Leakage Paper Dataset (Yang et al. ASE'22)

The leakage paper's [data repository](https://github.com/malusamayo/GitHubAPI-Crawler/tree/master/evaluation_materials) has two distinct parts with different rules:

**1. Labeled notebooks (99) — BENCHMARK ONLY, not for feedback**

The 99 manually labeled notebooks (`sample-notebooks.zip` + `manual-labels.csv`) have ground truth labels (`pre`, `overlap`, `multi`). These are a held-out benchmark for measuring precision/recall/F1.

- **Allowed:** Run scicode-lint and compare results to ground truth (`compare_ground_truth.py`)
- **NOT allowed:** Using ground truth labels to decide how to change detection questions (this would overfit to the benchmark)

**2. Unlabeled notebooks and repos (~5,000) — available for meta loop**

The same dataset contains a much larger pool of unlabeled real-world ML code:

| Dataset | Count | Source |
|---------|-------|--------|
| Kaggle notebooks | ~1,043 | `kaggle_list.txt` (house-prices, titanic competitions) |
| GitHub repos | ~4,000 | `repos.txt` |
| Paper's full analysis | 100,000+ | Analyzed by paper's static tool, not hand-labeled |

These **unlabeled** notebooks and repos can be used in the meta loop as additional real-world ML code. Use them the same way as PapersWithCode: run detection, verify with Sonnet, use verification feedback (not ground truth labels) to improve patterns.

#### TODO: Kaggle Notebooks Pipeline

The ~1,087 Kaggle notebooks are ideal for leakage pattern testing (self-contained ML notebooks from house-prices and titanic competitions). Not yet implemented — requires Kaggle API setup.

**Path format in `kaggle_list.txt`:**
```
kaggle-notebooks/<competition>/<username>/<kernel-slug>/<filename>.ipynb
```

**Implementation plan:**
1. Download `kaggle_list.txt` from GitHub
2. Parse paths → extract `<username>/<kernel-slug>` for Kaggle API
3. Download notebooks via `kaggle kernels pull <username>/<kernel-slug>` (requires `~/.kaggle/kaggle.json`)
4. Exclude the 99 labeled notebooks (match by notebook ID from `ground_truth.csv`)
5. Sample N random notebooks (`--sample N --seed S`)
6. Create manifest (reuse `prepare_manifest()` from leakage paper source)
7. Run standard pipeline: `run_analysis` → `verify_findings` → `analyze_errors`

**Setup required:**
- `pip install kaggle` (add to `[project.optional-dependencies]` in `pyproject.toml` under `real-world-demo`)
- API key from https://www.kaggle.com/settings → "Create New Token" → saves `~/.kaggle/kaggle.json`

**Pipeline integration:**
- New module: `real_world_demo/sources/kaggle_notebooks/` (similar to `leakage_paper/`)
- Add `kaggle_notebooks` to `DATA_SOURCE_CONFIGS` in `real_world_demo/config.py`
- Add `data_source` column value `kaggle_notebooks` for DB isolation
- Reuse existing pipeline: `run_analysis --source kaggle_notebooks` → `verify_findings` → `analyze_errors`
- Default patterns: all 66 (unlike leakage_paper which defaults to 4 leakage patterns only)
- Prefilter not needed (Kaggle competition notebooks are already self-contained ML code)

**Value:** These notebooks are specifically from ML competitions where data leakage is common, making them a high-signal data source for testing `ml-001`, `ml-002`, `ml-007`, `ml-009`, `ml-010` patterns.

### Separate PapersWithCode Samples

Papers used in meta loop should NOT overlap with papers used for final accuracy testing. This prevents overfitting patterns to the specific papers used for feedback.

Exclude papers **at the abstract filter stage** (before cloning) using `--exclude-from-prefilter-run` on `filter_abstracts.py`. This avoids wasting time cloning and processing already-used repos:

```bash
# 1. Download papers (shared pool)
python -m real_world_demo.sources.papers_with_code --download --force --papers 200

# 2. Filter abstracts, excluding papers from previous runs
python -m real_world_demo.sources.papers_with_code.filter_abstracts \
    --force --sample 60 --seed 99 --exclude-from-prefilter-run 4 6

# 3. Clone, filter files, prefilter, analyze
python -m real_world_demo.sources.papers_with_code --clone --force
python -m real_world_demo.sources.papers_with_code.filter_files --force
python -m real_world_demo.sources.papers_with_code.prefilter_files --force
python -m real_world_demo.run_analysis --source papers_with_code --from-prefilter-run <RUN_ID>
```

Exclusion operates at paper level: if a paper had any file in the excluded prefilter runs, it is removed before sampling.

The `prefilter_files.py` also supports `--exclude-from-prefilter-run` as a safety net when working directly with qualifying files.

### Paper Set Reference Files

Paper lists for each data pool are stored in `real_world_demo/paper_sets/` (git-tracked) as `.md` (human-readable) + `.json` (machine-readable with `repo_urls`). These are the **authoritative record** of each paper set — the JSON files contain `repo_urls` needed to re-clone repos and rebuild the database from scratch.

- `meta_loop_set.json` / `.md` — 38 papers used for pattern refinement (prefilter run 4, created before seeded sampling was added — **not reproducible from pipeline, JSON is the only record**)
- `holdout_set.json` / `.md` — 35 papers for final precision measurement (seed 99, excluding runs 4+6)

When creating a new data pool, always save its paper list to this directory. These files are the recovery mechanism if the database is lost.

### Non-overlapping Data Pools

1. **Meta loop set (PapersWithCode):** Used for pattern refinement (can look at verification feedback)
2. **Holdout set (PapersWithCode):** Used for final precision measurement (no peeking during development)
3. **Leakage paper benchmark:** 99 labeled notebooks for precision/recall/F1 against ground truth (never use labels for feedback)
4. **Kaggle notebooks (TODO):** ~1,087 unlabeled ML competition notebooks — high-signal for leakage patterns (requires Kaggle API setup)

## Comparison with Fast Loop

| Aspect | Fast Loop | Meta Loop |
|--------|-----------|-----------|
| **File** | [CONTINUOUS_IMPROVEMENT.md](CONTINUOUS_IMPROVEMENT.md) | This file |
| **Code** | Synthetic test files | Real Papers with Code |
| **Verifier** | Evals (pattern-specific) | Sonnet (general reasoning) |
| **Time** | Minutes | Hours |
| **Tokens** | Local vLLM only | ~1.5M Sonnet tokens |
| **Feedback** | Pass/fail | Structured reasoning |
| **Frequency** | Every change | Monthly / after major changes |
