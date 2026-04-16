# Papers With Code Data Source

Data source from PapersWithCode via HuggingFace datasets archive.

**HuggingFace Datasets:**
- Papers: https://huggingface.co/datasets/pwc-archive/papers-with-abstracts
- Links: https://huggingface.co/datasets/pwc-archive/links-between-paper-and-code

---

## Pipeline Overview

The pipeline finds scientific ML repositories through a two-stage filtering approach:

1. **Wide keyword net** - Cast a broad net using domain keywords
2. **LLM semantic filter** - vLLM classifies abstracts and assigns accurate domains

This approach ensures we don't miss papers due to imperfect keyword matching, while getting reliable domain categorization from the LLM.

---

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. KEYWORD FILTER (wide net)                                       │
│     - Match papers by scientific domain keywords                    │
│     - Exclude ML venues (NeurIPS, ICML, etc.)                       │
│     - Balanced sampling across keyword domains                      │
│     Output: filtered_papers.json (~500 papers)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. LLM ABSTRACT FILTER (semantic classification)                   │
│     - vLLM reads each abstract                                      │
│     - Classifies: is_ai_science (true/false)                        │
│     - Assigns: science_domain, confidence, application_type         │
│     - Replaces unreliable keyword-based domain                      │
│     Output: ai_science_papers.json + abstract_excluded.json         │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. BALANCED SAMPLING (by LLM domain)                               │
│     - Sample N papers balanced by LLM-assigned domain               │
│     - Prioritize high-confidence classifications                    │
│     Output: sampled_papers.json (~50 papers)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. CLONE REPOS                                                     │
│     - Clone only sampled repos (not all approved)                   │
│     Output: cloned_repos/, clone_results.json                       │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  5. FILTER ML FILES                                                 │
│     - Find Python files with ML imports                             │
│     - Apply size/line count filters                                 │
│     Output: qualifying_files.json                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  6. GENERATE MANIFEST                                               │
│     - Copy files to collected_code/                                 │
│     - Enrich with paper metadata                                    │
│     Output: collected_code/manifest.csv                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LLM Abstract Classification

The vLLM abstract filter provides structured classification:

```python
class AbstractFilterResult:
    is_ai_science: bool      # True if AI applied to real science
    confidence: float        # 0.0 to 1.0
    science_domain: str      # biology, chemistry, medicine, physics,
                             # materials, neuroscience, earth_science,
                             # astronomy, economics, social_science,
                             # engineering, mathematics, or 'none'
    application_type: str    # prediction, analysis, discovery,
                             # simulation, diagnosis, or 'methodology'
    explanation: str         # Brief reasoning
```

**Benefits over keyword matching:**
- "Speech Synthesis" → correctly classified as `none` (not chemistry)
- "Drug Response Prediction" → correctly classified as `medicine`/`chemistry`
- New domains: economics, social_science, engineering, mathematics

---

## Usage

```bash
# Step 1: Download and filter papers (wide net)
PYTHONPATH=. python -m real_world_demo.sources.papers_with_code --download --papers 500

# Step 2: LLM abstract filter + sample 50 high-confidence papers
PYTHONPATH=. python -m real_world_demo.sources.papers_with_code.filter_abstracts \
    --sample 50 --min-confidence 0.7

# Step 3: Clone sampled repos
PYTHONPATH=. python -m real_world_demo.sources.papers_with_code.clone_repos

# Step 4: Filter ML files
PYTHONPATH=. python -m real_world_demo.sources.papers_with_code.filter_files

# Step 5: Generate manifest
PYTHONPATH=. python -m real_world_demo.sources.papers_with_code.generate_manifest \
    --input-file real_world_demo/data/qualifying_files.json \
    --papers-file real_world_demo/data/sampled_papers.json

# Step 6: Run analysis (uses source-specific default: 60 concurrent)
PYTHONPATH=. python real_world_demo/run_analysis.py \
    --manifest real_world_demo/collected_code/manifest.csv \
    --base-dir real_world_demo/collected_code
```

---

## Output Files

```
real_world_demo/
├── data/
│   ├── filtered_papers.json    # Papers after keyword filter (wide net)
│   ├── ai_science_papers.json  # Papers approved by LLM (with classification)
│   ├── abstract_excluded.json  # Papers rejected by LLM
│   ├── sampled_papers.json     # Final balanced sample for cloning
│   ├── clone_results.json      # Clone status per repo
│   └── qualifying_files.json   # Files with ML imports
├── cloned_repos/               # Cloned GitHub repositories
├── collected_code/
│   ├── files/                  # Copied files organized by repo
│   └── manifest.csv            # Files to analyze with metadata
└── reports/
    └── findings_YYYY-MM-DD_HHMM.md
```

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_ABSTRACT_FILTER_CONCURRENT` | 100 | High concurrency for abstract filter |
| `EXCLUDE_VENUES` | 31 venues | ML conferences to exclude |
| `SCIENTIFIC_DOMAINS` | 8 domains | Keywords for initial filtering |

---

## Benchmark Results

**Sample run (2026-03-12):** 45 papers → 61 repos → 884 files collected.

Quick test on 15 files (full analysis pending):

### Detection Summary

| Metric | Value |
|--------|-------|
| Files analyzed | 15 |
| Files with findings | 8 (53.3%) |
| Total findings | 19 |
| **Verified precision** | **15.8%** |

### Verification by Pattern

| Pattern | Category | Valid | Invalid | Uncertain | Precision |
|---------|----------|-------|---------|-----------|-----------|
| pt-003 | ai-training | 1 | 0 | 0 | **100%** |
| rep-002 | reproducibility | 2 | 3 | 0 | **40%** |
| rep-001 | reproducibility | 0 | 2 | 0 | 0% |
| ml-010 | ai-training | 0 | 1 | 1 | 0% |
| rep-004 | reproducibility | 0 | 1 | 1 | 0% |
| Others | various | 0 | 7 | 0 | 0% |

### Known Issues

1. **Empty snippets**: Some detections have empty code snippets (bug in extraction)
2. **Missing context**: Detector doesn't see full file (e.g., seeds set in other function)
3. **Broad patterns**: Some patterns flag acceptable code (rep-003 hardcoded defaults)

### By Domain

| Domain | Files | With Issues | Rate |
|--------|-------|-------------|------|
| economics | 6 | 4 | 66.7% |
| neuroscience | 9 | 4 | 44.4% |

---

## Comparison with Leakage Paper Source

| Aspect | Papers With Code | Leakage Paper |
|--------|-----------------|---------------|
| Source | Live GitHub repos | Static notebook dataset |
| Size | Configurable (--papers N) | Fixed 99 notebooks |
| Code type | Full repositories | Jupyter notebooks only |
| Domain assignment | LLM semantic classification | Manual labels |
| Ground truth | None | Manual labels (pre, overlap, multi) |
| Use case | Real-world validation | Precision/recall evaluation |
