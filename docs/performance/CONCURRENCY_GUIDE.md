# Concurrency Guide for vLLM

## Overview

scicode-lint maximizes GPU utilization by sending concurrent requests to vLLM. vLLM handles batching and queuing internally.

## Linter Concurrency

**All patterns checked concurrently:**
- All 66 patterns are sent to vLLM simultaneously
- vLLM's internal scheduler batches them efficiently
- Prefix caching works best with concurrent requests

**Implementation:**
```python
# Send all requests at once, let vLLM queue them
tasks = [llm.async_complete_structured(...) for pattern in patterns]
results = await asyncio.gather(*tasks)
```

## Evaluation Concurrency

**All patterns AND test files evaluated concurrently:**
- All ~66 patterns evaluated in parallel (not sequentially)
- Each pattern's ~5 test files also run in parallel
- Semaphore limits max concurrent to avoid overwhelming system

**Implementation:**
```python
# Evaluate ALL patterns concurrently
pattern_tasks = [evaluator.evaluate_pattern(pid) for pid in pattern_ids]
all_results = await asyncio.gather(*pattern_tasks)

# Each pattern evaluates test files concurrently (with semaphore)
async with self._semaphore:  # Default: 100 concurrent
    linter_output = await self.run_linter(test_file)
    verdict = await self.llm.async_complete_structured(...)
```

**Configure concurrency:**
```bash
# Default: 100 concurrent evaluations (tuned for Qwen3-8B ~9GB model)
python -m evals.run_eval_llm_judge

# Lower for memory-constrained systems or larger models
python -m evals.run_eval_llm_judge --max-concurrent 32
```

## Performance

**Linter (single file):**
- 66 patterns, sequential: ~132s
- 66 patterns, concurrent: ~15-20s
- **Speedup: 6-8x**

**Evaluation (all patterns):**
- 66 patterns × 5 tests, sequential patterns: ~10 min
- 66 patterns × 5 tests, concurrent patterns: ~2-3 min
- **Speedup: 3-5x**

**Why concurrent is better:**
- vLLM automatically batches requests
- Prefix caching reuses common code across patterns
- Internal scheduler manages GPU resources
- No idle GPU time between requests

---

## Monitoring

See [VLLM_MONITORING.md](VLLM_MONITORING.md) for:
- Streamlit dashboard (`./tools/start_dashboard.sh`)
- CLI metrics logging (`--monitor-interval`)
- Key metrics interpretation
- Performance tuning tips
