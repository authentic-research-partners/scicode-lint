# vLLM Monitoring Guide

## Overview

scicode-lint includes built-in tools for monitoring vLLM server performance during linting and evaluations.

## Quick Start

### Streamlit Dashboard (Recommended)

Live visual dashboard with charts:

```bash
# Start dashboard
./tools/start_dashboard.sh

# Or directly
streamlit run tools/vllm_dashboard.py

# Custom vLLM URL
./tools/start_dashboard.sh --url http://remote-server:5001
```

Open http://localhost:8501 in your browser.

**Features:**
- Server config: Model, Max Tokens (setting), KV Cache Capacity, KV Blocks, Block Size, GPU Utilization
- Live metrics: Running, Queued, Throughput, KV Cache %
- Time-series charts with configurable moving average
- Clear log file button (for `evals/reports/vllm_metrics.log`)
- All vLLM metrics expandable

**Install dependencies:**
```bash
pip install scicode-lint[dashboard]
# or: pip install streamlit pandas
```

### CLI Metrics Monitor

For headless/CI environments, the eval script includes built-in monitoring:

```bash
# Run eval with metrics logging (default: every 5s)
python -m evals.run_eval_llm_judge

# Custom interval
python -m evals.run_eval_llm_judge --monitor-interval 2

# Disable monitoring
python -m evals.run_eval_llm_judge --monitor-interval 0
```

**Output:**
- Console: `[  60s] vLLM: 5 running, 27 queued, 1.2 req/s`
- CSV file: `evals/reports/vllm_metrics.log`

**CSV format:**
```csv
timestamp,elapsed_s,running,waiting,finished,throughput_req_s
2026-03-07T12:34:56+00:00,5.0,5,27,6,1.20
```

## Key Metrics

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| Running | Requests currently on GPU | 5-15 (depends on model) |
| Queued | Requests waiting | 0-50 (higher = GPU saturated) |
| Throughput | Requests/second | Varies by model/hardware |
| KV Cache % | GPU cache utilization | 50-95% (>95% may cause delays) |

## Eval Performance Tuning

### Parallelization

Evals run all patterns concurrently with a semaphore to limit concurrent requests:

```bash
# Default: 100 concurrent test evaluations (tuned for Qwen3-8B ~9GB model)
python -m evals.run_eval_llm_judge

# Lower for memory-constrained systems or larger models
python -m evals.run_eval_llm_judge --max-concurrent 32
```

### Context Length

Evals use shorter context (8K) by default for faster throughput:

```bash
# Default: 8K context for evals
python -m evals.run_eval_llm_judge

# Custom context length
python -m evals.run_eval_llm_judge --max-model-len 4000
```

## Programmatic Monitoring

Use `VLLMMetricsMonitor` in your own scripts:

```python
import asyncio
from scicode_lint.vllm import VLLMMetricsMonitor

async def main():
    monitor = VLLMMetricsMonitor(
        base_url="http://localhost:5001",
        interval=5.0,
        output_file="metrics.log",
        console=True,
    )
    monitor.start()

    try:
        # ... your vLLM workload ...
        await asyncio.sleep(60)
    finally:
        await monitor.stop()

asyncio.run(main())
```

## Troubleshooting

### Dashboard shows 0 for all metrics
- Check vLLM is running: `curl http://localhost:5001/health`
- Verify URL: `./tools/start_dashboard.sh --url http://correct-host:port`

### High queue, low running
- KV cache exhausted - reduce `max_model_len` or requests
- Check: `curl -s localhost:5001/metrics | grep kv_cache_usage`

### Throughput spike on start
- Normal if dashboard started mid-run
- Click "Clear History" in sidebar

## Architecture

### Data Sources

The monitoring system has two data paths:

| Source | Used By | Persistence | Latency |
|--------|---------|-------------|---------|
| **vLLM HTTP `/metrics`** | Dashboard (live stats) | None - lost on close | Real-time |
| **CSV log file** | CLI monitor, post-analysis | Persistent | 5s interval |

**Dashboard** fetches directly from vLLM for real-time display. Data is kept in-memory only.

**CLI Monitor** (`VLLMMetricsMonitor`) writes to CSV during evals for later analysis.

## Files

| File | Purpose |
|------|---------|
| `tools/vllm_dashboard.py` | Streamlit dashboard |
| `tools/start_dashboard.sh` | Dashboard launcher script |
| `evals/reports/vllm_metrics.log` | CSV metrics from eval runs |
| `src/scicode_lint/vllm/__init__.py` | VLLMMetricsMonitor class |
