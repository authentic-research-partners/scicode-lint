# Benchmarking Guide

Run benchmarks to gather performance metrics for your system.

## Quick Start

```bash
# Ensure vLLM server is running
scicode-lint vllm-server start

# Speed & concurrency metrics
python benchmarks/speed_benchmark.py

# Token limit vs accuracy
python benchmarks/max_tokens_experiment.py

# Accuracy evaluation
python evals/run_eval.py
```

## Benchmarks

| Benchmark | Measures | Output |
|-----------|----------|--------|
| `speed_benchmark.py` | Throughput, latency, KV cache, preemptions | `benchmarks/reports/speed/` |
| `max_tokens_experiment.py` | Accuracy at different token limits | `benchmarks/reports/max_tokens/` |

## See Also

- [`benchmarks/README.md`](../../benchmarks/README.md) - Full metric descriptions and latest results
- [`evals/README.md`](../../evals/README.md) - Eval suite documentation
