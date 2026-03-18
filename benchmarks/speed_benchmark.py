#!/usr/bin/env python3
"""Speed benchmark using vLLM metrics.

Runs an eval workload and captures vLLM server metrics to measure:
- Time to first token (TTFT) distribution
- Token generation speed (tokens/second)
- Prefix cache hit rate
- Throughput (requests/second)
- Peak concurrency and KV cache utilization

Usage:
    python benchmarks/speed_benchmark.py

Results saved to: benchmarks/reports/speed/benchmark_YYYYMMDD_HHMMSS.json
"""

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).parent.parent
VLLM_METRICS_URL = "http://localhost:5001/metrics"


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus metrics text into a dict, summing labeled metrics."""
    metrics: dict[str, float] = {}
    sum_metrics: dict[str, float] = {}

    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            full_name = parts[0]
            base_name = full_name.split("{")[0]
            try:
                value = float(parts[-1])
                if "{" in full_name:
                    sum_metrics[base_name] = sum_metrics.get(base_name, 0) + value
                else:
                    metrics[base_name] = value
            except ValueError:
                pass

    # Merge: prefer explicit metrics, fall back to summed
    for k, v in sum_metrics.items():
        if k not in metrics:
            metrics[k] = v

    return metrics


@dataclass
class VLLMSnapshot:
    """Snapshot of vLLM metrics at a point in time."""

    timestamp: float
    prompt_tokens: int
    generation_tokens: int
    prefix_cache_queries: int
    prefix_cache_hits: int
    requests_finished: int
    ttft_sum: float
    ttft_count: int
    preemptions: int
    running: int = 0
    waiting: int = 0
    kv_cache_pct: float = 0.0

    @classmethod
    def fetch(cls) -> "VLLMSnapshot":
        """Fetch current metrics from vLLM server."""
        resp = httpx.get(VLLM_METRICS_URL, timeout=5.0)
        resp.raise_for_status()
        metrics = parse_prometheus_metrics(resp.text)

        return cls(
            timestamp=time.time(),
            prompt_tokens=int(metrics.get("vllm:prompt_tokens_total", 0)),
            generation_tokens=int(metrics.get("vllm:generation_tokens_total", 0)),
            prefix_cache_queries=int(metrics.get("vllm:prefix_cache_queries_total", 0)),
            prefix_cache_hits=int(metrics.get("vllm:prefix_cache_hits_total", 0)),
            requests_finished=int(metrics.get("vllm:request_success_total", 0)),
            ttft_sum=metrics.get("vllm:time_to_first_token_seconds_sum", 0),
            ttft_count=int(metrics.get("vllm:time_to_first_token_seconds_count", 0)),
            preemptions=int(metrics.get("vllm:num_preemptions_total", 0)),
            running=int(metrics.get("vllm:num_requests_running", 0)),
            waiting=int(metrics.get("vllm:num_requests_waiting", 0)),
            kv_cache_pct=metrics.get("vllm:kv_cache_usage_perc", 0) * 100,
        )


@dataclass
class RuntimeStats:
    """Statistics collected during benchmark run."""

    peak_running: int = 0
    peak_waiting: int = 0
    peak_kv_cache_pct: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)


class MetricsMonitor:
    """Background monitor for collecting runtime stats during eval."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.stats = RuntimeStats()
        self._stop = False
        self._thread: threading.Thread | None = None

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop:
            try:
                snap = VLLMSnapshot.fetch()
                self.stats.peak_running = max(self.stats.peak_running, snap.running)
                self.stats.peak_waiting = max(self.stats.peak_waiting, snap.waiting)
                self.stats.peak_kv_cache_pct = max(self.stats.peak_kv_cache_pct, snap.kv_cache_pct)
                self.stats.samples.append(
                    {
                        "t": snap.timestamp,
                        "running": snap.running,
                        "waiting": snap.waiting,
                        "kv_pct": snap.kv_cache_pct,
                    }
                )
                # Print live status
                print(
                    f"  [live] running={snap.running}, waiting={snap.waiting}, "
                    f"KV={snap.kv_cache_pct:.1f}%"
                )
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self) -> None:
        """Start background monitoring."""
        self._stop = False
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> RuntimeStats:
        """Stop monitoring and return collected stats."""
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5.0)
        return self.stats


def fetch_histogram_buckets(metric_name: str) -> list[tuple[float, int]]:
    """Fetch histogram bucket data for a metric."""
    resp = httpx.get(VLLM_METRICS_URL, timeout=5.0)
    resp.raise_for_status()

    buckets: list[tuple[float, int]] = []
    for line in resp.text.split("\n"):
        if f"{metric_name}_bucket" in line and 'le="' in line:
            le_start = line.find('le="') + 4
            le_end = line.find('"', le_start)
            le_val = line[le_start:le_end]
            count = int(float(line.split()[-1]))
            if le_val != "+Inf":
                buckets.append((float(le_val), count))

    return sorted(buckets, key=lambda x: x[0])


def compute_percentile_from_histogram(
    buckets: list[tuple[float, int]], percentile: float
) -> float | None:
    """Estimate percentile from histogram buckets."""
    if not buckets:
        return None

    total = buckets[-1][1]
    if total == 0:
        return None

    target = total * percentile / 100.0
    for le, count in buckets:
        if count >= target:
            return le
    return buckets[-1][0]


def run_eval() -> tuple[dict[str, Any], float]:
    """Run eval and return report + elapsed time."""
    start = time.time()
    subprocess.run(
        ["python", "evals/run_eval.py"],
        cwd=REPO_ROOT,
        check=True,
    )
    elapsed = time.time() - start

    with open(REPO_ROOT / "evals/reports/judge/llm_judge_report.json") as f:
        return json.load(f), elapsed


def format_speed_report(results: dict[str, Any]) -> str:
    """Format results as markdown."""
    lines = [
        "# Speed Benchmark Report",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Run Date** | {results['timestamp'][:10]} |",
        f"| **Total Duration** | {results['total_time_seconds']:.1f}s |",
        f"| **Requests** | {results['requests_completed']} |",
        "",
        "## Throughput",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Request Throughput** | {results['throughput_req_s']:.2f} req/s |",
        f"| **Total Tokens/sec** | {results['total_tokens_per_second']:.1f} tok/s |",
        f"| **Generation Tokens/sec** | {results['generation_tokens_per_second']:.1f} tok/s |",
        "",
        "## Latency",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Avg TTFT** | {results['avg_ttft_seconds']:.2f}s |",
        f"| **TTFT p50** | {results['ttft_p50_seconds']:.2f}s |",
        f"| **TTFT p90** | {results['ttft_p90_seconds']:.2f}s |",
        "",
        "## Concurrency (during run)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Peak Running** | {results['peak_running']} |",
        f"| **Peak Waiting** | {results['peak_waiting']} |",
        f"| **Peak KV Cache** | {results['peak_kv_cache_pct']:.1f}% |",
        "",
        "## Token Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Prompt Tokens** | {results['prompt_tokens']:,} |",
        f"| **Generation Tokens** | {results['generation_tokens']:,} |",
        f"| **Total Tokens** | {results['total_tokens']:,} |",
        f"| **Avg Tokens/Request** | {results['avg_tokens_per_request']:.0f} |",
        "",
        "## Cache Efficiency",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Prefix Cache Hit Rate** | {results['prefix_cache_hit_rate'] * 100:.1f}% |",
        f"| **Cached Tokens** | {results['prefix_cache_hits']:,} |",
        f"| **Preemptions** | {results['preemptions']} |",
        "",
        "## Accuracy (from eval)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Overall Accuracy** | {results['overall_accuracy'] * 100:.1f}% |",
        f"| **Positive Accuracy** | {results['positive_accuracy'] * 100:.1f}% |",
        f"| **Negative Accuracy** | {results['negative_accuracy'] * 100:.1f}% |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Run speed benchmark and save results."""
    print("Fetching initial vLLM metrics...")
    try:
        before = VLLMSnapshot.fetch()
    except Exception as e:
        print(f"Error: Cannot connect to vLLM server at {VLLM_METRICS_URL}")
        print(f"Details: {e}")
        print("\nMake sure vLLM server is running:")
        print("  scicode-lint vllm-server start")
        return

    # Start background monitoring
    monitor = MetricsMonitor(interval=2.0)
    monitor.start()

    print("Running eval suite...")
    eval_report, total_time = run_eval()

    # Stop monitoring and get stats
    runtime_stats = monitor.stop()

    print("Fetching final vLLM metrics...")
    after = VLLMSnapshot.fetch()

    # Fetch TTFT histogram for percentiles
    ttft_buckets = fetch_histogram_buckets("vllm:time_to_first_token_seconds")

    # Compute deltas
    prompt_tokens = after.prompt_tokens - before.prompt_tokens
    generation_tokens = after.generation_tokens - before.generation_tokens
    total_tokens = prompt_tokens + generation_tokens
    requests = after.ttft_count - before.ttft_count
    ttft_delta = after.ttft_sum - before.ttft_sum
    cache_queries = after.prefix_cache_queries - before.prefix_cache_queries
    cache_hits = after.prefix_cache_hits - before.prefix_cache_hits
    preemptions = after.preemptions - before.preemptions

    # Compute derived metrics
    throughput = requests / total_time if total_time > 0 else 0
    avg_ttft = ttft_delta / requests if requests > 0 else 0
    gen_speed = generation_tokens / total_time if total_time > 0 else 0
    total_tok_speed = total_tokens / total_time if total_time > 0 else 0
    cache_hit_rate = cache_hits / cache_queries if cache_queries > 0 else 0
    avg_tokens_per_req = total_tokens / requests if requests > 0 else 0

    results = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": round(total_time, 1),
        "requests_completed": requests,
        "throughput_req_s": round(throughput, 3),
        "avg_ttft_seconds": round(avg_ttft, 2),
        "ttft_p50_seconds": round(compute_percentile_from_histogram(ttft_buckets, 50) or 0, 2),
        "ttft_p90_seconds": round(compute_percentile_from_histogram(ttft_buckets, 90) or 0, 2),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_request": round(avg_tokens_per_req, 0),
        "generation_tokens_per_second": round(gen_speed, 1),
        "total_tokens_per_second": round(total_tok_speed, 1),
        "prefix_cache_queries": cache_queries,
        "prefix_cache_hits": cache_hits,
        "prefix_cache_hit_rate": round(cache_hit_rate, 3),
        "preemptions": preemptions,
        "peak_running": runtime_stats.peak_running,
        "peak_waiting": runtime_stats.peak_waiting,
        "peak_kv_cache_pct": round(runtime_stats.peak_kv_cache_pct, 1),
        "overall_accuracy": eval_report["overall_accuracy"],
        "positive_accuracy": eval_report["positive_accuracy"],
        "negative_accuracy": eval_report["negative_accuracy"],
    }

    # Save results
    out_dir = REPO_ROOT / "benchmarks/reports/speed"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = out_dir / f"benchmark_{ts}.json"
    md_file = out_dir / "BENCHMARK_SUMMARY.md"

    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    with open(md_file, "w") as f:
        f.write(format_speed_report(results))

    print(f"\n{'=' * 60}")
    print("Speed Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Total time:         {total_time:.1f}s")
    print(f"Requests:           {requests}")
    print(f"Throughput:         {throughput:.2f} req/s")
    print(f"Total tok/s:        {total_tok_speed:.1f}")
    print(f"Gen tok/s:          {gen_speed:.1f}")
    print(f"Avg TTFT:           {avg_ttft:.2f}s")
    print(f"Peak running:       {runtime_stats.peak_running}")
    print(f"Peak waiting:       {runtime_stats.peak_waiting}")
    print(f"Peak KV cache:      {runtime_stats.peak_kv_cache_pct:.1f}%")
    print(f"Cache hit rate:     {cache_hit_rate * 100:.1f}%")
    print(f"Overall accuracy:   {eval_report['overall_accuracy'] * 100:.1f}%")
    print(f"{'=' * 60}")
    print(f"\nResults saved to: {json_file}")
    print(f"Summary saved to: {md_file}")


if __name__ == "__main__":
    main()
