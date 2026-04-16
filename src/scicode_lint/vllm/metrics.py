"""vLLM Prometheus metrics fetching.

Parses the ``/metrics`` endpoint exposed by vLLM and returns a flat dict
of key–value pairs. Used by the CLI monitor (``scicode-lint vllm-server monitor``).
"""

from __future__ import annotations

import re

import httpx
from loguru import logger

# Value pattern handles scientific notation (e.g. 1.572691e+06) from Prometheus.
_NUM = r"[\d.]+(?:[eE][+-]?\d+)?"
_GAUGE_PATTERNS: list[tuple[str, str]] = [
    (rf"vllm:kv_cache_usage_perc\{{[^}}]*\}}\s+({_NUM})", "kv_cache_pct"),
    (rf"vllm:num_requests_running\{{[^}}]*\}}\s+({_NUM})", "requests_running"),
    (rf"vllm:num_requests_waiting\{{[^}}]*\}}\s+({_NUM})", "requests_waiting"),
    (rf"vllm:num_requests_swapped\{{[^}}]*\}}\s+({_NUM})", "requests_swapped"),
    (rf"vllm:prompt_tokens_total\{{[^}}]*\}}\s+({_NUM})", "prompt_tokens_total"),
    (rf"vllm:generation_tokens_total\{{[^}}]*\}}\s+({_NUM})", "generation_tokens_total"),
    (rf"vllm:prefix_cache_hits_total\{{[^}}]*\}}\s+({_NUM})", "prefix_cache_hits"),
    (rf"vllm:prefix_cache_queries_total\{{[^}}]*\}}\s+({_NUM})", "prefix_cache_queries"),
    (rf"vllm:num_preemptions_total\{{[^}}]*\}}\s+({_NUM})", "num_preemptions"),
    (rf"vllm:e2e_request_latency_seconds_sum\{{[^}}]*\}}\s+({_NUM})", "e2e_latency_sum"),
    (rf"vllm:e2e_request_latency_seconds_count\{{[^}}]*\}}\s+({_NUM})", "e2e_latency_count"),
    (rf"vllm:time_to_first_token_seconds_sum\{{[^}}]*\}}\s+({_NUM})", "ttft_sum"),
    (rf"vllm:time_to_first_token_seconds_count\{{[^}}]*\}}\s+({_NUM})", "ttft_count"),
    (rf"vllm:inter_token_latency_seconds_sum\{{[^}}]*\}}\s+({_NUM})", "itl_sum"),
    (rf"vllm:inter_token_latency_seconds_count\{{[^}}]*\}}\s+({_NUM})", "itl_count"),
]

_REQUEST_REASONS = ("stop", "length", "abort", "error")


def parse_metrics(text: str) -> dict[str, float | int]:
    """Parse Prometheus metrics text into a flat dict.

    Pure function — no network. Useful for testing and when metrics text is
    fetched through a different path.
    """
    result: dict[str, float | int] = {}

    for pattern, key in _GAUGE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            result[key] = float(m.group(1))

    # Request outcomes by finished_reason
    for reason in _REQUEST_REASONS:
        m = re.search(
            rf'vllm:request_success_total\{{[^}}]*finished_reason="{reason}"[^}}]*\}}\s+({_NUM})',
            text,
        )
        if m:
            result[f"req_success_{reason}"] = float(m.group(1))

    # Label-embedded config values
    m = re.search(r'num_gpu_blocks="(\d+)"', text)
    if m:
        result["num_gpu_blocks"] = int(m.group(1))
    m = re.search(r'block_size="(\d+)"', text)
    if m:
        result["block_size"] = int(m.group(1))
    m = re.search(r'gpu_memory_utilization="([\d.]+)"', text)
    if m:
        result["gpu_util_cap"] = float(m.group(1))

    return result


def fetch_metrics(base_url: str, timeout: float = 3.0) -> dict[str, float | int]:
    """Fetch and parse vLLM Prometheus metrics.

    Args:
        base_url: vLLM server URL (e.g. ``http://localhost:5001``).
        timeout: HTTP timeout in seconds.

    Returns:
        Dict with keys like ``kv_cache_pct``, ``requests_running``,
        ``prefix_cache_hits``, ``e2e_latency_sum``, ``req_success_stop``, etc.
        Returns an empty dict if the server is unreachable.
    """
    url = f"{base_url.rstrip('/')}/metrics"
    try:
        resp = httpx.get(url, timeout=timeout)
        if resp.status_code != 200:
            return {}
        return parse_metrics(resp.text)
    except (httpx.HTTPError, ConnectionError) as e:
        logger.debug(f"vLLM metrics fetch failed ({type(e).__name__}: {e})")
        return {}
