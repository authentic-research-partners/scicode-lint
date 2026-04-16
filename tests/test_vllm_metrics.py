"""Tests for vLLM Prometheus metrics parser."""

from __future__ import annotations

from scicode_lint.vllm.metrics import parse_metrics

SAMPLE_PROMETHEUS = """\
# HELP vllm:kv_cache_usage_perc KV cache usage percentage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="qwen3-8b-fp8",num_gpu_blocks="1024",block_size="16",gpu_memory_utilization="0.9"} 0.234
vllm:num_requests_running{model_name="qwen3-8b-fp8"} 7.0
vllm:num_requests_waiting{model_name="qwen3-8b-fp8"} 0.0
vllm:num_requests_swapped{model_name="qwen3-8b-fp8"} 0.0
vllm:prompt_tokens_total{model_name="qwen3-8b-fp8"} 1.572691e+06
vllm:generation_tokens_total{model_name="qwen3-8b-fp8"} 245000.0
vllm:prefix_cache_hits_total{model_name="qwen3-8b-fp8"} 500000.0
vllm:prefix_cache_queries_total{model_name="qwen3-8b-fp8"} 1000000.0
vllm:num_preemptions_total{model_name="qwen3-8b-fp8"} 0.0
vllm:e2e_request_latency_seconds_sum{model_name="qwen3-8b-fp8"} 120.5
vllm:e2e_request_latency_seconds_count{model_name="qwen3-8b-fp8"} 50.0
vllm:time_to_first_token_seconds_sum{model_name="qwen3-8b-fp8"} 5.0
vllm:time_to_first_token_seconds_count{model_name="qwen3-8b-fp8"} 50.0
vllm:inter_token_latency_seconds_sum{model_name="qwen3-8b-fp8"} 10.0
vllm:inter_token_latency_seconds_count{model_name="qwen3-8b-fp8"} 500.0
vllm:request_success_total{model_name="qwen3-8b-fp8",finished_reason="stop"} 42.0
vllm:request_success_total{model_name="qwen3-8b-fp8",finished_reason="length"} 3.0
vllm:request_success_total{model_name="qwen3-8b-fp8",finished_reason="abort"} 1.0
vllm:request_success_total{model_name="qwen3-8b-fp8",finished_reason="error"} 0.0
"""


class TestParseMetrics:
    """Pure-function tests — no network."""

    def test_kv_cache_pct(self) -> None:
        m = parse_metrics(SAMPLE_PROMETHEUS)
        assert m["kv_cache_pct"] == 0.234

    def test_request_counts(self) -> None:
        m = parse_metrics(SAMPLE_PROMETHEUS)
        assert m["requests_running"] == 7.0
        assert m["requests_waiting"] == 0.0
        assert m["requests_swapped"] == 0.0

    def test_throughput_counters(self) -> None:
        m = parse_metrics(SAMPLE_PROMETHEUS)
        assert m["prompt_tokens_total"] == 1_572_691.0
        assert m["generation_tokens_total"] == 245_000.0

    def test_scientific_notation(self) -> None:
        """Prometheus emits large counters in scientific notation."""
        m = parse_metrics(SAMPLE_PROMETHEUS)
        # 1.572691e+06 should parse as 1572691
        assert m["prompt_tokens_total"] == 1_572_691.0

    def test_prefix_cache(self) -> None:
        m = parse_metrics(SAMPLE_PROMETHEUS)
        assert m["prefix_cache_hits"] == 500_000.0
        assert m["prefix_cache_queries"] == 1_000_000.0

    def test_latency_sums_and_counts(self) -> None:
        m = parse_metrics(SAMPLE_PROMETHEUS)
        assert m["e2e_latency_sum"] == 120.5
        assert m["e2e_latency_count"] == 50.0
        assert m["ttft_sum"] == 5.0
        assert m["ttft_count"] == 50.0
        assert m["itl_sum"] == 10.0
        assert m["itl_count"] == 500.0

    def test_request_outcomes(self) -> None:
        m = parse_metrics(SAMPLE_PROMETHEUS)
        assert m["req_success_stop"] == 42.0
        assert m["req_success_length"] == 3.0
        assert m["req_success_abort"] == 1.0
        assert m["req_success_error"] == 0.0

    def test_label_embedded_config(self) -> None:
        m = parse_metrics(SAMPLE_PROMETHEUS)
        assert m["num_gpu_blocks"] == 1024
        assert m["block_size"] == 16
        assert m["gpu_util_cap"] == 0.9

    def test_empty_text(self) -> None:
        m = parse_metrics("")
        assert m == {}

    def test_unrelated_text(self) -> None:
        m = parse_metrics("# Some unrelated content\nfoo bar baz")
        assert m == {}

    def test_partial_text(self) -> None:
        """Missing metrics should be omitted, not raise."""
        partial = 'vllm:num_requests_running{model_name="x"} 3.0\n'
        m = parse_metrics(partial)
        assert m == {"requests_running": 3.0}
