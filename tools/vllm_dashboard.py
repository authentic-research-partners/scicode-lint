"""Simple Streamlit dashboard for vLLM monitoring.

Run with:
    streamlit run tools/vllm_dashboard.py

Or with custom URL:
    streamlit run tools/vllm_dashboard.py -- --url http://localhost:5001
"""

import argparse
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

# Optional GPU monitoring via nvidia-ml-py (provides pynvml module)
try:
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False


def get_gpu_utilization(device_index: int = 0) -> int | None:
    """Get GPU core utilization percentage."""
    if not PYNVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return int(util.gpu)
    except Exception:
        return None


# Default metrics log path (written by VLLMMetricsMonitor during evals)
METRICS_LOG_PATH = Path(__file__).parent.parent / "evals" / "reports" / "vllm_metrics.log"

# Page config
st.set_page_config(
    page_title="vLLM Monitor",
    page_icon="🚀",
    layout="wide",
)

# Custom CSS for compact metrics
st.markdown(
    """
<style>
    .config-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
    .config-item {
        display: inline-block;
        margin-right: 24px;
    }
    .config-label {
        font-size: 12px;
        color: #666;
        margin-bottom: 2px;
    }
    .config-value {
        font-size: 14px;
        font-weight: 600;
        color: #333;
    }
    .small-metric .metric-label {
        font-size: 12px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus metrics format."""
    metrics = {}
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        try:
            value = float(line.split()[-1])

            if "{" in line:
                base_name = line.split("{")[0]
                full_name = line.split()[0]
                metrics[full_name] = value
                # Aggregate by base name
                if base_name in metrics:
                    metrics[base_name] += value
                else:
                    metrics[base_name] = value
            else:
                parts = line.split()
                if len(parts) >= 2:
                    metrics[parts[0]] = value
        except (ValueError, IndexError):
            continue
    return metrics


def parse_cache_config(text: str) -> dict[str, str]:
    """Extract cache config from vllm:cache_config_info labels."""
    config = {}
    for line in text.split("\n"):
        if "vllm:cache_config_info{" in line:
            # Extract labels
            match = re.search(r"\{([^}]+)\}", line)
            if match:
                labels = match.group(1)
                for item in labels.split(","):
                    if "=" in item:
                        key, val = item.split("=", 1)
                        config[key.strip()] = val.strip().strip('"')
            break
    return config


def fetch_metrics(
    base_url: str,
) -> tuple[dict[str, float] | None, dict[str, str]]:
    """Fetch metrics from vLLM server."""
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=2)
        if resp.status_code == 200:
            return parse_metrics(resp.text), parse_cache_config(resp.text)
    except Exception:  # noqa: S110 — best-effort poll; server-down rendered as N/A
        pass
    return None, {}


def fetch_server_info(base_url: str) -> dict[str, str | int] | None:
    """Fetch server configuration from vLLM."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                model_info = data["data"][0]
                return {
                    "model": model_info.get("id", "Unknown"),
                    "max_model_len": model_info.get("max_model_len", "N/A"),
                }
    except Exception:  # noqa: S110 — best-effort poll; server-down rendered as N/A
        pass
    return None


def moving_average(values: list[float], window: int = 5) -> list[float]:
    """Calculate moving average."""
    if len(values) < 2:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start : i + 1]) / (i - start + 1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:5001", help="vLLM server URL")
    args, _ = parser.parse_known_args()
    base_url = args.url

    st.title("🚀 vLLM Monitor")

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = {
            "time": deque(maxlen=120),
            "running": deque(maxlen=120),
            "waiting": deque(maxlen=120),
            "throughput": deque(maxlen=120),
            "tokens_per_sec": deque(maxlen=120),
            "cache": deque(maxlen=120),
            "gpu_util": deque(maxlen=120),
            "prefix_delta_hits": deque(maxlen=120),
            "prefix_delta_queries": deque(maxlen=120),
            "last_finished": None,  # None = not initialized
            "last_tokens": None,  # None = not initialized
            "last_prefix_queries": None,  # None = not initialized
            "last_prefix_hits": None,  # None = not initialized
            "last_time": time.time(),
        }

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        refresh_interval = st.slider("Refresh (s)", 1, 10, 2)
        ma_window = st.slider("Moving average window", 1, 20, 5)
        show_prefix_hit = st.checkbox("Show Prefix Hit %", value=False)
        st.markdown("---")
        if st.button("Reset Charts"):
            for key in st.session_state.history:
                if isinstance(st.session_state.history[key], deque):
                    st.session_state.history[key].clear()
            st.session_state.history["last_finished"] = None
            st.session_state.history["last_tokens"] = None
            st.session_state.history["last_time"] = time.time()

        st.markdown("---")
        st.caption("Metrics Log")
        if METRICS_LOG_PATH.exists():
            log_size = METRICS_LOG_PATH.stat().st_size
            log_lines = len(METRICS_LOG_PATH.read_text().splitlines())
            st.text(f"{log_lines} rows ({log_size / 1024:.1f} KB)")
            if st.button("Clear Log File"):
                METRICS_LOG_PATH.write_text("")
                st.success("Log cleared!")
        else:
            st.text("No log file yet")

    # Fetch data
    result = fetch_metrics(base_url)
    if result[0] is None:
        st.error(f"Cannot connect to vLLM at {base_url}")
        time.sleep(refresh_interval)
        st.rerun()
        return

    metrics, cache_config = result
    assert metrics is not None  # Checked above
    server_info = fetch_server_info(base_url)

    # Server config - compact single line
    model_name = server_info.get("model", "Unknown") if server_info else "Unknown"
    max_len = server_info.get("max_model_len", "N/A") if server_info else "N/A"
    gpu_blocks = cache_config.get("num_gpu_blocks", "N/A")
    gpu_mem_setting = cache_config.get("gpu_memory_utilization", "N/A")
    block_size = cache_config.get("block_size", "16")

    # Calculate total KV cache capacity in tokens
    try:
        kv_cache_tokens = int(gpu_blocks) * int(block_size)
        kv_cache_display = f"{kv_cache_tokens:,}"
    except (ValueError, TypeError):
        kv_cache_display = "N/A"

    config_html = f"""
    <div class="config-box">
        <span class="config-item">
            <span class="config-label">Model</span><br>
            <span class="config-value">{model_name}</span>
        </span>
        <span class="config-item">
            <span class="config-label">Max Tokens (setting)</span><br>
            <span class="config-value">{max_len:,}</span>
        </span>
        <span class="config-item">
            <span class="config-label">KV Cache Capacity</span><br>
            <span class="config-value">{kv_cache_display} tokens</span>
        </span>
        <span class="config-item">
            <span class="config-label">KV Blocks</span><br>
            <span class="config-value">{gpu_blocks}</span>
        </span>
        <span class="config-item">
            <span class="config-label">Block Size</span><br>
            <span class="config-value">{block_size} tokens</span>
        </span>
        <span class="config-item">
            <span class="config-label">GPU Mem Setting</span><br>
            <span class="config-value">{float(gpu_mem_setting) * 100:.0f}%</span>
        </span>
        <span class="config-item">
            <span class="config-label">Server</span><br>
            <span class="config-value">{base_url.replace("http://", "")}</span>
        </span>
    </div>
    """
    st.markdown(config_html, unsafe_allow_html=True)

    # Extract metrics
    running = int(metrics.get("vllm:num_requests_running", 0))
    waiting = int(metrics.get("vllm:num_requests_waiting", 0))
    finished = metrics.get("vllm:request_success_total", 0)
    total_tokens = metrics.get("vllm:generation_tokens_total", 0)
    cache_pct = metrics.get("vllm:kv_cache_usage_perc", 0) * 100
    prefix_queries = metrics.get("vllm:prefix_cache_queries_total", 0)
    prefix_hits = metrics.get("vllm:prefix_cache_hits_total", 0)

    # Calculate throughput (avoid spike on first run)
    now = time.time()
    elapsed = now - st.session_state.history["last_time"]

    if st.session_state.history["last_finished"] is None:
        # First run - initialize without calculating throughput
        throughput = 0.0
        tokens_per_sec = 0.0
        delta_queries = 0.0
        delta_hits = 0.0
    elif elapsed > 0:
        throughput = (finished - st.session_state.history["last_finished"]) / elapsed
        tokens_per_sec = (total_tokens - st.session_state.history["last_tokens"]) / elapsed
        # Prefix cache deltas for windowed hit rate
        delta_queries = prefix_queries - st.session_state.history["last_prefix_queries"]
        delta_hits = prefix_hits - st.session_state.history["last_prefix_hits"]
    else:
        throughput = 0.0
        tokens_per_sec = 0.0
        delta_queries = 0.0
        delta_hits = 0.0

    st.session_state.history["last_finished"] = finished
    st.session_state.history["last_tokens"] = total_tokens
    st.session_state.history["last_prefix_queries"] = prefix_queries
    st.session_state.history["last_prefix_hits"] = prefix_hits
    st.session_state.history["last_time"] = now

    # Get GPU utilization
    gpu_util = get_gpu_utilization()

    # Update history
    st.session_state.history["time"].append(datetime.now())
    st.session_state.history["running"].append(running)
    st.session_state.history["waiting"].append(waiting)
    st.session_state.history["throughput"].append(throughput)
    st.session_state.history["tokens_per_sec"].append(tokens_per_sec)
    st.session_state.history["cache"].append(cache_pct)
    st.session_state.history["gpu_util"].append(gpu_util if gpu_util is not None else 0)
    st.session_state.history["prefix_delta_hits"].append(delta_hits)
    st.session_state.history["prefix_delta_queries"].append(delta_queries)

    # Live stats
    cols = st.columns(9)
    with cols[0]:
        st.metric("Running", running, help="Requests currently being processed by vLLM")
    with cols[1]:
        st.metric("Queued", waiting, help="Requests waiting in queue for processing")
    with cols[2]:
        throughput_list = list(st.session_state.history["throughput"])
        smoothed = moving_average(throughput_list, ma_window)[-1] if throughput_list else 0
        st.metric("Req/s", f"{smoothed:.1f}")
    with cols[3]:
        tokens_list = list(st.session_state.history["tokens_per_sec"])
        smoothed_tokens = moving_average(tokens_list, ma_window)[-1] if tokens_list else 0
        st.metric("Tok/s", f"{smoothed_tokens:.0f}")
    with cols[4]:
        st.metric("KV Cache", f"{cache_pct:.0f}%", help="KV cache usage (100% = memory-bound)")
    with cols[5]:
        # Compute hit rate over the MA window (sum of hits / sum of queries)
        hits_list = list(st.session_state.history["prefix_delta_hits"])
        queries_list = list(st.session_state.history["prefix_delta_queries"])
        window_hits = sum(hits_list[-ma_window:]) if hits_list else 0
        window_queries = sum(queries_list[-ma_window:]) if queries_list else 0
        hit_rate = (window_hits / window_queries * 100) if window_queries > 0 else 0
        st.metric("Prefix Hit", f"{hit_rate:.0f}%", help="Prefix cache hit rate (over MA window)")
    with cols[6]:
        if gpu_util is not None:
            st.metric("GPU", f"{gpu_util}%", help="GPU compute utilization")
        else:
            st.metric("GPU", "N/A", help="pip install nvidia-ml-py")
    with cols[7]:
        st.metric("Total Reqs", f"{int(finished):,}", help="Completed since server start")
    with cols[8]:
        st.metric("Total Tokens", f"{int(total_tokens):,}", help="Tokens generated since start")

    # Charts - 3 in a row with aligned Y-axes
    if len(st.session_state.history["time"]) > 1:
        import altair as alt
        import pandas as pd

        throughput_ma = moving_average(list(st.session_state.history["throughput"]), ma_window)
        tokens_ma = moving_average(list(st.session_state.history["tokens_per_sec"]), ma_window)

        # Compute rolling prefix hit rate for chart
        hits_list = list(st.session_state.history["prefix_delta_hits"])
        queries_list = list(st.session_state.history["prefix_delta_queries"])
        prefix_hit_for_chart = []
        for i in range(len(hits_list)):
            start = max(0, i - ma_window + 1)
            window_hits = sum(hits_list[start : i + 1])
            window_queries = sum(queries_list[start : i + 1])
            rate = (window_hits / window_queries * 100) if window_queries > 0 else 0
            prefix_hit_for_chart.append(rate)

        df: pd.DataFrame = pd.DataFrame(
            {
                "Time": list(st.session_state.history["time"]),
                "Running": list(st.session_state.history["running"]),
                "Queued": list(st.session_state.history["waiting"]),
                "Throughput": throughput_ma,
                "Tokens/s": tokens_ma,
                "KV Cache %": list(st.session_state.history["cache"]),
                "Prefix Hit %": prefix_hit_for_chart,
            }
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.caption("Request Queue: Running (solid) / Queued (dashed)")
            # Melt for multi-line chart
            df_queue = df.melt(
                id_vars=["Time"],
                value_vars=["Running", "Queued"],
                var_name="Type",
                value_name="Count",
            )
            max_queue = max(df_queue["Count"].max(), 1)
            line = (
                alt.Chart(df_queue)
                .mark_line()
                .encode(
                    x=alt.X("Time:T", axis=alt.Axis(format="%H:%M", title=None)),
                    y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, max_queue * 1.1]), title=None),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(domain=["Running", "Queued"], range=["#1f77b4", "#e6a000"]),
                        legend=None,
                    ),
                    strokeDash=alt.StrokeDash(
                        "Type:N",
                        scale=alt.Scale(domain=["Running", "Queued"], range=[[1, 0], [4, 4]]),
                        legend=None,
                    ),
                )
            )
            points = (
                alt.Chart(df_queue)
                .mark_point(opacity=0, size=100)
                .encode(
                    x="Time:T",
                    y="Count:Q",
                    tooltip=[
                        alt.Tooltip("Time:T", format="%H:%M:%S", title="Time"),
                        alt.Tooltip("Type:N", title="Type"),
                        alt.Tooltip("Count:Q", title="Count"),
                    ],
                )
            )
            chart = (line + points).properties(height=180)
            st.altair_chart(chart, use_container_width=True)

        with c2:
            if show_prefix_hit:
                st.caption("KV Cache % (solid) / Prefix Hit % (dashed)")
                # Melt for dual-line chart
                df_cache = df.melt(
                    id_vars=["Time"],
                    value_vars=["KV Cache %", "Prefix Hit %"],
                    var_name="Metric",
                    value_name="Percent",
                )
                chart = (
                    alt.Chart(df_cache)
                    .mark_line()
                    .encode(
                        x=alt.X("Time:T", axis=alt.Axis(format="%H:%M", title=None)),
                        y=alt.Y("Percent:Q", scale=alt.Scale(domain=[0, 100]), title=None),
                        color=alt.Color(
                            "Metric:N",
                            scale=alt.Scale(
                                domain=["KV Cache %", "Prefix Hit %"],
                                range=["#2ca02c", "#9467bd"],
                            ),
                            legend=None,
                        ),
                        strokeDash=alt.StrokeDash(
                            "Metric:N",
                            scale=alt.Scale(
                                domain=["KV Cache %", "Prefix Hit %"],
                                range=[[1, 0], [4, 4]],
                            ),
                            legend=None,
                        ),
                    )
                    .properties(height=180)
                )
            else:
                st.caption("KV Cache %")
                chart = (
                    alt.Chart(df)
                    .mark_line(color="#2ca02c")
                    .encode(
                        x=alt.X("Time:T", axis=alt.Axis(format="%H:%M", title=None)),
                        y=alt.Y("KV Cache %:Q", scale=alt.Scale(domain=[0, 100]), title=None),
                    )
                    .properties(height=180)
                )
            st.altair_chart(chart, use_container_width=True)

        with c3:
            st.caption(f"Req/s ({ma_window}-pt moving avg)")
            max_tp = max(df["Throughput"].max(), 0.1)
            chart = (
                alt.Chart(df)
                .mark_line(color="#1f77b4")
                .encode(
                    x=alt.X("Time:T", axis=alt.Axis(format="%H:%M", title=None)),
                    y=alt.Y("Throughput:Q", scale=alt.Scale(domain=[0, max_tp * 1.1]), title=None),
                )
                .properties(height=180)
            )
            st.altair_chart(chart, use_container_width=True)

        with c4:
            st.caption(f"Tok/s ({ma_window}-pt moving avg)")
            max_tok = max(df["Tokens/s"].max(), 1)
            chart = (
                alt.Chart(df)
                .mark_line(color="#ff7f0e")
                .encode(
                    x=alt.X("Time:T", axis=alt.Axis(format="%H:%M", title=None)),
                    y=alt.Y("Tokens/s:Q", scale=alt.Scale(domain=[0, max_tok * 1.1]), title=None),
                )
                .properties(height=180)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Collecting data...")

    # All metrics expander
    with st.expander("All Metrics"):
        vllm_metrics = {
            k: v for k, v in sorted(metrics.items()) if k.startswith("vllm:") and "{" not in k
        }
        col1, col2 = st.columns(2)
        items = list(vllm_metrics.items())
        for i, (name, value) in enumerate(items):
            with col1 if i % 2 == 0 else col2:
                st.text(f"{name.replace('vllm:', '')}: {value:.2f}")

    # Auto-refresh
    time.sleep(refresh_interval)
    st.rerun()


if __name__ == "__main__":
    main()
