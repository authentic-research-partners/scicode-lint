#!/usr/bin/env python3
"""Benchmark: do wire-level schema bounds slow structured generation under load?

Background. Pydantic ``Field`` bounds — string ``maxLength``, list ``maxItems``,
numeric ``minimum``/``maximum`` — compile into vLLM's XGrammar grammar slow path
when sent on the wire. ``vllm_schema()`` (``src/scicode_lint/llm/models.py``) now
strips them and keeps them on the model for post-decode validation only; this eval is
the regression guard that justifies and protects that decision. The effect is
specific to XGrammar's grammar compilation — a single bounded field is enough to
collapse throughput at high concurrency on this stack.

This eval measures the effect on scicode-lint's actual ``DetectionResult`` schema,
which carries ``maxLength`` (200, 400) and numeric ``minimum``/``maximum`` (confidence).

Method. Fire ``--requests`` structured calls at concurrency ``--concurrency`` against
a live vLLM, twice over an *identical* prompt set: once with a bounded wire schema
(constraint keys present, ``$ref`` inlined) and once with those keys stripped (what
``vllm_schema`` now ships). Report success rate, p50/p95/max latency, wall-clock, and
throughput for each, plus the bounded-vs-unbounded speedup. The wire schema is the
only independent variable: both conditions reuse the client's own ``_build_api_params``
(same model, thinking budget, sampling) and bypass the retry/ladder loop so it can't
confound the latency.

Thinking is disabled by default so the JSON-decode phase — where the grammar mask
applies — dominates and the grammar cost is isolated. Re-run with
``--thinking-budget 3584`` for the production-realistic mix.

Requires vLLM server running::

    scicode-lint vllm-server start
    python evals/wire_bounds_throughput.py --concurrency 64 --requests 128
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel

from scicode_lint.config import load_llm_config
from scicode_lint.exceptions import LLMConnectionError
from scicode_lint.llm.client import VLLMClient, create_client
from scicode_lint.llm.models import DetectionResult

# Constraint keys XGrammar compiles into its grammar; stripping them is the
# unbounded condition.
_BOUND_KEYS = frozenset(
    {
        "maxLength",
        "minLength",
        "maxItems",
        "minItems",
        "pattern",
        "multipleOf",
        "maximum",
        "minimum",
        "exclusiveMaximum",
        "exclusiveMinimum",
    }
)

_SYSTEM_PROMPT = (
    "You are a Python code reviewer checking for machine-learning issues. "
    "Return a DetectionResult JSON object matching the schema."
)

# Distinct snippets so the user turn varies across requests — keeps the JSON-decode
# work real rather than letting prefix caching trivialize every call. The detection
# question is the same shape so response sizes are comparable.
_CODE_SNIPPETS: tuple[str, ...] = (
    "from sklearn.model_selection import KFold\n"
    "kf = KFold(n_splits=5, shuffle=False)\n"
    "for tr, te in kf.split(X):\n    model.fit(X[tr], y[tr])",
    "scaler = StandardScaler().fit(X)\n"
    "X = scaler.transform(X)\n"
    "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)",
    "df = df.fillna(df.mean())\n"
    "train, test = df[:800], df[800:]\n"
    "clf.fit(train.drop('y', axis=1), train['y'])",
    "X_train, X_test = train_test_split(X, test_size=0.3)\n"
    "pca = PCA(n_components=10).fit(X)\n"
    "X_train = pca.transform(X_train)",
    "for epoch in range(100):\n"
    "    model.train()\n"
    "    loss = criterion(model(X_val), y_val)\n"
    "    loss.backward()",
)


def _inline_refs(model: type[BaseModel]) -> dict[str, Any]:
    """Inline ``$ref`` and drop ``title`` but KEEP constraint keys — the bounded
    baseline. Production ``vllm_schema`` additionally strips the constraint keys;
    this helper deliberately does not, so the benchmark can compare bounded vs
    stripped rather than two identical (already-stripped) schemas.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                return _resolve(defs[obj["$ref"].rsplit("/", 1)[-1]])
            return {k: _resolve(v) for k, v in obj.items() if k != "title"}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj

    resolved: dict[str, Any] = _resolve(raw)
    return resolved


def _strip_bound_keys(obj: Any) -> Any:
    """Recursively drop XGrammar constraint keys from a JSON-schema dict tree."""
    if isinstance(obj, dict):
        return {k: _strip_bound_keys(v) for k, v in obj.items() if k not in _BOUND_KEYS}
    if isinstance(obj, list):
        return [_strip_bound_keys(v) for v in obj]
    return obj


def _build_prompts(count: int) -> list[str]:
    """Generate ``count`` distinct user prompts cycling the snippet pool."""
    prompts: list[str] = []
    for i in range(count):
        snippet = _CODE_SNIPPETS[i % len(_CODE_SNIPPETS)]
        prompts.append(
            f"Code (sample {i}):\n```python\n{snippet}\n```\n\n"
            "Question: does this code leak information from the test/validation "
            "set into training? Explain briefly."
        )
    return prompts


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (``pct`` in [0, 1]) over ``values``."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass
class CallResult:
    """Outcome of one structured call."""

    ok: bool
    latency: float
    finish_reason: str = "n/a"
    error: str | None = None


@dataclass
class ConditionStats:
    """Aggregated results for one wire-schema condition."""

    name: str
    wall_clock: float
    results: list[CallResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def success_rate(self) -> float:
        return self.ok_count / len(self.results) if self.results else 0.0

    @property
    def throughput(self) -> float:
        return self.ok_count / self.wall_clock if self.wall_clock else 0.0

    @property
    def ok_latencies(self) -> list[float]:
        return [r.latency for r in self.results if r.ok]

    def finish_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            key = r.error if r.error else r.finish_reason
            counts[key] = counts.get(key, 0) + 1
        return counts


async def _one_call(
    client: VLLMClient,
    semaphore: asyncio.Semaphore,
    user_prompt: str,
    json_schema: dict[str, Any],
    max_tokens: int,
    thinking_budget: int,
    timeout: float,
) -> CallResult:
    """Fire one structured call with a fixed wire schema; never raises."""
    async with semaphore:
        params = client._build_api_params(
            _SYSTEM_PROMPT,
            user_prompt,
            json_schema,
            "DetectionResult",
            max_tokens,
            thinking_budget=thinking_budget,
        )
        start = time.perf_counter()
        try:
            completion = await client._async_client.chat.completions.create(
                **params, timeout=timeout
            )
        except Exception as e:  # noqa: BLE001 - benchmark records every failure mode
            return CallResult(ok=False, latency=time.perf_counter() - start, error=type(e).__name__)
        latency = time.perf_counter() - start
        choice = completion.choices[0]
        content = choice.message.content
        finish = choice.finish_reason or "unknown"
        ok = content is not None and finish != "length"
        return CallResult(ok=ok, latency=latency, finish_reason=finish)


async def _run_condition(
    name: str,
    client: VLLMClient,
    prompts: list[str],
    json_schema: dict[str, Any],
    args: argparse.Namespace,
) -> ConditionStats:
    """Warm up, then fire all prompts at the target concurrency and time them."""
    semaphore = asyncio.Semaphore(args.concurrency)
    if args.warmup:
        logger.info(f"[{name}] warming up ({args.warmup} calls)")
        await asyncio.gather(
            *(
                _one_call(
                    client,
                    semaphore,
                    prompts[i % len(prompts)],
                    json_schema,
                    args.max_tokens,
                    args.thinking_budget,
                    args.timeout,
                )
                for i in range(args.warmup)
            )
        )

    logger.info(f"[{name}] firing {len(prompts)} calls at concurrency {args.concurrency}")
    start = time.perf_counter()
    results = await asyncio.gather(
        *(
            _one_call(
                client,
                semaphore,
                prompt,
                json_schema,
                args.max_tokens,
                args.thinking_budget,
                args.timeout,
            )
            for prompt in prompts
        )
    )
    wall_clock = time.perf_counter() - start
    return ConditionStats(name=name, wall_clock=wall_clock, results=list(results))


def _print_report(bounded: ConditionStats, unbounded: ConditionStats) -> None:
    """Print the side-by-side comparison and a verdict."""

    def row(label: str, b: str, u: str) -> str:
        return f"{label:<22} {b:>16} {u:>16}"

    print("\n" + "=" * 56)
    print(f"{'metric':<22} {'BOUNDED (wire)':>16} {'UNBOUNDED':>16}")
    print("-" * 56)
    print(row("requests", str(len(bounded.results)), str(len(unbounded.results))))
    print(
        row(
            "success rate",
            f"{bounded.success_rate:.0%}",
            f"{unbounded.success_rate:.0%}",
        )
    )
    print(row("wall-clock (s)", f"{bounded.wall_clock:.1f}", f"{unbounded.wall_clock:.1f}"))
    print(
        row(
            "throughput (req/s)",
            f"{bounded.throughput:.2f}",
            f"{unbounded.throughput:.2f}",
        )
    )
    for label, pct in (("p50 latency (s)", 0.50), ("p95 latency (s)", 0.95)):
        print(
            row(
                label,
                f"{_percentile(bounded.ok_latencies, pct):.1f}",
                f"{_percentile(unbounded.ok_latencies, pct):.1f}",
            )
        )
    print(
        row(
            "max latency (s)",
            f"{max(bounded.ok_latencies, default=float('nan')):.1f}",
            f"{max(unbounded.ok_latencies, default=float('nan')):.1f}",
        )
    )
    print("-" * 56)
    print(f"bounded finish/error breakdown:   {bounded.finish_breakdown()}")
    print(f"unbounded finish/error breakdown: {unbounded.finish_breakdown()}")
    print("=" * 56)

    speedup = bounded.wall_clock / unbounded.wall_clock if unbounded.wall_clock else float("nan")
    success_drop = unbounded.success_rate - bounded.success_rate
    print(f"\nWall-clock ratio (bounded / unbounded): {speedup:.2f}x")
    print(f"Success-rate drop from bounds:          {success_drop:+.0%}")
    if success_drop >= 0.20 or speedup >= 1.3:
        print(
            "\nVERDICT: REPRODUCED — wire-level schema bounds materially slow or "
            "break generation on this stack. Stripping the constraint keys from the "
            "wire schema is warranted."
        )
    else:
        print(
            "\nVERDICT: NOT REPRODUCED at this concurrency on DetectionResult. "
            "Bounds appear safe here; re-check at higher --concurrency and with "
            "list-valued schemas before concluding."
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=64, help="concurrent in-flight calls")
    parser.add_argument("--requests", type=int, default=128, help="total timed calls per condition")
    parser.add_argument("--warmup", type=int, default=4, help="untimed warmup calls per condition")
    parser.add_argument("--max-tokens", type=int, default=1024, help="response token cap")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="thinking tokens (0 = disabled, isolates the grammar cost)",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-call timeout (s)")
    args = parser.parse_args()

    config = load_llm_config()
    try:
        client = create_client(config)
    except LLMConnectionError as e:
        print(f"vLLM unreachable: {e}", file=sys.stderr)
        return 2
    if not isinstance(client, VLLMClient):
        print(f"Expected VLLMClient, got {type(client).__name__}", file=sys.stderr)
        return 2

    print(
        f"Scenario: concurrency={args.concurrency}, requests={args.requests}, "
        f"max_tokens={args.max_tokens}, thinking_budget={args.thinking_budget}, "
        f"timeout={args.timeout}s"
    )
    print(f"vLLM base_url: {client.config.base_url}\n")

    bounded_schema = _inline_refs(DetectionResult)
    unbounded_schema = _strip_bound_keys(bounded_schema)
    if bounded_schema == unbounded_schema:
        print(
            "ERROR: stripping changed nothing — DetectionResult carries no wire "
            "bounds. Benchmark would compare identical schemas.",
            file=sys.stderr,
        )
        return 1

    prompts = _build_prompts(args.requests)

    # Bounded first, then unbounded; each condition warms up independently so
    # order and one-time grammar-compile costs don't bias the timed run.
    bounded = await _run_condition("bounded", client, prompts, bounded_schema, args)
    unbounded = await _run_condition("unbounded", client, prompts, unbounded_schema, args)

    _print_report(bounded, unbounded)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
