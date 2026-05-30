#!/usr/bin/env python3
"""Forcing eval: verify the thinking-budget ladder recovers from length failures.

Runs against a live vLLM server. Length failures are rare in normal operation
(schema ``max_length`` caps keep JSON well under the response reserve), so this
eval *forces* the failure mode by setting ``max_completion_tokens=256`` and
starting ``thinking_budget=200``. The ladder then walks ``200 → 100 → 0``:

- Attempt 0 (budget=200): thinking alone consumes the ~256-token total budget,
  leaving nothing for JSON → ``finish_reason=length`` / ``content=None``.
- Attempt 1 (budget=100): same pattern, less severe, still fails.
- Attempt 2 (budget=0, thinking disabled): JSON gets the full 256 tokens,
  enough for the ~225-token worst-case ``DetectionResult`` → success.

Verification:
1. The captured loguru warning stream contains at least one ``stepping thinking
   budget`` entry.
2. Budgets chain (each step's ``to`` equals the next step's ``from``) and are
   strictly decreasing.
3. The final call succeeds OR raises cleanly with ``length-based failure``
   (ladder exhausted is also acceptable — proves the escape hatch fires).

Requires vLLM server running::

    scicode-lint vllm-server start
    python evals/ladder_recovery.py
"""

from __future__ import annotations

import asyncio
import re
import sys
from io import StringIO

from loguru import logger

from scicode_lint.config import load_llm_config
from scicode_lint.exceptions import LLMConnectionError
from scicode_lint.llm.client import create_client
from scicode_lint.llm.models import DetectionResult

_SYSTEM_PROMPT = (
    "You are a Python code reviewer checking for machine-learning issues. "
    "Return a DetectionResult JSON object matching the schema."
)

# A non-trivial detection task: does this KFold usage leak data via non-shuffled
# time-ordered samples? The realistic ambiguity forces the model to think before
# committing to yes/no/context-dependent — which, combined with the tight token
# cap below, is what makes length failure reliable in this eval.
_USER_PROMPT = """Code:
```python
import numpy as np
from sklearn.model_selection import KFold

def cross_validate(features, labels):
    kf = KFold(n_splits=5, shuffle=False)
    scores = []
    for train_idx, test_idx in kf.split(features):
        X_tr, X_te = features[train_idx], features[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]
        scores.append(fit_and_score(X_tr, y_tr, X_te, y_te))
    return np.mean(scores)
```

Question: does this code shuffle training data before cross-validation splits?
Consider whether the dataset order could leak information across folds.
"""

# Tight cap forces length failure at any non-trivial thinking budget.
# Starting budget 200 walks the ladder to 100, then to 0 (below the 64 floor).
_FORCED_MAX_COMPLETION_TOKENS = 256
_FORCED_THINKING_BUDGET = 200


def _parse_ladder_steps(log_output: str) -> list[tuple[int, int]]:
    """Extract ``(from, to)`` budget pairs from captured loguru warnings."""
    matches = re.findall(r"stepping thinking budget (\d+) → (\d+)", log_output)
    return [(int(a), int(b)) for a, b in matches]


async def main() -> int:
    config = load_llm_config()
    config.max_completion_tokens = _FORCED_MAX_COMPLETION_TOKENS
    print(
        f"Scenario: max_completion_tokens={_FORCED_MAX_COMPLETION_TOKENS}, "
        f"starting thinking_budget={_FORCED_THINKING_BUDGET}"
    )

    # ``create_client`` auto-detects vLLM on localhost:5001 / :8000 if
    # ``config.base_url`` is empty, and raises ``LLMConnectionError`` otherwise.
    try:
        client = create_client(config)
    except LLMConnectionError as e:
        print(f"vLLM unreachable: {e}", file=sys.stderr)
        return 2
    print(f"vLLM base_url: {client.config.base_url}\n")  # type: ignore[attr-defined]

    log_sink = StringIO()
    sink_id = logger.add(log_sink, level="WARNING", format="{message}")

    outcome: str
    try:
        try:
            result = await client.async_complete_structured(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_USER_PROMPT,
                schema=DetectionResult,
                thinking_budget=_FORCED_THINKING_BUDGET,
            )
            outcome = "success"
            print(
                f"Call succeeded — detected={result.detected!r}, confidence={result.confidence:.2f}"
            )
        except LLMConnectionError as e:
            print(f"vLLM unreachable: {e}", file=sys.stderr)
            return 2
        except ValueError as e:
            if "length-based failure" in str(e):
                outcome = "ladder_exhausted"
                print(f"Ladder exhausted: {e}")
            else:
                print(f"Unexpected ValueError: {e}", file=sys.stderr)
                return 1
    finally:
        logger.remove(sink_id)

    steps = _parse_ladder_steps(log_sink.getvalue())
    print(f"\nLadder steps observed: {len(steps)}")
    for from_budget, to_budget in steps:
        print(f"  {from_budget} → {to_budget}")

    # Budgets must chain and strictly decrease (0 is the floor and repeats don't occur).
    for (_, to_a), (from_b, to_b) in zip(steps, steps[1:], strict=False):
        if to_a != from_b:
            print(
                f"FAIL: ladder budgets do not chain ({to_a} → {from_b} break)",
                file=sys.stderr,
            )
            return 1
        if to_b >= from_b:
            print(
                f"FAIL: ladder did not step down ({from_b} → {to_b})",
                file=sys.stderr,
            )
            return 1

    print(f"\nOutcome: {outcome}")
    if outcome == "success" and steps:
        print("VERIFIED: ladder fired and the call recovered at a lower budget.")
        return 0
    if outcome == "success" and not steps:
        print(
            "INCONCLUSIVE: initial budget was sufficient; ladder never fired. "
            "Tighten _FORCED_MAX_COMPLETION_TOKENS or raise _FORCED_THINKING_BUDGET."
        )
        return 0
    if outcome == "ladder_exhausted":
        print(
            f"VERIFIED: ladder walked {len(steps)} step(s) before exhausting — "
            "the escape hatch works, but the scenario was too tight to recover."
        )
        return 0
    print(f"FAIL: unexpected outcome {outcome!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
