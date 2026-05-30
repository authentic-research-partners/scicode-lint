# Constrained Decoding with `response_format: json_schema`

We use the OpenAI-standard `response_format: json_schema` for structured output from vLLM.

### How it works

vLLM's constrained decoding (XGrammar/Outlines backend):

1. Parse schema → build FSM (finite state machine)
2. Track valid next tokens at each step
3. Mask invalid tokens before sampling

```
Schema: {"detected": bool}

After '{"detected": ' → only allow: true, false
After '{"detected": true' → only allow: , or }
```

### Reasoning parser (Qwen3-specific)

Qwen3 outputs `<think>...</think>` blocks before the JSON answer. vLLM's reasoning parser
(`--reasoning-parser qwen3`) separates these server-side:

- `message.content` → clean JSON
- `message.reasoning` → thinking content

Without the reasoning parser, constrained decoding applies from the first token,
suppressing the thinking phase and dropping accuracy from ~98% to ~78%.

**The `--reasoning-parser qwen3` flag is Qwen3-specific.** Other thinking models need
their own parser (e.g., `deepseek` for DeepSeek-R1). Non-thinking models don't need
any parser at all.

> **Note on vLLM version.** Setting `--reasoning-parser` alone enables reasoning
> implicitly. In older vLLM versions (≤0.17.x) reasoning also required a separate
> `--enable-reasoning` flag, which was removed in v0.18.0 — scicode-lint pins v0.18.0.

### Thinking controls

Two complementary controls for thinking depth, passed via `extra_body.thinking`:

- **`budget`** (int) — hard cap on thinking tokens. Abruptly stops thinking when reached.
  Prevents runaway reasoning from exhausting `max_completion_tokens`.
- **`effort`** (float, 0.0-1.0) — soft guide for thinking depth. The model aims for
  shorter (0.0) or deeper (1.0) reasoning but may exceed the target.

Use both together: effort guides depth, budget prevents worst-case.

| Use case | budget | effort | Rationale |
|----------|--------|--------|-----------|
| Detection (lint) | 3584 | (default) | Complex code analysis, let model decide depth |
| Classification | 200 | 0.3 | Simple binary decisions, minimize thinking |

`thinking_budget` is configured in `config.toml`. `thinking_effort` defaults to None
(model's own behavior) and should only be passed per-call for simple tasks.

### Why `response_format` (not `guided_json`)

Both use the same XGrammar/Outlines backend. `guided_json` (passed via `extra_body`) was vLLM-specific and is deprecated as of vLLM v0.12.0. `response_format: json_schema` is the OpenAI-standard API, portable across providers.

### Pydantic schemas are mandatory

Always use Pydantic models to generate JSON schemas. Hand-written schemas may have subtle issues that cause unreliable constraint enforcement.

### Wire schema via `vllm_schema()`

**Always use `vllm_schema()` instead of `model.model_json_schema()`** when passing
schemas to vLLM. It applies two transforms that make the schema safe and fast for
XGrammar:

1. **Inlines `$ref`.** Pydantic emits `$ref` for nested models (e.g.
   `DetectionResult.location` → `NamedLocation`). XGrammar may not resolve `$defs`,
   so refs are inlined into a flat schema. (It also drops `title` metadata.)
2. **Strips length/count constraint keys** — `maxLength`, `maxItems`, `minLength`,
   `minItems`. See the next section for why. Numeric ranges (`minimum`/`maximum`)
   are *kept* — they don't trigger the slow path.

```python
from scicode_lint.llm.models import vllm_schema, DetectionResult

# Use vllm_schema() — NOT model_json_schema()
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "DetectionResult",
        "schema": vllm_schema(DetectionResult),
        "strict": True,
    },
}
```

Regression tests in `tests/test_schema_bounds.py` verify the wire schema has no
`$ref` and none of the banned constraint keys, while the Pydantic model still
enforces them.

### The wire schema carries NO length/count bounds

Length/count keys (`maxLength`, `maxItems`, `minLength`, `minItems`) compile into
XGrammar's grammar **slow path**. Shipping them on the wire collapses throughput
under concurrency. Measured on this stack (qwen3-8b-fp8, vLLM v0.18.0) via
`evals/wire_bounds_throughput.py`, `DetectionResult` at concurrency 64:

| wire schema | success rate | wall-clock | p50 latency |
|---|---|---|---|
| **bounded** (keys present) | 8% (92% timed out) | 361 s | 239 s |
| **unbounded** (length keys stripped) | 100% | 6.7 s | 4.1 s |

**~50× slower, 92-point success collapse.** `lint_concurrency` defaults to 100 and a
single file fans out across all patterns, so this is the common path, not a corner
case. `vllm_schema()` therefore strips the length/count keys.

**Only length/count keys are stripped — not numeric ranges.** An isolation
measurement removed `maxLength` while keeping `minimum`/`maximum` on the wire and
recovered full speed (100% / 7.9 s), proving numeric ranges don't trigger the slow
path. So `confidence`'s `ge=0/le=1`
range stays decoder-enforced, and enums/`Literal` are kept too (XGrammar handles
both fine).

The Pydantic `Field(max_length=N)` constraints **stay on the model**; they are
post-decode guards, not wire constraints. Output size is controlled by three layers
that never touch the decoder:

1. **Prompt guidance** — every bounded free-text field's description states the cap
   in words ("1-2 sentences, under ~N words") and every list says "at most N". The
   model is asked to stay under, so it writes short by default.
2. **`max_completion_tokens`** caps total output; over-runs surface as
   `finish_reason=length` and recover via the thinking-budget ladder (below).
3. **Rerun on over-run.** If the model still exceeds a Pydantic bound, `model_validate`
   raises and the client **resamples** (see § Transient retry). A fresh sample at
   temp>0 almost always complies. Over-runs are surfaced and re-attempted — never
   silently truncated (that would hide the model misbehaving).

### The three failure modes

With thinking enabled, vLLM generates in two phases that share a single
`max_completion_tokens` budget:

```
<think>reasoning tokens...</think>{"detected":"yes","reasoning":"...",...}
|____________ phase 1: thinking ________||______ phase 2: JSON response _____|
```

| Limit hit | `finish_reason` | `content` |
|---|---|---|
| thinking budget | `stop` | valid JSON |
| `max_completion_tokens` during thinking | `length` | **`None`** |
| `max_completion_tokens` during JSON | `length` | **partial/invalid JSON** |

The surprising case is the second: if thinking consumes the entire budget
before JSON generation starts, you don't get partial JSON — you get
`content=None` and the model is still in the `<think>` parser state when
generation stops. Empirical findings on Qwen3 8B / vLLM v0.18.0:

```
max_tokens=100, budget=200  → content=None      (still thinking at token 100)
max_tokens=300, budget=4096 → content=partial   (JSON started but truncated)
max_tokens=500, budget=4096 → content=valid JSON
```

### Transient retry

The client retries on three recoverable failure modes, each with its own recovery
branch. All share a single budget: `_TRANSIENT_RETRIES = 2` (3 total attempts).

**1. Length-based failure — thinking-budget ladder.**
Covers both `content=None` (thinking consumed the entire `max_completion_tokens`
budget before JSON started) and `finish_reason='length'` with truncated JSON.
Same-budget retries here are almost guaranteed to fail again — if 3584 thinking
tokens exhausted the budget once, they will next time too. Instead, the client
steps the thinking budget down via `_step_down_thinking`:

```
3584 → 1792 → 896 → 448 → 224 → 112 → 0 (off)
```

Each step halves the current budget; once halving would fall below
`_THINKING_FLOOR = 64`, the next step drops straight to `0` (thinking disabled).
A short 0.2s delay is used between retries. Per-call `thinking_budget` overrides
are respected — the ladder starts from whatever the caller passed, not the
config default. When the ladder is exhausted (budget already `0`), the error
is raised immediately.

**2. Invalid JSON — same-budget exponential backoff.**
`JSONDecodeError` is a rare network/streaming glitch; constrained decoding
should prevent this, but transient failures occur in practice. Here the
thinking budget is *not* stepped — the problem isn't thinking, it's the wire
or the server. Retries use exponential backoff: 0.5s, 1.0s.

**3. Schema-validation over-run — same-budget resample.**
Because length caps are stripped from the wire (§ "The wire schema carries no
length/count bounds"), the model can exceed a Pydantic `max_length`/`max_items`.
`model_validate` raises, and the client resamples the same prompt — over-runs are
re-attempted, never clipped. Like branch 2 the thinking budget is *not* stepped; a
fresh sample at temp>0 almost always complies. Everything still enforced by the
decoder (types, required fields, enums, numeric ranges) cannot reach this branch, so
a residual validation failure is almost always a length over-run.

The branches compose and share one budget: a length failure at attempt 0 steps the
thinking budget down; a JSON-parse or over-run failure at attempt 1 keeps that
stepped-down budget. After the budget is exhausted the error is raised loudly. Tests
in `tests/test_llm_client.py` lock in all three branches and their interaction.

Missing-location errors are handled separately (not by this loop): they trigger a
correction-prompt retry, and on final failure flip the result to `detected="no"`.

### Prompt mirror pattern

The cap lives on the Pydantic model but not on the wire, so the **prompt is the
primary size control** — each bounded field's description states its cap and asks
the model to stay under it:

- List cap → "at most N items, most important first"
- String cap → "under ~N words" or "1-2 sentences"

Keep the prompt hint in sync with the Pydantic `max_length` (~8 chars/word, so
`max_length=400` → "under ~50 words"). The model writes below the limit by default;
a rare over-run is caught post-decode and resampled (§ Transient retry), not enforced
mid-token by the decoder.

### `max_length` is free on token count but lethal on throughput

Two different costs, easy to conflate:

- **Token count: none.** `maxLength` counts characters, not tokens, and acts as a
  ceiling above natural output. Single-request measurement on Qwen3 8B
  (n=10/setting, thinking=low) shows output length is identical with or without it:

  ```
  maxLength=none: 438   maxLength=150: 404   maxLength=500: 452   maxLength=1000: 436
  ```
  (all within one standard deviation)

- **Throughput under concurrency: severe.** The same key *on the wire* triggers
  XGrammar's grammar slow path — 54× slower and 92% timeouts at concurrency 64 (table
  above). The single-request token-count benchmark cannot see this; it never issues
  concurrent requests.

An earlier "bounds are free, so no truncation retry is needed" claim conflated the
two — it measured token count, not throughput. Bounds stay on the Pydantic model
(post-decode validation) and are stripped from the wire by `vllm_schema()`.

### Sizing for scicode-lint

scicode-lint allocates `max_completion_tokens=4096` with
`thinking_budget=3584` for detection (see `config.py`), leaving ~512 tokens
for the JSON response. Current schemas (see `llm/models.py`):

| Schema | Worst-case response tokens | Headroom |
|---|---|---|
| `DetectionResult` | ~225 | ~287 tokens |
| `FileClassification` | ~380 | ~130 tokens |

Both fit comfortably, and thinking can use its full 3584-token budget without
risk of phase-2 truncation.

### Checklist when adding a new vLLM call site

1. **Define the response schema** as a Pydantic `BaseModel`. Every `str` field
   gets `Field(max_length=N)` at ~2x natural output; every `list[X]` field gets
   `Field(max_length=N)` and per-item bounds if `X` is `str`
   (`Annotated[str, StringConstraints(max_length=M)]`). These are **post-decode
   guards** — `vllm_schema()` strips them from the wire (XGrammar slow path).
2. **Use `vllm_schema()`** to generate the JSON schema — not `model_json_schema()`.
3. **Compute the worst-case token count** (`chars / 3 + JSON overhead`) and verify
   it's ≤ `max_completion_tokens − thinking_budget`.
4. **Numerics** get `ge=/le=` bounds where meaningful (e.g., confidence 0-1).
5. **Nullable fields populated post-hoc** from side channels (e.g.,
   `DetectionResult.thinking`) stay unbounded.
6. **Add schema-bounds regression tests** in `tests/test_schema_bounds.py`. Assert
   the wire schema (`vllm_schema(Model)`) has none of the banned constraint keys,
   the model still rejects over-runs at construction, and enums are preserved.
7. **Add a prompt mirror.** For every list cap, the prompt says "at most N, most
   important first". For string caps, "under N words".
8. **Add retry-behavior tests** if using a custom call path (the standard
   `async_complete_structured()` handles transient retries automatically).
9. **Avoid the budget valley of death.** Mid-range thinking budgets (512-1024) cause
   more truncation than low (200) or high (2048+). Stick to low or high.

### Other vLLM constrained decoding options

| Option | Constrains to | Use case |
|--------|---------------|----------|
| `json_schema` | JSON schema | Structured output |
| `guided_choice` | One of N strings | Simple classification |
| `guided_regex` | Regex pattern | IDs, formatted strings |
| `guided_grammar` | Context-free grammar | SQL, custom formats |

All use same XGrammar/Outlines backend.

### References

- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/
- vLLM reasoning outputs: https://docs.vllm.ai/en/latest/features/reasoning_outputs/
- XGrammar: https://github.com/mlc-ai/xgrammar
