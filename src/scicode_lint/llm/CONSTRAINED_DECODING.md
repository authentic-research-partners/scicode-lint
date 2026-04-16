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

### `$ref` inlining via `vllm_schema()`

Pydantic generates `$ref` entries for nested models (e.g., `DetectionResult.location`
references `NamedLocation` via `$ref: "#/$defs/NamedLocation"`). vLLM's XGrammar
backend may not resolve `$defs` — it expects a flat schema.

**Always use `vllm_schema()` instead of `model.model_json_schema()`** when passing
schemas to vLLM. This function inlines all `$ref` references and strips Pydantic
metadata (`title` fields).

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

Regression tests in `tests/test_schema_bounds.py` verify no `$ref` entries remain
after inlining.

### Schema bounds: `max_length` and `max_items` are mandatory

Every string field in a response schema gets `Field(max_length=N)`, and every
`list[X]` field gets both `Field(max_length=N)` (list size) and per-item bounds
via `Annotated[str, StringConstraints(max_length=M)]`. These are enforced at
the decoder level by XGrammar and make the response size bounded and
predictable. This is **not optional** — without it, a verbose field or a
runaway list can blow through the JSON-response portion of `max_completion_tokens`.

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

The client retries on two transient failure modes:

1. **Empty content** (`content=None`) — thinking consumed the entire `max_completion_tokens`
   budget before JSON generation started. This is the dominant failure mode.
2. **Invalid JSON** (`JSONDecodeError`) — rare network glitch or vLLM streaming bug.
   Constrained decoding should prevent this, but transient failures occur in practice.

`_TRANSIENT_RETRIES = 2` (3 total attempts). Exponential backoff: 0.5s, 1.0s.
After all transient retries are exhausted, the error is raised.

Non-transient errors (schema validation, missing location) are handled separately
and are not retried by this loop. Missing-location errors trigger a correction prompt
retry (different mechanism).

### Prompt mirror pattern

For every decoder-enforced cap (list `maxItems`, string `maxLength`), the prompt
carries matching soft guidance. The decoder enforces the hard cap; the prompt
decides which content survives when the cap bites. Examples:

- List cap → "at most N items, most important first"
- String cap → "under N words" or "1-2 sentences"

If the cap never bites, the model writes naturally below the limit.

### Why `max_length` has no token cost

The `maxLength` JSON-schema keyword counts **characters**, not tokens, and
behaves as a ceiling when set generously above natural output length.
Empirical measurement on Qwen3 8B (n=10 runs per setting, thinking=low, max_tokens=2048):

```
maxLength=none:  mean=438  stdev=41
maxLength=150:   mean=404  stdev=30
maxLength=500:   mean=452  stdev=54
maxLength=1000:  mean=436  stdev=49
```

All means within one standard deviation — `max_length` neither saves nor
costs tokens when set at ~2x natural output. The value of bounding every
field is **predictability**: guaranteed JSON completion inside the response
budget, so no retry mechanism is needed for truncation. The schema itself
prevents it.

### Sizing for scicode-lint

scicode-lint allocates `max_completion_tokens=4096` with
`thinking_budget=3584` for detection (see `config.py`), leaving ~512 tokens
for the JSON response. Current bounded schemas (see `llm/models.py`):

| Schema | Worst-case response tokens | Headroom |
|---|---|---|
| `DetectionResult` | ~225 | ~287 tokens |
| `FileClassification` | ~380 | ~130 tokens |

Both fit comfortably, and thinking can use its full 3584-token budget without
risk of phase-2 truncation.

### Checklist when adding a new vLLM call site

1. **Define the response schema** as a Pydantic `BaseModel`. Every `str` field
   gets `Field(max_length=N)` at ~2x natural output. Every `list[X]` field gets
   `Field(max_length=N)` (emits `maxItems: N`) AND per-item bounds if `X` is `str`
   (use `Annotated[str, StringConstraints(max_length=M)]`).
2. **Use `vllm_schema()`** to generate the JSON schema — not `model_json_schema()`.
3. **Compute the worst-case token count** (`chars / 3 + JSON overhead`) and verify
   it's ≤ `max_completion_tokens − thinking_budget`.
4. **Numerics** get `ge=/le=` bounds where meaningful (e.g., confidence 0-1).
5. **Nullable fields populated post-hoc** from side channels (e.g.,
   `DetectionResult.thinking`) stay unbounded.
6. **Add schema-bounds regression tests** in `tests/test_schema_bounds.py`. Assert
   `maxLength` and `maxItems` values with failure messages pointing here.
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
