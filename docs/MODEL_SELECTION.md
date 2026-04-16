# Model Selection: Why Qwen3-8B-FP8

## TL;DR

**Qwen3-8B-FP8** was chosen because:
1. **Thinking mode support** - essential for understanding conceptual pattern descriptions
2. **Fits in 16GB VRAM** with room for 20K context (16K input + 4K response)
3. **FP8 quantization** - 50% memory savings, 2x throughput, no accuracy loss

**Gemma3 was rejected** because it lacks thinking mode and requires grep-like explicit matching rules.

## Key Requirement: Thinking Mode

Pattern detection requires understanding code at a conceptual level, not matching explicit rules. This requires **thinking mode** - step-by-step reasoning before producing output.

### Why Thinking Mode Matters

Without thinking mode, models cannot:
- Understand code structure and intent
- Reason about data flow
- Recognize variations (different variable names, code styles)

| Model | Thinking Mode | Pattern Understanding |
|-------|---------------|----------------------|
| Qwen3 | ✅ Native support | Understands "detect missing zero_grad()" |
| Gemma3 | ❌ Not supported | Needs grep-like step-by-step rules |

### Gemma3 Failure Case

**Gemma3-12B** was evaluated but failed. Without thinking mode, it required almost grep-like pattern descriptions:

```
Look for:
1. A for loop iterating over a dataloader
2. Inside that loop, find loss.backward() call
3. Check if optimizer.zero_grad() appears before backward()
4. If not found within 5 lines before backward(), report issue
```

This defeats the purpose of using an LLM - we could just write regex patterns instead.

**Qwen3 with thinking mode** understands the same pattern from: "detect when gradients aren't cleared before backward pass" - no explicit matching rules needed.

## Why Qwen3-8B-FP8 Specifically

Given thinking mode as the key requirement, Qwen3-8B-FP8 was selected for efficiency and performance.

### VRAM Efficiency

| Component | Size |
|-----------|------|
| Weights (FP8) | ~9GB |
| KV Cache (20K context, FP8) | ~1.5GB |
| Overhead | ~2GB |
| **Total** | **~13GB** |

Fits comfortably on 16GB+ GPUs (RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, L4, A10).

### FP8 Dynamic Quantization

We use [RedHatAI/Qwen3-8B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3-8B-FP8-dynamic) for efficiency gains with no accuracy loss:

| Metric | Improvement |
|--------|-------------|
| GPU Memory | ~50% reduction (16-bit → 8-bit) |
| Compute Throughput | ~2x faster matrix operations |
| Disk Size | ~50% smaller model files |

Accuracy is fully preserved:

| Benchmark | Base | FP8-dynamic | Recovery |
|-----------|------|-------------|----------|
| MMLU (5-shot) | 71.95 | 72.30 | 100% |
| GSM-8K (5-shot) | 75.97 | 80.52 | 100% |
| MMLU-Pro (5-shot) | 34.57 | 37.82 | 100% |

**How it works:**
- Weights: Symmetric static per-channel FP8
- Activations: Symmetric dynamic per-token FP8
- Only transformer linear layers quantized (lm_head excluded)

### Code Benchmarks

Qwen3-8B outperforms larger models on code-related tasks:

| Benchmark | Qwen3-8B | Gemma3-12B | Difference |
|-----------|----------|------------|------------|
| **EvalPlus** (code quality) | 67.65 | 52.65 | **+15** |
| **MBPP** (Python coding) | 69.80 | 60.60 | **+9** |
| **MATH** (reasoning) | 60.80 | 44.43 | **+16** |

## Configuration

### vLLM Setup

The default model, optimization flags, and sampling parameters are all configured in
the bundled `config.toml` and launched via the container command:

```bash
scicode-lint vllm-server start
```

See [INSTALLATION.md](../INSTALLATION.md) for the container-runtime prerequisites. The
flags the container passes to vLLM are documented in the next section.

**Optimization flags:**

| Flag | Effect |
|------|--------|
| `--kv-cache-dtype fp8` | Stores KV cache in FP8 instead of FP16. ~50% cache memory savings, allowing more concurrent sequences or longer contexts. Minimal precision impact since model already uses FP8 for weights/activations. |
| `--enable-chunked-prefill` | Splits long prompts into chunks that interleave with decode steps. Prevents one long prompt from blocking others. Better tail latency with concurrent requests. |

### Sampling Parameters

Based on [Qwen3 best practices](https://huggingface.co/Qwen/Qwen3-8B-FP8), configured in `config.toml`:

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | 0.6 | Thinking mode requirement. **DO NOT use 0.0** (greedy) - causes endless repetitions |
| `top_p` | 0.95 | Nucleus sampling for thinking mode |
| `top_k` | 20 | Limits token selection for coherent reasoning |
| `max_completion_tokens` | 4096 | Optimal accuracy (see benchmarks/) |

**Why not greedy decoding (temperature=0)?**

Qwen3 explicitly warns:
> "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions."

Thinking mode requires randomness to explore reasoning paths.

**Why 4K max tokens?**

Benchmarked for optimal accuracy (see `benchmarks/reports/max_tokens/`). For pattern detection:
- Thinking typically uses 500-2000 tokens
- JSON output is ~100 tokens
- 4K achieves best accuracy; higher values show diminishing returns

Increase to 8K or 16K in `config.toml` if you see truncated thinking warnings.

## Alternatives

### More VRAM (24GB full utilization)

| Model | Weights | Use Case |
|-------|---------|----------|
| **Qwen3-14B-FP8** | ~14GB | Better accuracy, same architecture |
| **Qwen3-Coder-30B-A3B** | ~15-16GB | Code-specific MoE |
| **Devstral-Small-2-24B** | ~24GB | Best SWE-Bench for size |

To use a non-default model, override `[llm].model` and related keys in
`~/.config/scicode-lint/config.toml`, then restart the container:

```toml
# ~/.config/scicode-lint/config.toml
[llm]
model = "Qwen/Qwen3-14B-FP8"

[vllm]
max_model_len = 8192
gpu_memory_utilization = 0.95
```

```bash
scicode-lint vllm-server restart
```

Non-Qwen3 models also need their own reasoning parser (or none). Set
`[vllm].reasoning_parser = ""` for non-thinking models, or set the model-specific
parser value vLLM expects.

### Less VRAM (< 16GB)

Use GGUF quantized models with llama.cpp:
- [bartowski/Qwen3-8B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-8B-GGUF) - Q5_K_M recommended

## Links

| Model | HuggingFace |
|-------|-------------|
| Qwen3-8B-FP8-dynamic | [RedHatAI/Qwen3-8B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3-8B-FP8-dynamic) |
| Qwen3-8B-FP8 | [Qwen/Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) |
| Qwen3-14B-FP8 | [Qwen/Qwen3-14B-FP8](https://huggingface.co/Qwen/Qwen3-14B-FP8) |
| Qwen3-Coder-30B-A3B-FP8 | [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8) |
| Devstral-Small-2-24B | [mistralai/Devstral-Small-2-24B-Instruct-2512](https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512) |
