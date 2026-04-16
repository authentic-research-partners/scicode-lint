# VRAM Requirements

scicode-lint requires **minimum 16GB VRAM** with **native FP8 support** (compute capability >= 8.9).

**Disk space:** Also requires **~15GB free disk space** for model weights, which are downloaded to `~/.cache/huggingface/hub/` on first launch. See [INSTALLATION.md](../INSTALLATION.md#model-storage) for details.

## Quick Reference

**Requirements:**
- 16GB+ VRAM
- Native FP8 support (compute capability >= 8.9)

**Configuration:** 40K max context (typical: ~20K per request), FP8 KV cache, default model: RedHatAI/Qwen3-8B-FP8-dynamic

## Setup

**Just run the script** - it automatically verifies your hardware meets requirements:

```bash
scicode-lint vllm-server start
```

The command verifies:
1. **VRAM**: Minimum 16GB available
2. **Compute capability**: >= 8.9 (native FP8 support)
3. **GPU availability**: Rejects CPU-only systems

If requirements aren't met, the script exits with clear error messages and hardware recommendations.

## Configuration

**Default Model:** Qwen3-8B-FP8 (`RedHatAI/Qwen3-8B-FP8-dynamic`)

**Specs (from config.toml):**
- Model size: ~9GB
- Max sequence length: 40,960 tokens (model native max — ceiling, not pre-allocated)
- Typical input: ~16K tokens (code + prompt); response: ~4K tokens
- Max input: ~1,400 lines (after ~2,000 token prompt overhead)
- VRAM usage: ~13GB total (fits comfortably on 16GB GPU)

**Memory breakdown:**
- Model weights (FP8): ~9GB
- KV cache (FP8, allocated on demand by PagedAttention): ~1.5GB typical
- Framework overhead: ~2GB
- Total: ~13GB typical

**Customization:** All context/VRAM settings are configurable in `config.toml`. Qwen3-8B supports up to 40,960 tokens natively (`max_position_embeddings`).

## Supported GPUs

**Native FP8 GPUs (compute capability >= 8.9):**
- **Consumer:** RTX 4060 Ti 16GB, RTX 4070+ (16GB+), RTX 4090 (24GB)
- **Workstation:** RTX 4000 Ada (20GB), RTX 5000 Ada (32GB)
- **Cloud/HPC inference:** L4 (24GB), L40 (48GB), A10 (24GB)

## Why 16GB Minimum?

With FP8 quantization for both weights and KV cache:
- Model weights (FP8): ~9GB
- KV cache (FP8, on demand): ~1.5GB for typical ~20K-token sequences
- Framework overhead: ~2GB
- **Total: ~12.5GB typical** (fits in 16GB with headroom)

## Context Window Sizing

The vLLM server accepts sequences up to **40,960 tokens** (Qwen3-8B native max). This is a ceiling, not a pre-allocation — vLLM's PagedAttention allocates KV cache blocks on demand. Raising the ceiling costs nothing for shorter sequences; the real concurrency constraint is actual tokens in flight.

**Typical token allocation per request:**
- Input: ~16,000 tokens max
  - System prompt: ~1,450 tokens
  - Detection question template: ~350–1,050 tokens (mean ~575)
  - Code content: ~14,000 tokens (~1,400 lines)
- Response: 4,096 tokens (thinking + JSON)

Based on analysis of 10M+ GitHub repositories:
- Median file: 258 lines (~2,600 tokens)
- Mean file: 879 lines (~8,800 tokens)
- 90th percentile: ~1,500 lines (~15,000 tokens)

**Why 4K response tokens?**
Benchmarked for optimal accuracy (see `benchmarks/reports/max_tokens/`). 4K achieves best accuracy; higher values show diminishing returns.

**PagedAttention:** Smaller files don't waste VRAM. Memory is allocated dynamically based on actual sequence length, not the 40K ceiling.

## Verification

Check your setup is working:

```bash
# 1. Verify requirements
scicode-lint vllm-server start
# Will exit with error if hardware doesn't meet requirements

# 2. Check VRAM usage with nvidia-smi (should be ~13GB)
nvidia-smi

# 3. Test linting
scicode-lint lint your_file.py
```

## Alternative Models

See **[MODEL_SELECTION.md](MODEL_SELECTION.md)** for:
- Benchmark comparisons (Qwen3 vs Gemma3 vs Devstral)
- vLLM commands for different models
- Recommendations by VRAM size

**Quick summary:** Qwen3-8B-FP8 outperforms Gemma3-12B on coding benchmarks (+15 EvalPlus) while using less VRAM.

---

## Summary

**Requirements:**
- 16GB+ VRAM
- Native FP8 support (compute cap >= 8.9)
- Examples: RTX 4060 Ti 16GB, RTX 4070+, RTX 4090, RTX 4000 Ada, L4, L40, A10

**Default Model:** RedHatAI/Qwen3-8B-FP8-dynamic
**Context:** 20K tokens (16K input + 4K response, supports ~1,400 line files)
