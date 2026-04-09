# Known Issues & Technical Debt

Detailed tracking of issues discovered during RDNA4 inference work.
Each issue includes root cause, impact, and proposed fix.

## Critical: MoE Models Blocked

### 1. AWQ MoE crashes on gfx1201 — hipErrorLaunchFailure
**Status:** Blocked — crashes during forward_extend on all configs
**Model:** Qwen3-Coder-30B-A3B AWQ, all MoE models
**Root cause:** The Triton AWQ GEMM kernel generates invalid HSACO when compiled within the full model forward pass on gfx1201. The same kernel configurations pass all tests in isolation (every M/K/N/split_k combination verified). The crash occurs during `forward_extend` (prefill) regardless of:
- Torch version (2.11 stable, 2.12 nightly)
- Attention backend (triton, torch_native)
- MoE dispatch method (HIP GEMV fused, Triton per-expert loop)
- Triton cache state (fresh or cached)
- TP size (1 or 2)
**Impact:** All MoE AWQ models on SGLang crash during the first inference request.
**HIP GEMV kernel:** We ported and integrated the native HIP AWQ GEMV MoE kernel from `mgehre-amd/vllm` (`awq_gemv_moe_hip`). This kernel:
- Dispatches ALL experts in a single GPU launch (no Python loop)
- Is **151x faster** than the per-expert Triton loop in microbenchmarks
- Correctly produces output in isolated tests with model-realistic dimensions
- But the full model pipeline still crashes because other kernels (AWQ GEMM for attention projections or shared expert) hit the same Triton codegen bug during extend.
**Workaround:** Use vLLM Docker FP8 for MoE models (1,185 tok/s peak for Coder-30B).
**Proposed fix:** Port the remaining Triton AWQ GEMM calls (attention QKV/O projections) to HIP to eliminate ALL Triton kernel usage, or identify the specific Triton autoconfig that crashes in the multi-kernel context.

### FP8 MoE on SGLang — Arch Linux comgr bug
**Status:** Blocked, workaround via vLLM Docker
**Root cause:** Arch Linux `comgr` package generates invalid `.hsaco` binaries for FP8 WMMA instructions on gfx1201. Same kernel ISA works in Docker (Ubuntu ROCm).
**Impact:** All FP8 Triton kernels hang on SGLang. FP8 MoE models must use vLLM Docker.
**Workaround:** Docker proven (1,185 tok/s peak for Coder-30B).

## Fixed: Performance

### 2. Dense AWQ M=1 decode — HIP GEMV 30% faster
**Status:** FIXED — HIP GEMV integrated for M=1 decode
**Root cause:** Triton GEMM kernel has overhead for M=1 (tile setup, split-K reduction).
**Fix applied:** Ported native HIP AWQ GEMV kernel from `mgehre-amd/vllm` (`matthias.awq_gemv` branch). The kernel uses wave32-native FP16 bit-trick dequantization — reinterprets INT4 nibbles as FP16 via magic constant OR, avoiding int-to-float conversion.
**Integration:** `AWQLinearMethod.apply()` now dispatches M=1 to HIP GEMV, M>1 to Triton GEMM (hybrid approach, inspired by vLLM's `HybridW4A16LinearKernel`).
**Benchmark (Devstral-24B AWQ, TP=2):**
| Layer | K | N | HIP (μs) | Triton (μs) | Speedup |
|-------|-----|-------|---------|------------|---------|
| qkv_proj | 5120 | 6400 | 39.9 | 50.0 | 1.26x |
| gate_up | 5120 | 28672 | 146.5 | 186.2 | 1.27x |
| down | 14336 | 5120 | 73.2 | 97.2 | 1.33x |
| o_proj | 5120 | 5120 | 32.1 | 41.6 | 1.30x |
**Build:** `scripts/build_awq_gemv.sh --env <env-name>`

### 3. sgl_kernel not available on RDNA4
**Status:** FIXED — built from source for gfx1201
**Fix:** `scripts/setup_sgl_kernel.sh --env <env-name>`
**Result:** 6 native HIP ops active. See `patches/004-sgl-kernel-rdna4-fallbacks.patch`.

### 4. Dense AWQ garbage on torch 2.12 — sgl_kernel rotary_embedding fallback
**Status:** FIXED — install native sgl_kernel
**Root cause:** pip sgl_kernel's `__init__.py` overwrites native HIP `rotary_embedding` with Python fallback that produces wrong results on non-contiguous tensors from `qkv.split()`.
**Fix:** `scripts/setup_sgl_kernel.sh --env <env-name>` (installs patched `__init__.py` + native .so)

## Model Support

### 5. Gemma 4 — Mixed head_dim, blocked on weight quality
**Status:** Infrastructure FIXED, blocked on community weight quality
**Fix applied:** Per-layer head_dim (SWA=256, full=512), SWAKVPool, K=V weight duplication. Model loads and runs.
**Remaining:** Community compressed-tensors weights produce garbage (verified in SGLang AND vLLM Docker). Need properly calibrated GPTQ weights.

### 6. Coder-Next-80B AWQ — blocked on weight quality
**Status:** Infrastructure FIXED, blocked on community AWQ weight quality
**Fix applied:** DeltaNet cache sizing, conv_state TP fix, rotary_embedding fallback.
**Remaining:** Community AWQ weights produce garbage. Need GPTQ calibration on BF16 base (149 GB).

### 7. Community AWQ weight quality
**Status:** Known limitation
**Root cause:** Community models use RTN-style quantization (minmax/mse observers) without GPTQ calibration. This is too lossy for DeltaNet and Gemma 4 architectures.
**Affected:** Gemma 4 (26B, 31B), Coder-Next (80B)
**Not affected:** Standard MoE (Coder-30B AWQ quality is acceptable), Dense (Devstral, Qwen3.5)
**Fix:** Run GPTQ calibration via `scripts/quantize_moe_llmcompressor.py`

## Low: Infrastructure

### 8. CUDA graphs not compatible with MoE AWQ
**Status:** Known — `--disable-cuda-graph` required for MoE AWQ
**Root cause:** Expert dispatch has data-dependent control flow. Dense models can use CUDA graphs.

### 9. Qwen3.5 thinking budget
**Status:** FIXED — `--thinking-budget N` flag in eval script

## Completed (for reference)

- **sgl_kernel import crashes** — FIXED: patched `__init__.py` + built from source
- **AWQ MoE OOM (28GB/GPU)** — FIXED: INT4 packed weights (7.93 GB/GPU)
- **Coder-Next weight loader crash** — FIXED: skip packed weight_loader for AWQ
- **Coder-Next shared_expert load error** — FIXED: `modules_to_not_convert`
- **Gemma 4 expert weight format** — FIXED: converter handles 3D fused expert tensors
