# Known Issues & Technical Debt

Detailed tracking of issues discovered during RDNA4 inference work.
Each issue includes root cause, impact, and proposed fix.

## Critical: Performance

### 1. AWQ MoE per-expert dispatch is slow (~3.5 tok/s)
**Status:** Working but slow
**Model:** Qwen3-Coder-30B-A3B AWQ
**Root cause:** `AWQTritonMoEMethod.apply()` loops over experts in Python, calling `awq_gemm_triton` per-expert. This creates 768+ kernel launches per decode step (8 active experts × 2 GEMMs × 48 layers). No CUDA graph support.
**Impact:** ~3.5 tok/s vs ~94 tok/s (FP8 vLLM Docker)
**Proposed fix:** Two approaches:
1. **Fused AWQ MoE Triton kernel** — modify `fused_moe_kernel_gptq_awq` to support AWQ packing order. Requires debugging the AWQ→GPTQ weight repack (current repack produces garbage output — likely bit ordering mismatch).
2. **Graph-compatible dispatch** — replace Python loop with batched torch operations (gather tokens per expert, batched AWQ GEMM). Less optimal than fused kernel but enables CUDA graphs.

### 2. sgl_kernel not available on RDNA4
**Status:** Mitigated with torch fallbacks
**Root cause:** pip `sgl-kernel` package ships CUDA-only `.so` files. Same issue affects [NVIDIA SM121a/DGX Spark](https://github.com/sgl-project/sglang/issues/18203).
**Impact:** Activation functions (silu_and_mul, gelu_and_mul), MoE routing (topk_softmax, moe_align_block_size), and memory ops (weak_ref_tensor) use torch fallbacks instead of optimized kernels. Performance impact: ~5-10% on dense models, major on MoE (moe_align_block_size fallback uses Python loops).
**Proposed fix:** Build sgl-kernel from source for ROCm/gfx1201 using `pyproject_rocm.toml`. Requires HIP compilation infrastructure.

### 3. FP8 MoE on SGLang — Arch Linux comgr bug
**Status:** Blocked, workaround via vLLM Docker
**Root cause:** Arch Linux `comgr` package generates invalid `.hsaco` binaries for FP8 WMMA instructions on gfx1201. Same kernel ISA works in Docker (Ubuntu ROCm).
**Impact:** All FP8 Triton kernels hang on SGLang. FP8 MoE models must use vLLM Docker.
**Proposed fix:** Build ROCm from source, or wait for Arch package fix. Docker workaround is proven (1,882 tok/s peak for Coder-30B).

## Critical: Model Support

### 4. Gemma 4 — Triton attention crash with mixed head_dim
**Status:** Blocked
**Root cause:** Gemma 4 uses SWA head_dim=256 and full attention head_dim=512. SGLang's Triton attention backend assumes uniform head_dim. When both attention types are used in the same model, the head_dim mismatch causes a crash.
**Model code:** `gemma4_causal.py` — ported but untested due to attention crash.
**Impact:** Gemma 4 (26B MoE and 31B dense) cannot run on SGLang.
**Proposed fix:** Modify Triton attention backend to handle per-layer head_dim. Or split attention calls by head_dim group.

### 5. Coder-Next-80B AWQ — DeltaNet conv1d cache assertion
**Status:** Partially working (loads at 23.13 GB/GPU, crashes on first inference)
**Root cause:** `causal_conv1d_fn` asserts `dim == conv_states.shape[1]` but the DeltaNet conv states have wrong dimensions with AWQ. Likely the conv1d weight or state initialization doesn't account for the AWQ weight format.
**File:** `causal_conv1d_triton.py:467`
**Impact:** Coder-Next-80B AWQ loads but can't run inference.
**Proposed fix:** Debug conv_states initialization in `qwen3_next.py` — check if `self.conv1d.weight.squeeze(1)` produces correct dimensions when the module uses AWQ quantization. May need to pass conv1d weight through a different path.

## Medium: Quality

### 6. Community AWQ conversions may produce lower quality
**Status:** Documented, GPTQ pipeline ready as alternative
**Root cause:** Community models (stelterlab, bullpoint) use compressed-tensors with `minmax`/`mse` observers — simple RTN-style quantization without GPTQ calibration. For DeltaNet models (Qwen3.5, Coder-Next), this produces garbage output. For standard MoE (Coder-30B), quality appears acceptable but unverified against GPTQ baseline.
**Impact:** Unknown quality degradation for Coder-30B AWQ. Coder-Next community model quality untested.
**Proposed fix:** Run GPTQ calibration pipeline (`scripts/quantize_moe_llmcompressor.py`) on BF16 base models. Both base models downloaded to `/data/models/`:
- `Qwen3-Coder-30B-A3B-BF16` (57GB)
- `Qwen3-Coder-Next-BF16` (149GB)
Calibration takes ~6h for 30B (CPU), ~24h+ for 80B (needs disk offloading).

### 7. Gemma 4 expert weight format incompatibility
**Status:** Not addressed
**Root cause:** Gemma 4 stores expert weights in a fused format (`experts.down_proj_packed [128, 2816, 88]`) instead of per-expert format. Our `convert_moe_ct_to_awq.py` converter handles per-expert weights (`experts.0.down_proj.weight_packed`) but not Gemma 4's fused layout. Additionally, the checkpoint uses `_packed`/`_scale` suffixes instead of `weight_packed`/`weight_scale`.
**Impact:** Gemma 4 AWQ conversion produces incorrect weights.
**Proposed fix:** Write Gemma 4-specific AWQ converter that handles fused expert layout. Or use the native `gemma4_causal.py` model code which loads the fused format directly (blocked by #4 attention crash).

## Low: Infrastructure

### 8. CUDA graph incompatibility with Python control flow
**Status:** Affects MoE AWQ + moe_align_block_size fallback
**Root cause:** CUDA graphs require a fixed computation graph. Our torch fallbacks for `moe_align_block_size` and `AWQTritonMoEMethod.apply()` use Python for-loops that can't be captured. `--disable-cuda-graph` is required.
**Impact:** ~30-50% performance penalty on all operations (not just MoE).
**Proposed fix:** Replace Python loops with tensor operations (scatter/gather/index_select) that are graph-compatible. For moe_align_block_size, implement as a Triton kernel.

### 9. Qwen3.5 AWQ quality regression (35/39 vs 39/39)
**Status:** Minor
**Root cause:** v0.5.10 + sgl_kernel fallbacks. 4 test failures are from thinking mode reasoning consuming the 256-token budget. Not a quality regression per se.
**Impact:** Cosmetic — the model is correct, just verbose in thinking mode.
**Proposed fix:** Increase max_tokens in eval script for thinking-mode models, or strip thinking tokens from evaluation.

## Completed (for reference)

### ~~sgl_kernel import crashes~~
**Fixed:** Patched `sgl_kernel/__init__.py` with graceful degradation + torch fallbacks for silu_and_mul, gelu_and_mul, rmsnorm, topk_softmax, topk_sigmoid, moe_align_block_size.

### ~~AWQ MoE OOM (28GB/GPU)~~
**Fixed:** `AWQTritonMoEMethod` keeps expert weights packed in INT4 (7.93 GB/GPU). Per-expert `awq_gemm_triton` dispatch.

### ~~Coder-Next AWQ weight loader crash~~
**Fixed:** `qwen3_next.py` — skip packed weight_loader override when module has qweight (AWQ) instead of weight (FP16).

### ~~Coder-Next shared_expert load error~~
**Fixed:** Added `shared_expert`, `shared_expert_gate`, DeltaNet projections, attention projections to `modules_to_not_convert` in AWQ config.
