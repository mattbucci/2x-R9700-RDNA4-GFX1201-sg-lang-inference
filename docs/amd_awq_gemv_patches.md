# AMD AWQ GEMV Branch Analysis for RDNA4 SGLang Porting

**Date:** 2026-04-05
**Branch:** `mgehre-amd/vllm:matthias.awq_gemv`
**Branch HEAD:** `53c4a47b5e8f` (Apr 1, 2026)
**Target:** SGLang v0.5.10rc0 on 2x AMD R9700 (gfx1201, RDNA4)

## Executive Summary

Matthias Gehre's branch contains ~100 commits on top of vLLM main implementing
a comprehensive AWQ INT4 optimization stack for AMD RDNA3/3.5/4 GPUs. The key
innovations are:

1. **Hand-optimized HIP GEMV kernel** for M=1 decode (split-K, FP16 bit-trick dequant)
2. **HIP skinny GEMM kernels** (wvSplitK) for int4/int8, supporting wave32 on gfx11/gfx12
3. **Hybrid W4A16 dispatch** - HIP skinny for decode (M<=5), Triton for prefill (M>5)
4. **Fused MoE ExLlama GEMM** - 4-bit dequant fused with expert GEMM (27% decode speedup)
5. **Triton W4A16 kernel** - fused dequant GEMM reading ExLlama shuffle format directly

Our SGLang already has the critical fused AWQ Triton GEMM (the "4x fix"). The question
is which of these vLLM-specific kernel optimizations translate to additional speedups.

---

## Commit-by-Commit Analysis

### Phase 1: Triton AWQ GEMM Foundation (Dec 2025 - Jan 2026)

| SHA | Title | Files Changed | Relevance |
|-----|-------|--------------|-----------|
| `3d809d94e7c0` | [Bugfix] awq_gemm: fix argument order swap | awq_triton.py | Already fixed in SGLang |
| `bb1a2a64cdb5` | AWQ GEMV (initial Triton GEMV kernel) | awq_triton.py (+305) | **Superseded** by HIP kernel |
| `c8a1f60be78a` | AWQ: Evaluate fused vs unfused GEMM on actual shape | awq.py | **We have this** - our AWQLinearMethod.apply() already routes to fused GEMM |
| `10c733ed15e9` | Add VLLM_ALLOW_UNFUSED_AWQ_GEMM | env | Not needed - we hardcode fused |
| `3a8885d3c35b` | _awq_gemm_triton: Improve for small shapes | awq_triton.py | Low - our Triton GEMM already tuned |
| `e9add129ad9d` | [Bugfix] awq_gemm: fix argument order swap | awq_triton.py | Already in SGLang |
| `caebc31df6a8` | Use unfused AWQ GEMM for prefill | _custom_ops.py | **Opposite of our approach** - we use fused for both |
| `db432b576d8a` | fix(triton): use range() instead of tl.static_range() | awq_triton.py | **RELEVANT** - Triton 3.6 hang fix |

### Phase 2: HIP GEMV Kernel (Jan 2026)

| SHA | Title | Files Changed | Relevance |
|-----|-------|--------------|-----------|
| `c9426ddd9218` | Hand-optimized HIP AWQ GEMV | awq_gemv_hip.cu (+848) | **HIGH** - native HIP M=1 kernel |
| `c5b25f1d33aa` | HIP GEMV: acc in float | awq_gemv_hip.cu | Part of above |
| `be42a5b26f34` | Improve perf by using split-K | awq_gemv_hip.cu | Part of above |
| `e7dcbd8cbe10` | split_k 16 and pad weights | awq_gemv_hip.cu | Weight padding for alignment |
| `4a6d514db5b2` | Unified padding impl | awq_gemv_hip.cu | Part of above |
| `d70faed66b0e` | AWQ GEMV: non-power-of-2 split-k, per-GPU benchmarks | awq_gemv_hip.cu, benchmark | Tuning infrastructure |
| `4a259da3eff5` | AWQ GEMV: FP16 bit-trick for 4-bit dequantization | awq_gemv_hip.cu (+131) | **HIGH** - performance trick |
| `f92dc83695e9` | Externalize AWQ GEMV split-k into JSON config files | awq_gemv_config.py, JSON | Per-device tuning configs |
| `b5bd597bec83` | AWQ GEMV: retune with 256MB cache flush, fix padding | awq.py, JSON | Tuning accuracy fix |

**Architecture:** The HIP GEMV kernel is a hand-written CUDA/HIP kernel for M=1 (single-token
decode) that uses:
- Split-K parallelism: each thread block has SPLIT_K splits of 32 threads
- FP16 bit-trick: OR 4-bit nibbles into fp16(1024.0) bit patterns instead of int-to-float conversion
- Pipeline depth 16: K-elements per iteration with instruction-level parallelism
- 8 output elements per thread
- Per-device JSON tuning configs for split-K selection based on N dimension

### Phase 3: wvSplitK Skinny GEMM for RDNA (Feb-Mar 2026)

| SHA | Title | Files Changed | Relevance |
|-----|-------|--------------|-----------|
| `ff5a32747faf` | [ROCm] GFX11 support for wvSplitK skinny GEMM | skinny_gemms.cu (+154) | **HIGH** - wave32 fp16/bf16 skinny GEMM |
| `d3a40c4483f9` | skinny_gemms: add GFX11 RDNA3/4 support | Same | Same commit, earlier version |
| `583562b65956` | W4A16 int4 skinny GEMM kernel for gfx11 | skinny_gemms_int4.cu (+701) | **HIGH** - int4 wave32 skinny GEMM |
| `857070def9b3` | ExLlama bit-trick for int4->fp16 dequant | skinny_gemms_int4.cu | 2.3% faster dequant |
| `91dac7d310e3` | Replace fp16 dot2acc intrinsic with manual float | skinny_gemms.cu | GFX11 compatibility |
| `757de5b29262` | Restore hardware dot2 intrinsic in DOT2C macro | skinny_gemms_int4.cu | Revert to hw intrinsic |
| `ec7450df27d3` | Support N=5 in skinny GEMM for speculative decoding | skinny_gemms*.cu | **MEDIUM** - eagle3 specdec |
| `74ae947b90ef` | Tune GFX11 YTILE/UNRL heuristic | skinny_gemms.cu | Performance tuning |

**Architecture:** The wvSplitK kernels are wave-level split-K GEMM kernels that:
- Use wave32 execution on RDNA (vs wave64 on CDNA)
- Use `__builtin_amdgcn_fdot2` for fused FP16 dot products on RDNA
- Support YTILE={1,2,4} rows per thread block, UNRL={1,2,4} loop unrolling
- Handle int4 with ExLlama shuffle format and per-group scales
- Support M=1..5 (decode and speculative decode verification)

**Benchmark (gfx1151 - Strix Halo RDNA3.5):**
- Qwen3-4B bf16: TPOT 50ms -> 40ms (20% faster with skinny GEMM)
- Qwen3-4B AWQ: TPOT 172ms -> 170ms (1.6% faster for int4)
- gate_up (1x2560x19456): 1.21x faster than torch.nn.functional.linear

### Phase 4: Hybrid W4A16 Kernel (Mar 2026)

| SHA | Title | Files Changed | Relevance |
|-----|-------|--------------|-----------|
| `d0a692642fc7` | TritonW4A16LinearKernel for ROCm (from PR #37352) | triton_w4a16.py (+404) | **HIGH** - Triton fused dequant GEMM |
| `73b5c71e01fb` | Tune TritonW4A16 block sizes for gfx1151 | | RDNA3.5 tuning |
| `97df1e95578b` | Add HybridW4A16LinearKernel: Triton prefill, HIP decode | hybrid_w4a16.py (+464) | **HIGH** - hybrid dispatch |
| `a9a4d1b4605f` | Unify to single skinny-format weight copy | hybrid_w4a16.py | Saves ~5GB memory |
| `0c21704f4279` | Refine skinny-fmt triton tile sizes | | Neighbor sweep tuning |
| `0c658ecad78e` | Converge M>1024 tall-K tile size | | Large batch tuning |
| `14ffc02c8464` | Simplify: use on_gfx1x(), rename benchmark | hybrid_w4a16.py | Code cleanup |
| `3fe3022baf18` | **Final PR: HybridW4A16LinearKernel** | hybrid_w4a16.py (+464), tests | **KEY COMMIT** |
| `da4b0ce71467` | Pre-compute dequantized weights at load time | hip_w4a16_skinny.py | Memory optimization |

**Architecture:** The Hybrid kernel stores weights **once** in ExLlama shuffle format
`[N, K//8]` and dispatches:
- **M <= 5:** HIP `wvSplitK_int4_g` skinny kernel (decode)
- **M > 5:** Triton `_triton_w4a16_skinny_fmt_kernel` (prefill)

Both paths read from the same weight tensor. The ExLlama shuffle format packs 8
int4 values per uint32 with interleaved nibbles: `v0|v2<<4|v4<<8|v6<<12|v1<<16|v3<<20|v5<<24|v7<<28`.

**Benchmark (gfx1151, Qwen3-4B-quantized.w4a16):**
- Decode: 70.6 tok/s (HIP skinny, same as standalone skinny kernel)
- Prefill TTFT: 376ms (Triton skinny-fmt)
- Memory: saves ~5GB vs skinny kernel's fp16 dequant fallback copy

### Phase 5: AWQ Format Fixes and Zero-Point Support (Mar 2026)

| SHA | Title | Files Changed | Relevance |
|-----|-------|--------------|-----------|
| `b71b3e160078` | Fix HipW4A16 weight/scale layout for AWQ models | hip_w4a16.py (+33) | **MEDIUM** - AWQ layout fix |
| `ce0e6c722206` | [ROCm] Fix HipW4A16LinearKernel weight/scale transpose | hip_w4a16.py | PR version of above |
| `af513c22d608` | Add zero-point support to HipW4A16SkinnyLinearKernel | skinny_gemms_int4.cu (+229) | **MEDIUM** - asymmetric AWQ |
| `c47a73c43653` | **PR: zero-point support** | skinny_gemms_int4.cu, tests | Enables asymmetric quant |
| `a8b9a8f979df` | Fix build: is_gfx1x_int4, unified ExLlama shuffle | skinny_gemms_int4.cu, py | Build fix |
| `0cdafcc4e4bb` | **PR: unified ExLlama shuffle packing** | Same | PR version |

### Phase 6: Fused MoE ExLlama GEMM (Feb-Mar 2026)

| SHA | Title | Files Changed | Relevance |
|-----|-------|--------------|-----------|
| `0ab69e7950cd` | Fused MoE exllama GEMM for compressed-tensors | exllama_moe.py (+265) | **HIGH** - fused MoE int4 |
| `598c07ba01b7` | [ROCm] Add fused MoE exllama GEMM test | test_exllama_moe.py (+201) | Tests only |

**Architecture:** Replaces generic Triton fused_moe with ExLlama-based kernel that fuses
4-bit dequantization with GEMM. Two paths:
- **Decode (block_size_m=1):** Scattered kernel, handles gather/scatter internally
- **Prefill (block_size_m=2,4):** Contiguous layout via `moe_align_block_size`, amortizes dequant

**Benchmark (gfx1151, Qwen3-30B-A3B-AWQ-4bit):**
- Decode TPOT: 21.87ms -> 15.91ms (27% faster)
- Prefill TTFT: 934ms -> 1609ms (72% slower -- tradeoff)

### Phase 7: Attention and Other Optimizations

| SHA | Title | Files Changed | Relevance |
|-----|-------|--------------|-----------|
| `81c56f0daac0` | Default to TRITON_ATTN on RDNA gfx11/gfx12 | rocm.py (+14) | **HIGH** - 31% TPOT fix |
| `764d610055e9` | Same (pre-PR) | rocm.py | Same |
| `1a101e829849` | Tune triton_scaled_mm for gfx11 decode | triton_scaled_mm.py (+88) | **MEDIUM** - W8A8 tuning |
| `3228caef9ccb` | Pinned CPU tensor for async sampled token ID copy | gpu_model_runner.py (+10) | **MEDIUM** - 1.9x copy speedup |
| `19e96a9b0ba3` | Support quantized lm_head | compressed_tensors.py, conch.py | **LOW** for us |
| `410d30089310` | Enable AWQMarlinConfig on ROCm | awq_marlin.py (+226) | **LOW** - Marlin on ROCm |
| `de0411e030cf` | Fix compressed-tensors W4A16 asymmetric zeros | conch.py (+45) | **LOW** - CT-specific |
| `73419abfae97` | [Bugfix] Handle Asym W4A16 ConchLinearKernel for CT | conch.py (+42) | **LOW** - CT-specific |
| `7e65e596b352` | Transposed dequant GEMM | awq_triton.py (+219) | **LOW** - experimental |

---

## Porting Analysis for SGLang on RDNA4

### What We Already Have

Our SGLang codebase already includes:
1. **Fused AWQ Triton GEMM** (`awq_gemm_triton` in `awq_triton.py`) - the critical 4x decode fix
2. **Tuned block sizes** (bm=32, bn=64, bk=64) via our sweep_awq_blocks.py
3. **Split-K dispatch** (split_k=8 for M<=16 decode, split_k=4 for M>16 prefill)
4. **FP32 accumulation** in the Triton GEMM kernel
5. **RDNA4-specific FP8 paths** with num_stages workarounds

### What We Are Missing

1. **No HIP native GEMV kernel** - we use Triton for all batch sizes including M=1
2. **No skinny GEMM dispatch** - no wave32 wvSplitK path for decode
3. **No hybrid dispatch** - no batch-size-dependent kernel selection
4. **No fused MoE int4 kernel** - we use generic Triton fused_moe for MoE AWQ models
5. **No ExLlama shuffle format** - weights stored in standard AWQ packed format
6. **No per-device tuning configs** - our block sizes are static

---

## Priority Ranking for Porting

### P0: HIGH IMPACT, MODERATE EFFORT

#### 1. Triton AWQ GEMV Kernel for M=1 Decode
**Source commits:** `bb1a2a64cdb5` (initial), superseded by HIP kernel
**What:** Write a Triton-based GEMV kernel specialized for M=1 that avoids the overhead
of the general GEMM kernel (block tile setup, split-K reduction for small M).
**Why:** Our current Triton GEMM works for M=1 but wastes compute on tile management.
A dedicated GEMV can skip BLOCK_SIZE_M tiling entirely.
**Effort:** Medium - can adapt the Triton GEMV from the initial commit rather than
porting the entire HIP C++ kernel.
**Expected gain:** 10-30% decode TPOT improvement based on Strix Halo numbers.

**Implementation sketch:**
```python
# In awq_triton.py, add awq_gemv_triton_kernel for M=1
# Key differences from awq_gemm_kernel:
# - No BLOCK_SIZE_M dimension (single row)
# - Each program processes one output element range
# - Higher split-K (16) for better K-parallelism
# - Direct accumulation without tile bookkeeping
```

**In awq.py AWQLinearMethod.apply():**
```python
if _is_hip:
    M = reshaped_x.shape[0]
    if M == 1:
        out = awq_gemv_triton(reshaped_x, qweight, scales, qzeros)
    else:
        out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, ...)
```

#### 2. Batch-Size Dependent Dispatch (Simplified Hybrid)
**Source commits:** `3fe3022baf18` (HybridW4A16)
**What:** Implement batch-size-dependent block size selection in our existing Triton GEMM.
For M=1, use high split-K (16) and smaller tiles. For M>32, use lower split-K (2) and
larger tiles optimized for throughput.
**Why:** A single set of block sizes cannot be optimal across 3 orders of magnitude of M.
**Effort:** Low - just extend the existing `if M <= 16 else` dispatch in AWQLinearMethod.apply().
**Expected gain:** 5-15% across mixed workloads.

**Implementation sketch:**
```python
# In awq.py AWQLinearMethod.apply():
if M == 1:
    split_k, bm, bn, bk = 16, 16, 64, 64    # GEMV-like
elif M <= 8:
    split_k, bm, bn, bk = 8, 32, 64, 64     # Small decode batch
elif M <= 64:
    split_k, bm, bn, bk = 4, 32, 64, 64     # Current default
else:
    split_k, bm, bn, bk = 2, 64, 128, 64    # Prefill
```

### P1: HIGH IMPACT, HIGH EFFORT

#### 3. HIP wvSplitK Int4 Skinny GEMM Kernel
**Source commits:** `583562b65956`, `ff5a32747faf`, `857070def9b3`
**What:** Port the native HIP wvSplitK_int4_g kernel for gfx12 (wave32).
**Why:** Native HIP kernels avoid Triton compilation overhead and can use hardware
intrinsics (fdot2) directly. On Strix Halo, skinny GEMM is 1.2x faster than
torch.nn.functional.linear for gate_up projections.
**Effort:** High - requires C++/HIP kernel compilation infrastructure in SGLang's sgl-kernel,
plus ExLlama weight repacking at load time.
**Expected gain:** 15-25% decode TPOT for dense AWQ models.
**Blockers:** SGLang's sgl-kernel build system would need ROCm C++ compilation support
for custom HIP kernels. Currently sgl-kernel only compiles CUDA.

**Key design decisions:**
- Weights must be repacked to ExLlama shuffle format `[N, K//8]` at load time
- Need `is_gfx12()` / `is_gfx1x()` runtime detection helpers
- Wave32 butterfly reduction via `__shfl_xor` (not DPP row_shr as on GFX9)
- `__builtin_amdgcn_fdot2` for fused FP16 dot products
- Template parameters: THRDS=32, YTILE={1,2,4}, UNRL={1,2,4}

#### 4. Fused MoE Int4 Kernel (ExLlama-based)
**Source commits:** `0ab69e7950cd`, `598c07ba01b7`
**What:** Port the ExLlama MoE kernel that fuses 4-bit dequant with expert GEMM.
**Why:** 27% decode speedup on Qwen3-30B-A3B MoE model. MoE decode is particularly
sensitive because each token only activates a few experts, making the per-expert
overhead of dequant + GEMM proportionally large.
**Effort:** High - requires ExLlama CUDA/HIP kernel, integration with SGLang's MoE
dispatch (different from vLLM's), and expert routing compatibility.
**Expected gain:** 20-30% decode TPOT for AWQ MoE models (Devstral, Qwen3-30B-A3B).
**Blockers:** Same sgl-kernel HIP compilation issue. Also, SGLang's MoE dispatch
uses a different MoeRunner abstraction than vLLM's fused_moe.

### P2: MEDIUM IMPACT, LOW-MEDIUM EFFORT

#### 5. Triton GEMM tl.static_range() Fix
**Source commit:** `db432b576d8a`
**What:** Replace `tl.static_range()` with `range()` in AWQ GEMM kernel to fix
Triton 3.6 compilation hangs.
**Why:** Prevents kernel compilation hangs with Triton 3.6 + torch.compile/CUDA graphs.
**Effort:** Trivial - 4 lines changed.
**Expected gain:** Stability fix, not performance. Prevents hangs in compilation.

**Status: ALREADY FIXED.** Our awq_gemm_kernel uses `range()` at line 180 in awq_triton.py.
No action needed.

#### 6. Triton Attention Default for RDNA
**Source commit:** `81c56f0daac0`
**What:** Default to TRITON_ATTN instead of ROCM_ATTN on gfx11/gfx12.
**Why:** ROCM_ATTN's paged attention falls back to Triton internally on RDNA with
extra overhead, causing 31% TPOT regression in speculative decoding.
**Effort:** Low - configuration change.
**Expected gain:** 31% TPOT improvement for speculative decoding workloads.
**Status:** Need to verify our attention backend selection. SGLang may already prefer
Triton attention on RDNA4.

#### 7. FP16 Bit-Trick Dequantization in Triton
**Source commit:** `4a259da3eff5`
**What:** Replace int-to-float AWQ weight conversion with OR-ing nibbles into fp16(1024.0)
bit patterns, avoiding expensive conversion instructions.
**Why:** Eliminates `ushort2half_rn` conversion instructions.
**Effort:** Medium - need to modify the dequant path in awq_gemm_kernel.
**Expected gain:** "Unknown whether this provides a performance benefit on RDNA" per commit
message. Worth benchmarking.

**Implementation sketch:**
```python
# Instead of: weights = (packed >> shifts) & 0xF; weights = weights.to(float16)
# Use: OR nibbles into fp16(1024.0), subtract bias
FP16_1024 = 0x6400  # fp16 representation of 1024.0
low_nibbles = (packed & 0x000F000F) | (FP16_1024 << 16 | FP16_1024)
# Result: fp16(1024 + nibble_value), bias cancels when subtracting zeros
```

#### 8. Pinned CPU Tensor for Async Token Copy
**Source commit:** `3228caef9ccb`
**What:** Use pinned CPU tensor + .copy_() instead of .to("cpu", non_blocking=True).
**Why:** Avoids DtoD staging overhead in PyTorch's `_to_copy` path.
**Effort:** Low - ~10 lines in model runner.
**Expected gain:** TPOT 18.5ms -> 14.5ms on Strix Halo (21% reduction).
**Applicability:** SGLang's model runner uses a different architecture than vLLM v1.
Need to find the equivalent token sampling output copy path.

### P3: LOW IMPACT OR NOT APPLICABLE

#### 9. Quantized lm_head Support
**Source commit:** `19e96a9b0ba3`
**What:** Allow quantized linear for lm_head in compressed-tensors.
**Impact:** Low - most AWQ models have unquantized lm_head.

#### 10. AWQMarlinConfig on ROCm
**Source commit:** `410d30089310`
**What:** Enable Marlin kernel for AWQ on ROCm.
**Impact:** Low - Marlin requires CDNA (gfx9) for optimal performance.
gfx12 (RDNA4) lacks the matrix core instructions Marlin targets.

#### 11. Compressed-Tensors Fixes (ConchLinearKernel, zero-point unpacking)
**Source commits:** `de0411e030cf`, `73419abfae97`, `5e806bcf541c`, `56a62c310cc4`
**What:** Various fixes for compressed-tensors W4A16 on ROCm.
**Impact:** Low - SGLang uses AWQ format directly, not compressed-tensors.

#### 12. torch.compile Skinny GEMM Fix
**Source commit:** `73bb9fa2ea22`
**What:** Wrap skinny GEMM dispatch in custom op for torch.compile compatibility.
**Impact:** Not applicable - SGLang doesn't use torch.compile on RDNA4, and we
don't have the skinny GEMM kernel.

#### 13. OpenVLA Model Support
**Source commits:** `53c4a47b5e8f`, `8dd936afeba1`, etc.
**What:** Vision-Language-Action model support.
**Impact:** Not relevant to our AWQ optimization goals.

---

## Recommended Implementation Plan

### Week 1: Quick Wins (P2)
1. Verify `tl.static_range()` fix status (5 min)
2. Implement batch-size-dependent block size dispatch (P0.2) (2 hours)
3. Benchmark pinned CPU tensor approach in SGLang's token output path (P2.8) (4 hours)
4. Verify attention backend selection on RDNA4 (P2.6) (1 hour)

### Week 2: Triton GEMV Kernel (P0)
1. Write dedicated `awq_gemv_triton_kernel` for M=1 (P0.1)
2. Benchmark against current GEMM kernel at M=1
3. Integrate into AWQLinearMethod.apply() dispatch
4. Test with Devstral-24B and Qwen3-Coder-30B

### Week 3-4: HIP Native Kernels (P1) - If Triton GEMV Shows Gains
1. Evaluate sgl-kernel HIP compilation feasibility
2. Port wvSplitK_int4_g with wave32 support
3. Implement ExLlama shuffle weight repacking
4. Benchmark against Triton GEMV

### Future: MoE Optimizations (P1)
1. Port ExLlama MoE kernel framework
2. Integrate with SGLang's MoeRunner abstraction
3. Test with Devstral-24B (MoE model, our primary workload)

---

## Key Technical Differences: vLLM vs SGLang

| Aspect | vLLM (Matthias branch) | SGLang (our codebase) |
|--------|----------------------|---------------------|
| Kernel dispatch | MPLinearKernel registry with priority | Direct in AWQLinearMethod.apply() |
| Weight format | ExLlama shuffle `[N, K//8]` | Standard AWQ `[K, N//8]` |
| MoE framework | fused_moe with ExllamaExperts | MoeRunner with Triton backend |
| HIP kernels | Custom C++ in csrc/rocm/ | sgl-kernel (CUDA-only currently) |
| torch.compile | Supported with custom op wrappers | Not used on RDNA4 |
| Attention | TRITON_ATTN default for RDNA | Need to verify |
| FP8 | Working on gfx11 | Blocked by Triton GPU hangs on gfx1201 |

## Risk Assessment

- **HIP kernel porting (P1):** High risk. sgl-kernel's build system needs ROCm support.
  The wvSplitK kernels are ~700 lines of complex HIP C++ with architecture-specific
  intrinsics. Could take weeks.

- **Triton GEMV (P0.1):** Medium risk. The Triton GEMV approach was superseded by HIP
  in vLLM, suggesting Triton GEMV wasn't fast enough. However, on gfx1201 (newer than
  gfx1151), Triton may perform differently.

- **MoE kernel (P1.4):** Medium-high risk. Different MoE abstractions between vLLM and
  SGLang mean the integration layer needs rewriting, not just porting.

- **Block size tuning (P0.2):** Low risk. Simple parameter changes with clear benchmarking
  methodology.

---

## Source References

- Branch: https://github.com/mgehre-amd/vllm/tree/matthias.awq_gemv
- Key file: `csrc/quantization/awq/awq_gemv_hip.cu` (HIP GEMV kernel)
- Key file: `csrc/rocm/skinny_gemms_int4.cu` (wvSplitK int4 kernel)
- Key file: `vllm/model_executor/kernels/linear/mixed_precision/hybrid_w4a16.py` (Hybrid dispatch)
- Key file: `vllm/model_executor/layers/fused_moe/exllama_moe.py` (Fused MoE)
- Upstream Triton W4A16 PR: https://github.com/vllm-project/vllm/pull/37352
