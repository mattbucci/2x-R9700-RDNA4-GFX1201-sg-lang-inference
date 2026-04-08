# Torch 2.11 vs 2.12 Root Cause Analysis for RDNA4

## Summary

Dense AWQ models produce garbled output on torch 2.12 but not 2.11. MoE AWQ models crash on torch 2.11 but work on 2.12. This analysis identifies the exact divergence point.

## Environment Details

| Component | sglang-clean (2.11) | sglang-triton36 (2.12) |
|-----------|---------------------|------------------------|
| torch | 2.11.0+rocm7.2 | 2.12.0.dev20260310+rocm7.2 |
| triton | 3.6.0 | 3.6.0 |
| libtriton.so | d029398... (415.8 MB) | 9019d15... (415.6 MB) |
| bundled ld.lld | YES (100 MB) | NO (uses system /usr/bin/ld.lld) |
| HIP headers | 7.1.52802 | 7.1.25424 |
| transformers | 5.5.0 | 5.3.0 |

## What Was Tested

### Identical on Both Envs (NOT the bug)
1. **AWQ GEMM Triton kernel** — bit-identical output with synthetic data (all shapes/split_k)
2. **AWQ GEMM with real Devstral weights** — bit-identical for all projections (gate/up/down/qkv/o)
3. **AWQ GEMM in multiprocess context** — bit-identical on both GPUs via mp.spawn
4. **AWQ interleave+shift pattern** — identical unpacking
5. **AWQ dequantize kernel** — identical
6. **torch.compile/inductor kernels** — correct on both
7. **FP8 KV cache roundtrip** — equivalent precision loss
8. **Basic torch ops** (matmul, embedding, SiLU, RMSNorm) — identical
9. **torch SDPA** — correct in isolation

### What Diverges: Attention Computation in Full Model
Using AWQ debug instrumentation (logging x_sum/out_sum at each AWQ call):

- **Layer 0, QKV projection**: Input and output are **IDENTICAL** between envs (`out_sum=-1146.7246`)
- **Layer 0, O projection input** (= attention output): **DIVERGES**
  - torch 2.12: `x_sum=-8.7207`, `x_std=0.008705`
  - torch 2.11: `x_sum=-9.0468`, `x_std=0.010578`
  - Relative diff: ~3.7%
- **Layer 1, Gate+Up input**: diff = 97.8 (47× amplification from layer 0)
- **Layer 4, Gate+Up input**: diff = 178.0
- Result: by final layer, hidden states are completely decorrelated → garbage logits

## What the Bug Is NOT
- NOT the AWQ GEMM kernel (proven identical)
- NOT FP8 KV cache (garbled without it too)
- NOT the Triton attention backend specifically (garbled with `torch_native` too)
- NOT torch.compile (explicitly disabled, verified)
- NOT tensor parallelism or all-reduce (same x_sum on both TP ranks)
- NOT libtriton.so alone (swapping 2.11's libtriton.so into 2.12 didn't fix it)

## What the Bug IS
The attention computation in the full serving pipeline produces numerically different results on torch 2.12 vs 2.11. Since:
1. Simple SDPA is fine in isolation
2. Both Triton and torch_native attention backends show the bug
3. The bug is specific to the full serving pipeline (paged attention, KV cache management)

The root cause is likely in **torch's internal kernels for paged/indirect attention**, or in how torch 2.12 handles the tensor operations that SGLang uses to manage paged KV cache (scatter/gather/index operations on GPU memory).

## MoE Issue (Separate Bug)
MoE AWQ crashes on torch 2.11 with `hipErrorLaunchFailure` in `fused_moe_kernel_gptq_awq`. This is a **different** libtriton.so codegen bug — torch 2.11's Triton generates invalid HSACO for the fused MoE kernel's `tl.dot` pattern. Our workaround (sort-based per-expert dispatch with `awq_gemm_triton`) avoids the fused kernel entirely and works on both envs.

## Proposed Fixes

### Option A: Unify on torch 2.11 (recommended short-term)
The MoE crash on torch 2.11 is in the **fused** kernel. Our sort-based dispatch workaround already avoids it. If we verify the sort-based MoE dispatch works on torch 2.11, we can run BOTH dense and MoE models on a single env.

**Steps:**
1. Launch Coder-30B AWQ with sort-based dispatch on `sglang-clean` (torch 2.11)
2. If it works → single env, problem solved
3. If it crashes → the crash is in a different kernel path

### Option B: Fix torch 2.12 attention
Investigate which specific torch 2.12 internal kernel causes the attention divergence. Candidates:
- `torch.index_select` / `torch.gather` for paged KV cache
- `torch.nn.functional.scaled_dot_product_attention` with indirect indexing
- Some fused kernel in torch's CUDA/HIP backend for attention

### Option C: Build Triton from source
Build a single Triton with correct codegen for both attention and MoE kernels. Requires building LLVM/MLIR from Triton's source tree.

### Option D: Use Docker
ROCm Docker (Ubuntu-based) has correct comgr. Both dense and MoE work there.
