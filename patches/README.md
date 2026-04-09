# SGLang v0.5.10 RDNA4 Patches

Apply in order to SGLang v0.5.10 (`git checkout v0.5.10`):

## 001-rdna4-core-v0.5.10.patch
Core RDNA4 support — required for all models on gfx1201.

Includes:
- Fused AWQ Triton GEMM (4x decode speedup vs dequant+matmul)
- torch.compile disabled on HIP (prevents 30+ min stalls)
- Triton 3.6.0/3.7.0 support in MoE config
- RDNA4 MoE kernel + torch-native FP8 fallbacks
- sgl-kernel setup_rocm.py: gfx12xx support, FP8 E4M3, 48KB LDS
- Gemma4 model support (native, not transformers backend)
- FP8 block quant: BLOCK_SIZE_M=16, num_stages=2 for RDNA4
- Qwen3.5 TP=2: replicated DeltaNet+MLP layers, float32 allreduce
- Devstral: chat template BOS fix, text-only VLM warmup
- CUDA-only import guards for aiter, flashinfer, quark

## 002-awq-performance-tuning.patch
AWQ GEMM performance tuning — batch-size-dependent block dispatch.

- M=1: split_k=16, bm=16 (decode optimization)
- M>32: split_k=2, bm=64, bn=128 (prefill throughput)
- +6% single-user decode, +13% throughput at 64 concurrent
- Also includes awq_gemv_triton kernel (unused, for future work)

## 003-hip-awq-gemv-kernel.patch
Native HIP AWQ GEMV kernel — **integrated into serving path**.

- Ported from mgehre-amd/vllm matthias.awq_gemv branch
- 966-line HIP C++ kernel, wave32 compatible, split-K parallelism
- `awq_gemv_hip`: M=1 decode GEMV — **30% faster** than Triton GEMM
- `awq_gemv_moe_hip`: Fused MoE dispatch — all experts in one GPU kernel (**151x faster** than per-expert loop in microbenchmarks)
- Build: `scripts/build_awq_gemv.sh --env <env-name>`
- `AWQLinearMethod.apply()` uses HIP GEMV for M=1, Triton GEMM for M>1
- `AWQTritonMoEMethod.apply()` uses HIP GEMV MoE when available

## 004-sgl-kernel-rdna4-fallbacks.patch
**CRITICAL** — sgl_kernel graceful degradation for RDNA4 (gfx1201).

Rewrites `sgl-kernel/python/sgl_kernel/__init__.py` to:
- Wrap all native op imports in try/except (RDNA4 lacks many CDNA ops)
- Provide torch fallbacks for: silu_and_mul, gelu_and_mul, rmsnorm,
  fused_add_rmsnorm, rotary_embedding, topk_softmax, topk_sigmoid,
  moe_align_block_size
- Preserve native elementwise ops when available (silu_and_mul, gelu_and_mul,
  rotary_embedding from HIP build)
- **Fix the fallback-override bug**: upstream `__init__.py` checks
  `torch.ops.sgl_kernel.{op}` registration and overwrites native elementwise
  imports with Python fallbacks if the op isn't registered as a torch custom op.
  Since the HIP build registers ops differently, `rotary_embedding` gets
  overwritten with `_fb_rotary_embedding` which produces wrong results on
  non-contiguous tensors from `qkv.split()`. This causes garbage output for
  dense AWQ models on torch 2.12.

Must be applied BEFORE building sgl_kernel native HIP ops.

## Apply

```bash
cd components/sglang
git checkout v0.5.10
git apply ../../../patches/001-rdna4-core-v0.5.10.patch
git apply ../../../patches/002-awq-performance-tuning.patch
git apply ../../../patches/003-hip-awq-gemv-kernel.patch  # optional
git apply ../../../patches/004-sgl-kernel-rdna4-fallbacks.patch
```

## Build native kernels after patching

```bash
# 1. Build and install sgl_kernel (CRITICAL — fixes rotary_embedding)
scripts/setup_sgl_kernel.sh --env <env-name>
scripts/setup_sgl_kernel.sh --env <env-name> --verify

# 2. Build and install AWQ GEMV HIP kernel (30% faster decode)
scripts/build_awq_gemv.sh --env <env-name>
```
