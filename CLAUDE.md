# RDNA4 Inference Project

## Overview
Custom SGLang v0.5.10rc0 with RDNA4 (gfx1201) patches for 2x AMD Radeon AI PRO R9700 GPUs.

## Agent Rules
**IMPORTANT**: Read and follow `rules-for-agents.md` before making any changes or running commands. It contains critical RDNA4 constraints, benchmarking requirements, and timeout rules.

## Project Structure
- `components/sglang/` — Patched SGLang (editable install, own git repo on branch `rdna4-v0.5.10rc0`)
- `components/sglang/sgl-kernel/` — sgl_kernel with native HIP build for gfx1201
- `benchmarks/` — Benchmark results (txt files with environment headers)
- `scripts/` — Launch, benchmark, evaluation, and conversion scripts
- `docs/` — Analysis documents and patch notes

## Key Environments
**CRITICAL**: Dense AWQ and MoE AWQ require DIFFERENT torch versions on RDNA4.

- **`sglang-clean`** (torch 2.11.0+rocm7.2) — USE THIS for dense AWQ models (Devstral, Qwen3.5)
  - Dense AWQ Triton kernels compile correctly
  - MoE AWQ crashes with hipErrorLaunchFailure on this torch version
- **`sglang-triton36`** (torch 2.12.0.dev20260310+rocm7.2) — USE THIS for AWQ MoE models (Coder-30B)
  - MoE AWQ kernels work
  - Dense AWQ produces garbage output (Triton codegen regression in torch 2.12 nightly)
- Both envs share the same sglang code from `components/sglang/`
- Both envs have native sgl_kernel (6 HIP ops + 4 torch fallbacks)
- Models: `~/AI/models/`

## sgl_kernel Native Build
Built from source for gfx1201. Native HIP ops: silu_and_mul, gelu_and_mul, gelu_tanh_and_mul, gelu_quick, topk_softmax, moe_align_block_size. Torch fallbacks: rmsnorm, fused_add_rmsnorm, rotary_embedding, topk_sigmoid.

To rebuild: `cd components/sglang/sgl-kernel && AMDGPU_TARGET=gfx1201 python setup_rocm.py build_ext --inplace`

## Quick Reference
```bash
# Launch Devstral AWQ (working — uses sglang-clean)
scripts/run_devstral_awq.sh

# Launch Coder-30B AWQ MoE (working — uses sglang-triton36)
scripts/run_coder30b_awq.sh

# Launch Coder-30B FP8 via vLLM Docker (working)
scripts/bench_vllm_docker.sh

# Run quality check
scripts/eval_comprehensive.py [--thinking-budget 512]

# Benchmark
scripts/bench_comprehensive.sh <label> auto <port>
```
