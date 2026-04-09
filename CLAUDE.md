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
- `patches/` — SGLang patches (4 patches, apply in order)
- `docs/` — Analysis documents and known issues

## Key Environments
Both environments use the same sglang code from `components/sglang/`.

- **`sglang-triton36`** (torch 2.12.0.dev20260310+rocm7.2) — Primary env for all dense AWQ models
- **`sglang-clean`** (torch 2.11.0+rocm7.2) — Alternative, same capabilities
- MoE AWQ on SGLang is **blocked** (Triton codegen crash on gfx1201) — use vLLM Docker FP8 for MoE
- Models: `~/AI/models/`

## Native Kernel Builds — CRITICAL

### sgl_kernel (rotary_embedding, activation functions)
```bash
scripts/setup_sgl_kernel.sh --env <env-name>
scripts/setup_sgl_kernel.sh --env <env-name> --verify
```

### AWQ GEMV HIP kernel (30% faster M=1 decode, fused MoE dispatch)
```bash
scripts/build_awq_gemv.sh --env <env-name>
```
Ported from `mgehre-amd/vllm` matthias.awq_gemv branch. Provides:
- `awq_gemv_hip`: Native HIP M=1 GEMV (FP16 bit-tricks, wave32)
- `awq_gemv_moe_hip`: Fused MoE expert dispatch (all experts, single GPU kernel)

## Quick Reference
```bash
# Launch Devstral AWQ (working, 841 tok/s @ 32 concurrent)
scripts/run_devstral_awq.sh

# Launch Coder-30B FP8 via vLLM Docker (working, 1185 tok/s peak)
scripts/bench_vllm_docker.sh

# Install native kernels to a new env
scripts/setup_sgl_kernel.sh --env <env-name>
scripts/build_awq_gemv.sh --env <env-name>

# Run quality check
scripts/eval_comprehensive.py [--thinking-budget 512]

# Benchmark
scripts/bench_comprehensive.sh <label> auto <port>
```

## Model Status
| Model | Quant | Engine | Status | Throughput |
|-------|-------|--------|--------|-----------|
| Devstral-24B | AWQ-4bit | SGLang | Working | 841 tok/s @ 32 |
| Qwen3.5-27B | AWQ-4bit | SGLang | Working | 129 tok/s @ 32 |
| Coder-30B-A3B | FP8 | vLLM Docker | Working | 1185 tok/s peak |
| Coder-30B-A3B | AWQ-4bit | SGLang | Blocked | Triton crash |
| Gemma 4 (26B/31B) | AWQ | SGLang | Blocked | Bad weights |
| Coder-Next-80B | AWQ | SGLang | Blocked | Bad weights |
