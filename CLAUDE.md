# RDNA4 Inference Project

## Overview
Custom SGLang v0.5.10rc0 with RDNA4 (gfx1201) patches for 2x AMD Radeon AI PRO R9700 GPUs.

## Agent Rules
**IMPORTANT**: Read and follow `rules-for-agents.md` before making any changes or running commands. It contains critical RDNA4 constraints, benchmarking requirements, and timeout rules.

## Project Structure
- `components/sglang/` — Patched SGLang (editable install, own git repo on branch `rdna4-v0.5.10rc0`)
- `benchmarks/` — Benchmark results (txt files with environment headers)
- `scripts/` — Launch, benchmark, evaluation, and conversion scripts
- `docs/` — Analysis documents and patch notes

## Key Environment
- Conda env: `sglang-clean` (torch 2.12.0.dev20260403+rocm7.2, Triton 3.6.0)
- Python: `/home/letsrtfm/miniforge3/envs/sglang-clean/bin/python`
- Models: `~/AI/models/`

## Quick Reference
```bash
# Launch Devstral AWQ (working)
scripts/run_devstral_awq.sh

# Launch Coder-30B FP8 via vLLM Docker (working)
scripts/bench_vllm_docker.sh

# Run quality check
scripts/eval_comprehensive.py

# Benchmark
scripts/bench_comprehensive.sh <label> auto <port>
```
