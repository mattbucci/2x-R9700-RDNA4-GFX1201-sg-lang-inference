# RDNA4 Inference Project

Custom SGLang v0.5.10 + RDNA4 patches for 2x AMD Radeon AI PRO R9700.

**All inference MUST use SGLang.** Other engines (vLLM Docker, llama.cpp) are for comparison benchmarks only.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support, known issues, architecture |
| [rules-for-agents.md](rules-for-agents.md) | RDNA4 constraints, launch flags, quantization rules |

## Key Commands
```bash
scripts/setup.sh                       # Full setup (applies all 5 patches)
scripts/setup_sgl_kernel.sh --env X    # Native sgl_kernel (required)
scripts/build_awq_gemv.sh --env X      # HIP GEMV kernel (required for MoE)
scripts/launch.sh devstral             # Devstral 24B AWQ (131K long-context)
scripts/launch.sh coder-30b            # Coder-30B MoE AWQ
scripts/launch.sh coder-next           # Coder-Next 80B AWQ (131K)
scripts/launch.sh coder-next-ream      # Coder-Next REAM 60B AWQ (131K, pruned)
scripts/launch.sh glm45-air            # GLM-4.5-Air REAP 82B MoE AWQ
scripts/launch.sh gemma4               # Gemma 4 26B MoE AWQ
scripts/launch.sh gemma4-31b           # Gemma 4 31B Dense AWQ
scripts/launch.sh qwen35               # Qwen3.5-27B DeltaNet AWQ (262K)
scripts/launch.sh qwen35-moe           # Qwen3.5-35B MoE GPTQ (262K) — supersedes → Qwen3.6 once calibrated
# Calibration + validation pipeline
scripts/quantize/run_full_pipeline.sh qwen35       # calib → CT→AWQ → merge vision → launch → validate
scripts/quantize/run_full_pipeline.sh gemma4-26b   # same, for Gemma4 26B MoE
scripts/eval/validate_capabilities.py --port 23334 # thinking + vision + basic QA gate
scripts/bench/bench_256k_sweep.sh                   # 256K single-user suite across all long-context models
```

## Critical Rules
- **SGLang only** — all models must run on SGLang with our RDNA4 patches
- **MoE quantization is hard** — standard GPTQ under-calibrates rare experts (see rules-for-agents.md)
- **DeltaNet/SSM layers cannot be INT4 quantized** — recurrent state error accumulation
- **HIP GEMV kernel required** — `scripts/common.sh` sets `LD_LIBRARY_PATH` and `PYTHONPATH`
- Always source `scripts/common.sh` + `activate_conda` + `setup_rdna4_env` before launching
- **Model status and benchmarks** are in README.md (single source of truth)

## Working Mode

**Operate autonomously.** The user reads all output and interrupts with feedback — do not stop for confirmation. Multi-hour calibrations and benchmark sweeps are allowed without asking.

**Primary optimization target: single-user 256K context performance** for all models in README. Multi-user throughput is secondary. When tuning, prioritize TPOT at large context over peak batch tok/s.

**Preserve during calibration:** thinking capabilities AND vision/image handling. Past calibrations have silently broken both — validate both on every requant. 3090 team confirmed this pattern. Use thinking-aware datasets (`AM-Thinking-v1-Distilled`, `glaiveai/reasoning-v1-20m`) plus image+text pairs (`LLaVA-Instruct`).

**Clean commits, shared learnings:**
- Commit + push as progress happens (don't batch).
- README.md is source of truth: what we've done + current known issues.
- 3090 team sister repo at `/home/letsrtfm/AI/2x-3090-GA102-300-A1-sglang-inference` — we can read their commits for ideas and push to their README to share learnings.

## Chat Template Rule

Chat templates matter. We've been burned by:
- Devstral AWQ: BOS token in template produced `<unk>` outputs → custom jinja template fix
- Gemma4 thinking: requires `{"chat_template_kwargs": {"enable_thinking": true}, "skip_special_tokens": false}` per-request
- Qwen3.5: thinking tags in template without calibrated thinking data → infinite `<think>` loop

Any new model: inspect its chat template BEFORE launching, check BOS/EOS behavior, verify thinking token handling.
