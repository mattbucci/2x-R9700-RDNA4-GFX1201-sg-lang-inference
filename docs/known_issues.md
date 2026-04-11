# Known Issues

## Active

### Gemma 4 26B — working but degraded quality (RTN expert quantization)
**Status:** FIXED. Model loads and generates output on TP=2 with AWQ GPTQ weights.
**Root cause was:** `load_weights()` used fused expert mapping (BF16 only). Per-expert AWQ keys (`experts.0.gate_proj.qweight`) were silently skipped. Fixed by adding `FusedMoE.make_expert_params_mapping()` for per-expert format.
**Remaining issue:** Expert weights are RTN-quantized (not GPTQ-calibrated) — causes repetition/artifacts in long outputs. Needs either calibrated GPTQ expert weights or official Google INT4 release.
**Weights:** `~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ` (16GB, group_size=32)
**BF16 base:** `~/AI/models/gemma-4-26B-A4B-it-BF16` (49GB, too large for 2x30GB)

### Coder-Next 80B — needs GPTQ calibration
**Status:** Infrastructure works, community AWQ weights produce garbage.
**Next step:** Run GPTQ on BF16 base (160GB, needs disk offloading, ~24h).
**BF16 base:** `~/AI/models/Qwen3-Coder-Next-BF16`

### FP8 MoE on SGLang — Arch Linux comgr bug
**Status:** Blocked. Arch `comgr` generates invalid HSACO for FP8 WMMA on gfx1201.
**Workaround:** vLLM Docker (1,185 tok/s peak for Coder-30B FP8).

### AWQ weight generation — experts not GPTQ-calibrated
GPTQ only calibrates `nn.Linear` modules. Fused expert tensors (`Qwen3MoeExperts`) are `nn.Parameter`, so they get RTN-quantized during conversion instead. Works for Coder-30B but may explain Gemma 4 quality issues.

