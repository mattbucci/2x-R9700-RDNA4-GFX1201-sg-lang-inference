# Known Issues

## Active

### Qwen3.5-27B AWQ — causal_conv1d shape mismatch at TP=2
**Status:** Server crashes during warmup. `conv_states.shape` has dim=5120 but expected 10240.
**Error:** `AssertionError: causal_conv1d shape mismatch: x.shape=torch.Size([10240, 1]), conv_states.shape=torch.Size([9, 5120, 3])`
**Root cause:** Likely a TP split issue in the Mamba conv1d state — the conv_states are being halved by TP but shouldn't be.
**Previous status:** Was working (129 tok/s @32). May have regressed from a code change.

### Gemma 4 31B Dense — needs GPTQ calibration
**Status:** cyankiwi AWQ loads and runs but RTN quality causes artifacts.
**Fix:** Standard GPTQ calibration (no monkey-patch needed, all nn.Linear). Script ready: `scripts/quantize_gemma4_31b_gptq.sh`.
**Note:** BF16 base `google/gemma-4-31B-it` is NOT gated — can download without HF token. Dense model, standard GPTQ works.

### FP8 MoE on SGLang — Arch Linux comgr bug
**Status:** Blocked. Arch `comgr` generates invalid HSACO for FP8 WMMA on gfx1201.
**Workaround:** vLLM Docker for comparison benchmarks (1,185 tok/s peak for Coder-30B FP8).

## Resolved

### Gemma 4 26B MoE — WORKING (GPTQ v3, forced-routing calibration)
**Status:** WORKING at 25 tok/s single-user, 29 tok/s @ 4 concurrent. Quality verified (knowledge, math, code).
**Root cause:** Standard GPTQ calibration fails for MoE due to inter-expert imbalance — router only sends
calibration data to popular experts. v1 calibrated 1/128 experts, rest got inf scales.
**Fix:** Forced-routing calibration — modified `UnfusedGemma4TextExperts.forward` to route ALL tokens
through ALL 128 experts with uniform weight during calibration. Every expert gets identical Hessian data.
**Additional fixes:** Weight naming regex, router dequant to BF16, activation fn (GELU), FP16 scale clamping.
**Model:** `~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed`

### Coder-Next 80B — working, bandwidth-limited by architecture
**Status:** WORKING at 15 tok/s. Community AWQ weights produce quality code output on SGLang TP=2.
DeltaNet BF16 weight reads (2.4 GB/token, 64% of total) are the architectural speed limit.
**Launch:** `scripts/run_coder_next_awq.sh`

## MoE Quantization Research (reference)

### Inter-expert imbalance (MoEQuant, ICML 2025)
Standard GPTQ/AWQ routes calibration data through the model's router, which activates experts unevenly.
Our fix: forced uniform routing during calibration ensures all experts see all data.
Also available: MoEQuant EBSS, GPTQModel MoE.Routing FailSafe.

### DeltaNet/SSM quantization sensitivity
Recurrent architectures accumulate quantization noise through state updates. DeltaNet layers MUST stay BF16.
