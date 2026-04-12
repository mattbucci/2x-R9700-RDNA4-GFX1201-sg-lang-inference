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

See [README.md](../README.md) for working models, benchmarks, and MoE quantization details.
