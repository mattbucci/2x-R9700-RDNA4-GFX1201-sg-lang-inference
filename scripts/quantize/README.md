# Quantization Scripts

All models use **AWQ 4-bit** format on SGLang. The pipeline is:

```
BF16 model → GPTQ calibration (llmcompressor) → compressed-tensors → CT→AWQ conversion → native AWQ
```

## Pipelines (shell wrappers)

Each shell script runs the full pipeline for one model: calibration, conversion, and post-processing.

| Script | Model | Notes |
|--------|-------|-------|
| `quantize_devstral_llmcompressor.sh` | Devstral-24B | Dense, skips vision tower |
| `quantize_qwen35_llmcompressor.sh` | Qwen3.5-27B | Skips DeltaNet layers (BF16) |
| `quantize_gemma4_gptq.sh` | Gemma 4 26B MoE | Forced-routing calibration for 128 experts |
| `quantize_gemma4_31b_gptq.sh` | Gemma 4 31B Dense | Standard GPTQ, no monkey-patching |

## Calibration (Python)

| Script | Purpose |
|--------|---------|
| `quantize_devstral_llmcompressor.py` | Devstral llmcompressor config |
| `quantize_qwen35_llmcompressor.py` | Qwen3.5 llmcompressor config |
| `quantize_gemma4_gptq.py` | Gemma 4 MoE GPTQ with expert unfusing |
| `quantize_gemma4_gptq_step1.py` | Gemma 4 step 1: GPTQ calibration |
| `quantize_moe_llmcompressor.py` | Generic MoE llmcompressor config |

## CT → AWQ Conversion

Each converter handles model-specific weight naming and layout.

| Script | Model | Special handling |
|--------|-------|-----------------|
| `convert_devstral_ct_to_awq.py` | Devstral | Vision tower + multi-modal projector (FP16) |
| `convert_qwen35_ct_to_awq.py` | Qwen3.5 | DeltaNet/SSM layers kept BF16 |
| `convert_gemma4_ct_to_awq.py` | Gemma 4 | Expert naming regex, router dequant |
| `convert_moe_ct_to_awq.py` | Generic MoE | CLI args: `src_dir`, `dst_dir`, `--group-size` |

## Post-Processing

| Script | Purpose |
|--------|---------|
| `fix_gemma4_awq_checkpoint.py` | Fix expert naming, dequant router to BF16, clamp scales |
| `create_gemma4_hybrid_awq.py` | Create hybrid BF16+AWQ checkpoint |
