# MoE Expert Compression: REAM & REAP

Two methods to shrink MoE models by reducing expert count. Both run on CPU with ~60GB RAM.

| Method | What it does | Quality | Code |
|--------|-------------|---------|------|
| **REAM** | Merges expert groups | >=94% retained | [SamsungSAILMontreal/ream](https://github.com/SamsungSAILMontreal/ream) |
| **REAP** | Prunes low-impact experts | Better on generative tasks | [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) |

## Why compress?

With 64GB VRAM (2x R9700), MoE models fit comfortably at AWQ 4-bit. Compression helps by:
- Reducing weight reads per token (faster decode for bandwidth-bound models)
- Freeing VRAM for larger KV cache (longer context or more concurrent users)
- Making DeltaNet hybrid models practical (DeltaNet layers stay BF16, so MoE expert savings matter more)

## Candidate models

### Primary target: Qwen3.5-35B-A3B (DeltaNet hybrid MoE)

35B total / 3B active, 256 experts, 40 layers (30 DeltaNet + 10 full attention).
Same hybrid architecture as Qwen3.5-27B but with MoE FFN blocks instead of dense MLP.

| Variant | Experts | ~Params | AWQ est. | Notes |
|---------|:-------:|:-------:|:--------:|-------|
| Full | 256 | 35B | ~18 GB | Fits 2x R9700, plenty of KV cache room |
| REAM 75% | 192 | ~27B | ~14 GB | Sweet spot: minimal quality loss |
| REAP 50% | 128 | ~20B | ~10 GB | Aggressive but may be OK for coding |

DeltaNet layers MUST stay BF16 — INT4 quantization destroys recurrent state quality (same as Qwen3.5-27B and Coder-Next).

### Pre-made REAP variants on HuggingFace

| Model | Experts | Pruned | Source |
|-------|:-------:|:------:|--------|
| [0xSero/Qwen-3.5-28B-A3B-REAP](https://huggingface.co/0xSero/Qwen-3.5-28B-A3B-REAP) | 205/256 | 20% | BF16, MMLU 80.9 (-3.4pp) |
| [atbender/Qwen3.5-REAP-20B-A3B](https://huggingface.co/atbender/Qwen3.5-REAP-20B-A3B) | 128/256 | 50% | BF16 + W4A16 AutoRound |

No AWQ versions exist — must self-calibrate with DeltaNet-aware pipeline.

### Other MoE models (pure MoE, no DeltaNet)

| Model | Params | Experts | After | AWQ est. | Method | Status |
|-------|--------|:-------:|:-----:|:--------:|--------|--------|
| Qwen3-Coder-30B | 30B | 128 | 96 (23B) | ~12 GB | REAM/REAP | Already working at 128 experts |
| Gemma 4 26B MoE | 26B | 128 | 103 (21B) | ~10 GB | REAP | Already working at 128 experts |
| Qwen3-Coder-REAP-25B | 25B | 103 | — | ~13 GB | REAP | [Pre-made](https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B) |

## REAM Setup (SamsungSAIL)

REAM merges experts instead of dropping them. Currently supports **Qwen3** and **GLM** only.
Qwen3.5 MoE may need model support added (similar architecture to Qwen3 MoE).

```bash
git clone https://github.com/SamsungSAILMontreal/ream.git
cd ream
conda create -n ream python=3.12 -y
conda activate ream
pip install -r requirements.txt
```

### REAM Qwen3.5-35B-A3B (256 -> 192 experts)

```bash
# Code-heavy calibration mix
CUDA_VISIBLE_DEVICES="" python merge.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --merge_size 192 \
    --saliency reap \
    --merging logits+weights \
    --grouping ream \
    --dataset "c4+math+code" \
    --mix_ratio "0.0,0.3,0.7" \
    --save_path ~/AI/models/Qwen3.5-35B-A3B-REAM-BF16
```

**Note:** If REAM doesn't support `qwen3_5_moe` yet, you may need to add it:
1. Check REAP's `src/reap/model_util.py` for the model's `MODEL_ATTRS` entry
2. Add the same tokenizer/model detection to REAM's `merge.py`
3. Port the MoE layer access patterns to REAM's `merger.py`

## REAP Setup (Cerebras)

REAP supports more model families out of the box. Wider architecture coverage.

```bash
git clone https://github.com/CerebrasResearch/reap.git
cd reap
bash scripts/build.sh
```

### REAP Qwen3.5-35B-A3B (256 -> 192 experts)

```bash
bash experiments/pruning-cli.sh 0 \
    Qwen/Qwen3.5-35B-A3B \
    reap 42 0.25 \
    "theblackcat102/evol-codealpaca-v1:4096,open-r1/Mixture-of-Thoughts[code]:4096,open-r1/Mixture-of-Thoughts[math]:4096" \
    true true false false false
```

## After compression: quantize to AWQ

The compressed BF16 model goes through the standard DeltaNet-aware pipeline:

```bash
source scripts/common.sh && activate_conda && setup_rdna4_env

# Full pipeline (GPTQ calibration + CT->AWQ conversion)
./scripts/quantize/quantize_qwen35_moe_ream.sh

# Or step by step:
# Step 1: GPTQ calibration (CPU, ~6-8h, skips DeltaNet layers)
CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_qwen35_moe_ream.py

# Step 2: CT -> AWQ conversion (preserves DeltaNet layers in BF16)
python scripts/quantize/convert_moe_ct_to_awq.py \
    ~/AI/models/Qwen3.5-35B-A3B-REAM-AWQ-CT \
    ~/AI/models/Qwen3.5-35B-A3B-REAM-AWQ
```

**DeltaNet handling:** The calibration script excludes `in_proj_a`, `in_proj_b` (DeltaNet gates, dim 48) from INT4 quantization. The CT->AWQ converter preserves all non-quantized layers (including DeltaNet recurrent weights) in their original dtype.

## Requirements

- **RAM**: ~70GB (BF16 model loaded on CPU + GPTQ Hessians)
- **GPU**: Optional — speeds up saliency computation but not required
- **Disk**: ~140GB (source BF16 + REAM BF16 + CT + AWQ outputs)
- **Time**: Several hours for REAM/REAP merging, then ~6-8 hours for GPTQ calibration
