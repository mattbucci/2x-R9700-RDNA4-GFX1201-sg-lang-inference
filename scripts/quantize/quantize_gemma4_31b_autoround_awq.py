#!/usr/bin/env python3
"""Quantize Gemma 4 31B with AutoRound → AWQ format.

Splits model across both GPUs with max_memory constraints to leave
headroom for calibration activations. Does NOT use low_gpu_mem_usage
(which tries to offload 59 GB to CPU, causing swap thrashing).

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    python scripts/quantize/quantize_gemma4_31b_autoround_awq.py
"""
import os
import torch

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
INPUT_MODEL = os.environ.get("INPUT_MODEL", f"{MODELS_DIR}/gemma-4-31B-it-BF16")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODELS_DIR}/gemma-4-31B-it-AutoRound-AWQ-native")

print(f"Input:  {INPUT_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print(f"torch:  {torch.__version__}, hip={torch.version.hip}")
print(f"GPUs:   {torch.cuda.device_count()}")

from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained(INPUT_MODEL, trust_remote_code=True)
tokenizer = processor.tokenizer

# Force split across BOTH GPUs + CPU overflow
# Model is 59 GB BF16, GPUs have 32+34=66 GB but need headroom
# GPU 0: 26 GB, GPU 1: 28 GB = 54 GB on GPU, ~5 GB on CPU
max_memory = {0: "26GiB", 1: "28GiB", "cpu": "30GiB"}

print(f"Loading model with max_memory={max_memory}...")
model = AutoModelForCausalLM.from_pretrained(
    INPUT_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    trust_remote_code=True,
)

for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  GPU {i}: {(total-free)/1e9:.1f} GB used, {free/1e9:.1f} GB free")

from auto_round import AutoRound

# Do NOT use low_gpu_mem_usage — it tries to move 59 GB to CPU, causing thrashing
# Instead, keep model on GPUs and calibrate in-place
autoround = AutoRound(
    model=model,
    tokenizer=tokenizer,
    processor=processor,
    bits=4,
    group_size=128,
    sym=False,
    iters=200,
    seqlen=2048,
    nsamples=128,
    batch_size=1,
    gradient_accumulate_steps=4,
)

print("Starting calibration (200 iterations × 60 layers)...")
autoround.quantize()

print(f"\nSaving to {OUTPUT_DIR} in auto_awq format...")
autoround.save_quantized(OUTPUT_DIR, format="auto_awq")

print(f"\nDone! Output: {OUTPUT_DIR}")
