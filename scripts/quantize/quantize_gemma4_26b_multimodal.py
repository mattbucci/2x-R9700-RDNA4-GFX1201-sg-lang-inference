#!/usr/bin/env python3
"""Quantize Gemma 4 26B MoE with multimodal calibration data.

Uses llm-compressor GPTQ with image-text pairs so MoE experts are
calibrated for BOTH text and vision token distributions.

Run in the quant conda env:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_gemma4_26b_multimodal.py
"""
import os
import torch

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
INPUT_MODEL = os.environ.get("INPUT_MODEL", f"{MODELS_DIR}/gemma-4-26B-A4B-it-BF16")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"{MODELS_DIR}/gemma-4-26B-A4B-it-CT-multimodal")
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "256"))

print(f"Input:   {INPUT_MODEL}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Samples: {NUM_SAMPLES}")

from transformers import AutoModelForImageTextToText, AutoProcessor
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(INPUT_MODEL, trust_remote_code=True)

print("Loading model (BF16, CPU)...")
model = AutoModelForImageTextToText.from_pretrained(
    INPUT_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
print(f"Model loaded: {type(model).__name__}")

# GPTQ recipe: exclude vision tower and multimodal projector
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        r"re:model\.vision_tower.*",
        r"re:model\.embed_vision.*",
        r"re:model\.multi_modal_projector.*",
    ],
)

print(f"\nStarting GPTQ calibration with multimodal data...")
print(f"  Dataset: lmms-lab/flickr30k (image-text pairs)")
print(f"  Excluding: vision_tower, embed_vision, multi_modal_projector, lm_head")

import shutil
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Use the same pattern as gemma3_example.py:
# Pass dataset as string, let oneshot handle multimodal loading
oneshot(
    model=model,
    dataset="flickr30k",
    splits={"calibration": f"test[:{NUM_SAMPLES}]"},
    processor=processor,
    recipe=recipe,
    output_dir=OUTPUT_DIR,
    max_seq_length=1024,
    num_calibration_samples=NUM_SAMPLES,
)

print(f"\nDone! Output: {OUTPUT_DIR}")
print(f"Next: convert CT → AWQ format")
