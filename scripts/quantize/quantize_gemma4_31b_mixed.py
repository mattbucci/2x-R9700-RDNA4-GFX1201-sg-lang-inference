#!/usr/bin/env python3
"""Mixed-precision GPTQ for Gemma 4 31B Dense: edge + global attention layers in BF16.

INT4 quantization noise compounds through 60 layers, causing quality degradation
after ~60-80 tokens. Research (APEX) shows keeping edge layers (first/last 8) and
global attention layers in higher precision eliminates most of the compounding error.

Gemma 31B layer pattern (60 layers):
  - sliding_attention: layers 0-4,6-10,12-16,18-22,24-28,30-34,36-40,42-46,48-52,54-58
  - full_attention: layers 5,11,17,23,29,35,41,47,53,59

Kept in BF16 (23 layers): 0-7, 52-59, plus global attention 11,17,23,29,35,41,47
Quantized to INT4 (37 layers): 8-10, 12-16, 18-22, 24-28, 30-34, 36-40, 42-46, 48-51

Requires separate conda env — llmcompressor conflicts with SGLang.
Runs CPU-only (~65GB RAM, ~3-4 hours — fewer layers to calibrate).

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_gemma4_31b_mixed.py
"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
MODEL_PATH = os.path.join(MODELS_DIR, "gemma-4-31B-it-BF16")
OUTPUT_DIR = os.path.join(MODELS_DIR, "gemma-4-31B-it-CT-GPTQ-mixed")

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

if not os.path.isdir(MODEL_PATH):
    MODEL_PATH = "google/gemma-4-31B-it"

# Edge layers (first 8, last 8) + global attention layers in the middle
EDGE_LAYERS = list(range(0, 8)) + list(range(52, 60))
GLOBAL_ATTN_LAYERS = [11, 17, 23, 29, 35, 41, 47]  # full_attention in middle
BF16_LAYERS = sorted(set(EDGE_LAYERS + GLOBAL_ATTN_LAYERS))
INT4_LAYERS = [i for i in range(60) if i not in BF16_LAYERS]

# Build ignore patterns for BF16 layers
ignore_patterns = [
    "lm_head",
    "re:.*vision_tower.*",
    "re:.*embed_vision.*",
]
for layer_idx in BF16_LAYERS:
    ignore_patterns.append(f"re:.*layers\\.{layer_idx}\\..*")

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:      {MODEL_PATH}")
print(f"Output:     {OUTPUT_DIR}")
print(f"RAM:        {ram_gb:.1f} GB")
print(f"BF16 layers ({len(BF16_LAYERS)}): {BF16_LAYERS}")
print(f"INT4 layers ({len(INT4_LAYERS)}): {INT4_LAYERS}")
print()

print("Loading model on CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cpu", torch_dtype="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.0f}s")

print(f"\nLoading calibration data ({NUM_CALIBRATION_SAMPLES} samples)...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=ignore_patterns,
    offload_hessians=True,
)

print(f"\nRunning GPTQ calibration (37 INT4 layers, 23 BF16 layers)...")
print(f"Ignore patterns: {len(ignore_patterns)} entries")
t0 = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=tokenizer,
)
elapsed = time.time() - t0
print(f"\nGPTQ completed in {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nSaving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Mixed-precision model saved to {OUTPUT_DIR}")
print(f"Next: python scripts/quantize/convert_gemma4_31b_ct_to_awq.py")
print(f"  CT_INPUT={OUTPUT_DIR} AWQ_OUTPUT={MODELS_DIR}/gemma-4-31B-it-AWQ-GPTQ-mixed")
