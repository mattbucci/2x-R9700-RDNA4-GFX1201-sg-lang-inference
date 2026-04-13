#!/usr/bin/env python3
"""GPTQ calibration for Gemma 4 31B Dense using llm-compressor.

Produces compressed-tensors format (weight_packed + weight_scale).
W4A16 scheme: 4-bit weights, 16-bit activations, group_size=128.
All dims divisible by 128: hidden=5376 (42 groups), intermediate=21504 (168 groups).

Requires separate conda env — llmcompressor conflicts with SGLang deps.
Runs CPU-only (~65GB RAM, ~4-6 hours).

Usage:
    conda activate quant
    CUDA_VISIBLE_DEVICES="" python scripts/quantize/quantize_gemma4_31b_llmcompressor.py
"""
import os
import sys
import time

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import load_dataset

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
MODEL_PATH = os.path.join(MODELS_DIR, "gemma-4-31B-it-BF16")
OUTPUT_DIR = os.path.join(MODELS_DIR, "gemma-4-31B-it-CT-GPTQ-128g")

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

if not os.path.isdir(MODEL_PATH):
    # Try HuggingFace
    MODEL_PATH = "google/gemma-4-31B-it"
    print(f"Local model not found, using HuggingFace: {MODEL_PATH}")

ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
print(f"Model:   {MODEL_PATH}")
print(f"Output:  {OUTPUT_DIR}")
print(f"RAM:     {ram_gb:.1f} GB")
print(f"Samples: {NUM_CALIBRATION_SAMPLES}")
print(f"Seq len: {MAX_SEQUENCE_LENGTH}")

if ram_gb < 60:
    print(f"WARNING: {ram_gb:.0f}GB RAM may be tight. Need ~65GB for BF16 model + Hessians.")

# Load model on CPU
print("\nLoading model on CPU...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="cpu", torch_dtype="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.0f}s")
print(f"Model type: {type(model).__name__}")

# Calibration data
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

# GPTQ recipe — standard W4A16 (group_size=128)
# All dims are divisible by 128:
#   hidden_size=5376 (42 groups), head_dim*num_heads=8192 (64 groups),
#   intermediate_size=21504 (168 groups)
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*vision_tower.*",    # Vision encoder (4304 cols not div by 128, text-only anyway)
        "re:.*embed_vision.*",    # Vision embedding projection
    ],
    offload_hessians=True,
)

print(f"\nRunning GPTQ calibration (this will take 4-6 hours on CPU)...")
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
print(f"Done! Compressed-tensors model saved to {OUTPUT_DIR}")
print(f"Next: python scripts/quantize/convert_gemma4_31b_ct_to_awq.py")
