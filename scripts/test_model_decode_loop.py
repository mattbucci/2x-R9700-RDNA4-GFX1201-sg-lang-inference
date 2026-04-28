#!/usr/bin/env python3
"""Test Gemma4-31B decode loop with triton attention — find exact crash point.

Loads the actual model and runs autoregressive decode step by step,
checking for crashes/NaN at each step. Runs on single GPU.

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    CUDA_VISIBLE_DEVICES=0 python scripts/test_model_decode_loop.py
"""
import os, sys, time, gc
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "components", "sglang", "python"))

MODEL_PATH = os.environ.get("MODEL", os.path.expanduser("~/AI/models/gemma-4-31B-it-AutoRound-AWQ"))

# Use TP=1 for simpler debugging
device = "cuda:0"

print(f"Loading model from {MODEL_PATH} on {device}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(device)}")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

processor = AutoProcessor.from_pretrained(
    os.path.expanduser("~/AI/models/gemma-4-31B-it-BF16"),
    trust_remote_code=True,
)
tokenizer = processor.tokenizer

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True,
    attn_implementation="eager",  # Use PyTorch attention, not flash
)
print(f"Model loaded. Memory: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

# Test prompt
prompt = "Write a detailed essay on the history of artificial intelligence."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)
input_ids = inputs["input_ids"]

print(f"Input tokens: {input_ids.shape[1]}")
print(f"Generating 800 tokens with greedy decoding...")

# Generate with step-by-step monitoring
with torch.no_grad():
    generated = input_ids
    for step in range(800):
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]

        # Check for NaN/Inf in logits
        has_nan = torch.isnan(next_token_logits).any().item()
        has_inf = torch.isinf(next_token_logits).any().item()

        if has_nan or has_inf:
            print(f"\nSTEP {step}: {'NaN' if has_nan else 'Inf'} in logits!")
            print(f"  Sequence length: {generated.shape[1]}")
            print(f"  Logits range: [{next_token_logits.min().item():.2f}, {next_token_logits.max().item():.2f}]")
            break

        # Greedy
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        # Check for repetition
        if step > 0 and step % 50 == 0:
            decoded = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
            last100 = decoded[-200:] if len(decoded) > 200 else decoded
            print(f"  Step {step}: len={generated.shape[1]}, last chars: {last100[-80:]!r}")

        # Stop on EOS
        if next_token.item() == tokenizer.eos_token_id:
            print(f"\nEOS at step {step}")
            break

    # Final output
    output_text = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nGenerated {step+1} tokens ({len(output_text.split())} words)")
    print(f"=== FIRST 300 chars ===")
    print(output_text[:300])
    print(f"\n=== LAST 300 chars ===")
    print(output_text[-300:])

    # Check for repetition
    if "l l l" in output_text or output_text.count(" l ") > 10:
        pos = output_text.find("l l l") if "l l l" in output_text else len(output_text)
        words = len(output_text[:pos].split())
        print(f"\n*** DEGRADATION detected at ~{words} words ***")
    else:
        print(f"\n*** CLEAN OUTPUT — no degradation detected ***")
