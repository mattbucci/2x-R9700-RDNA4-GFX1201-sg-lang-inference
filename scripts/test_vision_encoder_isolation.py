#!/usr/bin/env python3
"""Test Gemma4 vision encoder SDPA in isolation on RDNA4.

Progressively tests each component to find the crash point:
1. Raw SDPA with vision encoder dims
2. Vision encoder attention with RoPE
3. Full vision encoder forward

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    CUDA_VISIBLE_DEVICES=0 python scripts/test_vision_encoder_isolation.py
"""
import torch
import torch.nn.functional as F

device = "cuda:0"
dtype = torch.float16
print(f"Device: {torch.cuda.get_device_name(device)}")

# Gemma4 vision encoder dims
hidden_size = 1152
num_heads = 16
num_kv_heads = 16
head_dim = 72  # 1152 / 16
num_layers = 27
patch_size = 16

# For a 64x64 image: (64/16)^2 = 16 patches
# For a 256x256 image: (256/16)^2 = 256 patches
seq_lengths = [16, 64, 256, 1024]

print(f"\nVision encoder: {hidden_size}d, {num_heads}h, head_dim={head_dim}, {num_layers} layers")

# Test 1: Raw SDPA with vision dims
print("\n=== Test 1: Raw F.scaled_dot_product_attention ===")
for seq_len in seq_lengths:
    q = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    try:
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        torch.cuda.synchronize()
        has_nan = torch.isnan(out).any().item()
        print(f"  seq_len={seq_len:4d}: {'NaN!' if has_nan else 'PASS'} shape={out.shape}")
    except Exception as e:
        print(f"  seq_len={seq_len:4d}: CRASH: {e}")

# Test 2: SDPA with bidirectional attention mask (vision uses bidirectional, not causal)
print("\n=== Test 2: SDPA with bidirectional attention mask ===")
for seq_len in seq_lengths:
    q = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    # Bidirectional = no mask (all attend to all)
    try:
        out = F.scaled_dot_product_attention(q, k, v, scale=1.0, is_causal=False)
        torch.cuda.synchronize()
        has_nan = torch.isnan(out).any().item()
        print(f"  seq_len={seq_len:4d}: {'NaN!' if has_nan else 'PASS'}")
    except Exception as e:
        print(f"  seq_len={seq_len:4d}: CRASH: {e}")

# Test 3: SDPA with flatten_batch pattern (how vision backend uses it)
print("\n=== Test 3: SDPA with flatten_batch pattern ===")
for seq_len in seq_lengths:
    # The vision backend uses (bsz*seq_len, num_heads, head_dim) → reshape for SDPA
    bsz = 1
    q = torch.randn(bsz * seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(bsz * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(bsz * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Reshape for SDPA: (bsz, num_heads, seq_len, head_dim)
    q_4d = q.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k_4d = k.reshape(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v_4d = v.reshape(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    try:
        out = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=1.0, is_causal=False)
        torch.cuda.synchronize()
        has_nan = torch.isnan(out).any().item()
        print(f"  seq_len={seq_len:4d}: {'NaN!' if has_nan else 'PASS'}")
    except Exception as e:
        print(f"  seq_len={seq_len:4d}: CRASH: {e}")

# Test 4: SDPA with attention mask (4D mask like vision uses)
print("\n=== Test 4: SDPA with 4D boolean attention mask ===")
for seq_len in [16, 64, 256]:
    q = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    # Vision uses [1, 1, seq_len, seq_len] boolean mask
    attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=device)
    try:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        torch.cuda.synchronize()
        has_nan = torch.isnan(out).any().item()
        print(f"  seq_len={seq_len:4d}: {'NaN!' if has_nan else 'PASS'}")
    except Exception as e:
        print(f"  seq_len={seq_len:4d}: CRASH: {e}")

# Test 5: Import and test QKV_BACKEND_IMPL sdpa directly
print("\n=== Test 5: QKV_BACKEND_IMPL['sdpa'] ===")
try:
    import sys
    sys.path.insert(0, 'components/sglang/python')
    from sglang.srt.layers.attention.vision import QKV_BACKEND_IMPL
    backend = QKV_BACKEND_IMPL["sdpa"](
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        flatten_batch=True,
        softmax_in_single_precision=False,
        softmax_scale=1.0,
    )
    for seq_len in [16, 64, 256]:
        bsz = 1
        q = torch.randn(bsz * seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(bsz * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(bsz * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        try:
            out = backend.forward(q=q, k=k, v=v, cu_seqlens=None, bsz=bsz, seq_len=seq_len)
            torch.cuda.synchronize()
            has_nan = torch.isnan(out).any().item()
            print(f"  seq_len={seq_len:4d}: {'NaN!' if has_nan else 'PASS'} shape={out.shape}")
        except Exception as e:
            print(f"  seq_len={seq_len:4d}: CRASH: {type(e).__name__}: {e}")
except Exception as e:
    print(f"  Import failed: {e}")

# Test 6: Load actual vision weights and forward
print("\n=== Test 6: Actual vision encoder forward ===")
try:
    from safetensors import safe_open
    import os, json

    model_dir = os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed")
    vision_path = os.path.join(model_dir, "model-vision.safetensors")
    if os.path.exists(vision_path):
        sf = safe_open(vision_path, framework="pt")
        # Count vision weights
        vkeys = [k for k in sf.keys() if "vision_tower" in k]
        print(f"  Vision weights: {len(vkeys)}")
        # Check a few weight shapes
        for k in vkeys[:3]:
            t = sf.get_tensor(k)
            print(f"    {k}: {list(t.shape)} {t.dtype}")
    else:
        print(f"  No vision shard at {vision_path}")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone.")
