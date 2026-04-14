#!/usr/bin/env python3
"""Systematic isolation test for Triton decode attention on RDNA4 + Gemma4-31B.

Tests the decode attention kernel at various KV lengths to find the crash point.
Compares Triton kernel output against PyTorch reference.

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    python scripts/test_triton_attention_isolation.py
"""
import os
import sys
import torch
import time

# Add sglang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "components", "sglang", "python"))

from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd

def reference_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, scaling):
    """Pure PyTorch reference decode attention."""
    bs, num_heads, head_dim = q.shape
    v_dim = v_buffer.shape[-1]
    output = torch.zeros(bs, num_heads, v_dim, dtype=q.dtype, device=q.device)

    for b in range(bs):
        start = kv_indptr[b].item()
        end = kv_indptr[b + 1].item()
        indices = kv_indices[start:end]

        k = k_buffer[indices]  # [seq_len, num_kv_heads, head_dim]
        v = v_buffer[indices]  # [seq_len, num_kv_heads, v_dim]

        for h in range(num_heads):
            kv_h = h % k.shape[1]  # GQA mapping
            scores = torch.matmul(
                q[b, h].float().unsqueeze(0),
                k[:, kv_h].float().t()
            ) * scaling
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v[:, kv_h].float())
            output[b, h] = out.squeeze(0).to(q.dtype)

    return output


def test_decode_attention(kv_len, device="cuda:0"):
    """Test decode attention at a specific KV length."""
    # Gemma4-31B dimensions (per TP=2 shard)
    bs = 1
    num_q_heads = 16   # 32 / TP=2
    num_kv_heads = 8   # 16 / TP=2
    qk_head_dim = 256
    v_head_dim = 256    # same as qk for Gemma4
    max_kv_splits = 16
    scaling = 1.0 / (qk_head_dim ** 0.5)

    # Create random inputs
    q = torch.randn(bs, num_q_heads, qk_head_dim, dtype=torch.bfloat16, device=device)

    # KV cache: flat buffer of shape [max_tokens, num_kv_heads, dim]
    k_buffer = torch.randn(kv_len + 100, num_kv_heads, qk_head_dim, dtype=torch.bfloat16, device=device) * 0.1
    v_buffer = torch.randn(kv_len + 100, num_kv_heads, v_head_dim, dtype=torch.bfloat16, device=device) * 0.1

    # KV indices (simple 0..kv_len mapping)
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    kv_indices = torch.arange(kv_len, dtype=torch.int64, device=device)

    # Output buffer
    o = torch.empty(bs, num_q_heads, v_head_dim, dtype=torch.bfloat16, device=device)

    # Attention intermediate buffers
    attn_logits = torch.empty(bs, num_q_heads, max_kv_splits, v_head_dim, dtype=torch.float32, device=device)
    attn_lse = torch.empty(bs, num_q_heads, max_kv_splits, dtype=torch.float32, device=device)

    # Compute num_kv_splits
    num_kv_splits = torch.full((bs,), min(max(1, kv_len // 32), max_kv_splits), dtype=torch.int32, device=device)

    # Reference output
    ref_o = reference_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, scaling)

    # Triton kernel
    torch.cuda.synchronize()
    try:
        decode_attention_fwd(
            q, k_buffer, v_buffer, o,
            kv_indptr, kv_indices,
            attn_logits, attn_lse, num_kv_splits,
            max_kv_splits, scaling,
            1.0,  # k_scale
            1.0,  # v_scale
            logit_cap=0.0,
        )
        torch.cuda.synchronize()

        # Compare
        max_err = (o.float() - ref_o.float()).abs().max().item()
        mean_err = (o.float() - ref_o.float()).abs().mean().item()

        # Check for NaN/Inf
        has_nan = torch.isnan(o).any().item()
        has_inf = torch.isinf(o).any().item()

        status = "PASS" if max_err < 1.0 and not has_nan and not has_inf else "FAIL"
        if has_nan:
            status = "NaN"
        if has_inf:
            status = "Inf"

        return status, max_err, mean_err

    except Exception as e:
        return f"CRASH: {type(e).__name__}", 0, 0


def test_kv_indices_kernel(kv_len, device="cuda:0"):
    """Test the Triton kv_indices kernel in isolation."""
    from sglang.srt.layers.attention.triton_backend import _kv_indices_kernel

    bs = 1
    max_context = kv_len + 100

    req_to_token = torch.arange(max_context, dtype=torch.int64, device=device).unsqueeze(0)  # [1, max_context]
    req_pool_indices = torch.zeros(bs, dtype=torch.int64, device=device)
    page_kernel_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
    kv_indices = torch.empty(kv_len, dtype=torch.int64, device=device)

    try:
        _kv_indices_kernel[(bs,)](
            req_to_token, req_pool_indices, page_kernel_lens,
            kv_indptr,
            req_pool_indices,  # dummy
            kv_indices,
            req_to_token.stride(0),
            HAS_START_IDX=False,
        )
        torch.cuda.synchronize()

        expected = torch.arange(kv_len, dtype=torch.int64, device=device)
        match = (kv_indices == expected).all().item()
        return "PASS" if match else "MISMATCH"

    except Exception as e:
        return f"CRASH: {type(e).__name__}"


def test_awq_gemv(K, N, device="cuda:0"):
    """Test the Triton AWQ GEMV kernel in isolation."""
    from sglang.srt.layers.quantization.awq_triton import awq_gemv_triton

    group_size = 128
    num_groups = K // group_size

    # Create AWQ-format weights
    qweight = torch.randint(0, 2**31 - 1, (K, N // 8), dtype=torch.int32, device=device)
    scales = torch.randn(num_groups, N, dtype=torch.float16, device=device) * 0.01
    qzeros = torch.randint(0, 2**31 - 1, (num_groups, N // 8), dtype=torch.int32, device=device)

    # BF16 input
    x = torch.randn(1, K, dtype=torch.bfloat16, device=device) * 0.1

    try:
        out = awq_gemv_triton(x, qweight, scales, qzeros)
        torch.cuda.synchronize()

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()

        if has_nan:
            return "NaN"
        if has_inf:
            return "Inf"
        return "PASS"

    except Exception as e:
        return f"CRASH: {type(e).__name__}"


if __name__ == "__main__":
    device = "cuda:0"
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Test 1: kv_indices Triton kernel
    print("=" * 60)
    print("TEST 1: Triton kv_indices kernel")
    print("=" * 60)
    for kv_len in [10, 50, 100, 200, 300, 400, 500, 800, 1024, 2048]:
        result = test_kv_indices_kernel(kv_len, device)
        print(f"  kv_len={kv_len:5d}: {result}")
    print()

    # Test 2: AWQ Triton GEMV (Gemma4-31B dimensions per TP=2 shard)
    print("=" * 60)
    print("TEST 2: Triton AWQ GEMV (BF16 input, FP32 dequant)")
    print("=" * 60)
    # Gemma4 layer dims (per TP=2): hidden=2688, intermediate=14336/2=7168
    for K, N in [(2688, 2688), (2688, 7168), (7168, 2688), (2688, 512), (2688, 1024)]:
        result = test_awq_gemv(K, N, device)
        print(f"  K={K:5d}, N={N:5d}: {result}")
    print()

    # Test 3: Triton decode attention at increasing KV lengths
    print("=" * 60)
    print("TEST 3: Triton decode attention (Gemma4-31B dims)")
    print("=" * 60)
    for kv_len in [10, 50, 100, 200, 300, 350, 380, 400, 420, 450, 500, 600, 800, 1024]:
        result, max_err, mean_err = test_decode_attention(kv_len, device)
        if "CRASH" in str(result):
            print(f"  kv_len={kv_len:5d}: {result}")
        else:
            print(f"  kv_len={kv_len:5d}: {result}  max_err={max_err:.6f}  mean_err={mean_err:.6f}")
    print()

    # Test 4: Repeated decode to simulate autoregressive generation
    print("=" * 60)
    print("TEST 4: Simulated autoregressive decode (400 steps)")
    print("=" * 60)
    for target in [100, 200, 300, 400, 500]:
        all_pass = True
        for step in range(target):
            kv_len = step + 1
            result, max_err, mean_err = test_decode_attention(kv_len, device)
            if result != "PASS":
                print(f"  step={step:4d} (kv_len={kv_len}): {result}  max_err={max_err:.6f}")
                all_pass = False
                break
        if all_pass:
            print(f"  {target} steps: ALL PASS")

    print("\nDone.")
