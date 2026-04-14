#!/usr/bin/env python3
"""Isolate GLM-4.5-Air MoE AWQ crash on RDNA4.

Tests the HIP GEMV MoE kernel and Triton dequant+matmul fallback
with GLM's exact dimensions (96 experts, moe_intermediate=1408, group_size=32).

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    python scripts/test_glm_moe_isolation.py
"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "components", "sglang", "python"))

device = "cuda:0"
print(f"Device: {torch.cuda.get_device_name(device)}")

# GLM-4.5-Air dims (per TP=2 shard)
HIDDEN = 4096 // 2     # 2048 per shard
MOE_INTER = 1408 // 2  # 704 per shard
NUM_EXPERTS = 96
TOP_K = 8
GROUP_SIZE = 32
PACK_FACTOR = 8

print(f"GLM dims: hidden={HIDDEN}, moe_inter={MOE_INTER}, experts={NUM_EXPERTS}, topk={TOP_K}, gs={GROUP_SIZE}")

# ========== Test 1: Dense AWQ GEMV with GLM dims ==========
print("\n=== Test 1: Triton AWQ GEMV (dense, GLM dims) ===")
from sglang.srt.layers.quantization.awq_triton import awq_gemv_triton, awq_dequantize_decomposition

for K, N in [(HIDDEN, MOE_INTER * 2), (MOE_INTER, HIDDEN)]:
    num_groups = K // GROUP_SIZE
    qweight = torch.randint(0, 2**31 - 1, (K, N // PACK_FACTOR), dtype=torch.int32, device=device)
    scales = torch.randn(num_groups, N, dtype=torch.float16, device=device) * 0.01
    qzeros = torch.randint(0, 2**31 - 1, (num_groups, N // PACK_FACTOR), dtype=torch.int32, device=device)
    x = torch.randn(1, K, dtype=torch.float16, device=device) * 0.1

    try:
        out = awq_gemv_triton(x, qweight, scales, qzeros)
        torch.cuda.synchronize()
        has_nan = torch.isnan(out).any().item()
        print(f"  K={K}, N={N}: {'NaN!' if has_nan else 'PASS'}")
    except Exception as e:
        print(f"  K={K}, N={N}: CRASH: {e}")

# ========== Test 2: Unfused dequant + matmul (per-expert) ==========
print("\n=== Test 2: Unfused dequant+matmul (per-expert loop) ===")

K, N = HIDDEN, MOE_INTER * 2
num_groups = K // GROUP_SIZE
qweight = torch.randint(0, 2**31 - 1, (K, N // PACK_FACTOR), dtype=torch.int32, device=device)
scales = torch.randn(num_groups, N, dtype=torch.float16, device=device) * 0.01
qzeros = torch.randint(0, 2**31 - 1, (num_groups, N // PACK_FACTOR), dtype=torch.int32, device=device)

for num_tokens in [1, 4, 8]:
    x = torch.randn(num_tokens, K, dtype=torch.float16, device=device) * 0.1
    try:
        w = awq_dequantize_decomposition(qweight, scales, qzeros)
        out = torch.matmul(x, w)
        torch.cuda.synchronize()
        has_nan = torch.isnan(out).any().item()
        print(f"  M={num_tokens}: {'NaN!' if has_nan else 'PASS'}")
    except Exception as e:
        print(f"  M={num_tokens}: CRASH: {e}")

# ========== Test 3: HIP GEMV MoE kernel ==========
print("\n=== Test 3: HIP GEMV MoE kernel ===")
try:
    import awq_gemv_hip_ext
    has_hip = True
    print("  HIP GEMV extension loaded")
except ImportError:
    has_hip = False
    print("  HIP GEMV extension NOT available — skipping")

if has_hip:
    # Simulate MoE dispatch: 1 token, top_k=8 experts
    M = 1
    num_slots = M * TOP_K
    K, N = HIDDEN, MOE_INTER * 2  # gate+up projection

    # Create per-expert weights [E, K, N//8]
    w13_qweight = torch.randint(0, 2**31 - 1, (NUM_EXPERTS, K, N // PACK_FACTOR), dtype=torch.int32, device=device)
    num_groups = K // GROUP_SIZE
    w13_scales = torch.randn(NUM_EXPERTS, num_groups, N, dtype=torch.float16, device=device) * 0.01
    w13_qzeros = torch.randint(0, 2**31 - 1, (NUM_EXPERTS, num_groups, N // PACK_FACTOR), dtype=torch.int32, device=device)

    x = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
    sorted_ids = torch.arange(num_slots, dtype=torch.int32, device=device)
    expert_ids = torch.randint(0, NUM_EXPERTS, (num_slots,), dtype=torch.int32, device=device)
    weights = torch.ones(num_slots, dtype=torch.float32, device=device) / TOP_K

    gate_up = torch.empty(num_slots, N, dtype=torch.float16, device=device)

    try:
        awq_gemv_hip_ext.awq_gemv_moe_hip(
            x, w13_qweight, w13_scales, w13_qzeros,
            gate_up, sorted_ids, expert_ids, weights,
            TOP_K, False, 0)
        torch.cuda.synchronize()
        has_nan = torch.isnan(gate_up).any().item()
        print(f"  gate+up (K={K}, N={N}): {'NaN!' if has_nan else 'PASS'}")
    except Exception as e:
        print(f"  gate+up (K={K}, N={N}): CRASH: {e}")

    # Down projection
    K2, N2 = MOE_INTER, HIDDEN
    w2_qweight = torch.randint(0, 2**31 - 1, (NUM_EXPERTS, K2, N2 // PACK_FACTOR), dtype=torch.int32, device=device)
    num_groups2 = K2 // GROUP_SIZE
    w2_scales = torch.randn(NUM_EXPERTS, num_groups2, N2, dtype=torch.float16, device=device) * 0.01
    w2_qzeros = torch.randint(0, 2**31 - 1, (NUM_EXPERTS, num_groups2, N2 // PACK_FACTOR), dtype=torch.int32, device=device)

    activated = torch.randn(num_slots, K2, dtype=torch.float16, device=device) * 0.1
    down_out = torch.zeros(num_slots, N2, dtype=torch.float16, device=device)

    try:
        awq_gemv_hip_ext.awq_gemv_moe_hip(
            activated, w2_qweight, w2_scales, w2_qzeros,
            down_out, sorted_ids, expert_ids, weights,
            1, True, 0)
        torch.cuda.synchronize()
        has_nan = torch.isnan(down_out).any().item()
        print(f"  down (K={K2}, N={N2}): {'NaN!' if has_nan else 'PASS'}")
    except Exception as e:
        print(f"  down (K={K2}, N={N2}): CRASH: {e}")

# ========== Test 4: Full MoE forward with real weights ==========
print("\n=== Test 4: Load real GLM weights, single expert test ===")
try:
    from safetensors import safe_open
    import json, glob

    model_dir = os.path.expanduser("~/AI/models/GLM-4.5-Air-REAP-AWQ")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Find a MoE layer's weights
    moe_keys = [k for k in weight_map if "mlp.experts" in k and "w13_qweight" in k]
    if moe_keys:
        key = moe_keys[0]
        layer_prefix = key.rsplit(".", 1)[0]
        shard = weight_map[key]
        print(f"  Loading {layer_prefix} from {shard}")

        sf = safe_open(os.path.join(model_dir, shard), framework="pt")
        qw = sf.get_tensor(key).to(device)  # [E, K, N//8]
        print(f"  qweight shape: {qw.shape}")

        sc_key = layer_prefix.replace("qweight", "scales")
        if sc_key in weight_map:
            sc_shard = weight_map[sc_key]
            sf2 = safe_open(os.path.join(model_dir, sc_shard), framework="pt")
            sc = sf2.get_tensor(sc_key).to(device)
            print(f"  scales shape: {sc.shape}")

            # Test single expert dequant
            expert_0_qw = qw[0]  # [K, N//8]
            expert_0_sc = sc[0]  # [groups, N]
            print(f"  Expert 0: qweight={expert_0_qw.shape}, scales={expert_0_sc.shape}")

            # Check alignment
            K_dim, N_packed = expert_0_qw.shape
            N_dim = N_packed * 8
            groups = expert_0_sc.shape[0]
            print(f"  K={K_dim}, N={N_dim}, groups={groups}, gs={K_dim//groups}")
            print(f"  N % 8 = {N_dim % 8}, K % gs = {K_dim % (K_dim//groups)}")
    else:
        print("  No MoE expert weights found")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone.")
