#!/usr/bin/env python3
"""Convert Gemma 4 26B compressed-tensors GPTQ → AWQ format.

CT format: weight_packed (GPTQ sequential int32) + weight_scale (per-group BF16), symmetric
AWQ format: qweight (AWQ interleaved int32) + scales (FP16) + qzeros (int32), asymmetric

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    python scripts/quantize/convert_gemma4_26b_ct_to_awq.py
"""
import gc
import json
import os
import glob
import shutil

import torch
from collections import OrderedDict
from safetensors import safe_open
from safetensors.torch import save_file

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
SRC_DIR = os.environ.get("CT_INPUT", f"{MODELS_DIR}/gemma-4-26B-A4B-it-CT-multimodal")
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT", f"{MODELS_DIR}/gemma-4-26B-A4B-it-AWQ-v3")

W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
GROUP_SIZE = 32  # Must match calibration group_size

print(f"Source:     {SRC_DIR}")
print(f"Output:     {OUTPUT_DIR}")
print(f"Group size: {GROUP_SIZE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def unpack_gptq_sequential(packed: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ sequential int32 → raw int4 values.
    CT weight_packed: [out_features, in_features//8] int32
    Output: [out_features, in_features] int8 (values 0-15)
    """
    rows, cols_packed = packed.shape
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    # [rows, cols_packed, 8] → [rows, cols_packed * 8]
    return torch.stack(unpacked, dim=-1).reshape(rows, cols_packed * PACK_FACTOR).to(torch.int8)


def pack_awq_interleaved(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (0-15) into int32 with AWQ interleaved order.
    Input: [*, N] where N % 8 == 0
    Output: [*, N//8] int32
    """
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


def dequant_gptq_symmetric(packed: torch.Tensor, scales: torch.Tensor, group_size: int):
    """Dequantize GPTQ symmetric to float32."""
    w_int = unpack_gptq_sequential(packed)  # [out, in]
    out_features, in_features = w_int.shape
    num_groups = in_features // group_size

    w_float = torch.zeros(out_features, in_features, dtype=torch.float32)
    for g in range(num_groups):
        s, e = g * group_size, (g + 1) * group_size
        # Symmetric: zp = 8 for 4-bit, dequant = (val - 8) * scale
        w_float[:, s:e] = (w_int[:, s:e].float() - 8.0) * scales[:, g:g+1].float()

    return w_float


def requant_awq_asymmetric(w_float: torch.Tensor, group_size: int):
    """Re-quantize float32 weights as AWQ asymmetric."""
    out_features, in_features = w_float.shape
    num_groups = in_features // group_size

    new_w = torch.zeros_like(w_float, dtype=torch.int8)
    new_scales = torch.zeros(out_features, num_groups, dtype=torch.float32)
    new_zp = torch.zeros(out_features, num_groups, dtype=torch.int8)

    for g in range(num_groups):
        s, e = g * group_size, (g + 1) * group_size
        block = w_float[:, s:e]
        w_min = block.min(dim=1).values
        w_max = block.max(dim=1).values
        sc = (w_max - w_min) / 15.0
        sc = sc.clamp(min=1e-10)
        zp = torch.round(-w_min / sc).clamp(0, 15).to(torch.int8)
        w_q = torch.round(block / sc.unsqueeze(1) + zp.unsqueeze(1).float()).clamp(0, 15).to(torch.int8)
        new_w[:, s:e] = w_q
        new_scales[:, g] = sc
        new_zp[:, g] = zp

    return new_w, new_scales, new_zp


def convert_layer(packed: torch.Tensor, scales: torch.Tensor):
    """Convert one CT layer to AWQ format.

    CT format: weight_packed [N, K//8] (GPTQ sequential along K), weight_scale [N, K//G]
    AWQ format: qweight [K, N//8] (AWQ interleaved along N), scales [K//G, N], qzeros [K//G, N//8]
    """
    # Step 1: Dequant CT → float [N, K]
    w_float = dequant_gptq_symmetric(packed, scales, GROUP_SIZE)  # [N, K]

    # Step 2: Transpose to standard layout [K, N]
    w_float = w_float.T.contiguous()  # [K, N]
    K, N = w_float.shape

    # Step 3: Requant as AWQ asymmetric — groups along K, values along N
    num_groups = K // GROUP_SIZE
    new_w = torch.zeros(K, N, dtype=torch.int8)
    new_scales = torch.zeros(num_groups, N, dtype=torch.float32)  # [K//G, N]
    new_zp = torch.zeros(num_groups, N, dtype=torch.int8)  # [K//G, N]

    for g in range(num_groups):
        s, e = g * GROUP_SIZE, (g + 1) * GROUP_SIZE
        block = w_float[s:e]  # [G, N]
        w_min = block.min(dim=0).values  # [N]
        w_max = block.max(dim=0).values  # [N]
        sc = (w_max - w_min) / 15.0
        sc = sc.clamp(min=1e-10)
        zp = torch.round(-w_min / sc).clamp(0, 15).to(torch.int8)
        w_q = torch.round(block / sc.unsqueeze(0) + zp.unsqueeze(0).float()).clamp(0, 15).to(torch.int8)
        new_w[s:e] = w_q
        new_scales[g] = sc
        new_zp[g] = zp

    # Step 4: Pack with AWQ interleaved order along N dimension
    qweight = pack_awq_interleaved(new_w)  # [K, N//8]

    # Pad zero points along N to multiple of 8 for packing
    if N % PACK_FACTOR != 0:
        pad = PACK_FACTOR - (N % PACK_FACTOR)
        new_zp = torch.nn.functional.pad(new_zp, (0, pad), value=8)
    qzeros = pack_awq_interleaved(new_zp)  # [K//G, ceil(N/8)]

    awq_scales_fp16 = new_scales.clamp(-65504, 65504).to(torch.float16)  # [K//G, N]

    return qweight, awq_scales_fp16, qzeros


# Copy non-weight files
for fname in glob.glob(f"{SRC_DIR}/*.json") + glob.glob(f"{SRC_DIR}/*.txt") + \
             glob.glob(f"{SRC_DIR}/*.model") + glob.glob(f"{SRC_DIR}/*.jinja"):
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    if not os.path.exists(dst):
        shutil.copy2(fname, dst)
        print(f"  Copied {os.path.basename(fname)}")

# Load all tensors
print(f"\nLoading {SRC_DIR}/model.safetensors...")
sf = safe_open(os.path.join(SRC_DIR, "model.safetensors"), framework="pt")
all_keys = list(sf.keys())

converted = OrderedDict()
weight_map = {}
n_converted = 0
n_kept = 0
n_dequant_router = 0

for key in sorted(all_keys):
    if key.endswith(".weight_shape"):
        continue  # Skip shape metadata

    if key.endswith(".weight_packed"):
        base = key[:-len(".weight_packed")]
        scale_key = f"{base}.weight_scale"

        packed = sf.get_tensor(key)
        scales = sf.get_tensor(scale_key)

        # Check if this is the router — dequantize back to BF16
        if "router" in base:
            w_float = dequant_gptq_symmetric(packed, scales, GROUP_SIZE)
            # CT is [N, K], standard weight is [out, in] — keep as-is for router
            converted[f"{base}.weight"] = w_float.to(torch.bfloat16)
            n_dequant_router += 1
            if n_dequant_router <= 2:
                print(f"  Router dequant: {base}")
            continue

        # Convert GPTQ→AWQ
        qweight, awq_scales, qzeros = convert_layer(packed, scales)

        converted[f"{base}.qweight"] = qweight
        converted[f"{base}.scales"] = awq_scales
        converted[f"{base}.qzeros"] = qzeros
        n_converted += 1

        if n_converted % 500 == 0:
            print(f"  Converted {n_converted} layers...")

    elif key.endswith(".weight_scale"):
        continue  # Handled with weight_packed

    elif key.endswith(".weight"):
        # Non-quantized: cast BF16→FP16
        tensor = sf.get_tensor(key)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)
        converted[key] = tensor
        n_kept += 1

    else:
        # Other tensors (biases, norms, etc)
        converted[key] = sf.get_tensor(key)
        n_kept += 1

print(f"\nConverted {n_converted} quantized layers")
print(f"Kept {n_kept} non-quantized layers")
print(f"Dequantized {n_dequant_router} router layers")

# Save
print(f"\nSaving to {OUTPUT_DIR}...")
out_path = os.path.join(OUTPUT_DIR, "model-00001-of-00001.safetensors")
save_file(converted, out_path)

for k in converted:
    weight_map[k] = "model-00001-of-00001.safetensors"

# Save index
index = {
    "metadata": {"total_size": os.path.getsize(out_path)},
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

# Update config
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)
config["quantization_config"] = {
    "bits": 4,
    "group_size": GROUP_SIZE,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": ["model.vision_tower", "model.embed_vision"],
}
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

total_gb = os.path.getsize(out_path) / 1e9
print(f"\nDone!")
print(f"  Quantized: {n_converted}, Kept: {n_kept}, Router dequant: {n_dequant_router}")
print(f"  Size: {total_gb:.1f} GB")
print(f"  Output: {OUTPUT_DIR}")
print(f"\nNext: merge vision weights:")
print(f"  python scripts/quantize/merge_vision_weights.py --base ~/AI/models/gemma-4-26B-A4B-it-BF16 --awq {OUTPUT_DIR}")
