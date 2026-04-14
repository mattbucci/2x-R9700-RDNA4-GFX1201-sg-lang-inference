#!/usr/bin/env python3
"""Convert GPTQ-format model to native AWQ format for SGLang's Triton kernel.

GPTQ packing: sequential 4-bit along K (input) dimension
  qweight: [K//8, N] int32, qzeros: [groups, N//8] int32, scales: [groups, N] fp16

AWQ packing: interleaved 4-bit along N (output) dimension
  qweight: [K, N//8] int32, qzeros: [K//G, N//8] int32, scales: [K//G, N] fp16

Conversion: unpack GPTQ → raw 4-bit [K, N] → repack AWQ interleaved order.
Zero points: GPTQ stores zp-1 (actual=stored+1), AWQ stores actual zp.

Usage:
    GPTQ_INPUT=~/AI/models/gemma-4-31B-it-int4-AutoRound \\
    AWQ_OUTPUT=~/AI/models/gemma-4-31B-it-AutoRound-AWQ \\
    python scripts/quantize/convert_gptq_to_awq.py
"""
import gc
import os
import json
import glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
import shutil

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models"))
SRC_DIR = os.environ.get("GPTQ_INPUT", f"{MODELS_DIR}/gemma-4-31B-it-int4-AutoRound")
OUTPUT_DIR = os.environ.get("AWQ_OUTPUT", f"{MODELS_DIR}/gemma-4-31B-it-AutoRound-AWQ")

W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32

# AWQ packing order: interleaved for the GEMM kernel's unpack pattern
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

if not os.path.isdir(SRC_DIR):
    print(f"Source not found: {SRC_DIR}")
    exit(1)

# Read config
config_path = os.path.join(SRC_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)
qconfig = config.get("quantization_config", {})
GROUP_SIZE = qconfig.get("group_size", 128)

print(f"Source:     {SRC_DIR}")
print(f"Output:     {OUTPUT_DIR}")
print(f"Group size: {GROUP_SIZE}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def unpack_gptq_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ int32 tensor with sequential 4-bit packing along dim 0.
    Input: [K//8, N] int32 → Output: [K, N] int8"""
    K_packed, N = packed.shape
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    # Stack along dim=1 between K_packed and N, then reshape
    # [K_packed, 8, N] → [K, N]
    return torch.stack(unpacked, dim=1).reshape(K_packed * PACK_FACTOR, N).to(torch.int8)


def unpack_gptq_zeros(packed: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ qzeros: [groups, N//8] → [groups, N] with +1 correction."""
    groups, N_packed = packed.shape
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    # Sequential packing along N: [groups, N_packed, 8] → [groups, N]
    zp = torch.stack(unpacked, dim=-1).reshape(groups, N_packed * PACK_FACTOR)
    # GPTQ v1: actual_zp = stored_zp + 1
    return (zp + 1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (values 0-15) into int32 with AWQ interleaved order.
    Input: [*, N] where N is divisible by 8 → Output: [*, N//8] int32"""
    assert values.shape[-1] % PACK_FACTOR == 0
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


# Copy non-weight files
for fname in (
    glob.glob(f"{SRC_DIR}/*.json")
    + glob.glob(f"{SRC_DIR}/*.txt")
    + glob.glob(f"{SRC_DIR}/*.model")
    + glob.glob(f"{SRC_DIR}/*.jinja")
):
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    if not os.path.exists(dst):
        shutil.copy2(fname, dst)
        print(f"  Copied {os.path.basename(fname)}")

# Build global weight index for cross-shard tensor access
shard_files = sorted(glob.glob(f"{SRC_DIR}/model*.safetensors"))
if not shard_files:
    print(f"No model*.safetensors files found in {SRC_DIR}")
    exit(1)

# Read source weight map to handle cross-shard splits
src_index_path = os.path.join(SRC_DIR, "model.safetensors.index.json")
if os.path.exists(src_index_path):
    with open(src_index_path) as f:
        src_weight_map = json.load(f)["weight_map"]
else:
    # Single shard — build map from the file
    src_weight_map = {}
    for sp in shard_files:
        sf = safe_open(sp, framework="pt")
        for k in sf.keys():
            src_weight_map[k] = os.path.basename(sp)

# Cache open shard files for cross-shard access
shard_cache = {}
def get_tensor(name):
    """Load a tensor from any shard."""
    shard_name = src_weight_map[name]
    if shard_name not in shard_cache:
        shard_cache[shard_name] = safe_open(os.path.join(SRC_DIR, shard_name), framework="pt")
    return shard_cache[shard_name].get_tensor(name)

all_keys = list(src_weight_map.keys())
weight_map = {}
total_converted = 0
total_kept = 0
total_skipped_vision = 0

# Group keys by output shard (use same shard boundaries as source)
from collections import defaultdict
shard_keys = defaultdict(list)
for key in all_keys:
    shard_keys[src_weight_map[key]].append(key)

processed = set()

for shard_name in sorted(shard_keys.keys()):
    keys = shard_keys[shard_name]
    print(f"\n=== {shard_name} ===")

    converted = OrderedDict()

    for key in keys:
        if key in processed:
            continue

        # Skip vision tower weights
        if "vision_tower" in key or "embed_vision" in key:
            total_skipped_vision += 1
            processed.add(key)
            continue

        if key.endswith(".qweight"):
            # GPTQ quantized weight — convert to AWQ format
            base = key[: -len(".qweight")]

            qweight_gptq = get_tensor(key)  # [K//8, N] int32
            scale_key = f"{base}.scales"
            zeros_key = f"{base}.qzeros"

            if scale_key not in src_weight_map or zeros_key not in src_weight_map:
                print(f"  SKIP {base}: missing scales or qzeros")
                continue

            scales_gptq = get_tensor(scale_key)   # [groups, N] fp16
            qzeros_gptq = get_tensor(zeros_key)   # [groups, N//8] int32

            K = qweight_gptq.shape[0] * PACK_FACTOR
            N = qweight_gptq.shape[1]

            # 1. Unpack GPTQ sequential K-packed → raw 4-bit [K, N]
            w_int = unpack_gptq_to_4bit(qweight_gptq)

            # 2. Unpack zero points
            zp_raw = unpack_gptq_zeros(qzeros_gptq)  # [groups, N] actual zp
            if zp_raw.shape[1] > N:
                zp_raw = zp_raw[:, :N]

            # 3. Full FP32 dequant → re-quantize as asymmetric AWQ
            # This handles symmetric GPTQ (negative scales) correctly by
            # computing the actual float weights and re-quantizing with
            # positive-only scales and proper zero points.
            scales_fp32 = scales_gptq.to(torch.float32)
            has_neg = (scales_fp32 < 0).any().item()

            if has_neg:
                # Dequantize fully in FP32
                groups = scales_fp32.shape[0]
                w_float = torch.zeros(K, N, dtype=torch.float32)
                for g in range(groups):
                    s, e = g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, K)
                    w_float[s:e] = (w_int[s:e].float() - zp_raw[g].float()) * scales_fp32[g]

                # Re-quantize as asymmetric AWQ (positive scales, proper zp)
                new_w_int = torch.zeros_like(w_int)
                new_zp = torch.zeros_like(zp_raw)
                new_scales = torch.zeros_like(scales_fp32)
                for g in range(groups):
                    s, e = g * GROUP_SIZE, min((g + 1) * GROUP_SIZE, K)
                    block = w_float[s:e]  # [group_size, N]
                    # Per-column min/max
                    w_min = block.min(dim=0).values  # [N]
                    w_max = block.max(dim=0).values  # [N]
                    # Scale = (max - min) / 15
                    sc = (w_max - w_min) / 15.0
                    sc = sc.clamp(min=1e-10)  # avoid div by zero
                    # Zero point = round(-min / scale), clamped to 0-15
                    zp = torch.round(-w_min / sc).clamp(0, 15).to(torch.int8)
                    # Quantize: round((w / scale) + zp)
                    w_q = torch.round(block / sc.unsqueeze(0) + zp.unsqueeze(0).float())
                    w_q = w_q.clamp(0, 15).to(torch.int8)
                    new_w_int[s:e] = w_q
                    new_zp[g] = zp
                    new_scales[g] = sc
                w_int = new_w_int
                zp_raw = new_zp
                scales_fp32 = new_scales
                print(f"    Re-quantized (had negative scales)")

            # 4. Repack weights with AWQ interleaved order along N dimension
            qweight_awq = pack_4bit_to_int32_awq(w_int)  # [K, N//8]

            # 5. Clamp scales to FP16 range
            scales_awq = scales_fp32.clamp(0, 65504).to(torch.float16)

            # 6. Repack zero points with AWQ order
            qzeros_awq = pack_4bit_to_int32_awq(zp_raw)  # [groups, N//8]

            converted[f"{base}.qweight"] = qweight_awq
            converted[f"{base}.scales"] = scales_awq
            converted[f"{base}.qzeros"] = qzeros_awq

            processed.add(key)
            processed.add(scale_key)
            processed.add(zeros_key)
            processed.add(f"{base}.g_idx")  # Skip g_idx (AWQ doesn't use it)

            total_converted += 1
            print(
                f"  Q {base}: GPTQ[{qweight_gptq.shape[0]},{N}] -> "
                f"AWQ qw{list(qweight_awq.shape)} sc{list(scales_awq.shape)}"
            )

        elif key.endswith(".scales") or key.endswith(".qzeros") or key.endswith(".g_idx"):
            processed.add(key)
            continue  # Handled with qweight

        else:
            # Non-quantized weight — convert BF16 to FP16 for AWQ compatibility
            # (AWQ models expect FP16 norms; BF16 norms cause NaN in SGLang)
            tensor = get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            converted[key] = tensor
            processed.add(key)
            total_kept += 1

    # Save converted shard
    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)

    for k in converted:
        weight_map[k] = shard_name

    print(f"  Saved {len(converted)} tensors to {shard_name}")

    del converted
    gc.collect()

# Create model index
index = {
    "metadata": {
        "total_size": sum(
            os.path.getsize(os.path.join(OUTPUT_DIR, f_name))
            for f_name in os.listdir(OUTPUT_DIR)
            if f_name.endswith(".safetensors")
        )
    },
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

# Update config.json with AWQ quantization config
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as cfg_f:
    config = json.load(cfg_f)

config["quantization_config"] = {
    "bits": 4,
    "group_size": GROUP_SIZE,
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
    "modules_to_not_convert": [],
}

with open(config_path, "w") as cfg_f:
    json.dump(config, cfg_f, indent=2)

total_size_gb = index["metadata"]["total_size"] / (1024**3)
print(f"\nDone!")
print(f"  Quantized layers: {total_converted}")
print(f"  Kept layers: {total_kept}")
print(f"  Skipped vision: {total_skipped_vision}")
print(f"  Total size: {total_size_gb:.1f} GB")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Format: AWQ 4-bit, group_size={GROUP_SIZE}")
