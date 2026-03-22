#!/usr/bin/env python3
"""Convert compressed-tensors pack-quantized model to standard AWQ format.

Takes androiddrew/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit (compressed-tensors)
and converts to standard AWQ format that SGLang's Triton AWQ kernel can handle.

The compressed-tensors format stores:
  - weight_packed: int32 with 8x4-bit values packed sequentially along input dim
    Shape: [out_features, in_features // 8]
  - weight_scale: per-group BF16 scales
    Shape: [out_features, in_features // group_size]
  - weight_shape: original [out_features, in_features]

AWQ format stores:
  - qweight: int32 packed weights with AWQ interleaved packing along output dim
    Shape: [in_features, out_features // 8]
  - qzeros: int32 packed zero points
    Shape: [in_features // group_size, out_features // 8]
  - scales: FP16 per-group scales
    Shape: [in_features // group_size, out_features]

Key differences from compressed-tensors:
  1. Weight matrix is TRANSPOSED: [out, in] -> [in, out]
  2. Packing is along OUTPUT dim (dim=1), not input dim
  3. AWQ uses interleaved packing order: [0, 2, 4, 6, 1, 3, 5, 7]
  4. Scales and qzeros are also transposed

Run: conda activate sglang-7.2
     python convert_compressed_tensors_to_awq.py
"""
import os
import json
import glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict
import shutil

SRC_MODEL = "androiddrew/Devstral-Small-2-24B-Instruct-2512-AWQ-4bit"
SRC_DIR = os.path.expanduser(f"~/.cache/huggingface/hub/models--{SRC_MODEL.replace('/', '--')}")
SRC_SNAP = glob.glob(f"{SRC_DIR}/snapshots/*/model-00001-of-00004.safetensors")
if not SRC_SNAP:
    print(f"Model not found at {SRC_DIR}. Download it first:")
    print(f"  huggingface-cli download {SRC_MODEL}")
    exit(1)
SRC_SNAP_DIR = os.path.dirname(SRC_SNAP[0])
OUTPUT_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/AI/models")) + "/Devstral-Small-2-24B-AWQ-4bit"

GROUP_SIZE = 128
W_BIT = 4
PACK_FACTOR = 32 // W_BIT  # 8 values per int32

# AWQ packing order: value at output position i is stored at bit offset AWQ_PACK_ORDER[i] * 4.
# This is the REVERSE of the kernel's unpack order, so the kernel reads back the correct values.
# Kernel unpacks position i by reading bits at [0, 4, 1, 5, 2, 6, 3, 7][i] * 4.
# So we must STORE position i at those same bit offsets.
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

print(f"Source: {SRC_SNAP_DIR}")
print(f"Output: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Copy non-weight files (config, tokenizer, etc.)
for fname in glob.glob(f"{SRC_SNAP_DIR}/*.json") + glob.glob(f"{SRC_SNAP_DIR}/*.txt") + \
         glob.glob(f"{SRC_SNAP_DIR}/*.model") + glob.glob(f"{SRC_SNAP_DIR}/*.jinja"):
    dst = os.path.join(OUTPUT_DIR, os.path.basename(fname))
    if not os.path.exists(dst):
        shutil.copy2(fname, dst)
        print(f"  Copied {os.path.basename(fname)}")


def unpack_int32_to_4bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 tensor with 8x4-bit values (sequential order) to int8 tensor.

    Input: [..., N] int32
    Output: [..., N*8] int8 (values 0-15)
    """
    # Extract 8 x 4-bit values from each int32
    unpacked = []
    for i in range(PACK_FACTOR):
        unpacked.append((packed >> (i * W_BIT)) & 0xF)
    # Stack along last dim: each int32 becomes 8 values
    return torch.stack(unpacked, dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int8)


def pack_4bit_to_int32_awq(values: torch.Tensor) -> torch.Tensor:
    """Pack int8 tensor (values 0-15) into int32 with AWQ interleaved order.

    AWQ packing order: position i in the group of 8 goes to bit offset AWQ_PACK_ORDER[i]*4.
    Input: [..., N] int8 where N is divisible by 8
    Output: [..., N//8] int32
    """
    assert values.shape[-1] % PACK_FACTOR == 0
    # Reshape to groups of 8
    grouped = values.reshape(*values.shape[:-1], -1, PACK_FACTOR)  # [..., N//8, 8]
    packed = torch.zeros(*grouped.shape[:-1], dtype=torch.int32, device=values.device)
    for i in range(PACK_FACTOR):
        packed |= (grouped[..., i].to(torch.int32) & 0xF) << (AWQ_PACK_ORDER[i] * W_BIT)
    return packed


# Process each safetensors shard
shard_files = sorted(glob.glob(f"{SRC_SNAP_DIR}/model-*.safetensors"))
print(f"\nProcessing {len(shard_files)} shards...")

weight_map = {}  # For index.json

for shard_idx, shard_path in enumerate(shard_files):
    shard_name = os.path.basename(shard_path)
    print(f"\n=== {shard_name} ===")

    f = safe_open(shard_path, framework="pt")
    keys = list(f.keys())

    converted = OrderedDict()
    processed_packed = set()

    for key in keys:
        if key in processed_packed:
            continue

        if key.endswith(".weight_packed"):
            # This is a quantized weight — convert to AWQ format
            base = key[:-len(".weight_packed")]

            # Map weight names to match SGLang's expected format.
            # SGLang's LlavaForConditionalGeneration strips "language_model." prefix,
            # then passes to MistralForCausalLM which has a "model" submodule.
            # So keys must be: language_model.model.layers.* (not language_model.layers.*)
            # Source: model.language_model.layers.* -> language_model.model.layers.*
            # Source: model.language_model.embed_tokens.* -> language_model.model.embed_tokens.*
            # Source: model.language_model.lm_head.* -> language_model.lm_head.*
            awq_base = base
            if awq_base.startswith("model.language_model."):
                suffix = awq_base[len("model.language_model."):]
                if suffix.startswith("lm_head"):
                    awq_base = f"language_model.{suffix}"
                else:
                    awq_base = f"language_model.model.{suffix}"

            packed = f.get_tensor(key)  # [out_features, in_features // 8] int32
            scale_key = f"{base}.weight_scale"
            if scale_key not in keys:
                print(f"  SKIP {base} (scale in different shard)")
                continue
            scale = f.get_tensor(scale_key)  # [out_features, in_features // group_size] bf16

            out_features = packed.shape[0]
            in_features = packed.shape[1] * PACK_FACTOR

            # Step 1: Unpack compressed-tensors sequential packing to full int8
            # [out_features, in_features // 8] -> [out_features, in_features]
            unpacked = unpack_int32_to_4bit(packed)

            # Step 2: Transpose weight matrix [out, in] -> [in, out]
            unpacked_t = unpacked.T.contiguous()  # [in_features, out_features]

            # Step 3: Repack with AWQ interleaved order along output dim
            # [in_features, out_features] -> [in_features, out_features // 8]
            qweight = pack_4bit_to_int32_awq(unpacked_t)

            # Step 4: Transpose scales [out, in//G] -> [in//G, out]
            scales = scale.T.contiguous().to(torch.float16)

            # Step 5: Create qzeros with correct AWQ shape [in//G, out//8]
            # Symmetric quant: zero_point = 8 for 4-bit unsigned
            num_groups = in_features // GROUP_SIZE
            num_out_packed = out_features // PACK_FACTOR
            # Build 0x88888888 using AWQ packing order
            zp_val = torch.tensor([8], dtype=torch.int32)
            qzeros = torch.zeros((num_groups, num_out_packed), dtype=torch.int32)
            for i in range(PACK_FACTOR):
                qzeros |= (zp_val << (AWQ_PACK_ORDER[i] * W_BIT))

            converted[f"{awq_base}.qweight"] = qweight
            converted[f"{awq_base}.scales"] = scales
            converted[f"{awq_base}.qzeros"] = qzeros

            processed_packed.add(key)
            processed_packed.add(f"{base}.weight_scale")
            processed_packed.add(f"{base}.weight_shape")

            print(f"  {awq_base}: [{out_features}, {in_features}] -> "
                  f"qweight{list(qweight.shape)}, scales{list(scales.shape)}, "
                  f"qzeros{list(qzeros.shape)}")

        elif key.endswith(".weight_scale") or key.endswith(".weight_shape"):
            continue  # Already handled with weight_packed

        else:
            # Non-quantized weight (embeddings, layernorms, lm_head)
            # Same mapping as quantized weights above
            new_key = key
            if new_key.startswith("model.language_model."):
                suffix = new_key[len("model.language_model."):]
                if suffix.startswith("lm_head"):
                    new_key = f"language_model.{suffix}"
                else:
                    new_key = f"language_model.model.{suffix}"

            converted[new_key] = f.get_tensor(key)
            print(f"  {new_key}: {list(converted[new_key].shape)} {converted[new_key].dtype}")

    # Save converted shard
    out_path = os.path.join(OUTPUT_DIR, shard_name)
    save_file(converted, out_path)

    for k in converted:
        weight_map[k] = shard_name

    print(f"  Saved {len(converted)} tensors to {shard_name}")

# Create model index
index = {
    "metadata": {"total_size": sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, f))
        for f in os.listdir(OUTPUT_DIR) if f.endswith(".safetensors")
    )},
    "weight_map": weight_map,
}
with open(os.path.join(OUTPUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)

# Update config.json to use AWQ quantization instead of compressed-tensors
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path) as cfg_f:
    config = json.load(cfg_f)

# Replace quantization_config with AWQ format
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

print(f"\nDone! AWQ model saved to {OUTPUT_DIR}")
print(f"Quantization config: AWQ 4-bit, group_size={GROUP_SIZE}")
print(f"Layout: qweight [K, N//8], scales [K//G, N], qzeros [K//G, N//8]")
