#!/usr/bin/env python3
"""Merge vision weights from BF16 base model into AWQ quantized model.

For models where the AWQ conversion stripped vision encoder weights,
this script copies them from the original BF16 model and creates
a new safetensors shard.

Usage:
    python scripts/quantize/merge_vision_weights.py \
        --base ~/AI/models/gemma-4-31B-it-BF16 \
        --awq ~/AI/models/gemma-4-31B-it-AutoRound-AWQ \
        [--output ~/AI/models/gemma-4-31B-it-AutoRound-AWQ]  # in-place by default

Also works for replacing INT4-quantized vision weights with BF16 originals:
    python scripts/quantize/merge_vision_weights.py \
        --base ~/AI/models/gemma-4-26B-A4B-it-BF16 \
        --awq ~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed \
        --vision-prefix "model.embed_vision,model.vision_tower"
"""
import argparse
import gc
import json
import os
import shutil

import torch
from collections import OrderedDict
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Merge vision weights from BF16 base into AWQ model")
    parser.add_argument("--base", required=True, help="Path to BF16 base model")
    parser.add_argument("--awq", required=True, help="Path to AWQ model")
    parser.add_argument("--output", default=None, help="Output path (default: modify AWQ in-place)")
    parser.add_argument("--vision-prefix", default="model.vision_tower,model.embed_vision",
                        help="Comma-separated prefixes for vision weights")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"],
                        help="Dtype for vision weights (default: float16 for AWQ compat)")
    args = parser.parse_args()

    output_dir = args.output or args.awq
    vision_prefixes = tuple(args.vision_prefix.split(","))
    target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Base model:     {args.base}")
    print(f"AWQ model:      {args.awq}")
    print(f"Output:         {output_dir}")
    print(f"Vision prefix:  {vision_prefixes}")
    print(f"Target dtype:   {target_dtype}")

    # Load base model weight map
    base_index_path = os.path.join(args.base, "model.safetensors.index.json")
    with open(base_index_path) as f:
        base_wm = json.load(f)["weight_map"]

    # Find vision weights in base
    vision_keys = {k: v for k, v in base_wm.items() if k.startswith(vision_prefixes)}
    print(f"\nVision weights found in base: {len(vision_keys)}")
    if not vision_keys:
        print("No vision weights found! Check --vision-prefix")
        return

    # Load AWQ model weight map
    awq_index_path = os.path.join(args.awq, "model.safetensors.index.json")
    with open(awq_index_path) as f:
        awq_index = json.load(f)
    awq_wm = awq_index["weight_map"]

    # Check which vision keys already exist in AWQ
    existing = [k for k in vision_keys if k in awq_wm]
    missing = [k for k in vision_keys if k not in awq_wm]
    print(f"  Already in AWQ (will replace): {len(existing)}")
    print(f"  Missing from AWQ (will add):   {len(missing)}")

    # Copy vision weights from base into a new shard
    vision_shard_name = "model-vision.safetensors"
    vision_tensors = OrderedDict()

    # Load from base model shards
    base_shards_needed = set(vision_keys.values())
    shard_cache = {}

    for shard_name in sorted(base_shards_needed):
        shard_path = os.path.join(args.base, shard_name)
        print(f"  Loading {shard_name}...")
        sf = safe_open(shard_path, framework="pt")
        for key in vision_keys:
            if base_wm[key] == shard_name:
                tensor = sf.get_tensor(key)
                # Convert to target dtype
                if tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)
                vision_tensors[key] = tensor

    print(f"\nLoaded {len(vision_tensors)} vision tensors")

    # If output is different from AWQ, copy all files first
    if output_dir != args.awq:
        os.makedirs(output_dir, exist_ok=True)
        for fname in os.listdir(args.awq):
            src = os.path.join(args.awq, fname)
            dst = os.path.join(output_dir, fname)
            if not os.path.exists(dst):
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

    # Remove existing vision weights from their current shards
    if existing:
        # Find which AWQ shards contain vision weights
        shards_with_vision = {}
        for k in existing:
            shard = awq_wm[k]
            if shard not in shards_with_vision:
                shards_with_vision[shard] = []
            shards_with_vision[shard].append(k)

        for shard_name, keys_to_remove in shards_with_vision.items():
            shard_path = os.path.join(output_dir, shard_name)
            if not os.path.exists(shard_path):
                continue
            print(f"  Removing {len(keys_to_remove)} vision keys from {shard_name}...")
            sf = safe_open(shard_path, framework="pt")
            remaining = OrderedDict()
            for k in sf.keys():
                if k not in keys_to_remove:
                    remaining[k] = sf.get_tensor(k)
            if remaining:
                save_file(remaining, shard_path)
            else:
                os.remove(shard_path)
            del remaining
            gc.collect()

    # Save vision shard
    vision_path = os.path.join(output_dir, vision_shard_name)
    print(f"\nSaving {len(vision_tensors)} vision tensors to {vision_shard_name}...")
    save_file(vision_tensors, vision_path)

    # Update weight map
    for k in vision_tensors:
        awq_wm[k] = vision_shard_name
    # Remove keys that no longer exist in their old shards
    # (they've been moved to the vision shard)

    # Recalculate total size
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith(".safetensors")
    )
    awq_index["metadata"]["total_size"] = total_size
    awq_index["weight_map"] = awq_wm

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(awq_index, f, indent=2)

    vision_size_mb = os.path.getsize(vision_path) / 1e6
    print(f"\nDone!")
    print(f"  Vision shard: {vision_shard_name} ({vision_size_mb:.1f} MB)")
    print(f"  Total model size: {total_size / 1e9:.1f} GB")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
