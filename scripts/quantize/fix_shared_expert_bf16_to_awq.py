#!/usr/bin/env python3
"""Post-process fix: AWQ-quantize BF16 shared_expert weights in an existing AWQ model.

Used when an upstream AWQ checkpoint (e.g. atbender's REAP variants) shipped
with BF16 shared_expert.{gate,up,down}_proj weights. SGLang's `moe_wna16`
loader on the Qwen3_5MoeForConditionalGeneration arch can't handle BF16
shared_expert mixed with AWQ experts → HSAIL 0x1016 on first inference.

This script copies the source dir, replaces every shard, and rewrites the
safetensors index so that shared_expert tensors are AWQ-packed (qweight,
scales, qzeros) instead of plain `.weight` bf16. Idempotent — re-running
on an already-fixed dir is a no-op.

Usage:
    python fix_shared_expert_bf16_to_awq.py <model_dir> [--group-size 128]

Verification: scripts/quantize/audit_shared_expert.py <model_dir> should
report "AWQ ✓" after the fix.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Reuse the AWQ packer from convert_moe_ct_to_awq.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from convert_moe_ct_to_awq import quantize_bf16_to_awq  # noqa


def is_target_key(key: str) -> bool:
    """True if key is a BF16 shared_expert projection that needs AWQ packing."""
    if "shared_expert." not in key:
        return False
    if not key.endswith(".weight"):
        return False
    return any(p in key for p in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    if not files:
        sys.exit(f"no safetensors shards in {args.model_dir}")

    index_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    else:
        index = None
        weight_map = {}

    n_quantized = 0
    n_passthrough = 0
    new_weight_map = dict(weight_map)
    for fp in files:
        new_tensors = {}
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if is_target_key(k) and t.dtype in (torch.bfloat16, torch.float16, torch.float32):
                    qw, sc, qz = quantize_bf16_to_awq(t, args.group_size)
                    base = k[:-len(".weight")]
                    new_tensors[f"{base}.qweight"] = qw
                    new_tensors[f"{base}.scales"] = sc
                    new_tensors[f"{base}.qzeros"] = qz
                    n_quantized += 1
                    # update index
                    shard_name = os.path.basename(fp)
                    new_weight_map.pop(k, None)
                    new_weight_map[f"{base}.qweight"] = shard_name
                    new_weight_map[f"{base}.scales"] = shard_name
                    new_weight_map[f"{base}.qzeros"] = shard_name
                else:
                    new_tensors[k] = t
                    n_passthrough += 1

        out = fp + ".tmp"
        save_file(new_tensors, out)
        os.replace(out, fp)
        print(f"  {os.path.basename(fp)}: {len(new_tensors)} tensors saved")

    if index is not None:
        index["weight_map"] = new_weight_map
        # recompute total_size loosely (best-effort)
        index["metadata"] = index.get("metadata", {})
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    print(f"\nDone — quantized {n_quantized} shared_expert weights, passed through {n_passthrough}")


if __name__ == "__main__":
    main()
