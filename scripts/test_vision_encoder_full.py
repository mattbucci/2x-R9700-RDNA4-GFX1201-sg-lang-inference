#!/usr/bin/env python3
"""Test Gemma4 vision encoder with actual weights on RDNA4.

Loads the real vision encoder weights and runs a forward pass
with synthetic image data to reproduce the SDPA crash.

Usage:
    source scripts/common.sh && activate_conda && setup_rdna4_env
    CUDA_VISIBLE_DEVICES=0 python scripts/test_vision_encoder_full.py
"""
import os
import sys
import torch
import traceback

# Add sglang to path
sys.path.insert(0, 'components/sglang/python')

device = "cuda:0"
dtype = torch.float16
print(f"Device: {torch.cuda.get_device_name(device)}")

# Need to mock server args and TP state before importing SGLang modules
from unittest.mock import MagicMock, patch
import sglang.srt.server_args as server_args_mod
mock_args = MagicMock()
mock_args.mm_attention_backend = None  # let _select_backend choose
server_args_mod._global_server_args = mock_args

# Initialize minimal distributed state for TP=1
import sglang.srt.distributed.parallel_state as ps
# Mock the TP group
mock_group = MagicMock()
mock_group.world_size = 1
mock_group.rank_in_group = 0
mock_group.local_rank = 0
ps._TP = mock_group
ps._ATTN_TP = mock_group

print("\n=== Loading vision config ===")
from transformers import AutoConfig
model_dir = os.path.expanduser("~/AI/models/gemma-4-26B-A4B-it-AWQ-GPTQ-v2-fixed")
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
vision_config = config.vision_config
print(f"  hidden_size={vision_config.hidden_size}")
print(f"  num_heads={vision_config.num_attention_heads}")
print(f"  num_kv_heads={vision_config.num_key_value_heads}")
print(f"  head_dim={vision_config.head_dim}")
print(f"  num_layers={vision_config.num_hidden_layers}")
print(f"  patch_size={vision_config.patch_size}")
print(f"  intermediate_size={vision_config.intermediate_size}")
print(f"  pooling_kernel_size={vision_config.pooling_kernel_size}")

print("\n=== Building vision encoder ===")
from sglang.srt.models.gemma4_vision import Gemma4VisionEncoder
encoder = Gemma4VisionEncoder(vision_config, quant_config=None, prefix="model.vision_tower")
encoder = encoder.to(device=device, dtype=dtype)
print(f"  Encoder built, params: {sum(p.numel() for p in encoder.parameters()):,}")

print("\n=== Loading vision weights ===")
from safetensors import safe_open

# Find safetensor files with vision weights
import glob as glob_mod
st_files = sorted(glob_mod.glob(os.path.join(model_dir, "*.safetensors")))
loaded = 0
missing = []
for st_file in st_files:
    sf = safe_open(st_file, framework="pt", device="cpu")
    for key in sf.keys():
        if not key.startswith("model.vision_tower."):
            continue
        tensor = sf.get_tensor(key)
        # Strip prefix to match state dict keys
        short_key = key[len("model.vision_tower."):]
        try:
            # Navigate the module to find the parameter
            parts = short_key.split(".")
            mod = encoder
            for p in parts[:-1]:
                if p.isdigit():
                    mod = mod[int(p)]
                else:
                    mod = getattr(mod, p)
            param_name = parts[-1]
            param = getattr(mod, param_name, None)
            if param is not None:
                with torch.no_grad():
                    param.copy_(tensor.to(dtype=param.dtype))
                loaded += 1
            else:
                missing.append(short_key)
        except Exception as e:
            missing.append(f"{short_key}: {e}")
    sf = None

print(f"  Loaded: {loaded} tensors")
if missing:
    print(f"  Missing/failed: {len(missing)}")
    for m in missing[:5]:
        print(f"    {m}")

# Test with synthetic image data (simulating different image sizes)
patch_size = vision_config.patch_size
pooling_k = vision_config.pooling_kernel_size

test_cases = [
    ("64x64 image", 64),
    ("128x128 image", 128),
    ("256x256 image", 256),
    ("512x512 image", 512),
    ("768x768 image", 768),
]

for name, img_size in test_cases:
    num_patches_per_side = img_size // patch_size
    num_patches = num_patches_per_side * num_patches_per_side
    patch_pixels = 3 * patch_size * patch_size  # RGB

    print(f"\n=== Test: {name} ({num_patches} patches) ===")

    # Create synthetic patchified pixels [batch, num_patches, patch_pixels]
    pixel_values = torch.rand(1, num_patches, patch_pixels, dtype=dtype, device=device)

    # Create position IDs [batch, num_patches, 2] — (x, y) grid positions
    positions_x = torch.arange(num_patches_per_side, device=device).repeat(num_patches_per_side)
    positions_y = torch.arange(num_patches_per_side, device=device).repeat_interleave(num_patches_per_side)
    pixel_position_ids = torch.stack([positions_x, positions_y], dim=-1).unsqueeze(0)  # [1, num_patches, 2]

    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  pixel_position_ids: {pixel_position_ids.shape}")

    try:
        with torch.no_grad():
            pooled, mask = encoder(pixel_values, pixel_position_ids)
        torch.cuda.synchronize()
        has_nan = torch.isnan(pooled).any().item()
        print(f"  Output: {pooled.shape}, mask: {mask.shape}")
        print(f"  NaN: {has_nan}, max: {pooled.abs().max().item():.4f}")
    except Exception as e:
        print(f"  CRASH: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Try to sync and continue
        try:
            torch.cuda.synchronize()
        except:
            print("  GPU sync also failed — device may be in error state")
            break

print("\nDone.")
