#!/usr/bin/env python3
"""
Analyze quantization error characteristics of Gemma 4 31B RTN AWQ model.

Compares dequantized AWQ weights against BF16 reference to find:
- Per-layer and per-projection quantization error (cosine sim, max abs error, relative error)
- FP16 scale precision issues (scales exceeding FP16 max, dequantized overflow)
- Whether early/late layers have dramatically higher error

AWQ format:
  qweight: [in_features, out_features//8] int32, AWQ interleaved packing
  scales: [in_features//GROUP_SIZE, out_features] float16
  qzeros: [in_features//GROUP_SIZE, out_features//8] int32
  AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
  GROUP_SIZE = 128
  Dequant: (q_value - 8) * scale
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from safetensors import safe_open

# ── Config ──────────────────────────────────────────────────────────────────
BF16_DIR = Path.home() / "AI/models/gemma-4-31B-it-BF16"
AWQ_DIR = Path.home() / "AI/models/gemma-4-31B-it-AWQ-RTN-128g"
GROUP_SIZE = 128
AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
FP16_MAX = 65504.0

# Layers to check: early (0-2), late (57-59), and a middle sample
TARGET_LAYERS = [0, 1, 2, 29, 30, 57, 58, 59]
PROJ_TYPES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}


def unpack_awq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """
    Unpack int32 qweight into individual 4-bit values.
    qweight shape: [in_features, out_features // 8]
    returns: [in_features, out_features] uint8 (0-15)

    AWQ packs 8 x 4-bit values per int32 in a specific interleaved order.
    AWQ_PACK_ORDER[i] tells which packed nibble position to read for output column i.
    i.e., output column i (within each group of 8) reads from nibble AWQ_PACK_ORDER[i].
    """
    in_feat, packed_out = qweight.shape
    out_feat = packed_out * 8

    qweight_i64 = qweight.to(torch.int64)  # avoid sign issues
    unpacked = torch.zeros(in_feat, out_feat, dtype=torch.uint8)

    for out_within_group in range(8):
        pack_pos = AWQ_PACK_ORDER[out_within_group]
        nibble = ((qweight_i64 >> (pack_pos * 4)) & 0xF).to(torch.uint8)
        unpacked[:, out_within_group::8] = nibble

    return unpacked


def unpack_awq_qzeros(qzeros: torch.Tensor, out_features: int) -> torch.Tensor:
    """
    Unpack qzeros, same packing as qweight.
    qzeros shape: [n_groups, out_features // 8]
    returns: [n_groups, out_features] uint8
    """
    return unpack_awq_qweight(qzeros)


def dequantize_awq(qweight: torch.Tensor, scales: torch.Tensor,
                   qzeros: torch.Tensor) -> torch.Tensor:
    """
    Dequantize AWQ weights back to float32.

    qweight: [in_features, out_features//8] int32
    scales: [n_groups, out_features] float16
    qzeros: [n_groups, out_features//8] int32

    Returns: [in_features, out_features] float32
    """
    in_feat = qweight.shape[0]
    out_feat = scales.shape[1]

    # Unpack quantized values
    q_vals = unpack_awq_qweight(qweight).to(torch.float32)  # [in_feat, out_feat]
    z_vals = unpack_awq_qzeros(qzeros, out_feat).to(torch.float32)  # [n_groups, out_feat]
    s_vals = scales.to(torch.float32)  # [n_groups, out_feat]

    # Apply per-group dequantization: (q - zero) * scale
    # But RTN AWQ typically uses symmetric: (q - 8) * scale
    # Let's check what the zeros actually are first
    n_groups = scales.shape[0]

    result = torch.zeros(in_feat, out_feat, dtype=torch.float32)
    for g in range(n_groups):
        row_start = g * GROUP_SIZE
        row_end = min(row_start + GROUP_SIZE, in_feat)
        q_group = q_vals[row_start:row_end, :]     # [group_size, out_feat]
        z_group = z_vals[g:g+1, :]                  # [1, out_feat]
        s_group = s_vals[g:g+1, :]                  # [1, out_feat]
        result[row_start:row_end, :] = (q_group - z_group) * s_group

    return result


def compute_metrics(ref: torch.Tensor, deq: torch.Tensor):
    """Compute error metrics between reference and dequantized weights."""
    ref_f = ref.flatten().float()
    deq_f = deq.flatten().float()

    # Cosine similarity -- use float64 to avoid >1.0 on large vectors
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f.double().unsqueeze(0), deq_f.double().unsqueeze(0)
    ).item()

    # Absolute error
    abs_err = (ref_f - deq_f).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()

    # Relative error -- use a meaningful threshold to exclude near-zero weights
    # Weights below the quantization step will always have huge relative error
    ref_abs = ref_f.abs()
    # Use weights above 1% of max as the "significant" set
    threshold = ref_abs.max() * 0.01
    mask = ref_abs > threshold
    if mask.any():
        rel_err = (abs_err[mask] / ref_abs[mask])
        mean_rel_err = rel_err.mean().item()
        max_rel_err = rel_err.max().item()
        # Use numpy for p99 on large tensors (torch quantile has size limits)
        p99_rel_err = float(np.percentile(rel_err.numpy(), 99))
    else:
        mean_rel_err = max_rel_err = p99_rel_err = float('nan')

    # Fraction of weights with >50% and >100% relative error (using raw threshold 1e-8)
    mask_all = ref_abs > 1e-8
    if mask_all.any():
        rel_all = abs_err[mask_all] / ref_abs[mask_all]
        frac_gt50 = (rel_all > 0.5).float().mean().item()
        frac_gt100 = (rel_all > 1.0).float().mean().item()
    else:
        frac_gt50 = frac_gt100 = float('nan')

    # RMSE
    rmse = (abs_err ** 2).mean().sqrt().item()

    # Signal-to-noise ratio
    signal_power = (ref_f ** 2).mean()
    noise_power = ((ref_f - deq_f) ** 2).mean()
    snr_db = 10 * torch.log10(signal_power / noise_power).item() if noise_power > 0 else float('inf')

    return {
        "cos_sim": cos_sim,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "mean_rel_err": mean_rel_err,
        "max_rel_err": max_rel_err,
        "p99_rel_err": p99_rel_err,
        "frac_gt50": frac_gt50,
        "frac_gt100": frac_gt100,
        "snr_db": snr_db,
    }


def analyze_scales(scales: torch.Tensor, layer_idx: int, proj_name: str):
    """Analyze scale tensor for FP16 issues."""
    s = scales.float()
    s_abs = s.abs()
    info = {
        "scale_min": s_abs.min().item(),
        "scale_max": s_abs.max().item(),
        "scale_mean": s_abs.mean().item(),
        "scale_std": s_abs.std().item(),
        "n_exceed_fp16_max": (s_abs > FP16_MAX).sum().item(),
        "n_exceed_fp16_half": (s_abs > FP16_MAX / 2).sum().item(),
        "n_very_large": (s_abs > 1000).sum().item(),
        "n_very_small": (s_abs < 1e-6).sum().item(),
        "scale_dynamic_range": (s_abs.max() / s_abs[s_abs > 0].min()).item() if (s_abs > 0).any() else float('inf'),
    }
    return info


def check_dequant_overflow(deq: torch.Tensor):
    """Check if dequantized values would overflow FP16."""
    d = deq.float().abs()
    return {
        "deq_max": d.max().item(),
        "deq_exceed_fp16": (d > FP16_MAX).sum().item(),
        "deq_exceed_fp16_half": (d > FP16_MAX / 2).sum().item(),
        "deq_n_large": (d > 100).sum().item(),
    }


def load_safetensor_files(model_dir: Path):
    """Open all safetensor files in a model directory."""
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)

    # Get unique shard files
    shards = set(index["weight_map"].values())
    handles = {}
    for shard in shards:
        handles[shard] = safe_open(str(model_dir / shard), framework="pt")

    return index["weight_map"], handles


def get_tensor(weight_map, handles, key):
    """Get a tensor by key from the appropriate shard."""
    shard = weight_map[key]
    return handles[shard].get_tensor(key)


def main():
    print("=" * 100)
    print("Gemma 4 31B AWQ-RTN-128g Quantization Error Analysis")
    print("=" * 100)
    print(f"BF16 reference: {BF16_DIR}")
    print(f"AWQ RTN model:  {AWQ_DIR}")
    print(f"Group size:     {GROUP_SIZE}")
    print(f"Target layers:  {TARGET_LAYERS}")
    print()

    t0 = time.time()

    # Load model indices
    print("Loading model indices...")
    bf16_map, bf16_handles = load_safetensor_files(BF16_DIR)
    awq_map, awq_handles = load_safetensor_files(AWQ_DIR)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # First, check what zeros actually look like
    print("\n--- Zero-point analysis (layer 0, q_proj) ---")
    qz = get_tensor(awq_map, awq_handles, "model.language_model.layers.0.self_attn.q_proj.qzeros")
    z_unpacked = unpack_awq_qzeros(qz, 8192)
    z_unique = z_unpacked.unique()
    print(f"  Unique zero values: {z_unique.tolist()[:20]}")
    print(f"  Zero mean: {z_unpacked.float().mean():.4f}")
    print(f"  Number of unique zeros: {len(z_unique)}")
    if len(z_unique) == 1 and z_unique[0].item() == 8:
        print("  -> Confirmed symmetric quantization (zero=8 everywhere)")
    else:
        print("  -> Asymmetric quantization detected!")

    # Check layer_scalar values across all target layers
    print("\n--- Layer scalar values ---")
    for layer_idx in TARGET_LAYERS:
        key = f"model.language_model.layers.{layer_idx}.layer_scalar"
        try:
            scalar = get_tensor(bf16_map, bf16_handles, key)
            print(f"  Layer {layer_idx:2d}: layer_scalar = {scalar.item():.6f}")
        except KeyError:
            print(f"  Layer {layer_idx:2d}: no layer_scalar")

    # Storage for all results
    all_results = []
    fp16_issues = []

    proj_list = [
        ("self_attn", "q_proj"),
        ("self_attn", "k_proj"),
        ("self_attn", "v_proj"),
        ("self_attn", "o_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    ]

    print("\n--- Per-layer, per-projection analysis ---")
    for layer_idx in TARGET_LAYERS:
        print(f"\nLayer {layer_idx}:")
        for module, proj in proj_list:
            prefix = f"model.language_model.layers.{layer_idx}.{module}.{proj}"

            # Load BF16 reference
            bf16_key = f"{prefix}.weight"
            try:
                ref_weight = get_tensor(bf16_map, bf16_handles, bf16_key).float()
            except KeyError:
                print(f"  {proj}: SKIPPED (no BF16 weight)")
                continue

            # Load AWQ components
            try:
                qweight = get_tensor(awq_map, awq_handles, f"{prefix}.qweight")
                scales = get_tensor(awq_map, awq_handles, f"{prefix}.scales")
                qzeros = get_tensor(awq_map, awq_handles, f"{prefix}.qzeros")
            except KeyError:
                print(f"  {proj}: SKIPPED (no AWQ weight)")
                continue

            # Dequantize
            deq = dequantize_awq(qweight, scales, qzeros)

            # AWQ stores as [in_features, out_features], BF16 as [out_features, in_features]
            # So we need to transpose the dequantized result
            deq_t = deq.t()  # Now [out_features, in_features]

            # Verify shapes match
            if deq_t.shape != ref_weight.shape:
                print(f"  {proj}: SHAPE MISMATCH deq={deq_t.shape} ref={ref_weight.shape}")
                continue

            # Compute metrics
            metrics = compute_metrics(ref_weight, deq_t)

            # Analyze scales
            scale_info = analyze_scales(scales, layer_idx, proj)

            # Check dequantized overflow
            overflow_info = check_dequant_overflow(deq_t)

            # BF16 reference weight stats
            ref_stats = {
                "ref_max": ref_weight.abs().max().item(),
                "ref_mean": ref_weight.abs().mean().item(),
                "ref_std": ref_weight.std().item(),
            }

            result = {
                "layer": layer_idx,
                "proj": proj,
                "module": module,
                **metrics,
                **scale_info,
                **overflow_info,
                **ref_stats,
                "shape": list(ref_weight.shape),
            }
            all_results.append(result)

            # Flag FP16 issues
            if scale_info["n_exceed_fp16_max"] > 0:
                fp16_issues.append(f"Layer {layer_idx} {proj}: {scale_info['n_exceed_fp16_max']} scales exceed FP16 max!")
            if overflow_info["deq_exceed_fp16"] > 0:
                fp16_issues.append(f"Layer {layer_idx} {proj}: {overflow_info['deq_exceed_fp16']} dequantized values exceed FP16 max!")
            if scale_info["scale_max"] > 1000:
                fp16_issues.append(f"Layer {layer_idx} {proj}: scales reach {scale_info['scale_max']:.1f} (very large)")

            # Print compact per-projection results
            print(f"  {proj:10s}: cos={metrics['cos_sim']:.8f}  "
                  f"maxAE={metrics['max_abs_err']:.6f}  "
                  f"RMSE={metrics['rmse']:.6f}  "
                  f"SNR={metrics['snr_db']:.1f}dB  "
                  f"scale_max={scale_info['scale_max']:.4f}  "
                  f"ref_max={ref_stats['ref_max']:.4f}")

    # ── Summary tables ──────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY TABLE: Quantization Error by Layer and Projection")
    print("=" * 120)

    header = (f"{'Layer':>5} {'Proj':>10} {'CosSim':>12} {'MaxAbsErr':>12} {'RMSE':>10} "
              f"{'SNR(dB)':>8} {'RelErr>1%':>12} {'P99RelErr':>12} {'ScaleMax':>10} "
              f"{'RefMax':>10} {'>50%RE':>8} {'>100%RE':>8}")
    print(header)
    print("-" * 130)

    for r in all_results:
        line = (f"{r['layer']:5d} {r['proj']:>10s} {r['cos_sim']:12.8f} {r['max_abs_err']:12.6f} "
                f"{r['rmse']:10.6f} {r['snr_db']:8.1f} {r['mean_rel_err']:12.6f} "
                f"{r['p99_rel_err']:12.6f} {r['scale_max']:10.4f} {r['ref_max']:10.4f} "
                f"{r['frac_gt50']:8.4f} {r['frac_gt100']:8.4f}")
        print(line)

    # ── By-projection aggregation ───────────────────────────────────────
    print("\n" + "=" * 100)
    print("AGGREGATED BY PROJECTION TYPE (across all analyzed layers)")
    print("=" * 100)

    proj_agg = defaultdict(list)
    for r in all_results:
        proj_agg[r["proj"]].append(r)

    header2 = f"{'Proj':>10} {'AvgCosSim':>12} {'AvgRMSE':>10} {'AvgSNR':>8} {'MaxScaleMax':>12} {'MaxRefMax':>10} {'WorstCos':>12}"
    print(header2)
    print("-" * 100)

    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if proj_name not in proj_agg:
            continue
        entries = proj_agg[proj_name]
        avg_cos = np.mean([e["cos_sim"] for e in entries])
        avg_rmse = np.mean([e["rmse"] for e in entries])
        avg_snr = np.mean([e["snr_db"] for e in entries])
        max_scale = max(e["scale_max"] for e in entries)
        max_ref = max(e["ref_max"] for e in entries)
        worst_cos = min(e["cos_sim"] for e in entries)
        print(f"{proj_name:>10s} {avg_cos:12.8f} {avg_rmse:10.6f} {avg_snr:8.1f} "
              f"{max_scale:12.4f} {max_ref:10.4f} {worst_cos:12.8f}")

    # ── By-layer aggregation ────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("AGGREGATED BY LAYER (across all projection types)")
    print("=" * 100)

    layer_agg = defaultdict(list)
    for r in all_results:
        layer_agg[r["layer"]].append(r)

    header3 = f"{'Layer':>5} {'AvgCosSim':>12} {'AvgRMSE':>10} {'AvgSNR':>8} {'MaxScaleMax':>12} {'MaxRefMax':>10} {'WorstCos':>12} {'WorstProj':>12}"
    print(header3)
    print("-" * 100)

    for layer_idx in sorted(layer_agg.keys()):
        entries = layer_agg[layer_idx]
        avg_cos = np.mean([e["cos_sim"] for e in entries])
        avg_rmse = np.mean([e["rmse"] for e in entries])
        avg_snr = np.mean([e["snr_db"] for e in entries])
        max_scale = max(e["scale_max"] for e in entries)
        max_ref = max(e["ref_max"] for e in entries)
        worst = min(entries, key=lambda e: e["cos_sim"])
        print(f"{layer_idx:5d} {avg_cos:12.8f} {avg_rmse:10.6f} {avg_snr:8.1f} "
              f"{max_scale:12.4f} {max_ref:10.4f} {worst['cos_sim']:12.8f} {worst['proj']:>12s}")

    # ── Scale analysis deep-dive ────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SCALE ANALYSIS DEEP-DIVE")
    print("=" * 100)

    for r in all_results:
        flags = []
        if r["n_exceed_fp16_max"] > 0:
            flags.append(f"FP16_OVERFLOW({r['n_exceed_fp16_max']})")
        if r["n_very_large"] > 0:
            flags.append(f"LARGE_SCALES({r['n_very_large']})")
        if r["scale_dynamic_range"] > 1000:
            flags.append(f"HIGH_DYN_RANGE({r['scale_dynamic_range']:.0f})")
        if r["n_very_small"] > 0:
            flags.append(f"TINY_SCALES({r['n_very_small']})")
        if r["deq_exceed_fp16"] > 0:
            flags.append(f"DEQ_FP16_OVERFLOW({r['deq_exceed_fp16']})")

        if flags:
            print(f"  Layer {r['layer']:2d} {r['proj']:>10s}: {', '.join(flags)}")
            print(f"    scale_range=[{r['scale_min']:.6f}, {r['scale_max']:.4f}], "
                  f"dyn_range={r['scale_dynamic_range']:.0f}x, "
                  f"ref_max={r['ref_max']:.4f}")

    if not fp16_issues:
        print("  No FP16 overflow issues detected in scales or dequantized values.")

    # ── FP16 precision bottleneck check ─────────────────────────────────
    print("\n" + "=" * 100)
    print("FP16 PRECISION BOTTLENECK ANALYSIS")
    print("=" * 100)

    # FP16 has ~3.3 decimal digits of precision (10-bit mantissa + 1 implicit)
    # If scale * 15 (max q_value * scale) is large, the absolute quantization
    # step is also large, limiting precision
    for r in all_results:
        max_quant_step = r["scale_max"] * 1.0  # 1 quant level = 1 * scale
        # The quantization noise for uniform 4-bit is scale/sqrt(12)
        expected_noise = r["scale_mean"] / (12 ** 0.5)
        actual_noise = r["rmse"]
        ratio = actual_noise / expected_noise if expected_noise > 0 else float('inf')

        # Check if FP16 rounding of scale itself is an issue
        # FP16 mantissa is 10 bits -> relative precision ~2^-10 = 0.001
        fp16_scale_err = r["scale_max"] * (2**-10)
        # This error gets multiplied by q_value (up to 7 for symmetric around 8)
        fp16_induced_err = fp16_scale_err * 7

        if r["layer"] in [0, 1, 2, 57, 58, 59]:
            print(f"  Layer {r['layer']:2d} {r['proj']:>10s}: "
                  f"quant_step={max_quant_step:.6f}  "
                  f"expected_noise={expected_noise:.6f}  "
                  f"actual_RMSE={actual_noise:.6f}  "
                  f"ratio={ratio:.2f}x  "
                  f"fp16_induced_err={fp16_induced_err:.6f}")

    # ── Final diagnosis ─────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("DIAGNOSIS")
    print("=" * 100)

    # Check for layer-dependent trends
    cos_by_layer = {l: np.mean([e["cos_sim"] for e in layer_agg[l]]) for l in sorted(layer_agg.keys())}
    snr_by_layer = {l: np.mean([e["snr_db"] for e in layer_agg[l]]) for l in sorted(layer_agg.keys())}

    worst_layer = min(cos_by_layer, key=cos_by_layer.get)
    best_layer = max(cos_by_layer, key=cos_by_layer.get)
    print(f"  Best layer:  {best_layer} (avg cosine sim = {cos_by_layer[best_layer]:.8f})")
    print(f"  Worst layer: {worst_layer} (avg cosine sim = {cos_by_layer[worst_layer]:.8f})")

    # Check if early/late layers are worse
    early = [r for r in all_results if r["layer"] <= 2]
    late = [r for r in all_results if r["layer"] >= 57]
    middle = [r for r in all_results if 29 <= r["layer"] <= 30]
    if early and late and middle:
        early_cos = np.mean([r["cos_sim"] for r in early])
        late_cos = np.mean([r["cos_sim"] for r in late])
        mid_cos = np.mean([r["cos_sim"] for r in middle])
        print(f"\n  Early layers (0-2)  avg cosine: {early_cos:.8f}")
        print(f"  Middle layers (29-30) avg cosine: {mid_cos:.8f}")
        print(f"  Late layers (57-59)  avg cosine: {late_cos:.8f}")

        if late_cos < early_cos - 0.001:
            print("  -> Late layers have SIGNIFICANTLY worse quantization!")
        elif early_cos < late_cos - 0.001:
            print("  -> Early layers have SIGNIFICANTLY worse quantization!")
        else:
            print("  -> Error is relatively uniform across layers")

    # Check projection-level patterns
    worst_proj_overall = min(all_results, key=lambda r: r["cos_sim"])
    print(f"\n  Globally worst projection: Layer {worst_proj_overall['layer']} "
          f"{worst_proj_overall['proj']} (cosine={worst_proj_overall['cos_sim']:.8f})")

    # Check if any specific projection type is consistently worst
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        entries = proj_agg.get(proj_name, [])
        if entries:
            avg = np.mean([e["cos_sim"] for e in entries])
            if avg < 0.999:
                print(f"  WARNING: {proj_name} has low average cosine similarity: {avg:.8f}")

    # Scale precision summary
    all_scale_max = max(r["scale_max"] for r in all_results)
    all_dyn_range = max(r["scale_dynamic_range"] for r in all_results)
    any_fp16_overflow = any(r["n_exceed_fp16_max"] > 0 for r in all_results)
    any_deq_overflow = any(r["deq_exceed_fp16"] > 0 for r in all_results)

    print(f"\n  Max scale value across all layers: {all_scale_max:.4f}")
    print(f"  Max dynamic range of scales: {all_dyn_range:.0f}x")
    print(f"  Any FP16 scale overflow: {any_fp16_overflow}")
    print(f"  Any dequantized FP16 overflow: {any_deq_overflow}")

    if all_scale_max > FP16_MAX * 0.1:
        print("  -> Scales are approaching FP16 limits - this IS a precision bottleneck!")
    elif all_scale_max > 10:
        print("  -> Scales are moderate - FP16 precision may introduce minor errors")
    else:
        print("  -> Scales are well within FP16 range - NOT a precision bottleneck")

    elapsed = time.time() - t0
    print(f"\nTotal analysis time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
