# AWQ Triton Kernel Analysis on RDNA4 (gfx1201)

## Key Finding

**Triton IS using WMMA instructions.** The compiled AWQ GEMM kernel uses
`v_wmma_f32_16x16x16_f16` — the native gfx12 wave-matrix multiply-accumulate
instruction. This is the same instruction hipBLASLt uses for BF16/FP16 GEMM.

## Instruction Breakdown (918 total)

| Category | Count | Purpose |
|----------|-------|---------|
| WMMA (compute) | 8 | `v_wmma_f32_16x16x16_f16` — actual matrix multiply |
| Integer ALU | 128 | AWQ dequantization (shift, mask, subtract zeros) |
| FP16 ops | 128 | Scale multiplication after dequant |
| Memory loads | 87 | Loading weights, scales, zeros, inputs |
| Shared mem (LDS) | 80 | Data staging between global and register |
| Scalar ops | 405 | Address computation, loop control |
| Wait/sync | 113 | Synchronization barriers |
| Memory stores | 1 | Final result store |

## Analysis

The dequantization overhead (128 int + 128 fp16 = 256 ALU ops) vastly outnumbers
the actual compute (8 WMMA = 32,768 multiply-accumulate ops). This is the
fundamental cost of AWQ-4bit — for every WMMA, we do ~32 dequant operations.

hipBLASLt with BF16 skips ALL dequantization — it just loads and WMMAs.
That's why BF16 is ~10% faster despite reading 4x more data.

## Kernel Config

- `num_stages=2, num_warps=4, warp_size=32`
- Block sizes: 32x64x64 (RDNA4-tuned via sweep_awq_blocks.py)
- Split_k: 8 for M<=16 (decode), 4 for M>16 (batched)
- Accumulator: FP16 (matches output buffer dtype)
- v_bfe_u32 used for bitfield extraction: 48 instances

## Comparison with vLLM

vLLM on ROCm uses the **exact same Triton kernel** (awq_gemm_triton from vllm-project).
There is NO native C++ AWQ kernel for ROCm. vLLM's 27ms benchmark used BF16
(unquantized) model via hipBLASLt, not AWQ.

If vLLM loaded our AWQ model, it would use the same Triton kernel and get
the same ~30ms TPOT.

## A/B Test: Fused Triton vs Dequant+hipBLASLt (2026-03-31)

| Path | TPOT | Method |
|------|------|--------|
| **Fused Triton GEMM** | **30.1ms** | Dequant in registers → WMMA, no VRAM round-trip |
| Dequant + hipBLASLt | 114ms | Triton dequant → write FP16 to VRAM → hipBLASLt read + GEMM |
| vLLM BF16 (no quant) | 27ms | hipBLASLt GEMM directly on BF16 weights |

**Conclusion:** The fused Triton GEMM is 3.8x faster than dequant+hipBLASLt because it
avoids the memory round-trip. The 30ms fused TPOT is only 3ms slower than BF16 (27ms),
which is the irreducible cost of AWQ dequantization compute (256 ALU ops per 8 WMMA ops).

The fused kernel is the optimal AWQ path on RDNA4. Further optimization would require:
1. Reducing dequant ALU ops (e.g., precomputed lookup tables)
2. Better Triton codegen for the shift/mask/scale pattern
3. Native HIP AWQ GEMM kernel with hand-optimized dequant + WMMA interleaving
