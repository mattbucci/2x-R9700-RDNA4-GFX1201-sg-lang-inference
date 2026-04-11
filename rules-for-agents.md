# Rules for AI Agents

## Hardware
- 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4, Wave32)
- 32 GB VRAM each, ROCm 7.2.1, Arch Linux
- Consumer GPUs — NOT MI-series/CDNA. AITER not available.

## RDNA4 Constraints

### Triton
- Triton 3.6 generates valid gfx1201 ISA but is fragile in multi-kernel context
- AWQ GEMM uses dequant+matmul (no Triton) for M>1, HIP GEMV for M=1
- sgl_kernel.topk_softmax crashes — replaced with torch-native topk in topk.py
- FP8 WMMA instruction works but Arch comgr generates invalid HSACO for FP8 kernels

### Server Launch
- Always: `--disable-cuda-graph --disable-custom-all-reduce --disable-overlap-schedule`
- Always: `--attention-backend triton` (or `torch_native` as fallback)
- Always: `PYTHONDONTWRITEBYTECODE=1` and clear `__pycache__` before testing changes
- `@torch.compile` stalls 30+ min on ROCm — disabled in patches

### GPU Recovery
- After hang/reset, wait 10-15 seconds before retrying
- Check `dmesg` for amdgpu reset messages

## Benchmarking
- Concurrency sweep: 1, 4, 8, 16, 32
- Save to `benchmarks/` as `{model}_{quant}_{engine}.txt` with full env header
- Run `scripts/eval_comprehensive.py` after kernel changes
- Always use timeouts on GPU/Docker commands

## Model Status

### Working
| Model | Engine | tok/s @32 |
|-------|--------|-----------|
| Devstral-24B AWQ | SGLang | 841 |
| Qwen3.5-27B AWQ | SGLang | 129 |
| Coder-30B AWQ MoE | SGLang | 169 |
| Coder-30B FP8 | vLLM Docker | 1185 |

### Blocked
- **Gemma 4 26B**: GPTQ weights generated, model loads but outputs empty tokens. Weight loader investigation needed.
- **Coder-Next 80B**: Needs GPTQ calibration (~24h with disk offloading).
