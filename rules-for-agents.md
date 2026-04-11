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

## GPTQ Calibration
- Use a **clean conda env** for quantization — llmcompressor/auto-gptq have strict
  dependency requirements that conflict with the main sglang-triton36 env.
- Existing examples: `scripts/quantize_devstral_llmcompressor.sh`, `scripts/quantize_qwen35_llmcompressor.sh`
- Pattern: clean env runs llmcompressor oneshot → compressed-tensors output → convert to AWQ in main env
- Install: `pip install llmcompressor transformers==4.57.6 compressed-tensors accelerate datasets`
- For MoE models with fused expert Parameters (Gemma4TextExperts): monkey-patch to
  per-expert nn.Linear BEFORE loading, otherwise GPTQ skips expert calibration (RTN fallback)
- Dense models (no MoE) need no monkey-patch — all layers are nn.Linear
- CPU-only is fine for calibration: `CUDA_VISIBLE_DEVICES=""`

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

### Degraded Quality
- **Gemma 4 26B MoE**: AWQ working (weight loading fixed), RTN expert quantization causes artifacts. Fix: `scripts/quantize_gemma4_gptq.sh` (unfused expert GPTQ)
- **Gemma 4 31B Dense**: cyankiwi AWQ loads + runs, but RTN quality. Fix: standard GPTQ calibration (no monkey-patch needed, all nn.Linear — easy win)

### Blocked
- **Coder-Next 80B**: Needs GPTQ calibration (~24h with disk offloading).
