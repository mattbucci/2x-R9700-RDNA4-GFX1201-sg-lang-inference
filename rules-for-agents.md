# Rules for AI Agents Working on This Repo

## Hardware Context
- 2x AMD Radeon AI PRO R9700 (gfx1201, RDNA4, Wave32)
- 32 GB VRAM each (64 GB total across TP=2)
- ROCm 7.2.1, Arch Linux
- Consumer GPUs — NOT MI-series/CDNA. AITER is NOT available.

## Critical RDNA4 Constraints

### Triton Kernels
- Triton 3.6.0 can generate valid gfx1201 ISA but is fragile
- **Extra constexpr parameters cause GPU hangs**: SGLang's `swap_ab`, `c_sorted`, `FUSE_SUM_ALL_REDUCE`, `ROUTER_TOPK`, `a_desc`, `b_desc` all trigger Triton codegen bugs
- vLLM's simpler kernel (fewer constexpr) compiles and runs correctly
- Use `BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=2` as the known-good config from vLLM
- FP8 `v_wmma_f32_16x16x16_fp8_fp8` instruction works on RDNA4

### torch.compile
- `@torch.compile` stalls for 30+ min on ROCm. Disable with `disable=_is_hip` or `disable=is_hip()`
- Do NOT set `TORCHDYNAMO_DISABLE=1` globally — it breaks multiprocessing spawn

### Server Launch
- Always use `--disable-cuda-graph` (not supported on RDNA4)
- Always use `--disable-custom-all-reduce` (use NCCL instead)
- Use `--disable-overlap-schedule` for FP8 MoE models
- Use `--attention-backend triton` for AWQ models (proven working)
- Do NOT background-wait for startup — check health inline with error detection

### Stale .pyc Cache
- Multiprocessing spawn loads old bytecode despite source changes
- Always delete `__pycache__` AND `.pyc` files before testing changes:
  ```bash
  find components/sglang -name "__pycache__" -type d -exec rm -rf {} +
  find components/sglang -name "*.pyc" -delete
  ```
- Set `PYTHONDONTWRITEBYTECODE=1` in all launch scripts

## Benchmarking Rules

### Always Include
- Single request latency (200 tokens)
- Concurrency sweep: 1, 2, 4, 8, 16, 32
- Report tok/s for each concurrency level
- Context length used (default 4096, long context tests at 256K)
- Clear environment description (engine version, torch version, env name)

### Environment Description Must Include
- Whether using custom patched SGLang or official
- Exact torch version and ROCm version
- Which patches are active (list key ones)
- TP size and attention backend

### Quality Checks
- Run `scripts/eval_comprehensive.py` after kernel changes
- Run `scripts/test_tp2_quality.py` for TP=2 correctness
- Verify output is not garbage/repetitive before benchmarking

### Result Files
- Save to `benchmarks/` directory
- Use format: `{model}_{quant}_{engine}.txt`
- Include full hardware/software description in header
- Include date

## Command Execution Rules

### Always Use Timeouts
- All Docker commands: `--stop-timeout 300` or wrapper `timeout 300`
- All Triton compilation: `timeout 600000` (10 min max)
- All server launches: explicit health check loop with max iterations
- Never use `block=true` with unlimited timeout for GPU operations
- Background tasks: always set `timeout` parameter

### GPU Recovery
- After GPU hang/reset, wait 10-15 seconds before retrying
- Verify GPU health with `torch.cuda.device_count()` before launching
- Check `dmesg` for GPU reset messages to confirm recovery

## Model Status

### Working on SGLang
- Devstral-24B AWQ: 858 tok/s @ 32 concurrent
- Qwen3.5-27B AWQ: 129 tok/s @ 32 concurrent (bandwidth-limited)

### Working on vLLM Docker
- Qwen3-Coder-30B FP8: 1185 tok/s @ 32 concurrent
- Use `vllm/vllm-openai-rocm:gemma4` Docker image

### Blocked
- FP8 MoE on SGLang: GPU hang from Triton kernel (torch-native fallback available via `SGLANG_RDNA4_TORCH_MOE=1`)
- Gemma 4: Infrastructure fixed (mixed head_dim, SWAKVPool, K=V attention), but community AWQ weights produce garbage (verified in both SGLang and vLLM). Need properly calibrated GPTQ weights.
