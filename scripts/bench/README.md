# Benchmark Scripts

All SGLang benchmarks use `sglang.bench_serving` for proper TPOT/TTFT separation.
Never use wall-clock `completion_tokens / elapsed` — it mixes prefill and decode.

## Primary

| Script | Purpose |
|--------|---------|
| `bench_all_unified.py` | Context sweep + throughput sweep, JSON output |
| `bench_all_baselines.sh` | Comprehensive sweep with JSON output (shell version) |

```bash
# Start model server, then:
python scripts/bench/bench_all_unified.py \
    --name "Coder-30B AWQ" --port 23334 \
    --output benchmarks/coder-30b-awq/results.json
```

Runs two sweeps using `sglang.bench_serving`:
1. **Context sweep** — Single-user TPOT at context 128 to max
2. **Throughput sweep** — Concurrent throughput at 1/2/4/8/16/32

## SGLang Utilities

| Script | Purpose |
|--------|---------|
| `bench_comprehensive.sh` | Shell wrapper using `sglang.bench_serving` (256 in / 256 out) |
| `bench_all_models.sh` | Launches each model server sequentially and benchmarks all |
| `bench_quick.sh` | Fast 3-point check (1/8/16 concurrent) for A/B testing patches |
| `bench_long_context.py` | Context-length TPOT sweep via `sglang.bench_serving` |
| `bench_regression.sh` | Regression detection vs baselines |

## Comparison Engines

| Script | Purpose |
|--------|---------|
| `bench_vllm_docker.sh` | vLLM ROCm Docker (FP8, streaming TPOT) |
| `bench_llamacpp.sh` | llama.cpp Vulkan (`llama-bench` pp256/tg256, 2-GPU layer split) |
