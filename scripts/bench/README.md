# Benchmark Scripts

## Primary

| Script | Purpose |
|--------|---------|
| `bench_all_unified.py` | Primary benchmark — context sweep + throughput sweep, JSON output |

```bash
# Start model server, then:
python scripts/bench/bench_all_unified.py \
    --name "Coder-30B AWQ" --port 23334 \
    --output benchmarks/coder-30b-awq/results.json
```

Runs two sweeps:
1. **Context sweep** — Single-user, 100 output tokens, context 128 to max
2. **Throughput sweep** — Concurrency 1/2/4/8/16/32, 200 output tokens each

## SGLang Utilities

| Script | Purpose |
|--------|---------|
| `bench_comprehensive.sh` | Shell wrapper using `sglang.bench_serving` (256 in / 256 out) |
| `bench_all_models.sh` | Launches each model server sequentially and benchmarks all |
| `bench_quick.sh` | Fast 3-point check (1/8/16 concurrent) for A/B testing patches |
| `bench_long_context.py` | Context-length sweep via `/v1/completions` endpoint |

## Comparison Engines

| Script | Purpose |
|--------|---------|
| `bench_vllm_docker.sh` | vLLM ROCm Docker (FP8, `sglang.bench_serving`) |
| `bench_llamacpp.sh` | llama.cpp Vulkan (`llama-bench` pp256/tg256, 2-GPU layer split) |
