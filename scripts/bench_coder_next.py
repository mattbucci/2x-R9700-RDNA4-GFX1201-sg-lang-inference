#!/usr/bin/env python3
"""Benchmark Qwen3-Coder-Next-80B AWQ on SGLang.

Measures single-user latency at various context lengths and throughput
at various concurrency levels.
"""
import argparse
import json
import time
import sys
import os
import concurrent.futures
import requests

PORT = int(os.environ.get("PORT", 23334))
BASE = f"http://localhost:{PORT}"
MODEL = "coder-next"

# Warm up prompt (short code task)
WARMUP_PROMPT = "Write hello world in Python."
# Benchmark prompt template — pad with context
CODE_PROMPT = "Write a Python function that {task}. Only output code, no explanation."
TASKS = [
    "checks if a number is prime",
    "reverses a linked list",
    "implements binary search",
    "finds the longest common subsequence",
    "implements a basic stack using a list",
]


def generate(prompt: str, max_tokens: int = 100, timeout: int = 300) -> dict:
    """Send a generate request and return timing info."""
    start = time.time()
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=timeout,
    )
    elapsed = time.time() - start
    data = r.json()
    choice = data["choices"][0]
    usage = data["usage"]
    return {
        "elapsed": elapsed,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "content": choice["message"]["content"] or "",
        "finish_reason": choice["finish_reason"],
    }


def bench_single_user_context(context_lengths, output_tokens=100):
    """Benchmark single-user latency at various context lengths."""
    print(f"\n{'='*60}")
    print(f"Single-User Latency (output={output_tokens} tokens)")
    print(f"{'='*60}")
    print(f"{'Context':>10} {'Time':>8} {'Prompt':>8} {'Compl':>8} {'tok/s':>8} {'TPOT':>10}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    results = []
    for ctx_len in context_lengths:
        # Build prompt with padding to reach target context length
        # Each "word" is roughly 1.3 tokens; use repetitive filler
        base_prompt = f"Write a Python function that checks if a number is prime."
        filler = "x " * max(0, (ctx_len - 20) // 2)  # rough token estimate
        if ctx_len > 200:
            prompt = f"Context: {filler}\n\nNow, {base_prompt} Only output code."
        else:
            prompt = base_prompt

        try:
            r = generate(prompt, max_tokens=output_tokens, timeout=600)
            tps = r["completion_tokens"] / r["elapsed"] if r["elapsed"] > 0 else 0
            # TPOT = (elapsed - estimated_prefill) / completion_tokens
            # Rough: prefill ~= elapsed * (prompt_tokens / (prompt_tokens + completion_tokens * 10))
            tpot_ms = (r["elapsed"] / r["completion_tokens"] * 1000) if r["completion_tokens"] > 0 else 0

            print(f"{ctx_len:>10} {r['elapsed']:>7.1f}s {r['prompt_tokens']:>8} {r['completion_tokens']:>8} {tps:>7.1f} {tpot_ms:>9.1f}ms")
            results.append({
                "context_length": ctx_len,
                "elapsed": round(r["elapsed"], 2),
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "tok_per_sec": round(tps, 1),
                "tpot_ms": round(tpot_ms, 1),
            })
        except Exception as e:
            print(f"{ctx_len:>10} {'ERROR':>8} — {e}")
            results.append({"context_length": ctx_len, "error": str(e)})

    return results


def bench_throughput(concurrency_levels, prompt_tokens=128, output_tokens=200):
    """Benchmark throughput at various concurrency levels."""
    print(f"\n{'='*60}")
    print(f"Throughput (input≈{prompt_tokens}, output={output_tokens} tokens)")
    print(f"{'='*60}")
    print(f"{'Concur':>8} {'Time':>8} {'Total':>8} {'tok/s':>8} {'Avg TPOT':>10}")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    results = []
    for conc in concurrency_levels:
        prompts = [
            CODE_PROMPT.format(task=TASKS[i % len(TASKS)])
            for i in range(conc)
        ]

        start = time.time()
        total_tokens = 0
        errors = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
            futures = [
                pool.submit(generate, p, output_tokens, 600)
                for p in prompts
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    r = f.result()
                    total_tokens += r["completion_tokens"]
                except Exception:
                    errors += 1

        elapsed = time.time() - start
        tps = total_tokens / elapsed if elapsed > 0 else 0
        avg_tpot = (elapsed / (total_tokens / conc) * 1000) if total_tokens > 0 else 0

        status = f" ({errors} err)" if errors else ""
        print(f"{conc:>8} {elapsed:>7.1f}s {total_tokens:>8} {tps:>7.1f} {avg_tpot:>9.1f}ms{status}")
        results.append({
            "concurrency": conc,
            "elapsed": round(elapsed, 2),
            "total_tokens": total_tokens,
            "tok_per_sec": round(tps, 1),
            "avg_tpot_ms": round(avg_tpot, 1),
            "errors": errors,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--output", default=None, help="Save JSON results to file")
    args = parser.parse_args()

    global PORT, BASE
    PORT = args.port
    BASE = f"http://localhost:{PORT}"

    # Verify server is up
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        assert r.status_code == 200
    except Exception:
        print(f"Server not responding on port {PORT}")
        sys.exit(1)

    print(f"Benchmarking Qwen3-Coder-Next-80B AWQ on port {PORT}")

    # Warmup
    print("\nWarming up...")
    for _ in range(2):
        generate(WARMUP_PROMPT, max_tokens=20, timeout=120)
    print("Warmup done.")

    # Single-user context length sweep
    context_results = bench_single_user_context(
        [128, 512, 1024, 2048, 4096, 8192],
        output_tokens=100,
    )

    # Throughput sweep
    throughput_results = bench_throughput(
        [1, 2, 4, 8],
        prompt_tokens=128,
        output_tokens=200,
    )

    # Save results
    all_results = {
        "model": "Qwen3-Coder-Next-80B AWQ",
        "engine": "SGLang",
        "hardware": "2x R9700 TP=2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "context_sweep": context_results,
        "throughput_sweep": throughput_results,
    }

    out_path = args.output or f"benchmarks/coder_next_awq_sglang.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
