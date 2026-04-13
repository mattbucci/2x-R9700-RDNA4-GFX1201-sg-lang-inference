#!/usr/bin/env python3
"""Benchmark decode speed at various context lengths using sglang.bench_serving.

Measures TPOT (decode latency) and TTFT (prefill latency) separately at each
context length. Uses sglang.bench_serving for proper measurement.

Usage:
    python bench_long_context.py --port 23334
    python bench_long_context.py --port 23334 --output-tokens 50 --max-context 131072
"""
import argparse
import json
import re
import subprocess
import sys
import urllib.request


def run_bench_serving(port, model, input_len, output_len):
    """Run sglang.bench_serving and return parsed metrics."""
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", f"http://localhost:{port}",
        "--model", model,
        "--dataset-name", "random",
        "--random-input", str(input_len),
        "--random-output", str(output_len),
        "--num-prompts", "1",
        "--request-rate", "1",
        "--disable-ignore-eos",
        "--disable-tqdm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    def extract(field):
        m = re.search(rf"{field}:\s+([\d.]+)", output)
        return float(m.group(1)) if m else None

    return {
        "tpot_ms": extract("Mean TPOT"),
        "ttft_ms": extract("Mean TTFT"),
        "throughput": extract("Output token throughput"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23334)
    parser.add_argument("--output-tokens", type=int, default=50)
    parser.add_argument("--max-context", type=int, default=262144)
    args = parser.parse_args()

    # Check health
    try:
        with urllib.request.urlopen(f"http://localhost:{args.port}/health", timeout=5):
            pass
    except Exception as e:
        print(f"Server not ready: {e}")
        sys.exit(1)

    # Get model info
    try:
        with urllib.request.urlopen(f"http://localhost:{args.port}/v1/models", timeout=5) as r:
            models = json.loads(r.read())
        model = models["data"][0]["id"]
    except Exception:
        model = "default"

    out = args.output_tokens
    print(f"Model: {model}")
    print(f"Output tokens per test: {out}")
    print(f"Method: sglang.bench_serving (TPOT-based)")
    print(f"{'Context':>10s}  {'Input':>7s}  {'TPOT':>8s}  {'tok/s':>7s}  {'TTFT':>8s}")
    print("-" * 50)

    tests = [
        (128, "128"),
        (512, "512"),
        (1024, "1K"),
        (2048, "2K"),
        (4096, "4K"),
        (8192, "8K"),
        (16384, "16K"),
        (32768, "32K"),
        (65536, "64K"),
        (131072, "128K"),
        (262144, "256K"),
    ]

    results = []
    for input_len, label in tests:
        if input_len > args.max_context:
            break
        try:
            m = run_bench_serving(args.port, model, input_len, out)
            tpot = m["tpot_ms"]
            ttft = m["ttft_ms"]
            if tpot and tpot > 0:
                tok_s = round(1000.0 / tpot, 1)
                print(f"  {label:>8s}  {input_len:>7d}  {tpot:>6.1f}ms  {tok_s:>6.1f}  {ttft:>6.1f}ms")
                results.append({"context": input_len, "label": label,
                                "tpot_ms": tpot, "tok_per_sec": tok_s, "ttft_ms": ttft})
            else:
                print(f"  {label:>8s}  FAILED (no TPOT returned)")
        except Exception as e:
            print(f"  {label:>8s}  ERROR: {str(e)[:60]}")

    # Summary
    if results:
        print(f"\n{'='*50}")
        best = min(results, key=lambda r: r["tpot_ms"])
        worst = max(results, key=lambda r: r["tpot_ms"])
        print(f"Best:  {best['label']} — {best['tok_per_sec']} tok/s (TPOT {best['tpot_ms']:.1f}ms)")
        print(f"Worst: {worst['label']} — {worst['tok_per_sec']} tok/s (TPOT {worst['tpot_ms']:.1f}ms)")


if __name__ == "__main__":
    main()
