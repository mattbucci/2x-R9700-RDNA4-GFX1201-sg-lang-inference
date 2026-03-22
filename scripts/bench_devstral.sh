#!/bin/bash
# Benchmark Devstral-Small-2-24B on SGLang with ROCm 7.2
# Tests single-request latency and multi-concurrent throughput
#
# Usage:
#   ./bench_devstral.sh              # Run all tests
#   ./bench_devstral.sh short        # Quick test only (single + 8 concurrent)

# Source common configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="mistralai/Devstral-Small-2-24B-Instruct-2512"

activate_conda

# Wait for server
echo "Checking server at $BASE_URL..."
for i in $(seq 1 60); do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Server not responding at $BASE_URL"
        exit 1
    fi
    sleep 2
done

echo ""
echo "=============================================="
echo "Devstral-Small-2-24B Benchmark"
echo "=============================================="

# Test 1: Single request — measures decode latency (TPOT)
echo ""
echo "--- Test 1: Single request, 256 input, 256 output ---"
python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input 256 \
    --random-output 256 \
    --num-prompts 4 \
    --request-rate 1

# Test 2: 8 concurrent — moderate batch
echo ""
echo "--- Test 2: 8 concurrent, 256 input, 256 output ---"
python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input 256 \
    --random-output 256 \
    --num-prompts 32 \
    --request-rate inf

if [ "$1" = "short" ]; then
    echo ""
    echo "Short test complete."
    exit 0
fi

# Test 3: 16 concurrent — target throughput regime
echo ""
echo "--- Test 3: 16 concurrent, 256 input, 256 output ---"
python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input 256 \
    --random-output 256 \
    --num-prompts 64 \
    --request-rate inf

# Test 4: 32 concurrent — max throughput
echo ""
echo "--- Test 4: 32 concurrent, 256 input, 256 output ---"
python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input 256 \
    --random-output 256 \
    --num-prompts 96 \
    --request-rate inf

# Test 5: Longer context with high concurrency
echo ""
echo "--- Test 5: 16 concurrent, 4K input, 512 output ---"
python -m sglang.bench_serving \
    --backend sglang \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input 4096 \
    --random-output 512 \
    --num-prompts 48 \
    --request-rate inf

echo ""
echo "=============================================="
echo "Benchmark complete"
echo "=============================================="
