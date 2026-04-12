#!/bin/bash
# Comprehensive benchmark for all models: context sweep + concurrency sweep.
# Uses sglang.bench_serving for accurate TPOT measurement at each point.
# Saves results.json per model, baselines.json, and regenerates charts.
#
# Usage: ./scripts/bench/bench_all_baselines.sh           # All models
#        ./scripts/bench/bench_all_baselines.sh devstral   # One model
#
# Takes 30-60 minutes per model depending on context range.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
source "$REPO_DIR/scripts/common.sh"
activate_conda
setup_rdna4_env

PORT=23334
BASE_URL="http://localhost:$PORT"
BASELINES="$REPO_DIR/benchmarks/baselines.json"
MODEL_FILTER="${1:-all}"

# Model configs: bench_key -> (launch_key, results_dir, max_context, concurrency_levels)
declare -A MODEL_LAUNCH=(
    ["devstral"]="devstral"
    ["coder-30b"]="coder-30b"
    ["gemma4"]="gemma4"
    ["coder-next"]="coder-next"
    ["coder-next-ream"]="coder-next-ream"
    ["glm45-air"]="glm45-air"
    ["qwen35"]="qwen35"
)
declare -A MODEL_DIR=(
    ["devstral"]="devstral-24b-awq"
    ["coder-30b"]="coder-30b-awq"
    ["gemma4"]="gemma4-26b-awq"
    ["coder-next"]="coder-next-80b-awq"
    ["coder-next-ream"]="coder-next-ream-60b-awq"
    ["glm45-air"]="glm45-air-82b-awq"
    ["qwen35"]="qwen35-27b-awq"
)
declare -A MODEL_MAX_CTX=(
    ["devstral"]=32768
    ["coder-30b"]=32768
    ["gemma4"]=4096
    ["coder-next"]=8192
    ["coder-next-ream"]=32768
    ["glm45-air"]=32768
    ["qwen35"]=262144
)
declare -A MODEL_CONC_LEVELS=(
    ["devstral"]="1 2 4 8 16 32 64"
    ["coder-30b"]="1 2 4 8 16 32"
    ["gemma4"]="1 2 4 8 16 32"
    ["coder-next"]="1 2 4 8"
    ["coder-next-ream"]="1 2 4 8 16"
    ["glm45-air"]="1 2 4 8 16 32"
    ["qwen35"]="1 2 4 8"
)

OUTPUT_TOKENS=50

launch_and_wait() {
    local model="$1"
    echo "  Launching $model..."
    bash "$REPO_DIR/scripts/launch.sh" "$model" &>/dev/null &

    for i in $(seq 1 90); do
        if curl -s "$BASE_URL/v1/models" > /dev/null 2>&1; then
            echo "  Server ready after $((i*3))s"
            return 0
        fi
        sleep 3
    done
    echo "  ERROR: Server failed to start"
    return 1
}

warm_cache() {
    echo "  Warming Triton cache (sequential, short outputs)..."
    for i in $(seq 1 5); do
        curl -s --max-time 180 "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Warmup $i: what is $i + $i?\"}],\"max_tokens\":20,\"temperature\":0}" > /dev/null 2>&1 || true
        sleep 1
    done
    echo "  Cache warm."
}

get_served_model() {
    curl -s "$BASE_URL/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null
}

run_bench_serving() {
    local served_model="$1" input_len="$2" output_len="$3" num_prompts="$4" rate="$5"
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$served_model" \
        --dataset-name random \
        --random-input "$input_len" \
        --random-output "$output_len" \
        --num-prompts "$num_prompts" \
        --request-rate "$rate" \
        --disable-ignore-eos \
        --disable-tqdm 2>&1
}

extract() {
    local output="$1" field="$2"
    echo "$output" | grep "$field" | awk '{print $NF}' | sed 's/ms//'
}

context_sweep() {
    local key="$1" served_model="$2" max_ctx="$3"
    echo "  --- Context sweep (single-user, $OUTPUT_TOKENS output tokens) ---" >&2

    # Generate context lengths: powers of 2 from 128 to max_ctx
    local ctx=128
    local results="["
    local first=true
    while [ "$ctx" -le "$max_ctx" ]; do
        # Input length is half the context target (output fills the rest)
        local input_len=$((ctx / 2))
        [ "$input_len" -lt 32 ] && input_len=32

        local result
        result=$(run_bench_serving "$served_model" "$input_len" "$OUTPUT_TOKENS" 1 1 || true)
        local tpot throughput ttft
        tpot=$(extract "$result" "Mean TPOT")
        throughput=$(extract "$result" "Output token throughput")
        ttft=$(extract "$result" "Mean TTFT")

        if [ -n "$tpot" ] && [ "$tpot" != "0" ]; then
            local toks_per_sec
            toks_per_sec=$(python3 -c "print(round(1000.0 / float('$tpot'), 1))" 2>/dev/null)
            echo "    ctx=${ctx}: input=${input_len} TPOT=${tpot}ms = ${toks_per_sec} tok/s  TTFT=${ttft}ms" >&2

            [ "$first" = true ] && first=false || results+=","
            results+="{\"context\":$ctx,\"input_len\":$input_len,\"tpot_ms\":$tpot,\"tok_per_sec\":$toks_per_sec,\"ttft_ms\":$ttft,\"throughput\":$throughput}"
        else
            echo "    ctx=${ctx}: FAILED or server down" >&2
            # If server crashed, stop sweep
            if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
                echo "    Server down, stopping context sweep" >&2
                break
            fi
        fi

        # Next power of 2
        ctx=$((ctx * 2))
    done
    results+="]"
    echo "$results"
}

concurrency_sweep() {
    local key="$1" served_model="$2" conc_levels="$3"
    echo "  --- Concurrency sweep (128 input, $OUTPUT_TOKENS output) ---" >&2

    local results="["
    local first=true
    for conc in $conc_levels; do
        local num_prompts=$((conc * 2))
        [ "$num_prompts" -lt 4 ] && num_prompts=4

        local result
        result=$(run_bench_serving "$served_model" 128 "$OUTPUT_TOKENS" "$num_prompts" inf || true)
        local tpot throughput ttft
        tpot=$(extract "$result" "Mean TPOT")
        throughput=$(extract "$result" "Output token throughput")
        ttft=$(extract "$result" "Mean TTFT")

        if [ -n "$throughput" ] && [ "$throughput" != "0" ]; then
            echo "    conc=${conc}: total=${throughput} tok/s  TPOT=${tpot}ms  TTFT=${ttft}ms" >&2

            [ "$first" = true ] && first=false || results+=","
            results+="{\"concurrency\":$conc,\"tok_per_sec\":$throughput,\"tpot_ms\":$tpot,\"ttft_ms\":$ttft}"
        else
            echo "    conc=${conc}: FAILED (OOM or crash)" >&2
            [ "$first" = true ] && first=false || results+=","
            results+="{\"concurrency\":$conc,\"tok_per_sec\":0,\"error\":\"OOM\"}"

            if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
                echo "    Server down, stopping concurrency sweep" >&2
                break
            fi
        fi
    done
    results+="]"
    echo "$results"
}

bench_model() {
    local key="$1"
    local served_model max_ctx conc_levels results_dir
    served_model=$(get_served_model)
    max_ctx="${MODEL_MAX_CTX[$key]}"
    conc_levels="${MODEL_CONC_LEVELS[$key]}"
    results_dir="$REPO_DIR/benchmarks/${MODEL_DIR[$key]}"
    mkdir -p "$results_dir"

    echo "  Model: $served_model"
    echo "  Max context: $max_ctx"
    echo ""

    # Context sweep
    local ctx_json
    ctx_json=$(context_sweep "$key" "$served_model" "$max_ctx")
    # Last line is the JSON
    local ctx_results
    ctx_results=$(echo "$ctx_json" | tail -1)

    echo ""

    # Concurrency sweep
    local conc_json
    conc_json=$(concurrency_sweep "$key" "$served_model" "$conc_levels")
    local conc_results
    conc_results=$(echo "$conc_json" | tail -1)

    # Save results.json
    python3 -c "
import json, time
results = {
    'model': '$key',
    'engine': 'SGLang',
    'hardware': '2x R9700 TP=2',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'method': 'sglang.bench_serving (TPOT-based)',
    'output_tokens': $OUTPUT_TOKENS,
    'context_sweep': json.loads('''$ctx_results'''),
    'throughput_sweep': json.loads('''$conc_results'''),
}
with open('$results_dir/results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('  Saved $results_dir/results.json')
" 2>/dev/null

    # Save baseline
    local s_tpot s_tp
    s_tpot=$(python3 -c "
import json
ctx = json.loads('''$ctx_results''')
if ctx: print(ctx[0].get('tpot_ms', 0))
else: print(0)
" 2>/dev/null)
    s_tp=$(python3 -c "
import json
ctx = json.loads('''$ctx_results''')
if ctx: print(ctx[0].get('tok_per_sec', 0))
else: print(0)
" 2>/dev/null)
    local m_tp
    m_tp=$(python3 -c "
import json
conc = json.loads('''$conc_results''')
best = max((c.get('tok_per_sec',0) for c in conc), default=0)
print(best)
" 2>/dev/null)

    python3 -c "
import json, os
path = '$BASELINES'
baselines = {}
if os.path.exists(path):
    with open(path) as f:
        baselines = json.load(f)
baselines['$key'] = {
    'single_tpot_ms': float('${s_tpot:-0}'),
    'single_throughput': float('${s_tp:-0}'),
    'peak_throughput': float('${m_tp:-0}'),
}
with open(path, 'w') as f:
    json.dump(baselines, f, indent=2)
" 2>/dev/null
}

kill_server() {
    pkill -9 -f "python.*sglang" 2>/dev/null || true
    sleep 15
}

# Determine which models to run
if [ "$MODEL_FILTER" = "all" ]; then
    MODELS_TO_RUN=("devstral" "coder-30b" "gemma4" "coder-next" "coder-next-ream" "glm45-air" "qwen35")
else
    MODELS_TO_RUN=("$MODEL_FILTER")
fi

echo "============================================"
echo "RDNA4 Comprehensive Benchmark"
echo "Method: sglang.bench_serving (TPOT-based)"
echo "Models: ${MODELS_TO_RUN[*]}"
echo "============================================"

for model in "${MODELS_TO_RUN[@]}"; do
    echo ""
    echo "========================================"
    echo "=== $model ==="
    echo "========================================"
    kill_server
    if launch_and_wait "$model"; then
        warm_cache
        bench_model "$model"
    else
        echo "  SKIPPED (server failed)"
    fi
done

kill_server

echo ""
echo "============================================"
echo "Baselines:"
echo "============================================"
cat "$BASELINES" | python3 -m json.tool 2>/dev/null

echo ""
echo "Regenerating charts..."
python "$REPO_DIR/scripts/bench/generate_charts.py" 2>&1 || echo "(chart generation had errors)"

echo ""
echo "Done! Review results in benchmarks/*/results.json"
