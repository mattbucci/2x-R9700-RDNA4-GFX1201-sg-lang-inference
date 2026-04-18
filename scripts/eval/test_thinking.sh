#!/bin/bash
# Quick thinking format test for all models.
# Tests whether each model produces <think> tags and extracts answers correctly.
#
# Usage: bash scripts/eval/test_thinking.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"
source scripts/common.sh && activate_conda && setup_rdna4_env

PORT=23334

wait_for_server() {
    for i in $(seq 1 180); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

test_thinking() {
    local preset="$1"
    local tag="$2"

    echo ""
    echo "=== $tag ($preset) ==="

    pkill -9 -f sglang 2>/dev/null || true
    sleep 3

    bash scripts/launch.sh "$preset" > "/tmp/think_test_${preset}.log" 2>&1 &
    local pid=$!

    if ! wait_for_server; then
        echo "  FAILED to start"
        kill $pid 2>/dev/null || true
        return
    fi

    python3 -c "
import requests, re, json

url = 'http://localhost:${PORT}/v1/chat/completions'
prompts = [
    ('simple', 'What is 2+2? A. 3 B. 4 C. 5 D. 6 -- Answer with just the letter:'),
    ('reasoning', 'If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly? A. Yes B. No C. Cannot determine D. Only some -- Answer with just the letter:'),
    ('knowledge', 'What is the capital of Japan? A. Beijing B. Seoul C. Tokyo D. Bangkok -- Answer with just the letter:'),
]
think_count = 0
clean_count = 0
total_tokens = 0

for name, prompt in prompts:
    try:
        r = requests.post(url, json={
            'model': 'default',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 1024, 'temperature': 0,
        }, timeout=120).json()
        content = r['choices'][0]['message']['content'] or ''
        tokens = r['usage']['completion_tokens']
        finish = r['choices'][0]['finish_reason']
        has_tags = '<think>' in content and '</think>' in content
        after = content.split('</think>')[-1].strip() if '</think>' in content else content
        # Strip remaining think content for non-closed tags
        after = re.sub(r'<think>.*', '', after, flags=re.DOTALL).strip()
        letters = re.findall(r'\\b[ABCD]\\b', after)
        clean = len(letters) > 0

        if has_tags: think_count += 1
        if clean: clean_count += 1
        total_tokens += tokens

        status = []
        if has_tags: status.append('THINK')
        if clean: status.append(f'answer={letters[-1] if letters else \"?\"}')
        if finish == 'length': status.append('TRUNCATED')
        print(f'  {name:12s}: {\" \".join(status):30s} ({tokens} tok, {finish})')
    except Exception as e:
        print(f'  {name:12s}: ERROR {e}')

n = len(prompts)
print(f'  ---')
print(f'  Think tags: {think_count}/{n}  Clean answers: {clean_count}/{n}  Avg tokens: {total_tokens//n}')
" 2>&1

    pkill -f sglang 2>/dev/null || true
    sleep 2
}

echo "=== THINKING FORMAT TEST ==="
echo "Tests <think> tag production and answer extraction"
echo ""

test_thinking "devstral"        "Devstral-24B"
test_thinking "coder-30b"       "Coder-30B"
test_thinking "gemma4"          "Gemma4-26B"
test_thinking "gemma4-31b"      "Gemma4-31B"
test_thinking "qwen35"          "Qwen3.5-27B"
test_thinking "qwen35-moe"      "Qwen3.5-35B-MoE"

echo ""
echo "=== DONE ==="
