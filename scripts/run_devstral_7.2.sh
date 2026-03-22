#!/bin/bash
# Run Devstral-Small-2-24B FP8 — stock SGLang + system RCCL + triton 3.6
# 2x Radeon AI PRO R9700 (gfx1201)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

MODEL="mistralai/Devstral-Small-2-24B-Instruct-2512"

activate_conda
setup_rdna4_env
setup_rccl

echo "=============================================="
echo "Devstral-24B FP8 (stock SGLang, system RCCL, triton 3.6)"
echo "PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "Triton  $(python -c 'import triton; print(triton.__version__)')"
echo "=============================================="

# FP8 GEMM via hipBLASLt
export USE_TRITON_W8A8_FP8_KERNEL=0

exec python -m sglang.launch_server \
    --model-path $MODEL \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --context-length 262144 \
    --mem-fraction-static 0.70 \
    --cuda-graph-bs 1 2 4 \
    --max-running-requests 4 \
    --chunked-prefill-size 8192 \
    --disable-radix-cache \
    --attention-backend triton \
    --num-continuous-decode-steps 32 \
    --disable-custom-all-reduce \
    --port "$PORT" \
    --host 0.0.0.0 \
    --enable-metrics
