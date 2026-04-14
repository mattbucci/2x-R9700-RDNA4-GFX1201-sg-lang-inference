#!/bin/bash
# Quantize Gemma 4 31B with AutoRound → AWQ format
#
# Uses Intel AutoRound (SignSGD 200 iterations) to produce properly
# calibrated AWQ-format weights. This avoids the GPTQ→AWQ conversion
# path that produced NaN on RDNA4 due to intermediate overflow.
#
# Requires: awq-quant conda env with auto-round installed
# Runtime: ~2-4 hours on 2x R9700 (30 GB each)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
INPUT_MODEL="${INPUT_MODEL:-$MODELS_DIR/gemma-4-31B-it-BF16}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_DIR/gemma-4-31B-it-AutoRound-AWQ-native}"

if [ ! -d "$INPUT_MODEL" ]; then
    echo "ERROR: Input model not found: $INPUT_MODEL"
    exit 1
fi

echo "=============================================="
echo "AutoRound AWQ Quantization — Gemma 4 31B"
echo "Input:  $INPUT_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Format: auto_awq (W4A16, group_size=128, asymmetric)"
echo "=============================================="

init_conda
conda activate "${ENV_NAME:-sglang-triton36}"

# Set GPU visibility
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1}
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# AutoRound quantization
# --asym: asymmetric quantization (zero_point=True, needed for AWQ)
# --iters 200: SignSGD iterations per block
# --nsamples 128: calibration samples
# --seqlen 2048: calibration sequence length
# --format auto_awq: native AWQ output format
# --to_quant_block_names model.layers: only quantize text decoder layers
# --low_gpu_mem_usage: reduce VRAM during calibration
python -m auto_round \
    --model "$INPUT_MODEL" \
    --bits 4 \
    --group_size 128 \
    --asym \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --format "auto_awq" \
    --output_dir "$OUTPUT_DIR" \
    --to_quant_block_names "model.layers" \
    --low_gpu_mem_usage \
    --device_map auto \
    --batch_size 4

echo ""
echo "Done! Output: $OUTPUT_DIR"
echo "Test with: scripts/launch.sh gemma4-31b"
