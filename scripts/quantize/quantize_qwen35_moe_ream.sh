#!/bin/bash
# Quantize Qwen3.5-35B-A3B REAM/REAP to AWQ 4-bit
#
# Full pipeline: GPTQ calibration (DeltaNet-aware) + CT->AWQ conversion
#
# DeltaNet hybrid MoE: DeltaNet gate projections and recurrent layers
# stay in BF16. Only MoE expert MLPs + attention projections are INT4.
#
# Prerequisites (one-time, in sglang-triton36 env):
#   pip install git+https://github.com/vllm-project/llm-compressor.git --no-deps
#   pip install git+https://github.com/neuralmagic/compressed-tensors.git --no-deps
#
# Usage:
#   ./scripts/quantize/quantize_qwen35_moe_ream.sh
#   MODEL_SRC=~/AI/models/Qwen-3.5-28B-A3B-REAP ./scripts/quantize/quantize_qwen35_moe_ream.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

QUANT_ENV="${QUANT_ENV:-sglang-triton36}"
QUANT_PYTHON="$CONDA_BASE/envs/$QUANT_ENV/bin/python"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"

# Source model (REAM'd or REAP'd BF16)
MODEL_SRC="${MODEL_SRC:-$MODELS_DIR/Qwen3.5-35B-A3B-REAM-BF16}"
MODEL_NAME="$(basename "$MODEL_SRC")"

CT_OUTPUT="$MODELS_DIR/${MODEL_NAME}-AWQ-CT"
AWQ_OUTPUT="$MODELS_DIR/${MODEL_NAME}-AWQ"

# Check env exists
if [ ! -f "$QUANT_PYTHON" ]; then
    echo "ERROR: conda env '$QUANT_ENV' not found at $QUANT_PYTHON"
    echo "Create it with: conda create -n $QUANT_ENV python=3.12"
    exit 1
fi

if [ ! -d "$MODEL_SRC" ]; then
    echo "ERROR: Source model not found at $MODEL_SRC"
    echo "Run REAM/REAP first to create a compressed BF16 model."
    echo "See: scripts/quantize/REAM.md"
    exit 1
fi

echo "=============================================="
echo "Qwen3.5 MoE REAM/REAP AWQ Quantization"
echo "Conda env:  $QUANT_ENV"
echo "Source:     $MODEL_SRC"
echo "CT output:  $CT_OUTPUT"
echo "AWQ output: $AWQ_OUTPUT"
echo "=============================================="

# Step 1: GPTQ calibration (DeltaNet-aware)
if [ -d "$CT_OUTPUT" ] && ls "$CT_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 1: SKIP (compressed-tensors output exists at $CT_OUTPUT) ==="
    echo "    Delete $CT_OUTPUT to re-run calibration."
else
    echo ""
    echo "=== Step 1: GPTQ calibration (DeltaNet-aware) ==="
    echo ""
    MODELS_DIR="$MODELS_DIR" "$QUANT_PYTHON" \
        "$SCRIPT_DIR/quantize_qwen35_moe_ream.py" \
        --model "$MODEL_SRC" \
        --output "$CT_OUTPUT"
fi

# Step 2: CT -> AWQ conversion
if [ -d "$AWQ_OUTPUT" ] && ls "$AWQ_OUTPUT"/model*.safetensors &>/dev/null; then
    echo ""
    echo "=== Step 2: SKIP (AWQ output exists at $AWQ_OUTPUT) ==="
    echo "    Delete $AWQ_OUTPUT to re-run conversion."
else
    echo ""
    echo "=== Step 2: Convert compressed-tensors -> native AWQ ==="
    echo ""
    "$QUANT_PYTHON" "$SCRIPT_DIR/convert_moe_ct_to_awq.py" \
        "$CT_OUTPUT" "$AWQ_OUTPUT"
fi

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "AWQ model: $AWQ_OUTPUT"
echo ""
echo "Run inference with:"
echo "  MODEL=$AWQ_OUTPUT ./scripts/launch.sh qwen35-moe"
echo "=============================================="
