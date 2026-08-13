#!/usr/bin/env bash
# Build a dynamic-batch FP16 TensorRT engine for the PP-FormulaNet_plus-M
# encoder (PPHGNetV2-B6 vision backbone: x[B,1,384,384] -> feats[B,144,2048]).
#
# INT8 is intentionally NOT offered: measured 24% SLOWER than FP16 on this B6
# backbone on Blackwell (sm_120) -- see the feasibility probe. FP16 is optimal.
# The encoder is ~1.4% of plus-M's per-crop cost, so this is a convenience for
# the engine route; the C++ decoder may equally load encoder.onnx via ORT-CUDA
# (the -S FAST-path convention) and skip the engine entirely.
#
# Usage: scripts/models/trt/build_plusm_encoder_trt.sh [min_bs] [opt_bs] [max_bs]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRT="${TENSORRT_ROOT:-$HOME/TensorRT-10.15.1.29}"
MODEL_DIR="${REPO_ROOT}/models/formula/ppformulanet_plus_m"
ONNX="${MODEL_DIR}/encoder.onnx"
ENGINE="${MODEL_DIR}/encoder_fp16.engine"

MIN_BS="${1:-1}"; OPT_BS="${2:-8}"; MAX_BS="${3:-32}"

[ -f "${ONNX}" ] || { echo "ERROR: ${ONNX} not found"; exit 1; }
[ -x "${TRT}/bin/trtexec" ] || { echo "ERROR: trtexec not at ${TRT}/bin/trtexec (set TENSORRT_ROOT)"; exit 1; }

export LD_LIBRARY_PATH="${TRT}/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
echo "[build] plus-M encoder FP16 engine: bs min=${MIN_BS} opt=${OPT_BS} max=${MAX_BS}"
"${TRT}/bin/trtexec" \
  --onnx="${ONNX}" \
  --minShapes=x:${MIN_BS}x1x384x384 \
  --optShapes=x:${OPT_BS}x1x384x384 \
  --maxShapes=x:${MAX_BS}x1x384x384 \
  --fp16 --useCudaGraph --noTF32 \
  --saveEngine="${ENGINE}"
echo "[build] wrote ${ENGINE}"
