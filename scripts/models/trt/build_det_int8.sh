#!/usr/bin/env bash
# INT8 PTQ + cache-key-scoped rebuild for the PaddleDet text detector.
# Per quant-research path F1 (lowest-risk INT8 target): the older CRNN-era
# det INT8 in paddle-highspeed-cpp-coppy delivered ~44 % single-request
# latency improvement. Aiming for the same here on the current det.onnx.
#
# Latency gate: build_int8_common refuses to keep the INT8 engine unless it
# is ≥5 % faster than the FP16 baseline.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_int8_common.sh
source "${SCRIPT_DIR}/_build_int8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="det"
ENV_FLAG_NAME="TURBO_OCR_DET_INT8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/det.onnx}"
CACHE_GLOB="det_*.trt"

run_int8_build
