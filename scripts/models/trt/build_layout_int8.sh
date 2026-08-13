#!/usr/bin/env bash
# INT8 PTQ + cache-key-scoped rebuild for PP-DocLayoutV3.
# PP-DocLayoutV3 is DETR-family with 3 inputs (image / im_shape / scale_factor)
# — make sure the modelopt calibration loader feeds all three (see
# scripts/models/trt/quantize_onnx_int8.py _preprocess_layout).
#
# Latency gate: ≥5 % faster than FP16 baseline or revert.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_int8_common.sh
source "${SCRIPT_DIR}/_build_int8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="layout"
ENV_FLAG_NAME="TURBO_OCR_LAYOUT_INT8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/layout/layout.onnx}"
CACHE_GLOB="layout_*.trt"

run_int8_build
