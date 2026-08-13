#!/usr/bin/env bash
# FP8 PTQ + cache-key-scoped rebuild for PP-DocLayoutV3.
# Good fit per /tmp/fp8_research.md §Q5: 123 MatMuls dominate the DETR-family
# attention head; the 27/118 group convs are minority. Expect 1.1–1.3×.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_fp8_common.sh
source "${SCRIPT_DIR}/_build_fp8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="layout"
ENV_FLAG_NAME="TURBO_OCR_LAYOUT_FP8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/layout/layout.onnx}"
CACHE_GLOB="layout_*.trt"

run_fp8_build
