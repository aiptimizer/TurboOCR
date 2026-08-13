#!/usr/bin/env bash
# FP8 PTQ + cache-key-scoped rebuild for the PP-FormulaNet-S / RapidLaTeXOCR
# encoder (HGNetV2-B4 backbone). Good fit per /tmp/fp8_research.md §Q5: zero
# group convs, 1×1/3×3 dominant, single 7×7 stem (well within the ≤32 limit).
# Expect 1.15–1.3×.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_fp8_common.sh
source "${SCRIPT_DIR}/_build_fp8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="formula_encoder"
ENV_FLAG_NAME="TURBO_OCR_FORMULA_ENCODER_FP8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/formula/encoder.onnx}"
CACHE_GLOB="formula_encoder_*.trt"

run_fp8_build
