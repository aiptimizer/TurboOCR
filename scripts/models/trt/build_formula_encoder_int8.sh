#!/usr/bin/env bash
# INT8 PTQ + cache-key-scoped rebuild for the formula encoder (HGNetV2-B4
# vision backbone). DOES NOT touch the formula decoder (#10) or the
# formula_resizer.
#
# The released encoder.onnx already carries some Q/DQ ops baked in per
# models/MANIFEST.txt + internal engineering notes §6 — modelopt
# will add per-tensor activation scales on top. If the resulting engine is
# the same size or slower than FP16, the upstream Q/DQ scales may already
# saturate the achievable INT8 gain on this architecture — drop the flag and
# move on.
#
# Latency gate: ≥5 % faster than FP16 baseline or revert.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_int8_common.sh
source "${SCRIPT_DIR}/_build_int8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="formula_encoder"
ENV_FLAG_NAME="TURBO_OCR_FORMULA_ENCODER_INT8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/formula/encoder.onnx}"
CACHE_GLOB="formula_encoder_*.trt"

run_int8_build
