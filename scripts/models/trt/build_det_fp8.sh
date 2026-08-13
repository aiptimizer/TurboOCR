#!/usr/bin/env bash
# FP8 PTQ + cache-key-scoped rebuild for the PaddleDet text detector.
# Marginal-fit per /tmp/fp8_research.md §Q5 (23% group convs → some layers
# fall back to non-FP8 kernels per TRT 10.15.1 release-note advisory). The
# latency gate is what decides whether to keep the engine.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_fp8_common.sh
source "${SCRIPT_DIR}/_build_fp8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="det"
ENV_FLAG_NAME="TURBO_OCR_DET_FP8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/det.onnx}"
CACHE_GLOB="det_*.trt"

run_fp8_build
