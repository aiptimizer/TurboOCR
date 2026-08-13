#!/usr/bin/env bash
# FP8 PTQ + cache-key-scoped rebuild for the NVIDIA Nemotron YOLOX table
# structure detector. Best-fit per /tmp/fp8_research.md §Q5: 131 convs but
# ZERO group convs, all small kernels. Expect 1.2–1.4×.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_fp8_common.sh
source "${SCRIPT_DIR}/_build_fp8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="table_struct_nemotron"
ENV_FLAG_NAME="TURBO_OCR_NEMOTRON_FP8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/table/table_struct_nemotron.onnx}"
CACHE_GLOB="table_struct_nemotron_*.trt"

run_fp8_build
