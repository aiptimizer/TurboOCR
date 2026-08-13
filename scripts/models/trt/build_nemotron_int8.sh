#!/usr/bin/env bash
# INT8 PTQ + cache-key-scoped rebuild for nvidia/nemotron-table-structure-v1
# (YOLOX-based, pure conv head, no Loop). Per quant-research: this architecture
# is one of the safer INT8 targets — no AR, no softmax sensitivity.
#
# Note: nemotron weights live in a split file pair (.onnx + .onnx.data). The
# modelopt pass needs to honor that — should work via onnx's external-data
# loader transparently, but watch the build log if the QDQ ONNX is suspiciously
# small (likely a missing external-data ref).
#
# Latency gate: ≥5 % faster than FP16 baseline or revert.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_build_int8_common.sh
source "${SCRIPT_DIR}/_build_int8_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_TYPE="table_struct_nemotron"
ENV_FLAG_NAME="TURBO_OCR_NEMOTRON_INT8"
ONNX_PATH="${ONNX_PATH:-${REPO_ROOT}/models/table/table_struct_nemotron.onnx}"
CACHE_GLOB="table_struct_nemotron_*.trt"

run_int8_build
