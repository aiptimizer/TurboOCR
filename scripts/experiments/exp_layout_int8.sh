#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELOPT_VENV="${REPO_ROOT}/.venv-modelopt"
ONNX="${REPO_ROOT}/models/layout/layout.onnx"
INT8_ONNX="${REPO_ROOT}/models/layout/layout.int8.onnx"

if [[ ! -f "${INT8_ONNX}" || -n "${FORCE_REQUANT:-}" ]]; then
  echo "==> quantize layout.onnx (modelopt + calibration)"
  "${MODELOPT_VENV}/bin/python" "${REPO_ROOT}/scripts/models/trt/quantize_onnx_int8.py" \
    --model-type=layout --onnx "${ONNX}" --out "${INT8_ONNX}"
else
  echo "==> reusing ${INT8_ONNX}"
fi
exec "${REPO_ROOT}/scripts/experiments/exp_runner.sh" layout_int8 -e TURBO_OCR_LAYOUT_INT8=1
