#!/usr/bin/env bash
# Path G rec INT8 via modelopt explicit-QDQ with surgical exclusions.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELOPT_VENV="${REPO_ROOT}/.venv-modelopt"
ONNX="${REPO_ROOT}/models/rec.onnx"
QDQ_ONNX="${REPO_ROOT}/models/rec_qdq.onnx"
DET_ENGINE="${DET_ENGINE:-/tmp/det_for_calib.trt}"

if [[ ! -f "${DET_ENGINE}" ]]; then
  echo "FAIL: det engine missing: ${DET_ENGINE}" >&2
  exit 1
fi

if [[ ! -f "${QDQ_ONNX}" || -n "${FORCE_REQUANT:-}" ]]; then
  echo "==> quantize rec.onnx → QDQ INT8 (modelopt with MatMul+Softmax+HardSigmoid exclusions)"
  "${MODELOPT_VENV}/bin/python" "${REPO_ROOT}/scripts/models/trt/quantize_onnx_int8.py" \
    --model-type=rec --onnx "${ONNX}" --out "${QDQ_ONNX}" \
    --det-engine "${DET_ENGINE}"
else
  echo "==> reusing ${QDQ_ONNX}"
fi
exec "${REPO_ROOT}/scripts/experiments/exp_runner.sh" rec_qdq -e TURBO_OCR_REC_INT8_QDQ=1
