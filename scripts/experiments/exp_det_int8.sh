#!/usr/bin/env bash
# Experiment: PaddleDet (text detection) INT8 via modelopt explicit-QDQ.
# Past precedent: ~44% latency drop on det INT8 in paddle-highspeed-cpp-coppy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELOPT_VENV="${REPO_ROOT}/.venv-modelopt"
DET_ONNX="${REPO_ROOT}/models/det.onnx"
DET_INT8_ONNX="${REPO_ROOT}/models/det.int8.onnx"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [[ ! -f "${DET_INT8_ONNX}" || -n "${FORCE_REQUANT:-}" ]]; then
  step "quantize det.onnx via modelopt (calibration: 200 OmniDocBench images)"
  "${MODELOPT_VENV}/bin/python" "${REPO_ROOT}/scripts/models/trt/quantize_onnx_int8.py" \
    --model-type=det \
    --onnx "${DET_ONNX}" \
    --out "${DET_INT8_ONNX}"
else
  echo "==> reusing existing ${DET_INT8_ONNX} (set FORCE_REQUANT=1 to redo)"
fi

step "score det_int8 via exp_runner"
exec "${REPO_ROOT}/scripts/experiments/exp_runner.sh" det_int8 -e TURBO_OCR_DET_INT8=1
