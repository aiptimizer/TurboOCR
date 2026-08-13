#!/usr/bin/env bash
# Run the PP-FormulaNet-S backend (FORMULA_BACKEND=ppformulanet_s).
#
# The fused PaddleX export (`models/formula/ppformulanet_s/inference.onnx`)
# has two upstream-export bugs blocking native TRT/ORT execution:
#   1. INT64 BitwiseAnd/BitwiseNot inside the Loop body.
#   2. Two Loop-carry shape mismatches (arange [3]→[1] from a buggy
#      Slice, and If.3 then-branch [1] vs the scalar [] outer carry).
# scripts/models/onnx/patch_ppformulanet_s_onnx.py rewrites those into
# `inference_trt.onnx` (idempotent, re-runs only if src is newer).
#
# The C++ PP-FormulaNet-S backend runs fully in-process (ORT-CUDA-13 on the
# GPU build, ORT-CPU on the CPU build) — no Python sidecar, no socket.
#
# Validation gate (5-doc OmniDocBench subset):
#   display_formula_cdm ≥ 0.30  (vs FormulaNet baseline 0.040)
#   display_formula_edit_dist < 0.60  (vs 0.897 baseline)
#   table_teds + text_block_edit_dist must not regress > 0.02.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -d "${REPO_ROOT}/.venv-modelopt" ]]; then
  "${REPO_ROOT}/.venv-modelopt/bin/python" \
      "${REPO_ROOT}/scripts/models/onnx/patch_ppformulanet_s_onnx.py"
fi

exec "${REPO_ROOT}/scripts/experiments/exp_runner.sh" ppformulanet_s \
  -e FORMULA_BACKEND=ppformulanet_s
