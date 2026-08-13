#!/usr/bin/env bash
# Experiment: pure-FP16 table engines (kOBEY_PRECISION_CONSTRAINTS + per-layer kHALF).
# Baseline (TURBO_OCR_TABLE_FP16 unset) is TRT-chosen weak-typed mixed-FP16.
# A/B is "mixed FP16" vs "pure FP16 with no FP32 fallback".

set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/exp_runner.sh" \
  table_fp16_pure \
  -e TURBO_OCR_TABLE_FP16=1
