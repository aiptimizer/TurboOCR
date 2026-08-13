#!/usr/bin/env bash
# TATR (Microsoft Table Transformer) table-struct backend — DETR-family
# alternative to Nemotron YOLOX. Boots the wtree image with
# TABLE_STRUCT_KIND=tatr so the dispatcher loads
# models/table/table_struct_tatr.onnx, runs the 5-doc scorer, and diffs vs
# baseline_wtree. Gate: must match or beat Nemotron's TEDS 0.51 / structure
# 0.68 baseline on the subset (recorded in /tmp/tatr_wiring_plan.md).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${REPO_ROOT}/scripts/experiments/exp_runner.sh" tatr_table -e TABLE_STRUCT_KIND=tatr
