#!/usr/bin/env bash
# Re-establish baseline_wtree with tables + formulas actually ENABLED.
# Prior baseline_wtree had TABLE_MODE/FORMULA_MODE off so table/formula scores
# were measuring an inactive code path.

set -euo pipefail
# Override BASELINE check inside exp_runner.sh — this script IS the baseline,
# so point the diff back at itself (or skip diff entirely via env override).
export BASELINE_JSON="${BASELINE_JSON:-/tmp/omnidoc_runs/baseline_wtree_latest.json}"
exec "$(dirname "${BASH_SOURCE[0]}")/exp_runner.sh" baseline_wtree_full
