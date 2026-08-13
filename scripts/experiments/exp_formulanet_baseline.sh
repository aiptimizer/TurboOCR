#!/usr/bin/env bash
# Phase 1 sanity check: run the (existing) FormulaNet behind the new
# IFormulaRecognizer interface. Score MUST match baseline_wtree_full.
# Sets FORMULA_BACKEND=formulanet explicitly to lock the default.

set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/exp_runner.sh" formulanet_baseline \
  -e FORMULA_BACKEND=formulanet
