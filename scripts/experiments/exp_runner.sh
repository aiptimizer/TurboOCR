#!/usr/bin/env bash
# Generic experiment runner: restart docker container with caller-supplied env,
# wait health, score against 5-doc OmniDocBench subset, diff vs baseline_wtree,
# report PASS/FAIL.
#
# Usage:
#   scripts/experiments/exp_runner.sh <experiment_name> [docker -e KEY=VAL ...]
#
# Example:
#   scripts/experiments/exp_runner.sh table_fp16_pure -e TURBO_OCR_TABLE_FP16=1

set -euo pipefail

EXP_NAME="${1:?usage: $0 <experiment_name> [-e KEY=VAL ...]}"
shift
EXTRA_ENV=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${TURBOOCR_IMAGE:-turboocr:wtree}"
PORT="${TURBOOCR_PORT:-8080}"
SERVER_URL="http://localhost:${PORT}"
BASELINE="${BASELINE_JSON:-/tmp/omnidoc_runs/baseline_wtree_latest.json}"
HEALTH_TIMEOUT=300
SCORE_TIMEOUT=600

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f "${BASELINE}" ]] || fail "baseline missing: ${BASELINE} — run baseline_wtree first"

step "stop any running turboocr container"
docker rm -f turboocr 2>/dev/null || true

step "start ${IMAGE} on :${PORT} for experiment '${EXP_NAME}'"
printf '    env overrides:'
for e in "${EXTRA_ENV[@]}"; do printf ' %s' "$e"; done
printf '\n'

docker run -d --rm --name turboocr --gpus all \
  -p "${PORT}:8000" \
  -e REC_MODE=multilingual \
  -e TABLE_MODE=1 \
  -e FORMULA_MODE=1 \
  "${EXTRA_ENV[@]}" \
  -v turbo-ocr-cache:/home/ocr/.cache/turbo-ocr \
  -v "${REPO_ROOT}/models:/app/models:rw" \
  "${IMAGE}" >/dev/null

step "wait for /health (deadline ${HEALTH_TIMEOUT}s)"
SECONDS=0
until curl -sf --max-time 2 "${SERVER_URL}/health" >/dev/null 2>&1; do
  (( SECONDS < HEALTH_TIMEOUT )) || fail "container did not become healthy within ${HEALTH_TIMEOUT}s"
  sleep 2
done
echo "    ready in ${SECONDS}s"

step "scan container logs for silent TRT build failures"
LOG_ERRORS=$(docker logs turboocr 2>&1 | grep -E "Failed to build engine|TRT Build\] IBuilder::buildSerializedNetwork: Error|Failed to load engine|table_mode\":(false|0).*TABLE_MODE|formula_mode\":(false|0).*FORMULA_MODE" || true)
if [[ -n "${LOG_ERRORS}" ]]; then
  echo "${LOG_ERRORS}" | sed 's/^/    /'
  fail "container reported TRT build / engine failures — refusing to score against a silently-broken server"
fi
ROUTER_STATE=$(docker logs turboocr 2>&1 | grep -oE '"router_table_active":[0-9]+,"router_formula_active":[0-9]+' | tail -1)
echo "    router state: ${ROUTER_STATE:-unknown}"

step "score ${EXP_NAME} against 5-doc subset (deadline ${SCORE_TIMEOUT}s)"
if ! timeout "${SCORE_TIMEOUT}" python3 "${REPO_ROOT}/scripts/eval/omnidoc_run_and_score.py" \
       --experiment-name "${EXP_NAME}" \
       --server-url "${SERVER_URL}"; then
  fail "scoring failed for ${EXP_NAME}"
fi

RESULT_JSON=$(ls -1t /tmp/omnidoc_runs/"${EXP_NAME}"_*.json 2>/dev/null | head -1)
[[ -n "${RESULT_JSON}" ]] || fail "no result JSON produced for ${EXP_NAME}"

step "diff ${EXP_NAME} vs baseline_wtree (threshold 1pp)"
DIFF_EXIT=0
python3 "${REPO_ROOT}/scripts/eval/omnidoc_diff.py" "${BASELINE}" "${RESULT_JSON}" --threshold 0.01 || DIFF_EXIT=$?

step "experiment '${EXP_NAME}' complete"
echo "    result   : ${RESULT_JSON}"
echo "    baseline : ${BASELINE}"
if (( DIFF_EXIT == 0 )); then
  printf '\033[1;32mPASS\033[0m — no metric regressed > 1%%\n'
else
  printf '\033[1;33mGATE-FAIL\033[0m — diff exited %d (a metric regressed > 1%%)\n' "${DIFF_EXIT}"
fi

exit "${DIFF_EXIT}"
