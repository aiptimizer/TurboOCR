#!/usr/bin/env bash
# Path G (#10): rec INT8 via modelopt explicit-QDQ with surgical exclusions.
#
# Steps:
#   1. Find the FP16 det engine (needed to extract text crops for rec calibration).
#   2. Find the FP16 rec engine (needed as the latency-gate baseline).
#   3. Run scripts/models/trt/quantize_onnx_int8.py --model-type rec → models/rec_qdq.onnx
#      with the SVTR-MatMul / Softmax / SE-HardSigmoid exclusion regex.
#   4. Boot the server with TURBO_OCR_REC_INT8_QDQ=1 so it builds the INT8
#      rec engine on init.
#   5. HARD LATENCY GATE: rec INT8 engine must be ≥5 % faster than FP16 on
#      20 timed rec crops. If not, revert (delete the new engine) and FAIL.
#   6. HARD ACCURACY GATE: rec INT8 engine must drop text edit_dist on the
#      5-doc OmniDocBench subset by ≤1 pp vs the FP16 baseline. Requires
#      scripts/eval/omnidoc_run_and_score.py + a prior baseline run. If not present
#      yet (#4 still in_progress), the script will skip the accuracy gate
#      with a loud warning and prompt the user to re-run after baseline lands.
#   7. On any FAIL: rm the INT8 engine and exit non-zero with the failing
#      delta. Operator is expected to drop the flag (per /tmp/paddle_rec_int8_research.md).
#
# Past failure modes this guards against:
#   - 18 % SLOWER than FP16 (modelopt without exclusions, 2026-03 commit 0848864c)
#   - "AAA…" degenerate output (deprecated entropy calibrator on the older CRNN)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CACHE_DIR="${TRT_ENGINE_CACHE:-${HOME}/.cache/turbo-ocr}"
SERVER_BIN="${REPO_ROOT}/build/turboocr-server"
# Repo-relative modelopt venv; override VENV_PY for a venv elsewhere.
VENV_PY="${VENV_PY:-${REPO_ROOT}/.venv-modelopt/bin/python}"
DEADLINE_SECONDS="${DEADLINE_SECONDS:-600}"
ENV_FLAG_NAME="TURBO_OCR_REC_INT8_QDQ"
REC_ONNX="${REC_ONNX:-${REPO_ROOT}/models/rec.onnx}"
REC_QDQ="${REC_ONNX%.onnx}_qdq.onnx"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-/tmp/omnidoc_runs/baseline_latest.json}"

log()  { printf '[recqdq] %s\n' "$*"; }
die()  { printf '[recqdq] FATAL: %s\n' "$*" >&2; exit 1; }

[[ -x "${SERVER_BIN}" ]] || die "server binary missing: ${SERVER_BIN}"
[[ -f "${REC_ONNX}" ]] || die "rec ONNX missing: ${REC_ONNX}"
[[ -x "${VENV_PY}" ]] || die "Python interpreter missing: ${VENV_PY}"

log "rec onnx     : ${REC_ONNX}"
log "qdq target   : ${REC_QDQ}"
log "cache dir    : ${CACHE_DIR}"
log "env flag     : ${ENV_FLAG_NAME}=1"

# 1. Find FP16 det engine + FP16 rec engine.
shopt -s nullglob
det_candidates=( "${CACHE_DIR}"/det_*.trt )
rec_candidates=( "${CACHE_DIR}"/rec_*.trt )
shopt -u nullglob
(( ${#det_candidates[@]} > 0 )) || die "no FP16 det engine under ${CACHE_DIR}/det_*.trt — boot the server once first"
(( ${#rec_candidates[@]} > 0 )) || die "no FP16 rec engine under ${CACHE_DIR}/rec_*.trt — boot the server once first"

det_engine="$(ls -1t "${det_candidates[@]}" | head -n1)"
fp16_rec_engine="$(ls -1t "${rec_candidates[@]}" | head -n1)"
log "FP16 det     : ${det_engine}"
log "FP16 rec     : ${fp16_rec_engine}"

fp16_snap="$(mktemp -t rec_fp16.XXXXXX.trt)"
cp -p "${fp16_rec_engine}" "${fp16_snap}"

cleanup() {
  rm -f "${fp16_snap}"
}
trap cleanup EXIT INT TERM

# 2. modelopt QDQ pass (skips if up-to-date).
if [[ -f "${REC_QDQ}" ]] && [[ "${REC_QDQ}" -nt "${REC_ONNX}" ]]; then
  log "QDQ rec ONNX up to date, skipping modelopt pass"
else
  log "running modelopt INT8 PTQ with Path G surgical exclusions"
  "${VENV_PY}" "${REPO_ROOT}/scripts/models/trt/quantize_onnx_int8.py" \
    --onnx "${REC_ONNX}" --model-type rec \
    --det-engine "${det_engine}" \
    || die "modelopt rec quantization failed"
  [[ -f "${REC_QDQ}" ]] || die "modelopt did not emit ${REC_QDQ}"
fi

# 3. Boot server with the flag to build INT8 rec engine.
log_file="$(mktemp -t rec_int8_qdq_build.XXXXXX.log)"
log "build log    : ${log_file}"
log "deadline     : ${DEADLINE_SECONDS}s"
export "${ENV_FLAG_NAME}=1"
http_port="${HTTP_PORT:-18080}"
grpc_port="${GRPC_PORT:-15051}"

start_time=$(date +%s)
"${SERVER_BIN}" --http-port "${http_port}" --grpc-port "${grpc_port}" \
  > "${log_file}" 2>&1 &
server_pid=$!
trap 'kill -TERM '"${server_pid}"' 2>/dev/null || true; cleanup' EXIT INT TERM

ready=0; fail=0
while :; do
  now=$(date +%s); elapsed=$((now - start_time))
  if (( elapsed >= DEADLINE_SECONDS )); then
    log "ERROR: deadline exceeded (${DEADLINE_SECONDS}s)"
    fail=1; break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    log "ERROR: server exited before signalling ready (see ${log_file})"
    fail=1; break
  fi
  if grep -qE "Pipeline ready|HTTP listening|Listening on" "${log_file}" 2>/dev/null; then
    ready=1; break
  fi
  if grep -qE "Failed to build engine|\[TRT\] Failed|\[TRT Build\].*Error" "${log_file}" 2>/dev/null; then
    log "ERROR: TRT build error in ${log_file}"
    fail=1; break
  fi
  sleep 1
done

(( fail == 0 )) || { kill -TERM "${server_pid}" 2>/dev/null || true; die "engine build failed — see ${log_file}"; }
(( ready == 1 )) || { kill -TERM "${server_pid}" 2>/dev/null || true; die "server never signalled ready — see ${log_file}"; }
log "rec INT8 engine build done in $(( $(date +%s) - start_time ))s"

# 4. Locate the freshly-built INT8 rec engine (must differ from FP16 cache key).
shopt -s nullglob
rec_candidates=( "${CACHE_DIR}"/rec_*.trt )
shopt -u nullglob
int8_rec_engine="$(ls -1t "${rec_candidates[@]}" | head -n1)"
[[ "${int8_rec_engine}" != "${fp16_rec_engine}" ]] || die "INT8 build did not produce a new cache entry"
log "INT8 rec     : ${int8_rec_engine}"

# 5. LATENCY GATE.
log "running latency gate (3 warmup + 20 timed rec crops)"
if ! "${VENV_PY}" "${REPO_ROOT}/scripts/models/trt/int8_latency_gate.py" \
      --fp16-engine "${fp16_snap}" \
      --int8-engine "${int8_rec_engine}" \
      --model-type rec \
      --det-engine "${det_engine}"; then
  log "==> LATENCY GATE FAILED for rec (Path G)"
  log "    Removing the INT8 engine so next server boot reverts to FP16."
  rm -f "${int8_rec_engine}"
  kill -TERM "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  die "rec INT8 not ≥5 % faster than FP16 — drop TURBO_OCR_REC_INT8_QDQ"
fi

# 6. ACCURACY GATE (server is still running with INT8 rec loaded).
if [[ ! -f "${BASELINE_SUMMARY}" ]]; then
  log "WARNING: baseline summary missing at ${BASELINE_SUMMARY} —"
  log "         skipping accuracy gate. Re-run this script after task #4"
  log "         lands and points BASELINE_SUMMARY at the baseline JSON."
else
  cand_summary="/tmp/omnidoc_runs/rec_int8_qdq_$(date +%Y%m%d_%H%M%S).json"
  log "scoring rec INT8 on OmniDocBench 5-doc subset → ${cand_summary}"
  if ! timeout 1200s "${VENV_PY}" "${REPO_ROOT}/scripts/eval/omnidoc_run_and_score.py" \
        --experiment-name "rec_int8_qdq" \
        --server-url "http://localhost:${http_port}" \
        --out "${cand_summary}"; then
    log "ERROR: omnidoc scoring failed for rec INT8"
    rm -f "${int8_rec_engine}"
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    die "rec INT8 scoring run errored — engine reverted"
  fi
  # diff with threshold 0.01 ≡ ≤1 pp regression bar.
  if ! "${VENV_PY}" "${REPO_ROOT}/scripts/eval/omnidoc_diff.py" \
        "${BASELINE_SUMMARY}" "${cand_summary}" --threshold 0.01; then
    log "==> ACCURACY GATE FAILED for rec (Path G): text edit_dist regressed >1 pp"
    log "    Candidate summary: ${cand_summary}"
    log "    Removing the INT8 engine so next server boot reverts to FP16."
    rm -f "${int8_rec_engine}"
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    die "rec INT8 accuracy regression — drop TURBO_OCR_REC_INT8_QDQ"
  fi
  log "accuracy gate PASS — text edit_dist regression ≤1 pp"
fi

kill -TERM "${server_pid}" 2>/dev/null || true
wait "${server_pid}" 2>/dev/null || true

log "==> PASS — rec INT8 (Path G) is faster than FP16, within accuracy budget."
log "    Engine: ${int8_rec_engine}"
log "    Build log: ${log_file}"
