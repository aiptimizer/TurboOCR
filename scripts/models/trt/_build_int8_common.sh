#!/usr/bin/env bash
# Shared helpers for build_<model>_int8.sh wrappers. Source this file from
# the per-model script and then call `run_int8_build`.
#
# Required env from caller:
#   MODEL_TYPE         — TRT type label (det / layout / table_struct_nemotron / formula_encoder)
#   ENV_FLAG_NAME      — TURBO_OCR_<MODEL>_INT8
#   ONNX_PATH          — absolute path to source FP ONNX
#   CACHE_GLOB         — glob (in CACHE_DIR) of stale engine files to prune

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CACHE_DIR="${TRT_ENGINE_CACHE:-${HOME}/.cache/turbo-ocr}"
SERVER_BIN="${REPO_ROOT}/build/turboocr-server"
# Repo-relative modelopt venv; override VENV_PY for a venv elsewhere.
VENV_PY="${VENV_PY:-${REPO_ROOT}/.venv-modelopt/bin/python}"
DEADLINE_SECONDS="${DEADLINE_SECONDS:-600}"

log()  { printf '[int8] %s\n' "$*"; }
die()  { printf '[int8] FATAL: %s\n' "$*" >&2; exit 1; }

run_int8_build() {
  : "${MODEL_TYPE:?MODEL_TYPE not set}"
  : "${ENV_FLAG_NAME:?ENV_FLAG_NAME not set}"
  : "${ONNX_PATH:?ONNX_PATH not set}"
  : "${CACHE_GLOB:?CACHE_GLOB not set}"

  [[ -x "${SERVER_BIN}" ]] || die "server binary missing: ${SERVER_BIN} (build first)"
  [[ -f "${ONNX_PATH}" ]] || die "source ONNX missing: ${ONNX_PATH}"
  [[ -x "${VENV_PY}" ]] || die "Python interpreter missing: ${VENV_PY}"

  local stem="${ONNX_PATH%.onnx}"
  local qdq_path="${stem}.int8.onnx"

  log "model_type   : ${MODEL_TYPE}"
  log "env flag     : ${ENV_FLAG_NAME}=1"
  log "source onnx  : ${ONNX_PATH}"
  log "qdq onnx     : ${qdq_path}"
  log "cache dir    : ${CACHE_DIR}"

  # 1. modelopt QDQ pass (skips if QDQ ONNX already exists and is newer than source)
  if [[ -f "${qdq_path}" ]] && [[ "${qdq_path}" -nt "${ONNX_PATH}" ]]; then
    log "QDQ ONNX up to date, skipping modelopt pass"
  else
    log "running modelopt INT8 PTQ via quantize_onnx_int8.py"
    "${VENV_PY}" "${REPO_ROOT}/scripts/models/trt/quantize_onnx_int8.py" \
      --onnx "${ONNX_PATH}" --model-type "${MODEL_TYPE}" \
      || die "modelopt quantization failed"
    [[ -f "${qdq_path}" ]] || die "modelopt did not emit ${qdq_path}"
  fi

  # 2. Locate the FP16 baseline engine BEFORE pruning anything (we need both
  #    for the latency gate). If multiple match, take the most recent.
  shopt -s nullglob
  local fp16_engine=""
  local candidates=( "${CACHE_DIR}"/${CACHE_GLOB} )
  if (( ${#candidates[@]} > 0 )); then
    fp16_engine="$(ls -1t "${candidates[@]}" 2>/dev/null | head -n1)"
  fi
  shopt -u nullglob
  if [[ -z "${fp16_engine}" ]]; then
    log "no FP16 baseline engine found under ${CACHE_DIR}/${CACHE_GLOB}"
    log "  → start the server once (without ${ENV_FLAG_NAME}) to build the FP16 baseline,"
    log "    then re-run this script. The latency gate needs both engines side-by-side."
    die "FP16 baseline engine missing"
  fi
  log "FP16 baseline: ${fp16_engine}"
  # Stash a copy so the upcoming server boot doesn't overwrite or pollute it.
  local fp16_snap
  fp16_snap="$(mktemp -t "${MODEL_TYPE}_fp16.XXXXXX.trt")"
  cp -p "${fp16_engine}" "${fp16_snap}"

  # 3. Boot the server with the INT8 flag set so it builds the INT8 engine
  #    on init. Same pattern as build_table_engines_fp16.sh.
  local log_file
  log_file="$(mktemp -t "${MODEL_TYPE}_int8_build.XXXXXX.log")"
  log "build log    : ${log_file}"
  log "deadline     : ${DEADLINE_SECONDS}s"

  export "${ENV_FLAG_NAME}=1"
  local http_port="${HTTP_PORT:-18080}"
  local grpc_port="${GRPC_PORT:-15051}"

  local extra_args=()
  case "${MODEL_TYPE}" in
    table_struct_nemotron) extra_args+=(--table-mode --table-struct-kind nemotron) ;;
    formula_encoder)       extra_args+=(--formula-mode) ;;
    layout)                : ;;
    det)                   : ;;
  esac

  local start_time
  start_time=$(date +%s)
  "${SERVER_BIN}" --http-port "${http_port}" --grpc-port "${grpc_port}" \
    "${extra_args[@]}" > "${log_file}" 2>&1 &
  local server_pid=$!
  trap 'kill -TERM '"${server_pid}"' 2>/dev/null || true; rm -f "${fp16_snap}"' EXIT INT TERM

  local ready=0
  local fail=0
  while :; do
    local now elapsed
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
  kill -TERM "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true

  if (( fail != 0 )); then
    die "engine build failed — see ${log_file}"
  fi
  if (( ready == 0 )); then
    die "server never signalled ready — see ${log_file}"
  fi
  local duration=$(( $(date +%s) - start_time ))
  log "engine build complete in ${duration}s"

  # 4. Find the freshly-built INT8 engine. Pick the most recent matching file
  #    that ISN'T the FP16 snapshot we just stashed.
  shopt -s nullglob
  local int8_engine=""
  candidates=( "${CACHE_DIR}"/${CACHE_GLOB} )
  if (( ${#candidates[@]} > 0 )); then
    int8_engine="$(ls -1t "${candidates[@]}" 2>/dev/null | head -n1)"
  fi
  shopt -u nullglob
  [[ -n "${int8_engine}" ]] || die "INT8 engine not found in ${CACHE_DIR} after build"
  if [[ "${int8_engine}" == "${fp16_engine}" ]]; then
    die "INT8 build did not produce a new cache entry (cache key didn't change?)"
  fi
  log "INT8 engine  : ${int8_engine}"

  # 5. Latency gate: ≥5 % faster than FP16 baseline or fail loudly.
  log "running latency gate (3 warmup + 20 timed images)"
  if ! "${VENV_PY}" "${REPO_ROOT}/scripts/models/trt/int8_latency_gate.py" \
        --fp16-engine "${fp16_snap}" \
        --int8-engine "${int8_engine}" \
        --model-type "${MODEL_TYPE}"; then
    log "==> LATENCY GATE FAILED for ${MODEL_TYPE}"
    log "    Removing the INT8 engine so the next server boot reverts to FP16."
    rm -f "${int8_engine}"
    rm -f "${fp16_snap}"
    die "drop --model-type=${MODEL_TYPE} from the INT8 experiment"
  fi

  rm -f "${fp16_snap}"
  log "==> PASS — INT8 engine for ${MODEL_TYPE} is faster than FP16 and retained."
  log "    Engine: ${int8_engine}"
  log "    Build log: ${log_file}"
}
