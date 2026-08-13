#!/usr/bin/env bash
# Build all table-pipeline TRT engines with TURBO_OCR_TABLE_FP16=1 set.
# Deletes only the cached table_* engines (preserves det/rec/cls/layout/etc.),
# then boots turboocr-server in --table-mode so ensure_trt_engine() rebuilds
# each missing table engine, then kills the server once the build completes.
#
# Exits non-zero on build failure, timeout (>600s), or any "[TRT Build]"
# error line in the captured log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_BIN="${REPO_ROOT}/build/turboocr-server"
CACHE_DIR="${TRT_ENGINE_CACHE:-${HOME}/.cache/turbo-ocr}"
LOG_FILE="$(mktemp -t table_fp16_build.XXXXXX.log)"
DEADLINE_SECONDS=600

if [[ ! -x "${SERVER_BIN}" ]]; then
  echo "ERROR: server binary not found: ${SERVER_BIN}" >&2
  echo "Build it first: cmake --build ${REPO_ROOT}/build -j8" >&2
  exit 1
fi

echo "==> TURBO_OCR_TABLE_FP16=1 table-engine rebuild"
echo "    cache dir : ${CACHE_DIR}"
echo "    log file  : ${LOG_FILE}"
echo "    deadline  : ${DEADLINE_SECONDS}s"

# Step 1: delete cached table engines so they get rebuilt under the new flag.
# The cache key is now flag-scoped (:tfp16=1 suffix), so old FP32-mixed engines
# would NOT collide with new pure-FP16 engines — but pruning them keeps the
# cache directory tidy and forces fresh build artifacts for benchmarking.
deleted=0
if [[ -d "${CACHE_DIR}" ]]; then
  for f in "${CACHE_DIR}"/table_cls_*.trt \
           "${CACHE_DIR}"/table_struct_nemotron_*.trt \
           "${CACHE_DIR}"/table_struct_tatr_*.trt \
           "${CACHE_DIR}"/slanet_plus_*.trt \
           "${CACHE_DIR}"/slanext_*.trt \
           "${CACHE_DIR}"/table_cell_*.trt; do
    [[ -f "${f}" ]] || continue
    rm -f "${f}"
    deleted=$((deleted + 1))
  done
fi
echo "    pruned    : ${deleted} stale table engine(s)"

# Step 2: launch server in background. It blocks on engine build during init;
# once table engines are built it prints "[Pipeline] Table stage enabled" and
# binds to HTTP/gRPC ports. We watch for the bind log and then SIGTERM it.
export TURBO_OCR_TABLE_FP16=1
# Pick free-ish ports so we don't collide with a running server.
HTTP_PORT="${HTTP_PORT:-18080}"
GRPC_PORT="${GRPC_PORT:-15051}"
# Default to nemotron (the production table-struct), but allow override.
TABLE_KIND="${TABLE_KIND:-nemotron}"

start_time=$(date +%s)
"${SERVER_BIN}" \
  --table-mode \
  --table-struct-kind "${TABLE_KIND}" \
  --http-port "${HTTP_PORT}" \
  --grpc-port "${GRPC_PORT}" \
  > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    # Give it a couple seconds to drain.
    for _ in 1 2 3 4 5; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Step 3: tail the log waiting for either "HTTP listening" (success) or a
# TRT error / build failure / process exit. Cap at DEADLINE_SECONDS.
ready=0
fail=0
while :; do
  now=$(date +%s)
  elapsed=$((now - start_time))
  if (( elapsed >= DEADLINE_SECONDS )); then
    echo "ERROR: deadline exceeded (${DEADLINE_SECONDS}s) — aborting" >&2
    fail=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: server process exited before signalling ready" >&2
    fail=1
    break
  fi
  if grep -q "Pipeline ready" "${LOG_FILE}" 2>/dev/null || \
     grep -q "HTTP listening" "${LOG_FILE}" 2>/dev/null || \
     grep -q "Listening on" "${LOG_FILE}" 2>/dev/null; then
    ready=1
    break
  fi
  if grep -qE "Failed to build engine|\[TRT\] Failed|\[TRT Build\].*Error" \
       "${LOG_FILE}" 2>/dev/null; then
    echo "ERROR: TRT build error detected in ${LOG_FILE}" >&2
    fail=1
    break
  fi
  sleep 1
done

# Step 4: report.
end_time=$(date +%s)
duration=$((end_time - start_time))

echo
echo "==> build complete in ${duration}s (ready=${ready} fail=${fail})"

echo
echo "--- TABLE_FP16 / Built / TRT warning lines ---"
grep -E "TURBO_OCR_TABLE_FP16|^Built:|^Building TRT|\[TRT Build\]" \
  "${LOG_FILE}" || echo "(no matching lines)"

echo
echo "--- new table engines in ${CACHE_DIR} ---"
ls -lh "${CACHE_DIR}"/{table_cls,table_struct_nemotron,table_struct_tatr,slanet_plus,slanext,table_cell}_*.trt \
  2>/dev/null || echo "(no engines emitted — investigate ${LOG_FILE})"

if (( fail != 0 )); then
  echo
  echo "==> FAIL — see ${LOG_FILE} for full output" >&2
  exit 1
fi
if (( ready == 0 )); then
  echo
  echo "==> server never signalled ready, check ${LOG_FILE}" >&2
  exit 1
fi
echo
echo "==> OK — log retained at ${LOG_FILE}"
