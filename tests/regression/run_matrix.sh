#!/usr/bin/env bash
# Launch the OCR server, wait for /health/ready, run the endpoint x filetype
# matrix, then tear down. Auto-detects docker (preferred) vs native binary.
#
# Env knobs:
#   MATRIX_FAST=1            commit-gate mode: N=2 concurrency, no TIFF/WebP
#   MATRIX_CONCURRENCY=<N>   override stress concurrency (default 20)
#   MATRIX_HTTP_PORT=<port>  HTTP port (default 8000)
#   MATRIX_GRPC_PORT=<port>  gRPC port (default 50051)
#   MATRIX_KEEP_SERVER=1     do not tear the server down on exit (debug)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HTTP_PORT="${MATRIX_HTTP_PORT:-8000}"
GRPC_PORT="${MATRIX_GRPC_PORT:-50051}"
HEALTH_URL="http://localhost:${HTTP_PORT}/health/ready"
SERVER_URL="http://localhost:${HTTP_PORT}"
GRPC_TARGET="localhost:${GRPC_PORT}"
READY_TIMEOUT="${MATRIX_READY_TIMEOUT:-90}"

log() { printf "[run_matrix] %s\n" "$*" >&2; }

wait_ready() {
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while (( $(date +%s) < deadline )); do
    if curl -fs "${HEALTH_URL}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

server_already_up() {
  curl -fs "${HEALTH_URL}" >/dev/null 2>&1
}

SERVER_PID=""
COMPOSE_DOWN=0

teardown() {
  local rc=$?
  if [[ -n "${MATRIX_KEEP_SERVER:-}" ]]; then
    log "MATRIX_KEEP_SERVER set, leaving server running"
    exit "${rc}"
  fi
  if (( COMPOSE_DOWN == 1 )); then
    log "stopping docker compose stack"
    (cd "${REPO_ROOT}/docker" && docker compose down --remove-orphans) >/dev/null 2>&1 || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping server pid=${SERVER_PID}"
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap teardown EXIT INT TERM

if server_already_up; then
  log "server already responding at ${HEALTH_URL}, reusing it"
else
  NATIVE_BIN="${REPO_ROOT}/build/turboocr-server"
  if [[ -x "${NATIVE_BIN}" ]]; then
    log "launching native server: ${NATIVE_BIN}"
    PORT="${HTTP_PORT}" GRPC_PORT="${GRPC_PORT}" \
      setsid "${NATIVE_BIN}" >/tmp/run_matrix_server.log 2>&1 &
    SERVER_PID=$!
    log "server pid=${SERVER_PID}, waiting for ${HEALTH_URL} (timeout ${READY_TIMEOUT}s)"
    if ! wait_ready; then
      log "server did not become ready; tail of log:"
      tail -n 50 /tmp/run_matrix_server.log >&2 || true
      exit 1
    fi
  elif command -v docker >/dev/null 2>&1 && [[ -f "${REPO_ROOT}/docker/compose.yaml" ]]; then
    log "launching docker compose stack"
    (cd "${REPO_ROOT}/docker" && docker compose up -d --scale ocr=1) >&2
    COMPOSE_DOWN=1
    # docker-compose maps 8001-8007; first replica is 8001.
    HTTP_PORT=8001
    HEALTH_URL="http://localhost:${HTTP_PORT}/health/ready"
    SERVER_URL="http://localhost:${HTTP_PORT}"
    log "waiting for ${HEALTH_URL} (timeout ${READY_TIMEOUT}s)"
    if ! wait_ready; then
      log "docker server did not become ready"
      (cd "${REPO_ROOT}/docker" && docker compose logs --tail 100 ocr) >&2 || true
      exit 1
    fi
  else
    log "no native binary at ${NATIVE_BIN} and docker not available"
    exit 1
  fi
fi

log "server ready, running matrix (FAST=${MATRIX_FAST:-0}, CONCURRENCY=${MATRIX_CONCURRENCY:-default})"
cd "${REPO_ROOT}"
python -m pytest tests/regression/test_endpoint_matrix.py \
  --server-url "${SERVER_URL}" \
  --grpc-target "${GRPC_TARGET}" \
  -v "$@"
