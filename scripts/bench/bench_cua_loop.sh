#!/usr/bin/env bash
# CUA-router benchmark loop entrypoint (plan 06 §6).
#
# 1. health-probe /health/ready (≤2 s).
# 2. on success, run tests/benchmark/router/bench_cua_router.py.
# 3. exit: 0=PASS, 1=INFRA, 2=server-down, 3=ALERT, 4=HALT.

set -u

SERVER_URL="${OCR_SERVER_URL:-http://localhost:8000}"
OUT_DIR="${CUA_BENCH_OUT_DIR:-/tmp/cua_bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${PYTHON:-python3}"

mkdir -p "$OUT_DIR"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out_path="$OUT_DIR/$ts.json"

# 1. health probe
if ! curl -fs --max-time 2 "$SERVER_URL/health/ready" >/dev/null 2>&1; then
  cat > "$out_path" <<EOF
{
  "schema_version": 1,
  "timestamp_utc": "${ts:0:4}-${ts:4:2}-${ts:6:2}T${ts:9:2}:${ts:11:2}:${ts:13:2}Z",
  "server_url": "$SERVER_URL",
  "server_up": false,
  "duration_s": 0.0,
  "scenarios": {},
  "baseline": {},
  "verdict": {
    "overall": "INFRA",
    "text_only": "INFRA",
    "formula_heavy": "INFRA",
    "table_heavy": "INFRA",
    "consecutive_alerts": {},
    "reasons": ["health/ready unreachable"]
  }
}
EOF
  ln -sfn "$ts.json" "$OUT_DIR/latest.json"
  echo "$ts $SERVER_URL server_up:false (health probe failed)" \
    | tee "$OUT_DIR/latest.summary"
  exit 2
fi

# 2. run python orchestrator
cd "$REPO_DIR"
"$PY" tests/benchmark/router/bench_cua_router.py \
  --server-url "$SERVER_URL" \
  --out-dir "$OUT_DIR"
rc=$?

# 3. map exit code; orchestrator already wrote latest.json + summary
exit "$rc"
