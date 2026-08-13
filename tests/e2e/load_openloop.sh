#!/usr/bin/env bash
# Open-loop load test — the measurement soak.py cannot make.
#
# soak.py is CLOSED-loop: each worker waits for a response before sending the
# next request. Under saturation that self-throttles, so its latency numbers
# are queue-time at whatever rate the server absorbs — coordinated omission —
# and it can never offer MORE load than capacity, so overload behaviour is
# invisible to it. Use soak.py for its resource gates (fds, temp dirs, RSS
# settle); use THIS for latency and overload.
#
# Open loop (vegeta) sends at a fixed arrival rate regardless of responses.
# Measured on the RTX 5090 / tiny with a dense page, 2026-08-11:
#
#   rate 50/s   p50  33 ms   p99  38 ms       flat
#   rate 70/s   p50  33 ms   p99  43 ms       flat — capacity is ~80/s
#   rate 90/s   p50 2.3 s    max 4.6 s        queue grows without bound, all 200s
#
# The closed-loop soak at the same saturation reported p50 306 ms and looked
# healthy; the 33 ms above is the actual service time. Both are true — they
# answer different questions.
#
# OVERLOAD: the third row is the finding. WORK_QUEUE_DEPTH defaults to 8192,
# and past capacity the queue fills at (offered - capacity) req/s — at 10/s
# excess that is ~14 minutes to reach the bound, while REQUEST_TIMEOUT_MS
# (60 s) fires first. So an overloaded server climbs toward timeout instead of
# shedding. Size the queue to a latency budget instead:
#
#   WORK_QUEUE_DEPTH  =  capacity_req_s  x  max_acceptable_latency_s
#
# Verified with WORK_QUEUE_DEPTH=64 at 90/s offered: 91% served with p99 1.0 s
# (= 64 / 80 req/s), 9% shed as immediate 503 SERVER_BUSY (sub-ms), and zero
# false 503s below capacity.
#
# Usage:
#   tests/e2e/load_openloop.sh <url> <image> <rate1,rate2,...> [duration]
# e.g.
#   tests/e2e/load_openloop.sh http://127.0.0.1:8080 \
#       tests/fixtures/images/png/mixed_fonts.png 50,70,90 30s
set -euo pipefail
URL=${1:?url}; IMG=${2:?image}; RATES=${3:?rates}; DUR=${4:-30s}
command -v vegeta >/dev/null || { echo "vegeta not found — https://github.com/tsenart/vegeta/releases" >&2; exit 2; }
T=$(mktemp)
trap 'rm -f "$T"' EXIT
printf 'POST %s/ocr/raw\nContent-Type: image/png\n@%s\n' "$URL" "$(cd "$(dirname "$IMG")" && pwd)/$(basename "$IMG")" > "$T"
vegeta attack -targets="$T" -rate=10 -duration=8s -timeout=60s >/dev/null   # warmup
for R in ${RATES//,/ }; do
  echo "=== rate ${R}/s (${DUR}) ==="
  vegeta attack -targets="$T" -rate="$R" -duration="$DUR" -timeout=60s \
    | vegeta report | grep -E "Latencies|Success|Status"
done
