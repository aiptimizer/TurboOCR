#!/bin/bash
# TurboOCR CPU benchmark — measure throughput and latency against a running
# CPU-only server. No GPU required.
#
# Usage:
#   scripts/benchmark-cpu.sh [image_or_pdf] [seconds]
#
# Defaults:
#   image_or_pdf: tests/test_data/png/receipt.png (if present) or first .png/.pdf
#   seconds:      30
#
# Requires: curl, jq, bc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="${BASE_URL:-http://localhost:8000}"
INPUT_FILE="${1:-}"
DURATION="${2:-30}"

if [[ -z "$INPUT_FILE" ]]; then
  # Try to find a default sample image
  for candidate in \
      "${PROJECT_ROOT}/tests/test_data/png/receipt.png" \
      "${PROJECT_ROOT}/tests/test_data/receipt.png" \
      "${PROJECT_ROOT}/assets/sample.png"; do
    if [[ -f "$candidate" ]]; then
      INPUT_FILE="$candidate"
      break
    fi
  done
  if [[ -z "$INPUT_FILE" ]]; then
    # Fall back to any png/pdf in the project
    INPUT_FILE="$(find "${PROJECT_ROOT}" -maxdepth 3 -type f \( -iname '*.png' -o -iname '*.pdf' \) 2>/dev/null | head -n1 || true)"
  fi
fi

if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
  echo "[benchmark-cpu] ERROR: no input image/PDF found." >&2
  echo "  Provide one: scripts/benchmark-cpu.sh path/to/receipt.png [seconds]" >&2
  exit 1
fi

for cmd in curl jq bc; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[benchmark-cpu] ERROR: required command '$cmd' not found." >&2
    exit 1
  fi
done

EXT="${INPUT_FILE##*.}"
EXT_LOWER="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"

if [[ "$EXT_LOWER" == "pdf" ]]; then
  ENDPOINT="${BASE_URL}/ocr/pdf"
  MIME="application/pdf"
  LABEL="PDF pages"
else
  ENDPOINT="${BASE_URL}/ocr/raw?layout=1"
  MIME="image/${EXT_LOWER}"
  LABEL="images"
fi

echo "[benchmark-cpu] Endpoint: ${ENDPOINT}"
echo "[benchmark-cpu] File:     ${INPUT_FILE} ($(stat -c%s "$INPUT_FILE" 2>/dev/null || stat -f%z "$INPUT_FILE") bytes)"
echo "[benchmark-cpu] Duration: ${DURATION}s"
echo "[benchmark-cpu] Warm-up:  2 requests"

# Warm-up
for _ in 1 2; do
  curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$ENDPOINT" \
    --data-binary "@$INPUT_FILE" \
    -H "Content-Type: $MIME" >/dev/null || true
  sleep 0.5
done

START=$(date +%s.%N)
COUNT=0
FAIL=0
TOTAL_MS=0

while true; do
  NOW=$(date +%s.%N)
  ELAPSED=$(echo "$NOW - $START" | bc)
  if (( $(echo "$ELAPSED >= $DURATION" | bc -l) )); then
    break
  fi

  RESULT=$(curl -s -o /tmp/turboocr_bench.json -w "%{http_code}|%{time_total}" \
    -X POST "$ENDPOINT" \
    --data-binary "@$INPUT_FILE" \
    -H "Content-Type: $MIME")

  HTTP_CODE="${RESULT%|*}"
  TIME_TOTAL="${RESULT#*|}"

  if [[ "$HTTP_CODE" == "200" ]]; then
    COUNT=$((COUNT + 1))
    TOTAL_MS=$(echo "$TOTAL_MS + $TIME_TOTAL * 1000" | bc)
  else
    FAIL=$((FAIL + 1))
  fi
done

END=$(date +%s.%N)
TOTAL_ELAPSED=$(echo "$END - $START" | bc)

if [[ $COUNT -eq 0 ]]; then
  echo "[benchmark-cpu] ERROR: all requests failed (HTTP codes not 200)." >&2
  echo "  Sample response:"
  cat /tmp/turboocr_bench.json 2>/dev/null || true
  exit 1
fi

AVG_MS=$(echo "scale=2; $TOTAL_MS / $COUNT" | bc)
TPS=$(echo "scale=2; $COUNT / $TOTAL_ELAPSED" | bc)

# Count output elements for a rough "items per second" metric
if [[ "$EXT_LOWER" == "pdf" ]]; then
  ITEMS="$(jq -r '[.results[]?] | length' /tmp/turboocr_bench.json 2>/dev/null || echo 0)"
  ITEMS_PER_SEC="$(echo "scale=2; $ITEMS * $TPS" | bc)"
  echo "[benchmark-cpu] Results:"
  echo "  $LABEL:     $COUNT in ${TOTAL_ELAPSED}s"
  echo "  Throughput: $TPS ${LABEL}/s"
  echo "  Avg latency: ${AVG_MS}ms"
  echo "  Text items:  $ITEMS total (≈$ITEMS_PER_SEC items/s)"
else
  WORDS="$(jq -r '[.results[]?.text] | join(" ") | split(" ") | length' /tmp/turboocr_bench.json 2>/dev/null || echo 0)"
  WORDS_PER_SEC="$(echo "scale=2; $WORDS * $TPS" | bc)"
  echo "[benchmark-cpu] Results:"
  echo "  $LABEL:     $COUNT in ${TOTAL_ELAPSED}s"
  echo "  Throughput: $TPS ${LABEL}/s"
  echo "  Avg latency: ${AVG_MS}ms"
  echo "  Words:       $WORDS total (≈$WORDS_PER_SEC words/s)"
fi

if [[ $FAIL -gt 0 ]]; then
  echo "  Failed:      $FAIL"
fi

echo ""
echo "[benchmark-cpu] Tip: retry with different ORT_EP values:"
echo "  ORT_EP=cpu     scripts/benchmark-cpu.sh $INPUT_FILE $DURATION"
echo "  ORT_EP=xnnpack scripts/benchmark-cpu.sh $INPUT_FILE $DURATION"
echo "  ORT_EP=dnnl    scripts/benchmark-cpu.sh $INPUT_FILE $DURATION"
