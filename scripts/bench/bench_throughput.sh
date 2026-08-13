#!/bin/bash
# Benchmark throughput with hey (Go HTTP client)
# Usage: ./scripts/bench/bench_throughput.sh [image_file] [concurrency] [total_requests]

HEY=~/go/bin/hey
SERVER=${OCR_URL:-http://localhost:8000}
IMAGE=${1:-tests/fixtures/images/png/receipt.png}
CONCURRENCY=${2:-16}
TOTAL=${3:-1000}
CONTENT_TYPE="image/png"

# Auto-detect content type
if [[ "$IMAGE" == *.jpg ]] || [[ "$IMAGE" == *.jpeg ]]; then
  CONTENT_TYPE="image/jpeg"
fi

echo "=== Throughput Benchmark ==="
echo "Image: $IMAGE ($(du -h "$IMAGE" | cut -f1))"
echo "Concurrency: $CONCURRENCY"
echo "Total requests: $TOTAL"
echo ""

$HEY -n $TOTAL -c $CONCURRENCY -m POST \
  -T "$CONTENT_TYPE" \
  -D "$IMAGE" \
  "$SERVER/ocr/raw"
