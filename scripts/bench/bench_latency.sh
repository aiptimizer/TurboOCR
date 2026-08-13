#!/bin/bash
# Latency benchmark — sequential requests for accurate p50/p95/p99
HEY=~/go/bin/hey
SERVER=${OCR_URL:-http://localhost:8000}
IMAGE=${1:-tests/fixtures/images/png/receipt.png}
CONTENT_TYPE="image/png"

if [[ "$IMAGE" == *.jpg ]] || [[ "$IMAGE" == *.jpeg ]]; then
  CONTENT_TYPE="image/jpeg"
fi

echo "=== Latency Benchmark (sequential) ==="
echo "Image: $IMAGE"

# c=1 for pure latency measurement
$HEY -n 200 -c 1 -m POST \
  -T "$CONTENT_TYPE" \
  -D "$IMAGE" \
  "$SERVER/ocr/raw"
