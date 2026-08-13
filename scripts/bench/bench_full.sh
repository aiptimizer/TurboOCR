#!/bin/bash
# Full benchmark suite using hey
# Tests all image types at multiple concurrency levels
set -e

HEY=~/go/bin/hey
SERVER=${OCR_URL:-http://localhost:8000}
N=500

echo "============================================"
echo "  Turbo OCR Benchmark Suite (hey client)"
echo "============================================"
echo ""

# Warmup
echo "Warming up (50 requests)..."
$HEY -n 50 -c 8 -m POST -T "image/png" -D tests/fixtures/images/png/receipt.png "$SERVER/ocr/raw" > /dev/null 2>&1

echo ""
echo "--- Receipt (small, sparse) ---"
for c in 1 4 8 16 32; do
  echo ""
  echo "Concurrency: $c"
  $HEY -n $N -c $c -m POST -T "image/png" -D tests/fixtures/images/png/receipt.png "$SERVER/ocr/raw" 2>&1 | grep -E "Requests/sec|Latency|Total:|responses"
done

echo ""
echo "--- Dense document (mixed_fonts, 413 regions) ---"
for c in 1 4 8 16; do
  echo ""
  echo "Concurrency: $c"
  $HEY -n $((N/5)) -c $c -m POST -T "image/png" -D tests/fixtures/images/png/mixed_fonts.png "$SERVER/ocr/raw" 2>&1 | grep -E "Requests/sec|Latency|Total:|responses"
done

echo ""
echo "--- JPEG (restaurant menu) ---"
for c in 1 4 8 16; do
  echo ""
  echo "Concurrency: $c"
  $HEY -n $N -c $c -m POST -T "image/jpeg" -D tests/fixtures/images/jpeg/01_restaurant_menu.jpg "$SERVER/ocr/raw" 2>&1 | grep -E "Requests/sec|Latency|Total:|responses"
done

echo ""
echo "--- Format comparison (c=8, n=500) ---"
echo "PNG (receipt):"
$HEY -n $N -c 8 -m POST -T "image/png" -D tests/fixtures/images/png/receipt.png "$SERVER/ocr/raw" 2>&1 | grep "Requests/sec"
echo "JPEG (restaurant menu):"
$HEY -n $N -c 8 -m POST -T "image/jpeg" -D tests/fixtures/images/jpeg/01_restaurant_menu.jpg "$SERVER/ocr/raw" 2>&1 | grep "Requests/sec"

echo ""
echo "--- All test images (c=8, n=100 each) ---"
for f in tests/fixtures/images/png/*.png; do
  name=$(basename "$f" .png)
  result=$($HEY -n 100 -c 8 -m POST -T "image/png" -D "$f" "$SERVER/ocr/raw" 2>&1 | grep "Requests/sec")
  printf "  %-20s %s\n" "$name" "$result"
done
for f in tests/fixtures/images/jpeg/*.jpg; do
  name=$(basename "$f" .jpg)
  result=$($HEY -n 100 -c 8 -m POST -T "image/jpeg" -D "$f" "$SERVER/ocr/raw" 2>&1 | grep "Requests/sec")
  printf "  %-25s %s\n" "$name" "$result"
done

echo ""
echo "============================================"
echo "  Benchmark complete"
echo "============================================"
