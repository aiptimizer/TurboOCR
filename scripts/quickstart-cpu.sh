#!/bin/bash
# TurboOCR CPU quickstart — one-command build + run, no GPU required.
#
# Usage:
#   scripts/quickstart-cpu.sh                       # default tiny model
#   OCR_MODEL=small scripts/quickstart-cpu.sh       # choose tier
#   OCR_MODEL=medium ORT_EP=xnnpack scripts/quickstart-cpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OCR_MODEL="${OCR_MODEL:-tiny}"
ORT_EP="${ORT_EP:-cpu}"

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  cat <<'EOF'
TurboOCR CPU quickstart

Environment variables:
  OCR_MODEL          tiny (default) | small | medium
  ORT_EP             cpu (default) | xnnpack | dnnl | openvino
  ORT_NUM_THREADS    threads per inference session (default: 4)
  PIPELINE_POOL_SIZE concurrent pipelines (default: 4)
  DISABLE_LAYOUT     1 to skip layout model and save ~124 MB RAM
  TABLE_BACKEND      slanext to enable table → HTML
  FORMULA_BACKEND    ppformulanet_s to enable formula → LaTeX

Examples:
  scripts/quickstart-cpu.sh
  OCR_MODEL=small ORT_EP=xnnpack scripts/quickstart-cpu.sh
  DISABLE_LAYOUT=1 scripts/quickstart-cpu.sh
EOF
  exit 0
fi

cd "${PROJECT_ROOT}"

echo "[quickstart-cpu] Building CPU image (OCR_MODEL=${OCR_MODEL}, ORT_EP=${ORT_EP})..."
docker build -f docker/Dockerfile.cpu -t turboocr-cpu:latest .

echo "[quickstart-cpu] Starting container on http://localhost:8000 ..."
docker run -d --name turboocr-cpu \
  -p 8000:8000 \
  -p 50051:50051 \
  -e OCR_MODEL="${OCR_MODEL}" \
  -e ORT_EP="${ORT_EP}" \
  -e ORT_NUM_THREADS="${ORT_NUM_THREADS:-4}" \
  -e PIPELINE_POOL_SIZE="${PIPELINE_POOL_SIZE:-4}" \
  -e DISABLE_LAYOUT="${DISABLE_LAYOUT:-0}" \
  -e TABLE_BACKEND="${TABLE_BACKEND:-}" \
  -e FORMULA_BACKEND="${FORMULA_BACKEND:-}" \
  -e MAX_BODY_MB="${MAX_BODY_MB:-100}" \
  --restart unless-stopped \
  turboocr-cpu:latest

echo "[quickstart-cpu] Waiting for /health (up to 60s)..."
for i in {1..60}; do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "[quickstart-cpu] Ready. Test with:"
    echo "  curl -X POST http://localhost:8000/ocr/raw \\"
    echo "       --data-binary @tests/test_data/png/receipt.png -H 'Content-Type: image/png'"
    echo ""
    echo "  docker logs -f turboocr-cpu"
    exit 0
  fi
  sleep 1
done

echo "[quickstart-cpu] ERROR: server did not become healthy in 60s."
docker logs turboocr-cpu --tail 50 || true
exit 1
