#!/bin/bash
# Fetch a PP-OCRv5 language bundle from the TurboOCR GitHub Release
# (https://github.com/aiptimizer/TurboOCR/releases/tag/models-v2.1.0) and
# verify the bytes against SHA256SUMS before use.
#
# The release is authored by us from PaddlePaddle's official Baidu mirror
# (paddle-model-ecology.bj.bcebos.com) via `paddle2onnx`; dicts are verbatim
# from PaddleOCR/main:ppocr/utils/dict/ppocrv5_*.txt. Re-running the process
# end-to-end is reproducible with scripts/fetch_and_convert.sh.
#
# Usage:
#   ./scripts/download_models.sh --lang greek
#   ./scripts/download_models.sh --lang chinese
#   ./scripts/download_models.sh --lang chinese --server   # 84 MB server rec
#
# Env:
#   MODELS_RELEASE_URL — override the base URL (default: aiptimizer/TurboOCR
#                        release models-v2.1.0). Lets downstream forks pin
#                        their own release.

set -euo pipefail

SERVER=false
LANG_BUNDLE=""
OUT="models"
while (($#)); do
  case "$1" in
    --server) SERVER=true; shift ;;
    --lang)   LANG_BUNDLE="$2"; shift 2 ;;
    --lang=*) LANG_BUNDLE="${1#--lang=}"; shift ;;
    *)        OUT="$1"; shift ;;
  esac
done

if [[ -z "${LANG_BUNDLE}" ]]; then
  echo "Usage: $0 --lang <name> [--server] [out-dir]" >&2
  exit 2
fi

MODELS_RELEASE_URL="${MODELS_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v2.1.0}"

REC_DIR="${OUT}/rec/${LANG_BUNDLE}"
mkdir -p "${REC_DIR}"

# Atomic download + SHA256 verification against the release's SHA256SUMS.
fetch_verified() {
  local asset=$1 target=$2
  if [[ -f "$target" ]]; then
    echo "  $(basename "$target") exists, skipping"
    return 0
  fi
  local url="${MODELS_RELEASE_URL}/${asset}"
  echo "  fetch ${asset}"
  wget --tries=3 --timeout=30 --retry-connrefused -nv "$url" -O "${target}.part"

  # Grab the expected hash from SHA256SUMS (download once, cache alongside)
  local sums="${OUT}/SHA256SUMS.release.txt"
  if [[ ! -f "$sums" ]]; then
    wget --tries=3 --timeout=15 -nv "${MODELS_RELEASE_URL}/SHA256SUMS.txt" -O "$sums"
  fi
  local expected
  expected=$(awk -v a="$asset" '$2 == a {print $1}' "$sums")
  if [[ -z "$expected" ]]; then
    echo "  ERROR: no SHA256 entry for $asset in SHA256SUMS.txt" >&2
    rm -f "${target}.part"
    exit 1
  fi
  local actual
  actual=$(sha256sum "${target}.part" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "  ERROR: sha256 mismatch for $asset" >&2
    echo "    expected: $expected" >&2
    echo "    actual:   $actual" >&2
    rm -f "${target}.part"
    exit 1
  fi
  mv "${target}.part" "$target"
}

# Chinese: rec asset name depends on mobile vs server; dict is the same file.
if [[ "${LANG_BUNDLE}" == "chinese" ]]; then
  if [[ "${SERVER}" == true ]]; then
    echo "PP-OCRv5 chinese SERVER rec (84 MB) -> ${REC_DIR}/"
    fetch_verified "rec-chinese-server.onnx" "${REC_DIR}/rec.onnx"
  else
    echo "PP-OCRv5 chinese MOBILE rec (16 MB) -> ${REC_DIR}/"
    fetch_verified "rec-chinese.onnx" "${REC_DIR}/rec.onnx"
  fi
  fetch_verified "dict-chinese.txt" "${REC_DIR}/dict.txt"
else
  if [[ "${SERVER}" == true ]]; then
    echo "  NOTE: --server ignored (only chinese has a server variant)"
  fi
  echo "PP-OCRv5 ${LANG_BUNDLE} rec -> ${REC_DIR}/"
  fetch_verified "rec-${LANG_BUNDLE}.onnx" "${REC_DIR}/rec.onnx"
  fetch_verified "dict-${LANG_BUNDLE}.txt" "${REC_DIR}/dict.txt"
fi

echo ""
echo "Done. Run the server with:  OCR_LANG=${LANG_BUNDLE}"
ls -lh "${REC_DIR}"/{rec.onnx,dict.txt}
