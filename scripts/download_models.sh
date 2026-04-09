#!/bin/bash
# Download PP-OCR Latin models (ONNX format)
#
# Default: PP-OCRv4 det (mobile, 4.7MB) + PP-OCRv3 Latin rec (8MB)
# These give the best speed/accuracy tradeoff (F1=90.9%, 500+ img/s on GPU)
#
# Optional: --v5 flag downloads PP-OCRv5 models (slower det but newer rec)
#
# Usage: ./scripts/download_models.sh [--v5] [output_dir]

set -e

V5=false
OUT="models"
for arg in "$@"; do
  case $arg in
    --v5) V5=true ;;
    *) OUT="$arg" ;;
  esac
done

mkdir -p "$OUT"
dl() { [ -f "$OUT/$2" ] && echo "  $2 exists" || { echo "  Downloading $2..." && wget -q "$1" -O "$OUT/$2"; }; }

if [ "$V5" = true ]; then
  echo "Downloading PP-OCRv5 models to $OUT/"
  BASE="https://huggingface.co/monkt/paddleocr-onnx/resolve/main"
  dl "$BASE/detection/v5/det.onnx" "det.onnx"
  dl "$BASE/languages/latin/rec.onnx" "rec.onnx"
  dl "$BASE/languages/latin/dict.txt" "keys.txt"
  # No v5-specific cls, use v2
  dl "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx" "cls.onnx"
  echo "WARNING: v5 det is 84MB (server model) — much slower than v4 mobile (4.7MB)"
else
  echo "Downloading PP-OCRv4/v3 Latin models to $OUT/ (recommended for speed)"
  BASE="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0/onnx/PP-OCRv4"
  dl "$BASE/det/ch_PP-OCRv4_det_infer.onnx" "det.onnx"
  dl "$BASE/rec/latin_PP-OCRv3_rec_infer.onnx" "rec.onnx"
  dl "$BASE/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx" "cls.onnx"
  dl "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/dict/latin_dict.txt" "keys.txt"
fi

echo ""
echo "Models:"
ls -lh "$OUT"/{det,rec,cls}.onnx "$OUT"/keys.txt 2>/dev/null
echo ""
echo "TRT engines are auto-built from ONNX on first startup."
echo "CPU mode: use .onnx files directly."
