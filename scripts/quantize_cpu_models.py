#!/usr/bin/env python3
"""TurboOCR CPU INT8 quantization helper.

Dynamic INT8 quantization can reduce CPU inference latency by 1.2–2× on
x86_64/ARM64 with negligible accuracy loss on PP-OCRv6 models. This script
quantizes the standard TurboOCR ONNX models in-place (creating *_int8.onnx
siblings) and prints the env vars needed to use them.

Usage:
    python3 scripts/quantize_cpu_models.py --models-dir ./models
    python3 scripts/quantize_cpu_models.py --models-dir ./models --per-channel

After running, start the server with the quantized model paths:
    DET_MODEL=models/det_int8.onnx \
    REC_MODEL=models/rec_int8.onnx \
    CLS_MODEL=models/cls_int8.onnx \
    ./build_cpu/turboocr-cpu-server

Or with Docker, mount the quantized models over the baked ones:
    docker run -v $(pwd)/models:/app/models:ro ...

Requirements:
    pip install onnx onnxruntime

Note: This is an experimental optimization. Validate accuracy on your own
documents before deploying. Dynamic quantization does not require calibration
data; static quantization (not implemented here) can be more accurate but
needs a representative dataset.
"""

import argparse
import pathlib
import sys

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError as e:
    print("ERROR: onnxruntime quantization tools not installed.", file=sys.stderr)
    print("  pip install onnxruntime", file=sys.stderr)
    raise SystemExit(1) from e

DEFAULT_MODELS = {
    "det": ["det.onnx", "det_opt.onnx", "det_small.onnx", "det_small_opt.onnx", "det_tiny.onnx", "det_tiny_opt.onnx"],
    "rec": ["rec.onnx", "rec_opt.onnx", "rec_small.onnx", "rec_small_opt.onnx", "rec_tiny.onnx", "rec_tiny_opt.onnx"],
    "cls": ["cls.onnx", "cls_opt.onnx", "cls_x1_0.onnx", "cls_x1_0_opt.onnx"],
    "layout": ["layout/layout.onnx", "layout/layout_opt.onnx"],
    "doc_ori": ["doc_ori.onnx", "doc_ori_opt.onnx"],
}


def quantize_file(src: pathlib.Path, dst: pathlib.Path, per_channel: bool) -> bool:
    if not src.exists():
        print(f"  skip {src}: not found")
        return False
    if dst.exists():
        print(f"  skip {src}: {dst} already exists")
        return False
    print(f"  quantize {src} -> {dst}")
    try:
        quantize_dynamic(
            model_input=str(src),
            model_output=str(dst),
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            optimize_model=True,
        )
        return True
    except Exception as exc:
        print(f"  ERROR quantizing {src}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantize TurboOCR ONNX models for faster CPU inference."
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=pathlib.Path("models"),
        help="Directory containing the ONNX models (default: ./models)",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel quantization (often more accurate, slightly slower)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models that would be quantized without writing files",
    )
    args = parser.parse_args()

    models_dir: pathlib.Path = args.models_dir.resolve()
    if not models_dir.is_dir():
        print(f"ERROR: models directory not found: {models_dir}", file=sys.stderr)
        return 1

    print(f"Quantizing models in {models_dir} (per_channel={args.per_channel})")
    created = []
    for group, files in DEFAULT_MODELS.items():
        for fname in files:
            src = models_dir / fname
            if args.dry_run:
                if src.exists():
                    print(f"  would quantize {src}")
                else:
                    print(f"  skip {src}: not found")
                continue
            dst = src.with_stem(src.stem + "_int8")
            if quantize_file(src, dst, args.per_channel):
                created.append(dst)

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to write files.")
        return 0

    if not created:
        print("\nNo new models were quantized.")
        return 0

    print("\nQuantized models:")
    for p in created:
        print(f"  {p}")

    print("\nExample: run with the quantized models")
    print("  DET_MODEL=models/det_int8.onnx \\")
    print("  REC_MODEL=models/rec_int8.onnx \\")
    print("  CLS_MODEL=models/cls_int8.onnx \\")
    print("  ./build_cpu/turboocr-cpu-server")

    print("\nExample: Docker with volume-mounted quantized models")
    print("  docker run -d -p 8000:8000 \\")
    print("    -v $(pwd)/models:/app/models:ro \\")
    print("    -e DET_MODEL=/app/models/det_int8.onnx \\")
    print("    -e REC_MODEL=/app/models/rec_int8.onnx \\")
    print("    -e CLS_MODEL=/app/models/cls_int8.onnx \\")
    print("    turboocr-cpu:latest")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
