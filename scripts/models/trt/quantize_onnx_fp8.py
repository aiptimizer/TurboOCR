#!/usr/bin/env python3
"""Produce a QDQ-bearing FP8 ONNX from a source FP16/FP32 ONNX via nvidia-modelopt.

Mirrors scripts/models/trt/quantize_onnx_int8.py — only the modelopt arguments differ:
    quantize_mode = "fp8" (E4M3FN)
    calibration_method = "max" (per-tensor amax — preserves FP8's wider dynamic
                                range; entropy KL is INT8-tuned and throws that
                                advantage away — see /tmp/fp8_research.md §Q4)

Output is written next to the source as `<stem>_fp8.onnx`. The C++ TRT builder
picks this up automatically when the matching `TURBO_OCR_<MODEL>_FP8=1` env
var is set (see src/runtime/engine/trt/onnx_to_trt.cpp `fp8_onnx_path`).

Models supported (vision backbones + the formula encoder; per
/tmp/fp8_research.md §Q5 these have zero or moderate group-conv content and
should clear the ≥5% latency gate on Blackwell sm_120):
    det, layout, table_struct_nemotron, formula_encoder

Explicitly NOT supported (and why):
    rec               — 37% group convs, prior 18% INT8 regression + sibling
                        repo recorded prior FP8 as "slower or equivalent";
                        TRT 10.15.1 release notes advise INT8 over FP8 for
                        ConvNets containing group convolutions.
    formula_decoder   — Calibration requires feeding real encoder context
                        tensors; needs a dedicated build script that runs
                        the FP16 encoder over the calibration set first. The
                        env var TURBO_OCR_FORMULA_DECODER_FP8 is wired in the
                        C++ side, but operators must produce the QDQ ONNX
                        manually (or via a follow-on script) until that
                        plumbing lands.

Calibration data: scripts/models/trt/calibration_data.py — same 200 stratified
OmniDocBench images that the INT8 pipeline uses. Preprocessors are reused
verbatim from quantize_onnx_int8.py.

Usage:
    python3 quantize_onnx_fp8.py --onnx models/det.onnx --model-type det
    python3 quantize_onnx_fp8.py --onnx models/layout/layout.onnx --model-type layout
    python3 quantize_onnx_fp8.py --onnx models/table/table_struct_nemotron.onnx \\
        --model-type table_struct_nemotron
    python3 quantize_onnx_fp8.py --onnx models/formula/encoder.onnx \\
        --model-type formula_encoder

Env requirement: same as INT8 — nvidia-modelopt with onnx>=1.18 (in
.venv-paddle, fix per the int8 script's docstring).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from calibration_data import calibration_image_paths  # noqa: E402
from quantize_onnx_int8 import build_calibration_loader  # noqa: E402

# FP8 supports the same 4 backbones as INT8. Rec is excluded for the same
# architectural reasons + the TRT 10.15.1 group-conv advisory; formula_decoder
# is excluded pending a dedicated context-calibration script.
SUPPORTED_TYPES = (
    "det", "layout", "table_struct_nemotron", "formula_encoder",
)


def run_quantize(onnx_path: Path, model_type: str, out_path: Path,
                 n_calib: int = 200) -> int:
    try:
        import modelopt.onnx.quantization as moq
    except (ImportError, AttributeError) as exc:
        print("ERROR: nvidia-modelopt unavailable in this venv.", file=sys.stderr)
        print(f"  underlying: {exc}", file=sys.stderr)
        print("  fix: .venv-paddle/bin/pip install -U 'onnx>=1.18'", file=sys.stderr)
        print("       or create a dedicated .venv-modelopt", file=sys.stderr)
        return 2

    images = calibration_image_paths(n=n_calib)
    print(f"calibration: {len(images)} images")

    # modelopt 0.44 MHA-exclusion logic calls .get_first() — same fix as
    # quantize_onnx_int8.py.
    class _Reader:
        def __init__(self, gen):
            self._items = list(gen)
            self._idx = 0
        def get_first(self):
            return self._items[0] if self._items else None
        def get_next(self):
            if self._idx >= len(self._items):
                return None
            item = self._items[self._idx]
            self._idx += 1
            return item
        def rewind(self):
            self._idx = 0

    reader = _Reader(build_calibration_loader(model_type, images))

    print(f"quantizing {onnx_path} → {out_path} "
          f"(model_type={model_type}, mode=fp8, calibration=max)")
    moq.quantize(
        onnx_path=str(onnx_path),
        quantize_mode="fp8",
        calibration_data_reader=reader,
        calibration_method="max",
        output_path=str(out_path),
    )
    if not out_path.exists():
        print(f"ERROR: modelopt did not produce {out_path}", file=sys.stderr)
        return 3
    print(f"OK: wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--model-type", required=True, choices=SUPPORTED_TYPES)
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output path (default: <stem>_fp8.onnx beside source)")
    args = ap.parse_args()

    if not args.onnx.is_file():
        print(f"ERROR: source ONNX missing: {args.onnx}", file=sys.stderr)
        return 1

    out = args.out
    if out is None:
        out = args.onnx.with_name(args.onnx.stem + "_fp8.onnx")

    return run_quantize(args.onnx, args.model_type, out, n_calib=args.n_calib)


if __name__ == "__main__":
    sys.exit(main())
