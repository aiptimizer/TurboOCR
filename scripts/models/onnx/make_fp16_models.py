#!/usr/bin/env python3
"""Produce fp16 siblings of the ONNX models, for the providers whose fp16 lives
in the MODEL rather than in a session option.

WHY THIS EXISTS
---------------
"Use fp16" means three different things depending on the execution provider
(include/turbo_ocr/backend/engine_mode.h::Fp16Support):

  OpenVINO   Provider — one session option (precision=FP16). Nothing to do here.
  CoreML     Native   — already computes fp16 on ANE/GPU. Nothing to do here.
  CUDA/DML/  Model    — the EP runs whatever the graph declares, so fp16 needs
  MIGraphX              an fp16 GRAPH. That is this script.

engine::CpuEngine looks for `<stem>.fp16.onnx` next to the model it was asked
for and uses it when fp16 is requested on one of those providers; when it is
absent it says so and runs fp32 (never silently). Deliberately NOT done at load
time: converting weights on the first request would charge that request for it,
and a server should not write into its models tree.

This is a WEIGHT CAST, not a graph build — it takes seconds and the result is
portable across GPUs/drivers, unlike a TensorRT engine.

USAGE
-----
    pip install onnx onnxconverter-common
    python3 scripts/models/onnx/make_fp16_models.py                 # every model in models/
    python3 scripts/models/onnx/make_fp16_models.py models/det_tiny.onnx models/rec_tiny.onnx
    python3 scripts/models/onnx/make_fp16_models.py --dir models --force

ACCURACY
--------
fp16 is not free: it is a real numeric change. Detection/recognition on these
models tolerate it well, but VERIFY rather than assume — run the gate:

    ./build/turbo_bench --backend <vendor> --tier tiny --images <funsd_cache> \\
        --count 50 --words w.json && python3 tools/bench/score_funsd.py w.json

`--keep-io-fp32` (the default) leaves graph inputs/outputs fp32 so the surrounding
C++ — which feeds and reads float32 buffers — needs no change at all.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Ops that must stay fp32 or the graph degrades badly. Normalisation and
# reductions accumulate, and in fp16 that accumulation is where detection masks
# go blotchy; the CTC-facing softmax/argmax tail is likewise not worth the risk
# for the handful of ops involved.
OP_BLOCKLIST = [
    "ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper",
    "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder",
    "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder",
    "RandomUniformLike", "SVMClassifier", "SVMRegressor", "Scaler",
    "TreeEnsembleClassifier", "TreeEnsembleRegressor", "ZipMap",
    "NonMaxSuppression", "TopK", "RoiAlign", "Range", "CumSum", "Min", "Max",
    "Upsample",
    # Ours, on top of the converter's defaults:
    "Softmax",           # rec tail -> CTC; keep the distribution exact
    "ReduceMean",        # LayerNorm-ish accumulation
    "InstanceNormalization",
]


def fp16_path(src: str) -> str:
    stem, ext = os.path.splitext(src)
    return f"{stem}.fp16{ext}"


def convert_one(src: str, *, keep_io_fp32: bool, force: bool) -> bool:
    """Convert one .onnx to its fp16 sibling. Returns True when written."""
    import onnx
    from onnxconverter_common import float16

    dst = fp16_path(src)
    if os.path.exists(dst) and not force:
        print(f"  skip (exists): {dst}   [--force to overwrite]")
        return False

    model = onnx.load(src)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_fp32,
        op_block_list=OP_BLOCKLIST,
        disable_shape_infer=False,
    )
    onnx.save(converted, dst)
    a, b = os.path.getsize(src), os.path.getsize(dst)
    print(f"  {src} -> {dst}   ({a/1e6:.1f} MB -> {b/1e6:.1f} MB)")
    return True


def discover(root: str) -> List[str]:
    """Every .onnx under `root` that is not itself an fp16 output."""
    out: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".onnx") and not f.endswith(".fp16.onnx"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="specific .onnx files (default: scan --dir)")
    ap.add_argument("--dir", default="models", help="tree to scan when no files are given")
    ap.add_argument("--force", action="store_true", help="overwrite existing .fp16.onnx")
    ap.add_argument("--io-fp16", action="store_true",
                    help="also convert graph inputs/outputs to fp16 (the C++ "
                         "feeds float32, so this needs caller changes — off by default)")
    a = ap.parse_args()

    try:
        import onnx  # noqa: F401
        from onnxconverter_common import float16  # noqa: F401
    except ImportError:
        print("error: needs `pip install onnx onnxconverter-common`", file=sys.stderr)
        return 2

    srcs = a.models or discover(a.dir)
    if not srcs:
        print(f"no .onnx models found under {a.dir!r}", file=sys.stderr)
        return 1

    print(f"fp16 conversion ({len(srcs)} model(s)); "
          f"io={'fp16' if a.io_fp16 else 'fp32 (kept)'}")
    written = 0
    for s in srcs:
        if not os.path.exists(s):
            print(f"  MISSING: {s}", file=sys.stderr)
            continue
        try:
            written += bool(convert_one(s, keep_io_fp32=not a.io_fp16, force=a.force))
        except Exception as e:  # one bad model must not abort the rest
            print(f"  FAILED {s}: {e}", file=sys.stderr)

    print(f"\nwrote {written} fp16 model(s).")
    print("These are picked up automatically when fp16 is requested on a "
          "MODEL-class provider (CUDA / DirectML / MIGraphX).")
    print("VERIFY accuracy before shipping — see the docstring for the gate command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
