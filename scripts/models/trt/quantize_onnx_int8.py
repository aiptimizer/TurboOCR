#!/usr/bin/env python3
"""Produce a QDQ-bearing INT8 ONNX from a source FP16/FP32 ONNX via nvidia-modelopt.

Output is written next to the source as `<stem>.int8.onnx`. The C++ TRT builder
picks this up automatically when the matching `TURBO_OCR_<MODEL>_INT8=1` env
var is set (see src/runtime/engine/trt/onnx_to_trt.cpp `int8_onnx_path_for`).

Calibration source: scripts/models/trt/calibration_data.py picks 200 stratified
OmniDocBench images deterministically. Each model needs its own preprocess
(input shape + normalization), wired below per --model-type.

Models supported (low-risk vision backbones, per quant-research path F1):
    det, layout, table_struct_nemotron, formula_encoder

NOT supported here (AR / CTC paths — go through #10/#11/#12):
    rec, formula_decoder, slanext_wired, slanext_wireless

Usage:
    python3 quantize_onnx_int8.py --onnx models/det.onnx --model-type det
    python3 quantize_onnx_int8.py --onnx models/layout/layout.onnx --model-type layout
    python3 quantize_onnx_int8.py --onnx models/table/table_struct_nemotron.onnx \\
        --model-type table_struct_nemotron
    python3 quantize_onnx_int8.py --onnx models/formula/encoder.onnx \\
        --model-type formula_encoder

Env requirements (gap as of 2026-05-17): nvidia-modelopt 0.37 in .venv-paddle
triggers `AttributeError: FLOAT4E2M1` on `modelopt.onnx` import because
onnx<1.17 lacks the FLOAT4E2M1 tensor type. Fix before running:
    .venv-paddle/bin/pip install -U 'onnx>=1.18'
or create a dedicated .venv-modelopt with onnx>=1.18 + nvidia-modelopt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from calibration_data import calibration_image_paths  # noqa: E402

SUPPORTED_TYPES = (
    "det", "layout", "table_struct_nemotron", "formula_encoder",
    "rec",
)

# Path G (#10): surgical exclusion regex for rec.onnx. Per
# /tmp/paddle_rec_int8_research.md §3, the SVTR_LCNet attention path and the
# PP-LCNetV3 SE gates are the specific failure modes:
#   - 13 MatMul (SVTR Q/K/V projections + attention QK^T/AV + CTC head)
#   - 3 Softmax (SVTR attention)
#   - 30 HardSigmoid (PP-LCNetV3 SE blocks — piecewise-linear knee outliers)
# Names in the current export are bare `MatMul.<n>` / `Softmax.<n>` /
# `HardSigmoid.<n>` (verified via onnx.load). The regex below tolerates a
# leading slash in case a future re-export uses path-style names.
REC_EXCLUDE_REGEX = [
    r"^/?MatMul(\.|/|$).*",
    r"^/?Softmax(\.|/|$).*",
    r"^/?HardSigmoid(\.|/|$).*",
]


def _preprocess_det(path: Path, max_side: int = 960) -> np.ndarray:
    # Matches src/models/detection/paddle_det.cpp + the fused_resize_normalize_chw kernel:
    # ImageNet mean/std, BGR→RGB, round to /32, dynamic shape.
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    h, w = img.shape[:2]
    r = min(1.0, max_side / max(h, w))
    rh = max(int(round(h * r / 32.0) * 32), 32)
    rw = max(int(round(w * r / 32.0) * 32), 32)
    resized = cv2.resize(img, (rw, rh))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = ((rgb - mean) / std).transpose(2, 0, 1)
    return chw[np.newaxis]  # [1,3,H,W]


def _letterbox(img: np.ndarray, size: int, pad: int = 114) -> np.ndarray:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas


def _preprocess_layout(path: Path, size: int = 800) -> dict:
    # PP-DocLayoutV3: 3 inputs (image, im_shape, scale_factor), 800×800 padded.
    # Profile lives in src/runtime/engine/trt/onnx_to_trt.cpp:266-292.
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    canvas = _letterbox(img, size)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    chw = ((rgb - mean) / std).transpose(2, 0, 1)[np.newaxis]
    im_shape     = np.array([[size, size]], dtype=np.float32)
    scale_factor = np.array([[1.0, 1.0]],   dtype=np.float32)
    return {"image": chw, "im_shape": im_shape, "scale_factor": scale_factor}


def _preprocess_nemotron(path: Path, size: int = 1024) -> np.ndarray:
    # nvidia/nemotron-table-structure-v1: letterbox to 1024 with pad=114, 0..255
    # range (NOT ImageNet-normalized). Per models/MANIFEST.txt notes.
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    canvas = _letterbox(img, size, pad=114).astype(np.float32)  # keep 0..255
    rgb = canvas[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1)[np.newaxis]
    return chw


REC_WIDTH_BUCKETS = (160, 320, 480, 800, 1200, 1600, 2000, 2500, 3200, 4000)


def _snap_width(w: int) -> int:
    for b in REC_WIDTH_BUCKETS:
        if w <= b:
            return b
    return REC_WIDTH_BUCKETS[-1]


def _preprocess_rec_crop(crop_bgr: np.ndarray, target_h: int = 48) -> np.ndarray:
    # Matches src/models/recognition/paddle_rec.cpp + the fused batch_roi_warp kernel:
    #   resize → height=48 preserving aspect, snap width to bucket, pad right
    #   with zeros, normalize (mean=0.5 std=0.5 per PaddleOCR rec recipe).
    h, w = crop_bgr.shape[:2]
    aspect = w / max(h, 1)
    target_w = _snap_width(max(int(target_h * aspect), 1))
    new_w = min(int(target_h * aspect), target_w)
    new_w = max(new_w, 1)
    resized = cv2.resize(crop_bgr, (new_w, target_h))
    if new_w < target_w:
        pad = np.zeros((target_h, target_w - new_w, 3), dtype=np.uint8)
        resized = np.concatenate([resized, pad], axis=1)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    chw = norm.transpose(2, 0, 1)
    return chw[np.newaxis]  # [1,3,48,target_w]


def _det_engine_extract_crops(image_paths: list[Path],
                              det_engine_path: Path,
                              max_crops_per_image: int = 60) -> list[np.ndarray]:
    # Lazy: only imported when rec calibration is actually run, to keep the
    # other model paths free of pycuda/tensorrt deps.
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    import tensorrt as trt

    print(f"loading FP16 det engine: {det_engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(det_engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {det_engine_path}")
    context = engine.create_execution_context()
    in_name  = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)

    all_crops: list[np.ndarray] = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        inp = _preprocess_det(path)
        context.set_input_shape(in_name, inp.shape)
        out_shape = context.get_tensor_shape(out_name)
        in_buf  = cuda.mem_alloc(inp.nbytes)
        out_buf = cuda.mem_alloc(int(np.prod(out_shape)) * 4)
        out_host = np.empty(out_shape, dtype=np.float32)
        context.set_tensor_address(in_name,  int(in_buf))
        context.set_tensor_address(out_name, int(out_buf))
        cuda.memcpy_htod(in_buf, np.ascontiguousarray(inp))
        context.execute_async_v3(0)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(out_host, out_buf)
        in_buf.free(); out_buf.free()

        prob = out_host.squeeze()
        binary = (prob > 0.3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sx = w / inp.shape[3]
        sy = h / inp.shape[2]
        crops_for_image = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1 = max(0, int(x  * sx) - 2)
            y1 = max(0, int(y  * sy) - 2)
            x2 = min(w, int((x + bw) * sx) + 2)
            y2 = min(h, int((y + bh) * sy) + 2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = img[y1:y2, x1:x2].copy()
            if crop.size > 0:
                all_crops.append(crop)
                crops_for_image += 1
                if crops_for_image >= max_crops_per_image:
                    break
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(image_paths)} pages, {len(all_crops)} crops")
    print(f"extracted {len(all_crops)} text crops total")
    return all_crops


def _preprocess_formula_encoder(path: Path, h: int = 192, w: int = 672) -> np.ndarray:
    # RapidLaTeXOCR encoder.onnx: 1-channel grayscale at fixed 192×672 bucket
    # (the v1 single-profile pin in src/runtime/engine/trt/onnx_to_trt.cpp:381-390). Real
    # crops are formula sub-regions; OmniDocBench page-level images are still
    # a workable activation-distribution proxy here because the encoder is
    # backbone-only (HGNetV2-B4) — formula-vs-text is a head-side decision.
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    src_h, src_w = img.shape
    r = min(h / src_h, w / src_w)
    nh, nw = max(int(round(src_h * r)), 1), max(int(round(src_w * r)), 1)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((h, w), 255, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    norm = canvas.astype(np.float32) / 255.0
    return norm[np.newaxis, np.newaxis]  # [1,1,H,W]


def build_calibration_loader(model_type: str, image_paths: list[Path],
                              det_engine: Path | None = None,
                              max_rec_crops: int = 500):
    """Yield dicts {input_name: ndarray} compatible with modelopt's DataLoader."""
    if model_type == "det":
        for p in image_paths:
            yield {"x": _preprocess_det(p)}
    elif model_type == "layout":
        for p in image_paths:
            yield _preprocess_layout(p)
    elif model_type == "table_struct_nemotron":
        for p in image_paths:
            # The nemotron ONNX input name is "images" per the upstream export
            # (verify if quantize call complains; the canonical name in
            # NVIDIA's release is "input" or "images"; fall back to picking
            # the first graph input at call site if needed).
            yield {"images": _preprocess_nemotron(p)}
    elif model_type == "formula_encoder":
        for p in image_paths:
            yield {"input": _preprocess_formula_encoder(p)}
    elif model_type == "rec":
        if det_engine is None or not det_engine.is_file():
            raise RuntimeError(
                "rec calibration needs the FP16 det engine to extract text "
                "crops — pass --det-engine /path/to/det_<hash>.trt"
            )
        crops = _det_engine_extract_crops(image_paths, det_engine)
        if not crops:
            raise RuntimeError("no text crops extracted — check det engine + images")
        if len(crops) > max_rec_crops:
            # Deterministic stride sample to cap calibration cost.
            stride = len(crops) // max_rec_crops
            crops = crops[::stride][:max_rec_crops]
            print(f"sampled {len(crops)} crops (cap={max_rec_crops})")
        for c in crops:
            yield {"x": _preprocess_rec_crop(c)}
    else:
        raise ValueError(f"Unsupported model-type: {model_type}")


def run_quantize(onnx_path: Path, model_type: str, out_path: Path,
                 n_calib: int = 200,
                 det_engine: Path | None = None,
                 extra_exclude_regex: list[str] | None = None,
                 rec_exclude: bool = True) -> int:
    # Lazy import — modelopt currently breaks on .venv-paddle (onnx<1.18 missing
    # FLOAT4E2M1). Surface a clean error pointing at the fix instead of a
    # 30-line traceback.
    try:
        import modelopt.onnx.quantization as moq  # noqa: F401
    except (ImportError, AttributeError) as exc:
        print("ERROR: nvidia-modelopt unavailable in this venv.", file=sys.stderr)
        print(f"  underlying: {exc}", file=sys.stderr)
        print("  fix: run with <repo>/"
              ".venv-modelopt/bin/python (preferred — pinned to a working "
              "onnx≥1.21 + modelopt 0.44)", file=sys.stderr)
        return 2

    images = calibration_image_paths(n=n_calib)
    print(f"calibration: {len(images)} images")

    # modelopt's onnx PTQ entrypoint takes a calibration_data_reader that
    # mirrors onnxruntime's CalibrationDataReader interface.
    # modelopt 0.44 MHA-exclusion logic calls .get_first() to peek at input
    # shapes without consuming the iterator. Materialize the loader so we can
    # serve both get_first (idempotent peek) and get_next (consume + advance).
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

    reader = _Reader(build_calibration_loader(model_type, images,
                                               det_engine=det_engine))

    # Path G surgical exclusions for rec (#10): see REC_EXCLUDE_REGEX above.
    # Other models pass no exclusions — modelopt's default coverage is fine.
    exclude_regex = list(extra_exclude_regex or [])
    if model_type == "rec" and rec_exclude:
        exclude_regex.extend(REC_EXCLUDE_REGEX)

    print(f"quantizing {onnx_path} → {out_path} (model_type={model_type})")
    if exclude_regex:
        print(f"  exclude regex: {exclude_regex}")
    kwargs = dict(
        onnx_path=str(onnx_path),
        quantize_mode="int8",
        calibration_data_reader=reader,
        calibration_method="entropy",
        output_path=str(out_path),
        # Keep non-quantized tensors (including the DQ scale initializers) in
        # FP32. modelopt's default "fp16" runs `convert_to_f16` which sweeps
        # the DQ scale tensors → FP16, and TRT 10.15.1 then refuses them with
        # "[SCALE] has invalid precision Int8, ignored" (TRT requires FP32
        # scales for INT8 DQ on weakly-typed networks; FP16 scales are only
        # accepted on strongly-typed). The TRT engine builder still applies
        # kFP16 to compute, so runtime perf is unchanged.
        high_precision_dtype="fp32",
    )
    if exclude_regex:
        # modelopt accepts both `op_types_to_exclude` (op-type filter) and
        # `nodes_to_exclude` (regex on node name). Path G needs name-regex.
        kwargs["nodes_to_exclude"] = exclude_regex
    moq.quantize(**kwargs)
    if not out_path.exists():
        print(f"ERROR: modelopt did not produce {out_path}", file=sys.stderr)
        return 3

    # Safety net — independent of modelopt version, walk every DQ + Q node
    # and force scale initializers to FP32 if any slipped through as FP16.
    coerced = _coerce_qdq_scales_to_fp32(out_path)
    if coerced:
        print(f"  post-process: cast {coerced} QDQ scale initializer(s) to FP32")

    print(f"OK: wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return 0


def _coerce_qdq_scales_to_fp32(onnx_path: Path) -> int:
    # TRT 10.15 has a parser bug: when a QuantizeLinear node has FP16 weight
    # inputs, the validator misidentifies the scale's required precision and
    # silently ignores ALL scales (gemini-confirmed against TRT 10.15.1).
    # Symptom: "[SCALE] has invalid precision Int8, ignored" even when DQ
    # scales are demonstrably FP32. Fix: cast both the scale tensors AND the
    # Q weight inputs to FP32.
    import onnx
    from onnx import numpy_helper, TensorProto

    model = onnx.load(str(onnx_path))
    targets = set()
    for node in model.graph.node:
        if node.op_type in ("DequantizeLinear", "QuantizeLinear"):
            if len(node.input) >= 2:
                targets.add(node.input[1])  # scale
        if node.op_type == "QuantizeLinear" and len(node.input) >= 1:
            targets.add(node.input[0])      # weight/data input (TRT 10.15 parser fix)

    changed = 0
    new_initializers = []
    for init in model.graph.initializer:
        if init.name in targets and init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype("float32")
            new_init = numpy_helper.from_array(arr, init.name)
            new_initializers.append(new_init)
            changed += 1
        else:
            new_initializers.append(init)

    if changed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(new_initializers)
        onnx.save(model, str(onnx_path))
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--model-type", choices=SUPPORTED_TYPES,
                    help="Required unless --repair-only is set.")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output path. Default: <stem>.int8.onnx for the "
                         "low-risk backbones, <stem>_qdq.onnx for rec (Path G).")
    ap.add_argument("--det-engine", type=Path, default=None,
                    help="Path to FP16 det TRT engine (rec calibration only — "
                         "used to extract text crops from the calibration pages).")
    ap.add_argument("--repair-only", action="store_true",
                    help="Skip modelopt; only run the FP32-scale safety pass "
                         "in place on --onnx. Use this to repair an existing "
                         "QDQ ONNX whose DQ scales are FP16 (TRT 10.15 reject).")
    ap.add_argument("--no-exclude", action="store_true",
                    help="rec only: quantize ALL eligible nodes (skip the Path G "
                         "MatMul/Softmax/HardSigmoid exclusions). Fully-contiguous "
                         "INT8 minimizes reformat islands — favors speed over accuracy.")
    args = ap.parse_args()

    if not args.onnx.is_file():
        print(f"ERROR: source ONNX missing: {args.onnx}", file=sys.stderr)
        return 1

    if args.repair_only:
        coerced = _coerce_qdq_scales_to_fp32(args.onnx)
        print(f"repair: cast {coerced} QDQ scale initializer(s) to FP32 in {args.onnx}")
        return 0

    if not args.model_type:
        print("ERROR: --model-type required (omit only with --repair-only)",
              file=sys.stderr)
        return 1

    out = args.out
    if out is None:
        if args.model_type == "rec":
            # Path G (#10): different suffix from the low-risk INT8 path so the
            # two never collide on disk or in the C++ cache lookup.
            out = args.onnx.with_name(args.onnx.stem + "_qdq.onnx")
        else:
            out = args.onnx.with_name(args.onnx.stem + ".int8.onnx")

    return run_quantize(args.onnx, args.model_type, out,
                        n_calib=args.n_calib,
                        det_engine=args.det_engine,
                        rec_exclude=not args.no_exclude)


if __name__ == "__main__":
    sys.exit(main())
