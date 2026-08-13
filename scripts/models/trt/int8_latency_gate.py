#!/usr/bin/env python3
"""Compare wall-clock latency of two TRT engines (FP16 vs INT8) for the same model.

Per-task hard gate: INT8 engine MUST be ≥5 % faster (median over warmup-trimmed
runs) than the FP16 counterpart, or the script exits non-zero with a message
telling the user to drop the model from the INT8 experiment.

Inputs are taken from scripts/models/trt/calibration_data.py (first 20 — same distribution
as the calibration set, deterministic). Each model has a different shape, fed
via per-model preprocess from quantize_onnx_int8.py.

Usage:
    python3 int8_latency_gate.py \\
        --fp16-engine ~/.cache/turbo-ocr/det_<hash>.trt \\
        --int8-engine ~/.cache/turbo-ocr/det_<hash_int8>.trt \\
        --model-type det
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from calibration_data import calibration_image_paths  # noqa: E402
from quantize_onnx_int8 import build_calibration_loader  # noqa: E402

REQUIRED_SPEEDUP = 0.05  # 5 % faster, hard gate


def _load_engine(path: Path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {path}")
    return engine


def _bench(engine, inputs: list[dict]) -> float:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    import tensorrt as trt

    context = engine.create_execution_context()
    # Allocate per-binding device buffers sized to the largest input we will feed.
    # Output buffers sized from engine.get_tensor_shape after set_input_shape.
    bufs: dict[str, "cuda.DeviceAllocation"] = {}
    for sample in inputs[:1]:
        for name, arr in sample.items():
            context.set_input_shape(name, arr.shape)
            bufs[name] = cuda.mem_alloc(int(np.prod(arr.shape)) * arr.dtype.itemsize)
    # Output bindings.
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = context.get_tensor_shape(name)
            elt   = trt.nptype(engine.get_tensor_dtype(name))
            bufs[name] = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(elt).itemsize)
    for name, dptr in bufs.items():
        context.set_tensor_address(name, int(dptr))

    stream = cuda.Stream()
    times: list[float] = []
    # 3 warmup + 20 timed
    for k, sample in enumerate(inputs):
        for name, arr in sample.items():
            context.set_input_shape(name, arr.shape)
            cuda.memcpy_htod_async(bufs[name], np.ascontiguousarray(arr), stream)
        # Re-bind outputs after possible reshape.
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                context.set_tensor_address(name, int(bufs[name]))
        stream.synchronize()
        t0 = time.perf_counter()
        context.execute_async_v3(stream.handle)
        stream.synchronize()
        elapsed = time.perf_counter() - t0
        if k >= 3:
            times.append(elapsed)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16-engine", type=Path, required=True)
    ap.add_argument("--int8-engine", type=Path, required=True)
    ap.add_argument("--model-type", required=True)
    ap.add_argument("--n", type=int, default=23, help="3 warmup + 20 timed")
    ap.add_argument("--det-engine", type=Path, default=None,
                    help="FP16 det engine — required for --model-type=rec to "
                         "extract text crops (input to rec is crops, not pages).")
    args = ap.parse_args()

    for p in (args.fp16_engine, args.int8_engine):
        if not p.is_file():
            print(f"ERROR: engine missing: {p}", file=sys.stderr)
            return 1

    # For rec we need fewer pages — each page yields many crops, and the
    # loader caps at max_rec_crops anyway.
    if args.model_type == "rec":
        n_pages = 5
        samples = list(build_calibration_loader(
            args.model_type,
            calibration_image_paths(n=n_pages),
            det_engine=args.det_engine,
            max_rec_crops=args.n,
        ))
    else:
        imgs = calibration_image_paths(n=args.n)
        samples = list(build_calibration_loader(args.model_type, imgs))
    samples = samples[: args.n]
    if not samples:
        print("ERROR: calibration loader yielded zero samples", file=sys.stderr)
        return 1

    fp16_med = _bench(_load_engine(args.fp16_engine), samples)
    int8_med = _bench(_load_engine(args.int8_engine), samples)
    speedup = (fp16_med - int8_med) / fp16_med
    print(f"FP16 median: {fp16_med*1000:.3f} ms")
    print(f"INT8 median: {int8_med*1000:.3f} ms")
    print(f"Speedup: {speedup*100:+.1f}% (gate ≥ {REQUIRED_SPEEDUP*100:.0f}%)")
    if speedup < REQUIRED_SPEEDUP:
        print(f"FAIL: INT8 not ≥{REQUIRED_SPEEDUP*100:.0f}% faster than FP16.",
              file=sys.stderr)
        print(f"      Drop --model-type={args.model_type} from the INT8 experiment.",
              file=sys.stderr)
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
