#!/usr/bin/env python3
"""Compare wall-clock latency of two TRT engines (FP16 vs FP8) for the same model.

Per task #11 hard gate: FP8 engine MUST be ≥5 % faster (median over warmup-
trimmed runs) than the FP16 counterpart, or exits non-zero. This is the
specific check for the "FP8 storage with FP16 compute" failure mode flagged
in /tmp/fp8_research.md §Q6 — group-conv-heavy models will fall back to
non-FP8 kernels per the TRT 10.15.1 release-note advisory, and that shows
up here as a flat or negative speedup.

Inputs reuse scripts/models/trt/calibration_data.py + scripts/models/trt/quantize_onnx_int8.py's
per-model preprocessors (det / layout / table_struct_nemotron / formula_encoder).
formula_decoder needs a dedicated harness — not handled here.

Usage:
    python3 fp8_latency_gate.py \\
        --fp16-engine ~/.cache/turbo-ocr/det_<hash>.trt \\
        --fp8-engine  ~/.cache/turbo-ocr/det_<hash_fp8>.trt \\
        --model-type  det
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

REQUIRED_SPEEDUP = 0.05  # 5 % faster, hard gate per task #11


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
    bufs: dict[str, "cuda.DeviceAllocation"] = {}
    for sample in inputs[:1]:
        for name, arr in sample.items():
            context.set_input_shape(name, arr.shape)
            bufs[name] = cuda.mem_alloc(int(np.prod(arr.shape)) * arr.dtype.itemsize)
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
    for k, sample in enumerate(inputs):
        for name, arr in sample.items():
            context.set_input_shape(name, arr.shape)
            cuda.memcpy_htod_async(bufs[name], np.ascontiguousarray(arr), stream)
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
    ap.add_argument("--fp8-engine",  type=Path, required=True)
    ap.add_argument("--model-type", required=True)
    ap.add_argument("--n", type=int, default=23, help="3 warmup + 20 timed")
    args = ap.parse_args()

    for p in (args.fp16_engine, args.fp8_engine):
        if not p.is_file():
            print(f"ERROR: engine missing: {p}", file=sys.stderr)
            return 1

    imgs = calibration_image_paths(n=args.n)
    samples = list(build_calibration_loader(args.model_type, imgs))

    fp16_med = _bench(_load_engine(args.fp16_engine), samples)
    fp8_med  = _bench(_load_engine(args.fp8_engine),  samples)
    speedup = (fp16_med - fp8_med) / fp16_med
    print(f"FP16 median: {fp16_med*1000:.3f} ms")
    print(f"FP8  median: {fp8_med*1000:.3f} ms")
    print(f"Speedup: {speedup*100:+.1f}% (gate ≥ {REQUIRED_SPEEDUP*100:.0f}%)")
    if speedup < REQUIRED_SPEEDUP:
        print(f"FAIL: FP8 not ≥{REQUIRED_SPEEDUP*100:.0f}% faster than FP16.",
              file=sys.stderr)
        print("      Likely cause: 'FP8 storage with FP16 compute' fallback "
              "(group convs / unsupported op).", file=sys.stderr)
        print(f"      Drop --model-type={args.model_type} from the FP8 experiment.",
              file=sys.stderr)
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
