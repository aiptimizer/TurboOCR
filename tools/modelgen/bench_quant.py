"""Isolated ORT latency benchmark: fp32 vs int8 OCR models on the CPU EP.

Reports median ms over N runs at representative widths, single-thread and
4-thread, so we can confirm AVX-VNNI actually accelerates the INT8 model.

Usage:
  bench_quant.py rec models/rec.onnx models/rec.int8.static.onnx [more.onnx ...]
  bench_quant.py det models/det.onnx models/det.int8.onnx models/det.int8.static.onnx
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_common as Q

REC_WIDTHS = [320, 640, 1280]
DET_SIZES = [(640, 640), (960, 960)]


def sess(path, intra):
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so,
                               providers=["CPUExecutionProvider"])


def timeit(s, feed, warmup=5, runs=30):
    name = s.get_inputs()[0].name
    for _ in range(warmup):
        s.run(None, {name: feed})
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, {name: feed})
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]  # median ms


def make_feed(target, shape):
    if target == "rec":
        h, w = 48, shape
        return (np.random.rand(1, 3, h, w).astype(np.float32) * 2 - 1)
    h, w = shape
    return (np.random.rand(1, 3, h, w).astype(np.float32) * 2 - 1)


def main():
    target = sys.argv[1]
    models = sys.argv[2:]
    shapes = REC_WIDTHS if target == "rec" else DET_SIZES

    for intra in (1, 4):
        print(f"\n=== {target}  intra_op_threads={intra} ===")
        hdr = "shape".ljust(12) + "".join(os.path.basename(m).ljust(28) for m in models)
        print(hdr)
        base_ms = {}
        for shp in shapes:
            feed = make_feed(target, shp)
            row = str(shp).ljust(12)
            ms0 = None
            for mi, m in enumerate(models):
                s = sess(m, intra)
                ms = timeit(s, feed)
                if mi == 0:
                    ms0 = ms
                    cell = f"{ms:7.2f}ms"
                else:
                    cell = f"{ms:7.2f}ms ({ms0/ms:4.2f}x)"
                row += cell.ljust(28)
            print(row)


if __name__ == "__main__":
    main()
