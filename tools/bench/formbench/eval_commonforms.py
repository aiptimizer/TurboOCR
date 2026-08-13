#!/usr/bin/env python3
"""Measure the ONNX decode on the CommonForms TEST split (Apache-2.0).

This is the paper's own held-out set with real ground-truth widgets, so it does
two jobs at once: it tells us whether the exported graph plus my decode
reproduces the published model, and it gives the baseline every later change
(fp16, C++ port, merge with the geometry detectors) has to be measured against.

Matching is greedy by descending score at a fixed IoU, which is what P/R at a
single operating point means; AP would need the full sweep and is not what we
are deciding with here.
"""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from decode_reference import decode, preprocess  # noqa: E402

CLASS_ORDER = ["text", "checkbox", "signature"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def match(pred, gt, iou_thr, class_aware=True):
    """Greedy by descending score. Returns (tp, fp, fn) counts."""
    if not pred:
        return 0, 0, len(gt)
    if not gt:
        return 0, len(pred), 0
    pb = np.array([p[2:6] for p in pred], np.float32)
    gb = np.array([g[1:5] for g in gt], np.float32)
    ious = iou_matrix(pb, gb)
    used = set()
    tp = 0
    for i in np.argsort([-p[1] for p in pred]):
        best, best_iou = -1, iou_thr
        for j in range(len(gt)):
            if j in used or ious[i, j] < best_iou:
                continue
            if class_aware and pred[i][0] != gt[j][0]:
                continue
            best, best_iou = j, ious[i, j]
        if best >= 0:
            used.add(best)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--dump", type=Path)
    args = ap.parse_args()

    path = hf_hub_download(repo_id="jbarrow/CommonForms",
                           filename=f"data/test-{args.shard:05d}-of-00024.parquet",
                           repo_type="dataset")
    table = pq.read_table(path)
    print(f"shard columns: {table.column_names}", flush=True)

    sess = ort.InferenceSession(str(args.onnx),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    fp16 = sess.get_inputs()[0].type == "tensor(float16)"

    agg = defaultdict(lambda: [0, 0, 0])   # class -> [tp, fp, fn]
    tot = [0, 0, 0]
    n = min(args.pages, table.num_rows)
    rows = table.slice(0, n).to_pylist()
    dumped = []

    for k, row in enumerate(rows):
        img_field = row.get("image")
        raw = img_field["bytes"] if isinstance(img_field, dict) else img_field
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        W, H = img.size

        gt = []
        objs = row.get("objects") or {}
        boxes = objs.get("bbox") or objs.get("boxes") or []
        cats = objs.get("category") or objs.get("categories") or objs.get("label") or []
        for b, c in zip(boxes, cats):
            # COCO xywh in absolute pixels is what the dataset card documents.
            x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            gt.append((CLASS_ORDER[int(c)] if int(c) < 3 else "text",
                       x, y, x + w, y + h))

        x = preprocess(img, 1024)
        if fp16:
            x = x.astype(np.float16)
        dets, logits = sess.run(None, {in_name: x})
        pred = decode(dets.astype(np.float32), logits.astype(np.float32),
                      W, H, args.conf, 0.10)

        for cls in CLASS_ORDER:
            p = [q for q in pred if q[0] == cls]
            g = [q for q in gt if q[0] == cls]
            tp, fp, fn = match(p, g, args.iou, class_aware=False)
            a = agg[cls]
            a[0] += tp; a[1] += fp; a[2] += fn
        tp, fp, fn = match(pred, gt, args.iou, class_aware=True)
        tot[0] += tp; tot[1] += fp; tot[2] += fn

        if args.dump is not None and len(dumped) < 5:
            dumped.append({"page": k, "size": [W, H], "gt": gt, "pred": pred})
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{n}", flush=True)

    print(f"\n=== {args.onnx.name} · {n} pages · conf {args.conf} · IoU {args.iou} ===")
    print(f"{'class':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    for cls in CLASS_ORDER:
        tp, fp, fn = agg[cls]
        p, r, f = prf(tp, fp, fn)
        print(f"{cls:10s} {p:7.3f} {r:7.3f} {f:7.3f} {tp:6d} {fp:6d} {fn:6d}")
    p, r, f = prf(*tot)
    print(f"{'OVERALL':10s} {p:7.3f} {r:7.3f} {f:7.3f} {tot[0]:6d} {tot[1]:6d} {tot[2]:6d}")

    if args.dump is not None:
        args.dump.write_text(json.dumps(dumped, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
