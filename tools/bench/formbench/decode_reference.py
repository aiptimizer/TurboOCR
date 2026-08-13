#!/usr/bin/env python3
"""Pin the ONNX decode against the reference implementation.

The C++ runner is written from this script's decode, so anything wrong here is
wrong there too — in particular the class-index order (the head has 91 slots
and only the first three are form classes) and whether the reference offsets
class ids at all. Guessing either would produce a detector that runs, emits
plausible boxes, and is silently mislabelled.
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pypdfium2 as pdfium
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASSES = {0: "text", 1: "checkbox", 2: "signature"}


def render(pdf: Path, dpi: int = 200) -> Image.Image:
    doc = pdfium.PdfDocument(str(pdf))
    return doc[0].render(scale=dpi / 72.0).to_pil().convert("RGB")


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    # The reference resizes to a SQUARE, ignoring aspect ratio, so normalised
    # output coords map straight back onto the original with no letterbox math.
    r = img.resize((size, size), Image.Resampling.LANCZOS)
    x = np.asarray(r, np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def nms(boxes, scores, iou_thr):
    """Class-agnostic, matching FFDetrDetector's with_nms(class_agnostic=True)."""
    idx = np.argsort(-scores)
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(i)
        if idx.size == 1:
            break
        rest = idx[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a_i + a_r - inter, 1e-9)
        idx = rest[iou <= iou_thr]
    return keep


def decode(dets, logits, w, h, conf, iou_thr, n_form_classes=3):
    """sigmoid → threshold → cxcywh→xyxy → class-agnostic NMS.

    DETR-with-focal-loss scores every (query, class) pair independently; the
    reference top-ks over the flattened product, so one query may survive under
    two classes. Thresholding gives the same survivors without the top-k.
    """
    prob = 1.0 / (1.0 + np.exp(-logits[0]))          # (Q, C)
    prob = prob[:, :n_form_classes]                   # slots 3.. are unused
    q, c = np.nonzero(prob >= conf)
    if q.size == 0:
        return []
    scores = prob[q, c]
    cx, cy, bw, bh = dets[0][q].T
    boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    keep = nms(boxes, scores, iou_thr)
    out = [(CLASSES[int(c[k])], float(scores[k]), *boxes[k].tolist()) for k in keep]
    out.sort(key=lambda r: (round(r[3] / 10), r[2]))  # reading order
    return out


def main() -> int:
    pdf = Path(sys.argv[1])
    onnx = Path(sys.argv[2])
    size, conf, iou_thr = 1024, 0.40, 0.10

    img = render(pdf)
    print(f"page {img.width}x{img.height}  model {onnx.name}")

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    inp = preprocess(img, size)
    if sess.get_inputs()[0].type == "tensor(float16)":
        inp = inp.astype(np.float16)
    dets, logits = sess.run(None, {sess.get_inputs()[0].name: inp})
    dets, logits = dets.astype(np.float32), logits.astype(np.float32)

    mine = decode(dets, logits, img.width, img.height, conf, iou_thr)
    print(f"\n--- raw ONNX decode: {len(mine)} field(s) ---")
    for t, s, x0, y0, x1, y1 in mine:
        print(f"  {t:9s} {s:.3f}  [{x0:7.1f} {y0:7.1f} {x1:7.1f} {y1:7.1f}]")

    # Sanity on the unused head slots: if classes >=3 ever fire above threshold
    # the "first three slots" assumption is wrong and everything is mislabelled.
    prob = 1.0 / (1.0 + np.exp(-logits[0]))
    tail = prob[:, 3:]
    print(f"\ntail slots 3..{prob.shape[1]-1}: max prob {tail.max():.4f} "
          f"({'OK - unused' if tail.max() < conf else 'PROBLEM - tail fires!'})")

    try:
        from commonforms.inference import FFDetrDetector
        from commonforms.utils import Page
    except ImportError:
        print("\n(reference not installed; skipping cross-check)")
        return 0

    det = FFDetrDetector("FFDetr", device="cpu")
    widgets = det.extract_widgets([Page(image=img, page=0)], confidence=conf)
    ref = []
    for wdg in widgets.get(0, []):
        b = wdg.bounding_box
        ref.append((wdg.widget_type, b.x0 * img.width, b.y0 * img.height,
                    b.x1 * img.width, b.y1 * img.height))
    ref.sort(key=lambda r: (round(r[2] / 10), r[1]))
    print(f"\n--- reference: {len(ref)} field(s) ---")
    for t, x0, y0, x1, y1 in ref:
        print(f"  {t:14s}        [{x0:7.1f} {y0:7.1f} {x1:7.1f} {y1:7.1f}]")

    print(f"\ncount  mine={len(mine)}  ref={len(ref)}  "
          f"{'MATCH' if len(mine) == len(ref) else 'MISMATCH'}")
    if len(mine) == len(ref) and mine:
        d = max(max(abs(a - b) for a, b in zip(m[2:], r[1:]))
                for m, r in zip(mine, ref))
        print(f"max corner delta: {d:.2f} px  "
              f"{'OK' if d < 2.0 else 'CHECK PREPROCESSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
