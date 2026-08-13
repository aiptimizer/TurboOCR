"""Accuracy validation of quantized OCR models against fp32 reference.

rec: runs fp32 rec and each int8 rec on held-out FUNSD line crops, CTC-decodes
     with keys.txt, reports exact-match rate and character error rate (CER),
     using the fp32 output as the gold reference (quantization-induced error).
det: reports DB probability-map MSE / max-abs error vs fp32 + box-count match.

Usage:
  eval_quant.py rec models/rec.int8.static.onnx [models/rec.int8.qop.onnx ...]
  eval_quant.py det models/det.int8.static.onnx [models/det.int8.qop.onnx ...]
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_common as Q

# held-out pages: calibration used pages [0:20]; evaluate on [20:50]
EVAL_PAGES = Q.list_pages()[20:]


def edit_distance(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def eval_rec(int8_models):
    det = Q.make_cpu_session(f"{Q.REPO}/models/det.onnx")
    fp32 = Q.make_cpu_session(f"{Q.REPO}/models/rec.onnx")
    labels = Q.load_keys()
    crops = Q.extract_line_crops(det, EVAL_PAGES)
    print(f"[rec] {len(crops)} held-out crops from {len(EVAL_PAGES)} pages\n")

    rin = fp32.get_inputs()[0].name
    feeds = []
    refs = []
    for c in crops:
        nchw, _ = Q.rec_preprocess(c)
        feeds.append(nchw)
        out = fp32.run(None, {rin: nchw})[0]
        refs.append(Q.ctc_greedy_decode(out[0], labels))

    ref_chars = sum(len(r) for r in refs)
    print(f"{'model':<30}{'exact-match':>14}{'CER':>10}{'mismatch lines':>16}")
    for mp in int8_models:
        s = Q.make_cpu_session(mp)
        sin = s.get_inputs()[0].name
        exact = 0
        tot_ed = 0
        mismatch_samples = []
        for nchw, ref in zip(feeds, refs):
            out = s.run(None, {sin: nchw})[0]
            hyp = Q.ctc_greedy_decode(out[0], labels)
            if hyp == ref:
                exact += 1
            else:
                ed = edit_distance(ref, hyp)
                tot_ed += ed
                if len(mismatch_samples) < 5:
                    mismatch_samples.append((ref, hyp))
        cer = tot_ed / max(ref_chars, 1)
        em = exact / max(len(refs), 1)
        print(f"{os.path.basename(mp):<30}{em*100:>13.2f}%{cer*100:>9.3f}%"
              f"{len(refs)-exact:>16}")
        for ref, hyp in mismatch_samples:
            print(f"    ref={ref!r}\n    hyp={hyp!r}")
    print(f"\n(reference chars: {ref_chars}; CER = total edit distance / ref chars)")


def eval_det(int8_models):
    fp32 = Q.make_cpu_session(f"{Q.REPO}/models/det.onnx")
    din = fp32.get_inputs()[0].name
    feeds = []
    preds_fp32 = []
    metas = []
    for p in EVAL_PAGES:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        inp, rh, rw = Q.det_preprocess(img)
        feeds.append(inp)
        out = fp32.run(None, {din: inp})[0]
        preds_fp32.append(out[0, 0].astype(np.float32))
        metas.append((h, w, rh, rw))
    print(f"[det] {len(feeds)} held-out pages\n")

    print(f"{'model':<30}{'probmap MSE':>13}{'maskIoU@.3':>11}"
          f"{'boxcount':>10}{'lines f/i':>14}")
    for mp in int8_models:
        s = Q.make_cpu_session(mp)
        sin = s.get_inputs()[0].name
        se = 0.0
        n = 0
        iou_sum = 0.0
        box_match = 0
        lf = li = 0
        for inp, pf, meta in zip(feeds, preds_fp32, metas):
            out = s.run(None, {sin: inp})[0]
            pi = out[0, 0].astype(np.float32)
            se += float(np.sum((pi - pf) ** 2))
            n += pi.size
            mf = pf > Q.DET_DB_THRESH
            mi = pi > Q.DET_DB_THRESH
            inter = np.logical_and(mf, mi).sum()
            union = np.logical_or(mf, mi).sum()
            iou_sum += (inter / union) if union else 1.0
            h, w, rh, rw = meta
            nb_f = len(Q.det_postprocess(pf, h, w, rh, rw))
            nb_i = len(Q.det_postprocess(pi, h, w, rh, rw))
            lf += nb_f
            li += nb_i
            if nb_f == nb_i:
                box_match += 1
        mse = se / max(n, 1)
        miou = iou_sum / max(len(feeds), 1)
        print(f"{os.path.basename(mp):<30}{mse:>13.3e}{miou:>11.4f}"
              f"{box_match:>7}/{len(feeds):<2}{lf:>8}/{li:<5}")


def main():
    target = sys.argv[1]
    models = sys.argv[2:]
    if target == "rec":
        eval_rec(models)
    else:
        eval_det(models)


if __name__ == "__main__":
    main()
