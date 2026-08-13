#!/usr/bin/env python3
"""Endpoint-by-endpoint TurboOCR vs PaddleOCR comparison.

Fires identical inputs at both servers and compares the VALUES each endpoint
returns (texts, confidences, boxes, layout, tables, formulas, markdown, pdf
pages), plus per-endpoint latency. Prints one table per endpoint family.

Turbo:  http://127.0.0.1:8080   Paddle wrapper: http://127.0.0.1:9090
"""

import base64
import glob
import os
import json
import re
import statistics
import sys
import time

import requests

TURBO = "http://127.0.0.1:8080"
PADDLE = "http://127.0.0.1:9090"
IMAGES = sorted(glob.glob(os.path.join(
    os.environ.get("FUNSD_CACHE", "compare-ocrs/funsd_cache"), "funsd_*.png")))[:10]
PDF = os.environ.get("TEST_PDF", "tests/fixtures/pdf/test8.pdf")
N_LAT = 5  # latency reps per endpoint (after 1 warmup)


def tokens(text):
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", text) if len(t) >= 2}


def token_f1(a_texts, b_texts):
    A, B = set(), set()
    for t in a_texts: A |= tokens(t)
    for t in b_texts: B |= tokens(t)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    tp = len(A & B)
    p, r = tp / len(B), tp / len(A)
    return 2 * p * r / (p + r) if p + r else 0.0


def box_iou(b1, b2):
    def rect(b):
        xs = [p[0] for p in b]; ys = [p[1] for p in b]
        return min(xs), min(ys), max(xs), max(ys)
    ax0, ay0, ax1, ay1 = rect(b1); bx0, by0, bx1, by1 = rect(b2)
    ix = max(0, min(ax1, bx1) - max(ax0, bx0)); iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / union if union else 0.0


def matched_iou(a, b):
    """Greedy match a's boxes to b's; mean IoU of matches >0.3 + match rate."""
    used, ious = set(), []
    for ra in a:
        best, besti = 0.0, -1
        for i, rb in enumerate(b):
            if i in used: continue
            v = box_iou(ra["bounding_box"], rb["bounding_box"])
            if v > best: best, besti = v, i
        if best > 0.3:
            used.add(besti); ious.append(best)
    rate = len(ious) / max(len(a), 1)
    return (statistics.mean(ious) if ious else 0.0), rate


def timed(fn, n=N_LAT):
    fn()  # warmup
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); ts.append((time.perf_counter()-t0)*1000)
    return r, statistics.median(ts)


def norm_results(rs):
    return {
        "texts": [r["text"] for r in rs],
        "confs": [float(r["confidence"]) for r in rs],
        "n": len(rs),
        "raw": rs,
    }


def compare_pair(name, tu, pa, tu_ms, pa_ms, extra=""):
    f1 = token_f1(tu["texts"], pa["texts"])
    miou, mrate = matched_iou(tu["raw"], pa["raw"])
    mc_t = statistics.mean(tu["confs"]) if tu["confs"] else 0
    mc_p = statistics.mean(pa["confs"]) if pa["confs"] else 0
    print(f"| {name} | {tu['n']} / {pa['n']} | {f1:.3f} | {miou:.2f} @ {mrate:.0%} "
          f"| {mc_t:.3f} / {mc_p:.3f} | {tu_ms:.0f} / {pa_ms:.0f} |{extra}")
    return f1


def main():
    img_bytes = [open(p, "rb").read() for p in IMAGES]
    b64 = [base64.b64encode(b).decode() for b in img_bytes]

    print("## Plain OCR endpoints (page 0; latency median of %d)" % N_LAT)
    print("| endpoint | words T/P | token-F1 | box IoU@match | mean conf T/P | ms T/P |")
    print("|---|---|---|---|---|---|")

    # --- Turbo transports (internal consistency + vs paddle) ---
    def t_raw():
        return requests.post(f"{TURBO}/ocr/raw", data=img_bytes[0],
                             headers={"Content-Type": "image/png"}).json()["results"]
    def t_b64():
        return requests.post(f"{TURBO}/ocr", json={"image": b64[0]}).json()["results"]
    def t_batch():
        return requests.post(f"{TURBO}/ocr/batch", json={"images": [b64[0]]}).json()["batch_results"][0]["results"]
    def p_raw():
        return requests.post(f"{PADDLE}/ocr/raw", data=img_bytes[0]).json()["results"]
    def p_b64():
        return requests.post(f"{PADDLE}/ocr", json={"image": b64[0]}).json()["results"]
    def p_batch():
        return requests.post(f"{PADDLE}/ocr/batch", json={"images": [b64[0]]}).json()["results"][0]

    tr, tr_ms = timed(t_raw); pr, pr_ms = timed(p_raw)
    tb, tb_ms = timed(t_b64); pb, pb_ms = timed(p_b64)
    tba, tba_ms = timed(t_batch); pba, pba_ms = timed(p_batch)

    # Turbo internal consistency across transports
    same_rb = [r["text"] for r in tr] == [r["text"] for r in tb]
    same_rba = [r["text"] for r in tr] == [r["text"] for r in tba]
    compare_pair("/ocr/raw", norm_results(tr), norm_results(pr), tr_ms, pr_ms)
    compare_pair("/ocr (b64)", norm_results(tb), norm_results(pb), tb_ms, pb_ms)
    compare_pair("/ocr/batch[1]", norm_results(tba), norm_results(pba), tba_ms, pba_ms)
    print(f"\nTurbo transport consistency: raw==b64: {same_rb} · raw==batch: {same_rba}")

    # --- 10-page aggregate ---
    agg_f1, agg_t_words, agg_p_words = [], 0, 0
    for i, (bts, b6) in enumerate(zip(img_bytes, b64)):
        t = requests.post(f"{TURBO}/ocr/raw", data=bts,
                          headers={"Content-Type": "image/png"}).json()["results"]
        p = requests.post(f"{PADDLE}/ocr/raw", data=bts).json()["results"]
        agg_f1.append(token_f1([r["text"] for r in t], [r["text"] for r in p]))
        agg_t_words += len(t); agg_p_words += len(p)
    print(f"10-page aggregate: token-F1 mean {statistics.mean(agg_f1):.3f} "
          f"(min {min(agg_f1):.3f}) · words {agg_t_words} vs {agg_p_words}")

    # --- structure: layout / tables / formulas ---
    print("\n## Structure endpoints (page 0)")
    def t_struct():
        return requests.post(f"{TURBO}/ocr/raw?layout=1&tables=1&formulas=1",
                             data=img_bytes[0], headers={"Content-Type": "image/png"}).json()
    def p_struct():
        return requests.post(f"{PADDLE}/ocr/structure?tables=1&formulas=1",
                             data=img_bytes[0]).json()
    ts_, ts_ms = timed(t_struct, 3); ps_, ps_ms = timed(p_struct, 3)
    t_lay = ts_.get("layout", []); p_lay = ps_.get("layout", [])
    def label_counts(lay, key):
        c = {}
        for r in lay: c[r.get(key, "?")] = c.get(r.get(key, "?"), 0) + 1
        return dict(sorted(c.items()))
    print(f"layout regions: turbo {len(t_lay)} {label_counts(t_lay,'class')}")
    print(f"                paddle {len(p_lay)} {label_counts(p_lay,'label')}")
    print(f"tables: turbo {len(ts_.get('tables', []))} · paddle {len(ps_.get('tables', []))}")
    print(f"formulas: turbo {len(ts_.get('formulas', []))} · paddle {len(ps_.get('formulas', []))}")
    print(f"latency ms: turbo {ts_ms:.0f} · paddle {ps_ms:.0f}")

    # --- markdown ---
    print("\n## /ocr/markdown (page 0)")
    def t_md():
        r = requests.post(f"{TURBO}/ocr/markdown", data=img_bytes[0],
                          headers={"Content-Type": "image/png"})
        return {"markdown": r.text}  # turbo returns text/markdown directly
    def p_md():
        return requests.post(f"{PADDLE}/ocr/markdown", data=img_bytes[0]).json()
    tm, tm_ms = timed(t_md, 3); pm, pm_ms = timed(p_md, 3)
    tmd = tm.get("markdown", "") if isinstance(tm, dict) else ""
    pmd = pm.get("markdown", "")
    print(f"markdown chars: turbo {len(tmd)} · paddle {len(pmd)} · "
          f"token-F1 {token_f1([tmd],[pmd]):.3f} · ms {tm_ms:.0f}/{pm_ms:.0f}")

    # --- pdf ---
    print("\n## /ocr/pdf (test8.pdf)")
    pdf = open(PDF, "rb").read()
    def t_pdf():
        return requests.post(f"{TURBO}/ocr/pdf", data=pdf,
                             headers={"Content-Type": "application/pdf"}).json()
    def p_pdf():
        return requests.post(f"{PADDLE}/ocr/pdf", data=pdf).json()
    tp_, tp_ms = timed(t_pdf, 2); pp_, pp_ms = timed(p_pdf, 2)
    t_pages = tp_.get("pages", tp_.get("results", []))
    p_pages = pp_.get("pages", [])
    print(f"pages: turbo {len(t_pages)} · paddle {len(p_pages)} · ms {tp_ms:.0f}/{pp_ms:.0f}")
    for i in range(min(len(t_pages), len(p_pages), 8)):
        t_texts = [r["text"] for r in (t_pages[i].get("results") or [])]
        p_texts = [r["text"] for r in (p_pages[i].get("results") or [])]
        print(f"  page {i}: words {len(t_texts)}/{len(p_texts)} "
              f"token-F1 {token_f1(t_texts, p_texts):.3f}")

    # --- stream (turbo-only transport; values must equal /ocr/pdf) ---
    print("\n## /ocr/stream vs /ocr/pdf (turbo internal)")
    r = requests.post(f"{TURBO}/ocr/stream", data=pdf,
                      headers={"Content-Type": "application/pdf"}, stream=True)
    events = [json.loads(l) for l in r.iter_lines() if l]
    spages = [e for e in events if e.get("results") is not None or e.get("page") is not None]
    print(f"stream events with pages: {len(spages)} (pdf pages: {len(t_pages)})")


if __name__ == "__main__":
    sys.exit(main())
