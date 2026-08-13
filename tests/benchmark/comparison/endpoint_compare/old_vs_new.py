#!/usr/bin/env python3
"""OLD pre-multibackend TurboOCR (paddle-highspeed-cpp, :8081) vs NEW
the unified server (:8080) — same API, so every endpoint is compared
for EXACT value equality, with precise deltas where values differ, plus
median latency per endpoint.
"""

import base64
import glob
import os
import json
import re
import statistics
import time

import requests

NEW = "http://127.0.0.1:8080"
OLD = "http://127.0.0.1:8081"
IMAGES = sorted(glob.glob(os.path.join(
    os.environ.get("FUNSD_CACHE", "compare-ocrs/funsd_cache"), "funsd_*.png")))[:10]
PDF = os.environ.get("TEST_PDF", "tests/fixtures/pdf/test8.pdf")
N = 5


def tokens(t):
    return {x.lower() for x in re.split(r"[^A-Za-z0-9]+", t) if len(x) >= 2}


def token_f1(a, b):
    A, B = set(), set()
    for t in a: A |= tokens(t)
    for t in b: B |= tokens(t)
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    tp = len(A & B); p, r = tp/len(B), tp/len(A)
    return 2*p*r/(p+r) if p+r else 0.0


def timed(fn, n=N):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); ts.append((time.perf_counter()-t0)*1000)
    return r, statistics.median(ts)


def diff_results(new_rs, old_rs):
    """Exactness report between two turbo result lists."""
    texts_new = [r["text"] for r in new_rs]
    texts_old = [r["text"] for r in old_rs]
    same_n = len(new_rs) == len(old_rs)
    same_texts = texts_new == texts_old
    max_conf_d, max_box_d = 0.0, 0
    n_text_diff = 0
    if same_n:
        for a, b in zip(new_rs, old_rs):
            if a["text"] != b["text"]:
                n_text_diff += 1
            max_conf_d = max(max_conf_d, abs(a["confidence"] - b["confidence"]))
            for pa, pb in zip(a["bounding_box"], b["bounding_box"]):
                max_box_d = max(max_box_d, abs(pa[0]-pb[0]), abs(pa[1]-pb[1]))
    verdict = "IDENTICAL" if (same_n and same_texts and max_conf_d < 1e-4 and max_box_d == 0) \
        else "EQUAL-TEXT" if (same_n and same_texts) else "DIFFERS"
    return {
        "verdict": verdict, "n_new": len(new_rs), "n_old": len(old_rs),
        "text_diffs": n_text_diff if same_n else None,
        "max_conf_delta": round(max_conf_d, 6) if same_n else None,
        "max_box_delta_px": max_box_d if same_n else None,
        "token_f1": round(token_f1(texts_new, texts_old), 4),
    }


def row(name, d, new_ms, old_ms):
    print(f"| {name} | {d['verdict']} | {d['n_new']}/{d['n_old']} | "
          f"{d['text_diffs']} | {d['max_conf_delta']} | {d['max_box_delta_px']} | "
          f"{d['token_f1']} | {new_ms:.0f} / {old_ms:.0f} |")


def main():
    img = open(IMAGES[0], "rb").read()
    b64 = base64.b64encode(img).decode()
    hdr = {"Content-Type": "image/png"}

    print("## Same-endpoint value comparison — NEW (unified, :8080) vs OLD (pre-multibackend, :8081)")
    print("| endpoint | verdict | words N/O | text diffs | max conf Δ | max box Δpx | token-F1 | ms N/O |")
    print("|---|---|---|---|---|---|---|---|")

    n_raw, n_ms = timed(lambda: requests.post(f"{NEW}/ocr/raw", data=img, headers=hdr).json()["results"])
    o_raw, o_ms = timed(lambda: requests.post(f"{OLD}/ocr/raw", data=img, headers=hdr).json()["results"])
    row("/ocr/raw", diff_results(n_raw, o_raw), n_ms, o_ms)

    n_b, n_bms = timed(lambda: requests.post(f"{NEW}/ocr", json={"image": b64}).json()["results"])
    o_b, o_bms = timed(lambda: requests.post(f"{OLD}/ocr", json={"image": b64}).json()["results"])
    row("/ocr (b64)", diff_results(n_b, o_b), n_bms, o_bms)

    def batch(base):
        j = requests.post(f"{base}/ocr/batch", json={"images": [b64]}).json()
        br = j.get("batch_results")
        return br[0]["results"] if br else j["results"][0]
    n_ba, n_bams = timed(lambda: batch(NEW))
    o_ba, o_bams = timed(lambda: batch(OLD))
    row("/ocr/batch[1]", diff_results(n_ba, o_ba), n_bams, o_bams)

    # 10-page aggregate exactness
    ident, eqtext, diff = 0, 0, 0
    for p in IMAGES:
        b = open(p, "rb").read()
        n = requests.post(f"{NEW}/ocr/raw", data=b, headers=hdr).json()["results"]
        o = requests.post(f"{OLD}/ocr/raw", data=b, headers=hdr).json()["results"]
        v = diff_results(n, o)["verdict"]
        ident += v == "IDENTICAL"; eqtext += v == "EQUAL-TEXT"; diff += v == "DIFFERS"
    print(f"\n10-page /ocr/raw: IDENTICAL {ident} · EQUAL-TEXT {eqtext} · DIFFERS {diff}")

    # structure
    print("\n## layout / tables / formulas (page 0)")
    def struct(base):
        return requests.post(f"{base}/ocr/raw?layout=1&tables=1&formulas=1",
                             data=img, headers=hdr).json()
    ns, ns_ms = timed(lambda: struct(NEW), 3)
    os_, os_ms = timed(lambda: struct(OLD), 3)
    def lay_sig(j):
        return [(r.get("class") or r.get("label"), tuple(map(tuple, r["bounding_box"])))
                for r in j.get("layout", [])]
    same_lay = lay_sig(ns) == lay_sig(os_)
    print(f"layout: new {len(ns.get('layout', []))} vs old {len(os_.get('layout', []))} "
          f"regions · exact-equal: {same_lay} · ms {ns_ms:.0f}/{os_ms:.0f}")
    if not same_lay:
        from collections import Counter
        cn = Counter(c for c, _ in lay_sig(ns)); co = Counter(c for c, _ in lay_sig(os_))
        print(f"  class hist new: {dict(sorted(cn.items()))}")
        print(f"  class hist old: {dict(sorted(co.items()))}")
    print(f"tables: {len(ns.get('tables', []))}/{len(os_.get('tables', []))} · "
          f"formulas: {len(ns.get('formulas', []))}/{len(os_.get('formulas', []))}")

    # markdown
    print("\n## /ocr/markdown (page 0)")
    n_md, nmd_ms = timed(lambda: requests.post(f"{NEW}/ocr/markdown", data=img, headers=hdr).text, 3)
    o_md, omd_ms = timed(lambda: requests.post(f"{OLD}/ocr/markdown", data=img, headers=hdr).text, 3)
    print(f"byte-identical: {n_md == o_md} · chars {len(n_md)}/{len(o_md)} · "
          f"token-F1 {token_f1([n_md],[o_md]):.4f} · ms {nmd_ms:.0f}/{omd_ms:.0f}")

    # pdf
    print("\n## /ocr/pdf (test8.pdf)")
    pdf = open(PDF, "rb").read()
    ph = {"Content-Type": "application/pdf"}
    n_pdf, npdf_ms = timed(lambda: requests.post(f"{NEW}/ocr/pdf", data=pdf, headers=ph).json(), 2)
    o_pdf, opdf_ms = timed(lambda: requests.post(f"{OLD}/ocr/pdf", data=pdf, headers=ph).json(), 2)
    n_pages = n_pdf.get("pages", n_pdf.get("results", []))
    o_pages = o_pdf.get("pages", o_pdf.get("results", []))
    print(f"pages {len(n_pages)}/{len(o_pages)} · ms {npdf_ms:.0f}/{opdf_ms:.0f}")
    for i in range(min(len(n_pages), len(o_pages))):
        nr = n_pages[i].get("results") or []
        orr = o_pages[i].get("results") or []
        d = diff_results(nr, orr)
        print(f"  page {i}: {d['verdict']} words {d['n_new']}/{d['n_old']} "
              f"textΔ {d['text_diffs']} confΔ {d['max_conf_delta']} f1 {d['token_f1']}")

    # capabilities
    print("\n## /capabilities")
    nc = requests.get(f"{NEW}/capabilities").json()
    oc = requests.get(f"{OLD}/capabilities").json()
    nk, ok = set(str(nc)), set(str(oc))
    print(f"equal: {nc == oc}")
    if nc != oc:
        n_only = {k: v for k, v in nc.items() if oc.get(k) != v} if isinstance(nc, dict) else nc
        o_only = {k: v for k, v in oc.items() if nc.get(k) != v} if isinstance(oc, dict) else oc
        print(f"  new-differs: {json.dumps(n_only)[:300]}")
        print(f"  old-differs: {json.dumps(o_only)[:300]}")


if __name__ == "__main__":
    main()
