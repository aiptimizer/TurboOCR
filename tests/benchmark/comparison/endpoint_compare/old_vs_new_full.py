#!/usr/bin/env python3
"""OLD vs NEW TurboOCR — the endpoints the first pass could not exercise:
tables, formulas (real OmniDocBench pages), markdown on structured pages,
/ocr/pdf?markdown=1, /ocr/stream VALUES, /ocr/pixels, /infer, ?autorotate=1.
Prints actual output samples so the values themselves are inspectable."""

import base64
import difflib
import glob
import os
import json
import re

import cv2
import numpy as np
import requests

NEW = "http://127.0.0.1:8080"
OLD = "http://127.0.0.1:8081"
OMNI = sorted(glob.glob(os.path.join(
    os.environ.get("OMNIDOCBENCH", "omnidocbench/data/images"), "*.jpg")))
FUNSD0 = os.path.join(
    os.environ.get("FUNSD_CACHE", "compare-ocrs/funsd_cache"), "funsd_000.png")
PDF = os.environ.get("TEST_PDF", "tests/fixtures/pdf/test8.pdf")
HDR = {"Content-Type": "image/jpeg"}


def tokens(t):
    return {x.lower() for x in re.split(r"[^A-Za-z0-9]+", t) if len(x) >= 2}


def token_f1(a, b):
    A = tokens(a) if isinstance(a, str) else set().union(*[tokens(x) for x in a]) if a else set()
    B = tokens(b) if isinstance(b, str) else set().union(*[tokens(x) for x in b]) if b else set()
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    tp = len(A & B); p, r = tp/len(B), tp/len(A)
    return 2*p*r/(p+r) if p+r else 0.0


def structured(base, data, hdr=HDR):
    return requests.post(f"{base}/ocr/raw?layout=1&tables=1&formulas=1",
                         data=data, headers=hdr).json()


def find_pages():
    """Scan omnidocbench pages until we have one with tables and one with formulas."""
    table_page = formula_page = None
    for p in OMNI[:60]:
        data = open(p, "rb").read()
        j = structured(NEW, data)
        if table_page is None and j.get("tables"):
            table_page = (p, data)
        if formula_page is None and j.get("formulas"):
            formula_page = (p, data)
        if table_page and formula_page:
            break
    return table_page, formula_page


def show(tag, s, n=220):
    s = (s or "").replace("\n", "\\n")
    print(f"    {tag}: {s[:n]}{'…' if len(s) > n else ''}")


def main():
    print("## Finding structure-rich pages in OmniDocBench…")
    table_page, formula_page = find_pages()
    print(f"table page:   {table_page[0].split('/')[-1] if table_page else 'NONE FOUND'}")
    print(f"formula page: {formula_page[0].split('/')[-1] if formula_page else 'NONE FOUND'}")

    # ---- tables ----
    if table_page:
        print("\n## ?tables=1 — table HTML values")
        jn = structured(NEW, table_page[1]); jo = structured(OLD, table_page[1])
        tn, to = jn.get("tables", []), jo.get("tables", [])
        print(f"count new/old: {len(tn)}/{len(to)}")
        for i in range(min(len(tn), len(to))):
            hn, ho = tn[i].get("html", ""), to[i].get("html", "")
            cells_n = re.findall(r"<td[^>]*>(.*?)</td>", hn)
            cells_o = re.findall(r"<td[^>]*>(.*?)</td>", ho)
            print(f"  table {i}: byte-identical={hn == ho} · cells {len(cells_n)}/{len(cells_o)} "
                  f"· cell-text F1 {token_f1(' '.join(cells_n), ' '.join(cells_o)):.3f}")
            show("new html", hn); show("old html", ho)

    # ---- formulas ----
    if formula_page:
        print("\n## ?formulas=1 — LaTeX values")
        jn = structured(NEW, formula_page[1]); jo = structured(OLD, formula_page[1])
        fn, fo = jn.get("formulas", []), jo.get("formulas", [])
        print(f"count new/old: {len(fn)}/{len(fo)}")
        for i in range(min(len(fn), len(fo), 4)):
            ln, lo = fn[i].get("latex", ""), fo[i].get("latex", "")
            print(f"  formula {i}: identical={ln == lo} · token-F1 {token_f1(ln, lo):.3f}")
            show("new latex", ln, 150); show("old latex", lo, 150)

    # ---- markdown on the structured page ----
    page = table_page or formula_page
    if page:
        print("\n## /ocr/markdown on the structured page")
        mn = requests.post(f"{NEW}/ocr/markdown", data=page[1], headers=HDR).text
        mo = requests.post(f"{OLD}/ocr/markdown", data=page[1], headers=HDR).text
        print(f"byte-identical: {mn == mo} · chars {len(mn)}/{len(mo)} · "
              f"token-F1 {token_f1(mn, mo):.4f}")
        if mn != mo:
            d = list(difflib.unified_diff(mo.splitlines(), mn.splitlines(),
                                          "old", "new", lineterm=""))[:14]
            print("    diff head:"); [print("    " + l[:160]) for l in d]
        else:
            show("markdown head", mn, 300)

    # ---- pdf markdown ----
    print("\n## /ocr/pdf?markdown=1 (test8.pdf)")
    pdf = open(PDF, "rb").read()
    ph = {"Content-Type": "application/pdf"}
    rn = requests.post(f"{NEW}/ocr/pdf?markdown=1", data=pdf, headers=ph)
    ro = requests.post(f"{OLD}/ocr/pdf?markdown=1", data=pdf, headers=ph)
    print(f"status {rn.status_code}/{ro.status_code} · byte-identical: {rn.text == ro.text} "
          f"· chars {len(rn.text)}/{len(ro.text)} · token-F1 {token_f1(rn.text, ro.text):.4f}")

    # ---- stream VALUES ----
    print("\n## /ocr/stream — per-page values vs /ocr/pdf, both servers")
    for tag, base in (("new", NEW), ("old", OLD)):
        rp = requests.post(f"{base}/ocr/pdf", data=pdf, headers=ph).json()
        pages = rp.get("pages", rp.get("results", []))
        rs = requests.post(f"{base}/ocr/stream", data=pdf, headers=ph, stream=True)
        evs = [json.loads(l) for l in rs.iter_lines() if l]
        sp = {e.get("page"): e for e in evs if e.get("results") is not None}
        agree = 0
        for i, pg in enumerate(pages):
            a = [r["text"] for r in (pg.get("results") or [])]
            b = [r["text"] for r in ((sp.get(i) or {}).get("results") or [])]
            agree += a == b
        print(f"  {tag}: stream pages {len(sp)} · text-identical to /ocr/pdf on {agree}/{len(pages)} pages")

    # ---- pixels ----
    print("\n## /ocr/pixels (raw BGR buffer)")
    img = cv2.imread(FUNSD0)
    h, w = img.shape[:2]
    buf = np.ascontiguousarray(img).tobytes()
    pn = requests.post(f"{NEW}/ocr/pixels?width={w}&height={h}", data=buf).json().get("results", [])
    po = requests.post(f"{OLD}/ocr/pixels?width={w}&height={h}", data=buf).json().get("results", [])
    raw_n = requests.post(f"{NEW}/ocr/raw", data=open(FUNSD0, "rb").read(),
                          headers={"Content-Type": "image/png"}).json()["results"]
    print(f"words new/old: {len(pn)}/{len(po)} · new pixels==new raw texts: "
          f"{[r['text'] for r in pn] == [r['text'] for r in raw_n]} · cross F1 "
          f"{token_f1([r['text'] for r in pn], [r['text'] for r in po]):.4f}")

    # ---- /infer on one table crop ----
    if table_page:
        print("\n## /infer (single table crop)")
        jn = structured(NEW, table_page[1])
        box = (jn.get("tables") or [{}])[0].get("bounding_box")
        if box:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            im = cv2.imdecode(np.frombuffer(table_page[1], np.uint8), cv2.IMREAD_COLOR)
            crop = im[min(ys):max(ys), min(xs):max(xs)]
            ok, enc = cv2.imencode(".png", crop)
            cb = base64.b64encode(enc.tobytes()).decode()
            body = {"image": cb, "modality": "table", "backend": "slanext"}
            inew = requests.post(f"{NEW}/infer", json=body)
            iold = requests.post(f"{OLD}/infer", json=body)
            print(f"  status {inew.status_code}/{iold.status_code} · "
                  f"byte-identical: {inew.text == iold.text}")
            show("new", inew.text, 180); show("old", iold.text, 180)

    # ---- autorotate ----
    print("\n## ?autorotate=1 (page rotated 180°)")
    rot = cv2.rotate(cv2.imread(FUNSD0), cv2.ROTATE_180)
    ok, enc = cv2.imencode(".png", rot)
    rb = enc.tobytes()
    base_texts = [r["text"] for r in raw_n]
    for tag, base in (("new", NEW), ("old", OLD)):
        rr = requests.post(f"{base}/ocr/raw?autorotate=1", data=rb,
                           headers={"Content-Type": "image/png"}).json().get("results", [])
        f1 = token_f1([r["text"] for r in rr], base_texts)
        print(f"  {tag}: words {len(rr)} · token-F1 vs upright baseline {f1:.4f}")


if __name__ == "__main__":
    main()
