#!/usr/bin/env python3
"""Sanity-check Turbo-OCR endpoints/formats on a handful of FUNSD pages.

Sends each page through: /ocr (base64 JSON), /ocr/raw (PNG), /ocr/raw (JPEG),
/ocr/pixels (raw BGR), /ocr/batch (PNGs), /ocr/pdf (single-page PDF).
Reports per-endpoint word-level F1 against FUNSD ground truth, and verifies
all endpoints return the same text.
"""
import base64
import io
import json
import re

import numpy as np
import requests
from datasets import load_dataset
from PIL import Image

URL = "http://localhost:8000"
N_PAGES = 5


def tokens(text: str) -> set:
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", text) if len(t) >= 2}


def f1(pred: set, gt: set) -> float:
    if not gt:
        return 0.0
    tp = len(pred & gt)
    r = tp / len(gt)
    p = tp / len(pred) if pred else 0.0
    return 2 * r * p / (r + p) if (r + p) > 0 else 0.0


def extract(resp_json, key="results"):
    return [it["text"] for it in resp_json.get(key, [])]


def post_json(path, payload):
    r = requests.post(f"{URL}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def post_bytes(path, data, content_type, extra_headers=None):
    headers = {"Content-Type": content_type}
    if extra_headers:
        headers.update(extra_headers)
    r = requests.post(f"{URL}{path}", data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def png_to_pdf(png_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


def main():
    print("Loading FUNSD…")
    ds = load_dataset("nielsr/funsd", split="test").select(range(N_PAGES))

    pages = []
    for i, s in enumerate(ds):
        img = s["image"].convert("RGB")
        png_buf = io.BytesIO(); img.save(png_buf, format="PNG")
        jpg_buf = io.BytesIO(); img.save(jpg_buf, format="JPEG", quality=95)
        # BGR pixel bytes for /ocr/pixels
        arr = np.array(img)[:, :, ::-1].tobytes()  # RGB->BGR
        pages.append({
            "idx": i,
            "png": png_buf.getvalue(),
            "jpg": jpg_buf.getvalue(),
            "bgr": arr,
            "w": img.width, "h": img.height,
            "gt": tokens(" ".join(s["words"])),
        })

    endpoints = ["/ocr (b64 PNG)", "/ocr/raw PNG", "/ocr/raw JPEG",
                 "/ocr/pixels BGR", "/ocr/batch PNG", "/ocr/pdf"]
    f1_per_ep = {e: [] for e in endpoints}
    texts_per_page = []

    for p in pages:
        row = {}
        # /ocr (base64 JSON)
        res = post_json("/ocr", {"image": base64.b64encode(p["png"]).decode()})
        row["/ocr (b64 PNG)"] = extract(res)

        # /ocr/raw PNG
        res = post_bytes("/ocr/raw", p["png"], "image/png")
        row["/ocr/raw PNG"] = extract(res)

        # /ocr/raw JPEG
        res = post_bytes("/ocr/raw", p["jpg"], "image/jpeg")
        row["/ocr/raw JPEG"] = extract(res)

        # /ocr/pixels (raw BGR + size headers)
        res = post_bytes(
            "/ocr/pixels", p["bgr"], "application/octet-stream",
            extra_headers={"X-Width": str(p["w"]), "X-Height": str(p["h"]),
                           "X-Channels": "3"},
        )
        row["/ocr/pixels BGR"] = extract(res)

        # /ocr/batch — send this page as a 1-image batch
        res = post_json("/ocr/batch", {"images": [base64.b64encode(p["png"]).decode()]})
        br = res.get("batch_results") or res.get("results") or []
        if br and isinstance(br[0], dict):
            row["/ocr/batch PNG"] = [it["text"] for it in br[0].get("results", [])]
        else:
            row["/ocr/batch PNG"] = []

        # /ocr/pdf
        pdf_bytes = png_to_pdf(p["png"])
        res = post_bytes("/ocr/pdf", pdf_bytes, "application/pdf")
        pages_out = res.get("pages", [])
        pdf_texts = []
        if pages_out:
            pdf_texts = [it["text"] for it in pages_out[0].get("results", [])]
        row["/ocr/pdf"] = pdf_texts

        texts_per_page.append(row)
        for ep, texts in row.items():
            f1_per_ep[ep].append(f1(tokens(" ".join(texts)), p["gt"]))

    # Report
    print(f"\nPer-endpoint F1 (mean over {N_PAGES} FUNSD pages):")
    for ep in endpoints:
        vals = f1_per_ep[ep]
        print(f"  {ep:<20} F1={np.mean(vals):.1%}   per-page={['%.1f%%' % (v*100) for v in vals]}")

    # Consistency check: do raw/PNG and pixels produce identical word sets?
    print("\nConsistency (word-set overlap vs /ocr/raw PNG):")
    for p_idx, row in enumerate(texts_per_page):
        base = tokens(" ".join(row["/ocr/raw PNG"]))
        line = [f"page{p_idx}"]
        for ep in endpoints:
            cur = tokens(" ".join(row[ep]))
            if not base and not cur:
                sim = 1.0
            else:
                inter = len(base & cur)
                union = len(base | cur)
                sim = inter / union if union else 0.0
            line.append(f"{ep.split()[0]}={sim:.0%}")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
