#!/usr/bin/env python3
"""Minimal HTTP wrapper around PaddleOCR 3.x, endpoint-shaped like TurboOCR.

Exposes on :9090 —
  POST /ocr/raw        image bytes           -> {"results": [{text, confidence, bounding_box}]}
  POST /ocr            {"image": b64}        -> same
  POST /ocr/batch      {"images": [b64,...]} -> {"results": [[...], ...]}
  POST /ocr/structure  image bytes (?tables=1&formulas=1) -> layout regions (+tables html, formulas latex)
  POST /ocr/markdown   image bytes           -> {"markdown": "..."}
  POST /ocr/pdf        pdf bytes             -> {"pages": [{"page": n, "results": [...]}]}
  GET  /health

PP-OCRv5 mobile (det+rec, GPU) for text; PP-StructureV3 for structure/markdown.
"""

import base64
import os
import tempfile

import numpy as np
import cv2
from fastapi import FastAPI, Request
import uvicorn

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCR  # noqa: E402

app = FastAPI()

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    device="gpu",
)

_structure = None  # lazy: model download is slow and only some endpoints need it


def structure():
    global _structure
    if _structure is None:
        from paddleocr import PPStructureV3
        _structure = PPStructureV3(device="gpu",
                                   use_doc_orientation_classify=False,
                                   use_doc_unwarping=False)
    return _structure


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_ocr(img: np.ndarray):
    out = ocr.predict(img)
    results = []
    for page in out:
        d = page  # dict-like OCRResult
        texts = d["rec_texts"]
        scores = d["rec_scores"]
        polys = d["rec_polys"]
        for t, s, p in zip(texts, scores, polys):
            box = [[int(x), int(y)] for x, y in np.asarray(p).reshape(-1, 2).tolist()]
            results.append({"text": t, "confidence": float(s), "bounding_box": box})
    return results


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr/raw")
async def ocr_raw(request: Request):
    img = decode_image(await request.body())
    return {"results": run_ocr(img)}


@app.post("/ocr")
async def ocr_b64(request: Request):
    j = await request.json()
    img = decode_image(base64.b64decode(j["image"]))
    return {"results": run_ocr(img)}


@app.post("/ocr/batch")
async def ocr_batch(request: Request):
    j = await request.json()
    out = []
    for b in j["images"]:
        out.append(run_ocr(decode_image(base64.b64decode(b))))
    return {"results": out}


@app.post("/ocr/structure")
async def ocr_structure(request: Request):
    img = decode_image(await request.body())
    res = structure().predict(img)
    regions, tables, formulas = [], [], []
    for page in res:
        d = page
        lay = d.get("layout_det_res", {})
        for b in lay.get("boxes", []):
            regions.append({"label": b.get("label"),
                            "score": float(b.get("score", 0)),
                            "box": [int(v) for v in b.get("coordinate", [])]})
        for tr in d.get("table_res_list", []) or []:
            tables.append({"html": tr.get("pred_html", "")})
        for fr in d.get("formula_res_list", []) or []:
            formulas.append({"latex": fr.get("rec_formula", "")})
    return {"layout": regions, "tables": tables, "formulas": formulas}


@app.post("/ocr/markdown")
async def ocr_markdown(request: Request):
    data = await request.body()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        res = structure().predict(path)
        md = []
        for page in res:
            m = getattr(page, "markdown", None)
            if m is not None:
                md.append(m.get("markdown_texts", "") if isinstance(m, dict) else str(m))
        return {"markdown": "\n\n".join(md)}
    finally:
        os.unlink(path)


@app.post("/ocr/pdf")
async def ocr_pdf(request: Request):
    data = await request.body()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        out = ocr.predict(path)
        pages = []
        for i, page in enumerate(out):
            d = page
            results = []
            for t, s, p in zip(d["rec_texts"], d["rec_scores"], d["rec_polys"]):
                box = [[int(x), int(y)] for x, y in np.asarray(p).reshape(-1, 2).tolist()]
                results.append({"text": t, "confidence": float(s), "bounding_box": box})
            pages.append({"page": i, "results": results})
        return {"pages": pages}
    finally:
        os.unlink(path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9090, log_level="warning")
