"""Per-document speed + accuracy matrix.

For every fixture and every applicable endpoint, measures:
  - median latency (5 runs, warmed)
  - F1 word-overlap vs the committed ground truth

Emits a markdown matrix (endpoints x documents) plus a summary row and
writes to tests/benchmark/scoring/PER_DOCUMENT.md. Intended as the single
"see everything at once" report.
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
TESTS_DIR = HERE.parents[1]
FIXTURES = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR / "accuracy"))
from _scoring import load_json, score_image, score_pdf, tokenize, word_f1  # type: ignore

IMAGE_FIXTURES: list[Path] = sorted((FIXTURES / "images" / "png").glob("*.png")) \
    + sorted((FIXTURES / "images" / "jpeg").glob("*.jpg"))
PDF_FIXTURES: list[Path] = sorted((FIXTURES / "pdf").glob("*.pdf"))

IMAGE_EXP = FIXTURES / "images" / "expected"
PDF_EXP = FIXTURES / "pdf" / "expected"

PDF_MODES = ["ocr", "geometric", "auto", "auto_verified"]
RUNS_PER_CELL = 5
WARMUP = 2


async def _post(session: aiohttp.ClientSession, url: str, data: bytes, headers: dict) -> tuple[int, dict | None]:
    async with session.post(url, data=data, headers=headers) as r:
        if r.status != 200:
            await r.read()
            return r.status, None
        try:
            return 200, await r.json()
        except Exception:
            return 200, None


async def _measure(session, url, data, headers) -> tuple[float, dict | None]:
    for _ in range(WARMUP):
        await _post(session, url, data, headers)
    lats = []
    last_json = None
    for _ in range(RUNS_PER_CELL):
        t0 = time.perf_counter()
        status, j = await _post(session, url, data, headers)
        lats.append((time.perf_counter() - t0) * 1000.0)
        if status == 200 and j is not None:
            last_json = j
    return statistics.median(lats), last_json


async def bench_image(session, base_url, path: Path) -> dict[str, Any]:
    exp_path = IMAGE_EXP / f"{path.stem}.json"
    expected = load_json(exp_path) if exp_path.exists() else None
    png_bytes = path.read_bytes()
    suffix = path.suffix.lower()
    ct = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}[suffix]

    out = {"fixture": path.stem}

    # /ocr/raw
    lat, j = await _measure(session, f"{base_url}/ocr/raw", png_bytes, {"Content-Type": ct})
    out["raw_lat_ms"] = lat
    out["raw_f1"] = score_image(expected, j)["f1"] if (expected and j) else None

    # /ocr (JSON + base64)
    payload = json.dumps({"image": base64.b64encode(png_bytes).decode()})
    lat, j = await _measure(session, f"{base_url}/ocr", payload.encode(),
                             {"Content-Type": "application/json"})
    out["ocr_lat_ms"] = lat
    out["ocr_f1"] = score_image(expected, j)["f1"] if (expected and j) else None

    # /ocr/pixels (BGR)
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)[:, :, ::-1].copy()
    h, w = arr.shape[:2]
    lat, j = await _measure(
        session, f"{base_url}/ocr/pixels", arr.tobytes(),
        {"X-Width": str(w), "X-Height": str(h), "X-Channels": "3"},
    )
    out["pixels_lat_ms"] = lat
    out["pixels_f1"] = score_image(expected, j)["f1"] if (expected and j) else None

    return out


async def bench_pdf(session, base_url, path: Path) -> dict[str, Any]:
    exp_path = PDF_EXP / f"{path.stem}.json"
    expected = load_json(exp_path) if exp_path.exists() else None
    pdf_bytes = path.read_bytes()
    out = {"fixture": path.stem, "n_pages": 0}
    for mode in PDF_MODES:
        lat, j = await _measure(
            session,
            f"{base_url}/ocr/pdf?mode={mode}",
            pdf_bytes,
            {"Content-Type": "application/pdf"},
        )
        out[f"{mode}_lat_ms"] = lat
        out[f"{mode}_f1"] = score_pdf(expected, j)["f1"] if (expected and j) else None
        if j and "pages" in j:
            out["n_pages"] = max(out["n_pages"], len(j["pages"]))
    return out


def fmt_f1(v):
    return f"{v:.2f}" if v is not None else "—"


def fmt_ms(v):
    return f"{v:.0f}" if v is not None else "—"


def render_markdown(img_rows, pdf_rows) -> str:
    lines = [
        "# Turbo OCR — per-document speed + accuracy",
        "",
        f"runs per cell: {RUNS_PER_CELL} (median), warmup: {WARMUP}",
        "F1 is word-overlap vs the captured-baseline ground truth (tests/fixtures/*/expected/).",
        "",
    ]

    # Images
    lines += [
        "## Images",
        "",
        "| fixture | /ocr F1 | /ocr ms | /ocr/raw F1 | /ocr/raw ms | /ocr/pixels F1 | /ocr/pixels ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in img_rows:
        lines.append(
            f"| {r['fixture']} | "
            f"{fmt_f1(r.get('ocr_f1'))} | {fmt_ms(r.get('ocr_lat_ms'))} | "
            f"{fmt_f1(r.get('raw_f1'))} | {fmt_ms(r.get('raw_lat_ms'))} | "
            f"{fmt_f1(r.get('pixels_f1'))} | {fmt_ms(r.get('pixels_lat_ms'))} |"
        )
    # means
    def mean(key):
        vals = [r[key] for r in img_rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None
    lines += [
        f"| **mean** | **{fmt_f1(mean('ocr_f1'))}** | **{fmt_ms(mean('ocr_lat_ms'))}** | "
        f"**{fmt_f1(mean('raw_f1'))}** | **{fmt_ms(mean('raw_lat_ms'))}** | "
        f"**{fmt_f1(mean('pixels_f1'))}** | **{fmt_ms(mean('pixels_lat_ms'))}** |",
        "",
    ]

    # PDFs
    lines += [
        "## PDFs (`/ocr/pdf?mode=...`)",
        "",
        "F1/ms per mode. Latency is per-document, not per-page.",
        "",
        "| fixture | pages | ocr F1 | ocr ms | geometric F1 | geometric ms | auto F1 | auto ms | auto_verified F1 | auto_verified ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pdf_rows:
        lines.append(
            f"| {r['fixture']} | {r['n_pages']} | "
            + " | ".join(
                f"{fmt_f1(r.get(f'{m}_f1'))} | {fmt_ms(r.get(f'{m}_lat_ms'))}"
                for m in PDF_MODES
            )
            + " |"
        )
    def pmean(key):
        vals = [r[key] for r in pdf_rows if r.get(key) is not None]
        return statistics.mean(vals) if vals else None
    lines.append(
        "| **mean** | — | "
        + " | ".join(
            f"**{fmt_f1(pmean(f'{m}_f1'))}** | **{fmt_ms(pmean(f'{m}_lat_ms'))}**"
            for m in PDF_MODES
        )
        + " |"
    )
    lines.append("")
    return "\n".join(lines)


async def main(base_url: str, out_md: Path, out_json: Path):
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        print(f"images ({len(IMAGE_FIXTURES)})")
        img_rows = []
        for i, p in enumerate(IMAGE_FIXTURES, 1):
            row = await bench_image(session, base_url, p)
            img_rows.append(row)
            print(f"  [{i}/{len(IMAGE_FIXTURES)}] {p.stem}  "
                  f"ocr_f1={fmt_f1(row.get('ocr_f1'))}  raw_ms={fmt_ms(row.get('raw_lat_ms'))}")

        print(f"\npdfs ({len(PDF_FIXTURES)})")
        pdf_rows = []
        for i, p in enumerate(PDF_FIXTURES, 1):
            row = await bench_pdf(session, base_url, p)
            pdf_rows.append(row)
            print(f"  [{i}/{len(PDF_FIXTURES)}] {p.stem} ({row['n_pages']}pg)  "
                  f"ocr_f1={fmt_f1(row.get('ocr_f1'))} geo_f1={fmt_f1(row.get('geometric_f1'))}")

    md = render_markdown(img_rows, pdf_rows)
    out_md.write_text(md)
    out_json.write_text(json.dumps({"images": img_rows, "pdfs": pdf_rows}, indent=2))
    print(f"\nwrote {out_md}\nwrote {out_json}")
    print("\n" + md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default=os.environ.get("OCR_SERVER_URL", "http://localhost:8000"))
    ap.add_argument("--out-md", default=str(HERE / "PER_DOCUMENT.md"))
    ap.add_argument("--out-json", default=str(HERE / "PER_DOCUMENT.json"))
    args = ap.parse_args()
    asyncio.run(main(args.server_url, Path(args.out_md), Path(args.out_json)))
