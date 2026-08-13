"""Shared benchmark primitives: latency, throughput, stress harnesses.

Single source of truth for the endpoint registry (see ENDPOINTS) so all
bench scripts and stress tests measure the same cells.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiohttp

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
PDF_DIR = FIXTURES_DIR / "pdf"


def load_image_bytes() -> bytes:
    """Canonical mid-size A4-ish fixture for image benchmarks."""
    candidates = [
        IMAGES_DIR / "png" / "business_letter.png",
        IMAGES_DIR / "png" / "dense_text.png",
        IMAGES_DIR / "jpeg" / "08_document_scan.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p.read_bytes()
    raise FileNotFoundError("no benchmark image fixture available")


def load_pdf_bytes_list() -> list[bytes]:
    """Rotated list of small real PDFs (≤4 pages each) for bench loops."""
    names = ["mixed_text_images.pdf", "multi_column.pdf", "academic_paper.pdf", "formulas.pdf"]
    out = []
    for n in names:
        p = PDF_DIR / n
        if p.exists():
            out.append(p.read_bytes())
    if not out:
        raise FileNotFoundError("no benchmark PDF fixtures available")
    return out


@dataclass
class Endpoint:
    name: str
    path: str  # includes query string for PDF modes
    payload_kind: str  # 'image' | 'pdf' | 'batch'
    content_type: str
    unit: str  # 'imgs/s' | 'pages/s' | 'reqs/s'
    stress_concurrency: int
    p99_ceiling_ms: float


ENDPOINTS: list[Endpoint] = [
    Endpoint("POST /ocr",                       "/ocr",                        "json_img","application/json","imgs/s",  32,  150.0),
    Endpoint("POST /ocr/raw",                   "/ocr/raw",                    "image",   "image/png",       "imgs/s",  32,  100.0),
    Endpoint("POST /ocr/pixels",                "/ocr/pixels",                 "pixels",  "application/octet-stream","imgs/s",32,100.0),
    Endpoint("POST /ocr/batch",                 "/ocr/batch",                  "batch",   "application/json","imgs/s",   8,  500.0),
    Endpoint("POST /ocr/pdf?mode=ocr",          "/ocr/pdf?mode=ocr",           "pdf",     "application/pdf", "pages/s", 32,  500.0),
    Endpoint("POST /ocr/pdf?mode=geometric",    "/ocr/pdf?mode=geometric",     "pdf",     "application/pdf", "pages/s",  8, 3000.0),
    Endpoint("POST /ocr/pdf?mode=auto",         "/ocr/pdf?mode=auto",          "pdf",     "application/pdf", "pages/s", 16, 2000.0),
    Endpoint("POST /ocr/pdf?mode=auto_verified","/ocr/pdf?mode=auto_verified", "pdf",     "application/pdf", "pages/s",  8, 4000.0),
]


# ---------------------------------------------------------------------------
# Per-request senders
# ---------------------------------------------------------------------------

async def _send_image(session: aiohttp.ClientSession, url: str, ct: str, body: bytes):
    async with session.post(url, data=body, headers={"Content-Type": ct}) as r:
        await r.read()
        return r.status, 1


async def _send_json_img(session: aiohttp.ClientSession, url: str, _ct: str, body: bytes):
    import base64
    payload = json.dumps({"image": base64.b64encode(body).decode()})
    async with session.post(url, data=payload, headers={"Content-Type": "application/json"}) as r:
        await r.read()
        return r.status, 1


async def _send_pixels(session: aiohttp.ClientSession, url: str, _ct: str, body: bytes):
    """body is raw BGR pixel bytes; we stash (w,h) on the bytes via a module global."""
    w, h = _PIXELS_WH
    async with session.post(
        url, data=body,
        headers={"X-Width": str(w), "X-Height": str(h), "X-Channels": "3",
                 "Content-Type": "application/octet-stream"},
    ) as r:
        await r.read()
        return r.status, 1


async def _send_batch(session: aiohttp.ClientSession, url: str, _ct: str, body: bytes):
    import base64
    b64 = base64.b64encode(body).decode()
    payload = json.dumps({"images": [b64] * 8})
    async with session.post(url, data=payload, headers={"Content-Type": "application/json"}) as r:
        await r.read()
        return r.status, 8


async def _send_pdf(session: aiohttp.ClientSession, url: str, ct: str, body: bytes):
    async with session.post(url, data=body, headers={"Content-Type": ct}) as r:
        if r.status != 200:
            await r.read()
            return r.status, 0
        try:
            j = await r.json()
            return r.status, len(j.get("pages", []))
        except Exception:
            return r.status, 0


def _sender_for(kind: str):
    return {
        "image": _send_image,
        "json_img": _send_json_img,
        "pixels": _send_pixels,
        "batch": _send_batch,
        "pdf": _send_pdf,
    }[kind]


_PIXELS_WH: tuple[int, int] = (0, 0)


def _payloads_for(ep: Endpoint):
    global _PIXELS_WH
    if ep.payload_kind == "pdf":
        return load_pdf_bytes_list()
    if ep.payload_kind == "pixels":
        import numpy as np
        from PIL import Image
        path = next((IMAGES_DIR / "png").glob("business_letter.png"), None) \
               or next((IMAGES_DIR / "png").iterdir(), None)
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)[:, :, ::-1].copy()  # RGB->BGR
        _PIXELS_WH = (arr.shape[1], arr.shape[0])
        return [arr.tobytes()]
    return [load_image_bytes()]


# ---------------------------------------------------------------------------
# Harnesses
# ---------------------------------------------------------------------------

async def measure_latency(
    session: aiohttp.ClientSession,
    base_url: str,
    ep: Endpoint,
    n: int = 200,
    warmup: int = 20,
) -> dict[str, Any]:
    url = f"{base_url}{ep.path}"
    sender = _sender_for(ep.payload_kind)
    payloads = _payloads_for(ep)

    # warmup
    for i in range(warmup):
        await sender(session, url, ep.content_type, payloads[i % len(payloads)])

    latencies_ms: list[float] = []
    ok = 0
    err = 0
    for i in range(n):
        t0 = time.perf_counter()
        status, _ = await sender(session, url, ep.content_type, payloads[i % len(payloads)])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        if status == 200:
            ok += 1
        else:
            err += 1

    latencies_ms.sort()
    return {
        "endpoint": ep.name,
        "n": n,
        "ok": ok,
        "err": err,
        "p50_ms": latencies_ms[int(n * 0.50)],
        "p95_ms": latencies_ms[int(n * 0.95)],
        "p99_ms": latencies_ms[min(int(n * 0.99), n - 1)],
        "avg_ms": statistics.mean(latencies_ms),
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
    }


async def measure_throughput(
    session: aiohttp.ClientSession,
    base_url: str,
    ep: Endpoint,
    concurrency: int,
    n: int = 200,
) -> dict[str, Any]:
    url = f"{base_url}{ep.path}"
    sender = _sender_for(ep.payload_kind)
    payloads = _payloads_for(ep)
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int):
        async with sem:
            return await sender(session, url, ep.content_type, payloads[i % len(payloads)])

    t0 = time.perf_counter()
    results = await asyncio.gather(*[one(i) for i in range(n)])
    dt = time.perf_counter() - t0

    ok = sum(1 for s, _ in results if s == 200)
    err = n - ok
    units = sum(u for _, u in results)
    return {
        "endpoint": ep.name,
        "concurrency": concurrency,
        "n": n,
        "ok": ok,
        "err": err,
        "elapsed_s": dt,
        "reqs_per_s": ok / dt if dt else 0.0,
        "units_per_s": units / dt if dt else 0.0,
        "unit": ep.unit,
    }


async def measure_stress(
    session: aiohttp.ClientSession,
    base_url: str,
    ep: Endpoint,
    duration_s: float = 60.0,
    concurrency: int | None = None,
    warmup: int = 20,
) -> dict[str, Any]:
    concurrency = concurrency or ep.stress_concurrency
    url = f"{base_url}{ep.path}"
    sender = _sender_for(ep.payload_kind)
    payloads = _payloads_for(ep)

    for i in range(warmup):
        await sender(session, url, ep.content_type, payloads[i % len(payloads)])

    latencies_ms: list[float] = []
    ok = 0
    err = 0
    units = 0
    stop_at = time.monotonic() + duration_s
    counter = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal ok, err, units, counter
        while time.monotonic() < stop_at:
            async with lock:
                idx = counter
                counter += 1
            t0 = time.perf_counter()
            try:
                status, u = await sender(session, url, ep.content_type, payloads[idx % len(payloads)])
            except Exception:
                err += 1
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            if status == 200:
                ok += 1
                units += u
            else:
                err += 1

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    dt = time.perf_counter() - t0

    latencies_ms.sort()
    total = ok + err
    return {
        "endpoint": ep.name,
        "concurrency": concurrency,
        "duration_s": dt,
        "total": total,
        "ok": ok,
        "errors": err,
        "error_rate": err / total if total else 0.0,
        "reqs_per_s": ok / dt if dt else 0.0,
        "units_per_s": units / dt if dt else 0.0,
        "unit": ep.unit,
        "p50_ms": latencies_ms[int(len(latencies_ms) * 0.50)] if latencies_ms else 0.0,
        "p95_ms": latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0,
        "p99_ms": latencies_ms[min(int(len(latencies_ms) * 0.99), len(latencies_ms) - 1)] if latencies_ms else 0.0,
        "max_ms": latencies_ms[-1] if latencies_ms else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def emit_json(report: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, default=str))


def emit_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Turbo OCR benchmark",
        "",
        f"generated: {report.get('timestamp','')}",
        f"server:    {report.get('server_url','')}",
        "",
    ]
    if lat := report.get("latency"):
        lines += ["## Latency (c=1, n=200)", "",
                  "| endpoint | p50 ms | p95 ms | p99 ms | avg ms | err |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in lat:
            lines.append(
                f"| {r['endpoint']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | "
                f"{r['p99_ms']:.1f} | {r['avg_ms']:.1f} | {r['err']} |"
            )
        lines.append("")
    if tp := report.get("throughput"):
        lines += ["## Throughput", "",
                  "| endpoint | conc | unit | reqs/s | units/s | err |",
                  "|---|---:|---|---:|---:|---:|"]
        for r in tp:
            lines.append(
                f"| {r['endpoint']} | {r['concurrency']} | {r['unit']} | "
                f"{r['reqs_per_s']:.1f} | {r['units_per_s']:.1f} | {r['err']} |"
            )
        lines.append("")
    if st := report.get("stress"):
        lines += ["## Stress (60s soak)", "",
                  "| endpoint | conc | reqs/s | units/s | p99 ms | errors | err rate |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in st:
            lines.append(
                f"| {r['endpoint']} | {r['concurrency']} | {r['reqs_per_s']:.1f} | "
                f"{r['units_per_s']:.1f} | {r['p99_ms']:.1f} | {r['errors']} | "
                f"{r['error_rate']*100:.2f}% |"
            )
        lines.append("")
    Path(path).write_text("\n".join(lines))
