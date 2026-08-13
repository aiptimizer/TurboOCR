from __future__ import annotations

import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import httpx
import pytest

from turboocr import Client


def _server_url() -> str:
    return os.environ.get("TURBO_OCR_BASE_URL", "http://localhost:8080")


def _server_reachable(url: str) -> bool:
    try:
        with httpx.Client(base_url=url, timeout=2.0) as http:
            return http.get("/health").is_success
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session")
def server_url() -> str:
    url = _server_url()
    if not _server_reachable(url):
        pytest.skip(
            f"turbo-ocr server not reachable at {url}; "
            "start it (or set TURBO_OCR_BASE_URL) to run integration tests"
        )
    return url


@pytest.fixture
def client(server_url: str) -> Iterator[Client]:
    with Client(base_url=server_url) as c:
        yield c


@pytest.fixture(scope="session")
def sample_image() -> bytes:
    for env_var in ("TURBO_OCR_SAMPLE_IMAGE",):
        if (override := os.environ.get(env_var)) and Path(override).is_file():
            return Path(override).read_bytes()
    for rel in ("fixtures/sample.png", "fixtures/sample.jpg", "data/sample.png"):
        full = (Path(__file__).parent.parent / rel).resolve()
        if full.is_file():
            return full.read_bytes()
    return _render_sample_image()


def _render_sample_image() -> bytes:
    from io import BytesIO as _BytesIO

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen.canvas import Canvas

    pdf_buf = _BytesIO()
    canvas = Canvas(pdf_buf, pagesize=letter)
    canvas.setFont("Helvetica", 36)
    canvas.drawString(72, 720, "Turbo OCR Integration Test")
    canvas.setFont("Helvetica", 18)
    canvas.drawString(72, 670, "The quick brown fox jumps over the lazy dog.")
    canvas.drawString(72, 640, "Sphinx of black quartz, judge my vow.")
    canvas.showPage()
    canvas.save()

    try:
        import pypdfium2
    except ImportError:
        pytest.skip("install pypdfium2 to generate a sample image at integration test time")

    pdf = pypdfium2.PdfDocument(pdf_buf.getvalue())
    pil = pdf[0].render(scale=2).to_pil()
    out = _BytesIO()
    pil.save(out, format="PNG")
    return out.getvalue()


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen.canvas import Canvas

    buf = BytesIO()
    canvas = Canvas(buf, pagesize=letter)
    canvas.setFont("Helvetica", 24)
    canvas.drawString(100, 700, "Integration Test Page")
    canvas.drawString(100, 650, "The quick brown fox jumps over the lazy dog.")
    canvas.showPage()
    canvas.save()
    return buf.getvalue()
