"""Integration tests for the CPU-only server.

All endpoints: /health, /ocr, /ocr/raw, /ocr/pixels, /ocr/pdf.
Layout support via ONNX Runtime. Conservative timeouts for CPU inference.

Requires: running CPU server on port 8001 (or CPU_SERVER_URL env var).
"""

import base64
import io
import os

import pytest
import requests

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

CPU_URL = os.environ.get("CPU_SERVER_URL", "http://localhost:8001")


@pytest.fixture(scope="module")
def cpu_url():
    """CPU server URL. Skip all tests if server is not reachable."""
    try:
        r = requests.get(f"{CPU_URL}/health", timeout=5)
        if r.text != "ok":
            pytest.skip("CPU server not healthy")
    except requests.ConnectionError:
        pytest.skip(f"CPU server not reachable at {CPU_URL}")
    return CPU_URL


@pytest.fixture(scope="module")
def test_image():
    """Small test image with known text."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (300, 80), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "HELLO WORLD", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def test_jpeg(test_image):
    from PIL import Image
    img = Image.open(io.BytesIO(test_image))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_pdf(pages=1, text_fn=None):
    if not _HAS_REPORTLAB:
        pytest.skip("reportlab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        c.setFont("Helvetica", 24)
        text = text_fn(i) if text_fn else f"Page {i + 1}"
        c.drawString(100, 700, text)
        c.showPage()
    c.save()
    return buf.getvalue()


class TestCpuHealth:
    def test_health(self, cpu_url):
        r = requests.get(f"{cpu_url}/health", timeout=5)
        assert r.status_code == 200
        assert r.text == "ok"


class TestCpuOcr:
    def test_ocr_base64_png(self, cpu_url, test_image):
        b64 = base64.b64encode(test_image).decode()
        r = requests.post(
            f"{cpu_url}/ocr",
            json={"image": b64},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert len(data["results"]) > 0

    def test_ocr_raw_png(self, cpu_url, test_image):
        r = requests.post(
            f"{cpu_url}/ocr/raw",
            data=test_image,
            headers={"Content-Type": "image/png"},
            timeout=60,
        )
        assert r.status_code == 200
        assert len(r.json()["results"]) > 0

    def test_ocr_raw_jpeg(self, cpu_url, test_jpeg):
        r = requests.post(
            f"{cpu_url}/ocr/raw",
            data=test_jpeg,
            headers={"Content-Type": "image/jpeg"},
            timeout=60,
        )
        assert r.status_code == 200
        assert len(r.json()["results"]) > 0

    def test_ocr_pixels(self, cpu_url):
        from PIL import Image, ImageDraw
        import numpy as np
        img = Image.new("RGB", (200, 50), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "PIXELS", fill=(0, 0, 0))
        # Convert to BGR numpy
        arr = np.array(img)[:, :, ::-1].copy()
        r = requests.post(
            f"{cpu_url}/ocr/pixels",
            data=arr.tobytes(),
            headers={"X-Width": "200", "X-Height": "50"},
            timeout=60,
        )
        assert r.status_code == 200
        assert len(r.json()["results"]) > 0

    def test_ocr_empty_body(self, cpu_url):
        r = requests.post(f"{cpu_url}/ocr/raw", data=b"", timeout=10)
        assert r.status_code == 400


class TestCpuLayout:
    def test_layout_on_raw(self, cpu_url, test_image):
        r = requests.post(
            f"{cpu_url}/ocr/raw?layout=1",
            data=test_image,
            headers={"Content-Type": "image/png"},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert "layout" in data

    def test_layout_on_base64(self, cpu_url, test_image):
        b64 = base64.b64encode(test_image).decode()
        r = requests.post(
            f"{cpu_url}/ocr?layout=1",
            json={"image": b64},
            timeout=60,
        )
        assert r.status_code == 200
        assert "layout" in r.json()

    def test_no_layout_by_default(self, cpu_url, test_image):
        r = requests.post(
            f"{cpu_url}/ocr/raw",
            data=test_image,
            headers={"Content-Type": "image/png"},
            timeout=60,
        )
        assert r.status_code == 200
        assert "layout" not in r.json()


class TestCpuPdf:
    def test_pdf_raw(self, cpu_url):
        pdf = _make_pdf(2)
        r = requests.post(f"{cpu_url}/ocr/pdf", data=pdf, timeout=120)
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 2

    def test_pdf_multipart(self, cpu_url):
        pdf = _make_pdf(1)
        r = requests.post(
            f"{cpu_url}/ocr/pdf",
            files={"file": ("test.pdf", pdf, "application/pdf")},
            timeout=120,
        )
        assert r.status_code == 200
        assert len(r.json()["pages"]) == 1

    def test_pdf_base64(self, cpu_url):
        pdf = _make_pdf(1)
        r = requests.post(
            f"{cpu_url}/ocr/pdf",
            json={"pdf": base64.b64encode(pdf).decode()},
            timeout=120,
        )
        assert r.status_code == 200
        assert len(r.json()["pages"]) == 1

    def test_pdf_with_layout(self, cpu_url):
        pdf = _make_pdf(1)
        r = requests.post(
            f"{cpu_url}/ocr/pdf?layout=1", data=pdf, timeout=120
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 1
        assert "layout" in data["pages"][0]

    def test_pdf_page_schema(self, cpu_url):
        pdf = _make_pdf(1)
        r = requests.post(f"{cpu_url}/ocr/pdf", data=pdf, timeout=120)
        page = r.json()["pages"][0]
        assert "page" in page
        assert "page_index" in page
        assert "dpi" in page
        assert "width" in page
        assert "height" in page
        assert "results" in page
        assert "mode" in page

    def test_pdf_10_pages(self, cpu_url):
        pdf = _make_pdf(10, text_fn=lambda i: f"TEN_{i:02d}")
        r = requests.post(f"{cpu_url}/ocr/pdf", data=pdf, timeout=300)
        assert r.status_code == 200
        assert len(r.json()["pages"]) == 10

    def test_pdf_empty_body(self, cpu_url):
        r = requests.post(f"{cpu_url}/ocr/pdf", data=b"", timeout=10)
        assert r.status_code == 400

    def test_pdf_invalid_data(self, cpu_url):
        r = requests.post(f"{cpu_url}/ocr/pdf", data=b"not a pdf", timeout=10)
        assert r.status_code == 400


class TestCpuPdfLarge:
    """Large PDF tests — conservative timeouts for CPU inference."""

    def test_100_page_sparse_pdf(self, cpu_url):
        """100 pages with minimal text — tests streaming pipeline on CPU."""
        pdf = _make_pdf(100, text_fn=lambda i: f"P{i}")
        r = requests.post(f"{cpu_url}/ocr/pdf", data=pdf, timeout=600)
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 100
        # Verify ordering
        for i, page in enumerate(data["pages"]):
            assert page["page"] == i + 1


class TestCpuErrorHandling:
    def test_pixels_missing_headers(self, cpu_url):
        r = requests.post(f"{cpu_url}/ocr/pixels", data=b"\x00" * 100, timeout=10)
        assert r.status_code == 400

    def test_pixels_wrong_size(self, cpu_url):
        r = requests.post(
            f"{cpu_url}/ocr/pixels",
            data=b"\x00" * 100,
            headers={"X-Width": "10", "X-Height": "10"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_invalid_layout_param(self, cpu_url, test_image):
        r = requests.post(
            f"{cpu_url}/ocr/raw?layout=maybe",
            data=test_image,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 400
