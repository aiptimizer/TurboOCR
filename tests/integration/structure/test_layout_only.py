"""Integration tests for ?text=0 (layout-only / image-only runs, GPU build).

text=0 skips text detection/recognition entirely: with layout=1 the response
carries only the layout array (a layout-only pass); on /ocr/pdf with
images=inline it is the fast pdf->page-images path with zero OCR cost.
Everything text-derived (tables/formulas/blocks/reading_order) is rejected
loudly rather than returned silently empty.
"""
import io

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes, pil_to_jpeg_bytes


def _pdf_bytes(pages=2):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        c.setFont("Helvetica", 24)
        c.drawString(100, 700, f"Layout only page {i + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="module")
def gpu_layout(server_url):
    caps = requests.get(f"{server_url}/capabilities", timeout=5).json()
    if caps.get("build") != "gpu":
        pytest.skip("text=0 is GPU-only")
    if not caps.get("features", {}).get("layout"):
        pytest.skip("layout model not loaded")
    return True


class TestLayoutOnlyImage:
    def test_layout_only_returns_layout_no_text(self, server_url, gpu_layout, hello_image):
        r = requests.post(f"{server_url}/ocr/raw?text=0&layout=1",
                          data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png"}, timeout=30)
        assert r.status_code == 200, r.text[:200]
        d = r.json()
        assert d["results"] == [], "text=0 must not return OCR results"
        assert "layout" in d

    def test_layout_only_jpeg(self, server_url, gpu_layout, hello_image):
        """JPEG goes through the generic decode branch under text=0 (the
        nvJPEG GPU-direct fast path is OCR-only)."""
        r = requests.post(f"{server_url}/ocr/raw?text=0&layout=1",
                          data=pil_to_jpeg_bytes(hello_image),
                          headers={"Content-Type": "image/jpeg"}, timeout=30)
        assert r.status_code == 200, r.text[:200]
        assert r.json()["results"] == []

    def test_default_unaffected(self, server_url, gpu_layout):
        """No text param -> full OCR, exactly as before the feature."""
        img = make_text_image("DEFAULT STILL OCRS", width=420, height=100, font_size=32)
        r = requests.post(f"{server_url}/ocr/raw",
                          data=pil_to_png_bytes(img),
                          headers={"Content-Type": "image/png"}, timeout=30)
        assert r.status_code == 200
        text = " ".join(x["text"] for x in r.json()["results"]).lower()
        assert "default" in text or "still" in text

    def test_text0_alone_rejected(self, server_url, gpu_layout, hello_image):
        r = requests.post(f"{server_url}/ocr/raw?text=0",
                          data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png"}, timeout=15)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_PARAMETER"

    def test_text0_with_structure_rejected(self, server_url, gpu_layout, hello_image):
        for extra in ("tables=1", "formulas=1", "as_blocks=1", "reading_order=1"):
            r = requests.post(f"{server_url}/ocr/raw?text=0&layout=1&{extra}",
                              data=pil_to_png_bytes(hello_image),
                              headers={"Content-Type": "image/png"}, timeout=15)
            assert r.status_code == 400, f"{extra}: {r.status_code}"

    def test_batch_rejects_text0(self, server_url, gpu_layout, hello_image):
        import base64
        b64 = base64.b64encode(pil_to_png_bytes(hello_image)).decode()
        r = requests.post(f"{server_url}/ocr/batch?text=0&layout=1",
                          json={"images": [b64]}, timeout=15)
        assert r.status_code == 400
        assert "batch" in r.json()["error"]["message"].lower()


class TestLayoutOnlyPdf:
    def test_image_only_pdf(self, server_url, gpu_layout):
        """text=0&images=inline: page images with zero OCR cost."""
        r = requests.post(f"{server_url}/ocr/pdf?text=0&images=inline&format=jpeg",
                          data=_pdf_bytes(2),
                          headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200, r.text[:300]
        pages = r.json()["pages"]
        assert len(pages) == 2
        for p in pages:
            assert "image_b64" in p, "image-only page missing its image"
            assert p["results"] == [], "text=0 pdf page must carry no text"

    def test_layout_plus_image_pdf(self, server_url, gpu_layout):
        r = requests.post(
            f"{server_url}/ocr/pdf?text=0&layout=1&images=inline&format=jpeg",
            data=_pdf_bytes(2),
            headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200, r.text[:300]
        pages = r.json()["pages"]
        assert len(pages) == 2
        for p in pages:
            assert "image_b64" in p
            assert "layout" in p
            assert p["results"] == []

    def test_text0_geometric_drops_layer_text(self, server_url, gpu_layout):
        """Even when the PDF has a text layer (geometric mode), text=0 must
        honor the no-text contract."""
        r = requests.post(
            f"{server_url}/ocr/pdf?mode=geometric&text=0&images=inline",
            data=_pdf_bytes(1),
            headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200, r.text[:300]
        assert r.json()["pages"][0]["results"] == []

    def test_bare_text0_pdf_rejected(self, server_url, gpu_layout):
        r = requests.post(f"{server_url}/ocr/pdf?text=0",
                          data=_pdf_bytes(1),
                          headers={"Content-Type": "application/pdf"}, timeout=15)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_PARAMETER"
