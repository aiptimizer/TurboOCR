"""Integration tests for the /ocr/pdf endpoint."""

import io

import pytest
import requests

from conftest import make_text_image

# Try to import reportlab for PDF generation; skip tests if unavailable
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


def _make_simple_pdf(text="Hello PDF", pages=1):
    """Generate a simple PDF with text using reportlab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        c.setFont("Helvetica", 24)
        c.drawString(100, 700, f"{text} - Page {i + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed")
class TestPdfEndpoint:
    """Test /ocr/pdf endpoint for PDF document processing."""

    def test_single_page_pdf(self, server_url):
        """Single page PDF should return one page of results."""
        pdf_bytes = _make_simple_pdf("SINGLE PAGE", pages=1)
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "pages" in data
        assert len(data["pages"]) == 1
        assert data["pages"][0]["page"] == 1

    def test_multi_page_pdf(self, server_url):
        """Multi-page PDF should return results for each page in order."""
        pdf_bytes = _make_simple_pdf("MULTI PAGE", pages=3)
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=pdf_bytes,
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 3
        # Page numbering should be 1-indexed and sequential
        for i, page in enumerate(data["pages"]):
            assert page["page"] == i + 1

    def test_pdf_page_results_have_correct_schema(self, server_url):
        """Each page in the PDF response should have the standard results schema."""
        pdf_bytes = _make_simple_pdf("SCHEMA TEST", pages=1)
        r = requests.post(f"{server_url}/ocr/pdf", data=pdf_bytes, timeout=30)
        data = r.json()
        page = data["pages"][0]
        assert "results" in page
        if page["results"]:
            item = page["results"][0]
            assert "text" in item
            assert "confidence" in item
            assert "bounding_box" in item

    def test_pdf_empty_body(self, server_url):
        """Empty body to /ocr/pdf should return 400."""
        r = requests.post(f"{server_url}/ocr/pdf", data=b"", timeout=10)
        assert r.status_code == 400

    def test_pdf_invalid_data(self, server_url):
        """Non-PDF data to /ocr/pdf should return 400."""
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=b"this is not a pdf file",
            timeout=10,
        )
        assert r.status_code == 400

    def test_pdf_dpi_parameter(self, server_url):
        """DPI query parameter should be accepted."""
        pdf_bytes = _make_simple_pdf("DPI TEST", pages=1)
        r = requests.post(
            f"{server_url}/ocr/pdf?dpi=150",
            data=pdf_bytes,
            timeout=30,
        )
        assert r.status_code == 200

    def test_pdf_invalid_dpi(self, server_url):
        """DPI outside valid range should return 400."""
        pdf_bytes = _make_simple_pdf("BAD DPI", pages=1)
        r = requests.post(
            f"{server_url}/ocr/pdf?dpi=9999",
            data=pdf_bytes,
            timeout=10,
        )
        assert r.status_code == 400

    def test_pdf_page_ordering(self, server_url):
        """Pages should come back in the same order as in the PDF."""
        # Each page has a unique identifier
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        for i in range(5):
            c.setFont("Helvetica", 36)
            c.drawString(100, 700, f"PAGE{i + 1}")
            c.showPage()
        c.save()
        pdf_bytes = buf.getvalue()

        r = requests.post(f"{server_url}/ocr/pdf", data=pdf_bytes, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 5
        for i in range(5):
            assert data["pages"][i]["page"] == i + 1
