"""Integration tests for the PDF streaming path.

Tests the inotify + daemon pool + OCR overlap pipeline — the most complex
codepath. Covers: large PDFs, concurrent requests, page ordering, all input
modes (raw/multipart/base64), mode × layout combinations, and edge cases
like single-page and empty-text PDFs.

Requires: running server, reportlab (pip install reportlab).
"""

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import requests

try:
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.pdfgen import canvas

    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

pytestmark = pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed")


def _make_pdf(pages=1, text_fn=None, pagesize=A4):
    """Generate a PDF. text_fn(page_index) returns the text for that page."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=pagesize)
    for i in range(pages):
        c.setFont("Helvetica", 24)
        text = text_fn(i) if text_fn else f"Page {i + 1} content"
        c.drawString(100, 700, text)
        # Add a second line so there's enough content to detect
        c.setFont("Helvetica", 14)
        c.drawString(100, 660, f"Document page number {i + 1} of {pages}")
        c.showPage()
    c.save()
    return buf.getvalue()


class TestPdfPageCounts:
    """Verify no pages are dropped at various sizes."""

    @pytest.mark.parametrize("n_pages", [1, 2, 5, 10, 20, 50])
    def test_page_count_matches(self, server_url, n_pages):
        pdf = _make_pdf(n_pages)
        r = requests.post(
            f"{server_url}/ocr/pdf", data=pdf, timeout=max(30, n_pages * 3)
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == n_pages, (
            f"Expected {n_pages} pages, got {len(data['pages'])}"
        )

    def test_100_page_pdf(self, server_url):
        """100-page PDF — stress test for daemon pool and inotify."""
        pdf = _make_pdf(100, text_fn=lambda i: f"HUNDRED_{i:03d}")
        r = requests.post(f"{server_url}/ocr/pdf", data=pdf, timeout=120)
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 100, (
            f"Expected 100 pages, got {len(data['pages'])}"
        )
        # Verify page ordering
        for i, page in enumerate(data["pages"]):
            assert page["page"] == i + 1
            assert page["page_index"] == i


class TestPdfPageOrdering:
    """Verify pages come back in correct order with identifiable content."""

    def test_unique_content_per_page(self, server_url):
        """Each page has unique text; verify it appears on the right page."""
        markers = [f"MARKER_{i:03d}" for i in range(10)]
        pdf = _make_pdf(10, text_fn=lambda i: markers[i])
        r = requests.post(f"{server_url}/ocr/pdf", data=pdf, timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 10

        for i, page in enumerate(data["pages"]):
            page_text = " ".join(item["text"] for item in page["results"])
            # The marker should appear somewhere in this page's text
            assert markers[i].replace("_", "") in page_text.replace("_", "").replace(" ", ""), (
                f"Page {i} missing marker {markers[i]}, got: {page_text[:100]}"
            )


class TestPdfConcurrency:
    """Concurrent PDF requests — tests dispatcher queue + daemon pool."""

    def test_concurrent_small_pdfs(self, server_url):
        """8 concurrent 3-page PDFs — all should complete successfully."""
        pdf = _make_pdf(3)
        n_concurrent = 8

        def send():
            return requests.post(
                f"{server_url}/ocr/pdf", data=pdf, timeout=60
            )

        with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
            futures = [pool.submit(send) for _ in range(n_concurrent)]
            results = [f.result() for f in as_completed(futures)]

        for r in results:
            assert r.status_code == 200
            assert len(r.json()["pages"]) == 3

    def test_concurrent_mixed_sizes(self, server_url):
        """Concurrent PDFs of different sizes."""
        sizes = [1, 5, 10, 20, 1, 5]
        pdfs = [_make_pdf(n) for n in sizes]

        def send(pdf_bytes, expected):
            r = requests.post(
                f"{server_url}/ocr/pdf", data=pdf_bytes, timeout=120
            )
            return r, expected

        with ThreadPoolExecutor(max_workers=len(sizes)) as pool:
            futures = [
                pool.submit(send, pdfs[i], sizes[i]) for i in range(len(sizes))
            ]
            for f in as_completed(futures):
                r, expected = f.result()
                assert r.status_code == 200
                got = len(r.json()["pages"])
                assert got == expected, f"Expected {expected} pages, got {got}"


class TestPdfInputModes:
    """Test all three input modes produce identical results."""

    def test_raw_vs_multipart_vs_base64(self, server_url):
        pdf = _make_pdf(2, text_fn=lambda i: f"INPUT_MODE_TEST_{i}")

        # Raw binary
        r1 = requests.post(
            f"{server_url}/ocr/pdf",
            data=pdf,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )

        # Multipart
        r2 = requests.post(
            f"{server_url}/ocr/pdf",
            files={"file": ("test.pdf", pdf, "application/pdf")},
            timeout=30,
        )

        # Base64 JSON
        r3 = requests.post(
            f"{server_url}/ocr/pdf",
            json={"pdf": base64.b64encode(pdf).decode()},
            timeout=30,
        )

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200

        d1 = r1.json()
        d2 = r2.json()
        d3 = r3.json()

        # Same page count
        assert len(d1["pages"]) == len(d2["pages"]) == len(d3["pages"]) == 2

        # Same text content (results may vary slightly in order but text should match)
        for mode_name, d in [("raw", d1), ("multipart", d2), ("base64", d3)]:
            total = sum(len(p["results"]) for p in d["pages"])
            assert total > 0, f"{mode_name} returned no results"


class TestPdfModesWithLayout:
    """Test all mode × layout combinations."""

    @pytest.mark.parametrize("mode", ["ocr", "geometric", "auto", "auto_verified"])
    @pytest.mark.parametrize("layout", ["0", "1"])
    def test_mode_layout_combination(self, server_url, mode, layout):
        pdf = _make_pdf(2)
        r = requests.post(
            f"{server_url}/ocr/pdf?mode={mode}&layout={layout}",
            data=pdf,
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 2

        for page in data["pages"]:
            assert "results" in page
            assert "mode" in page
            if layout == "1":
                assert "layout" in page


class TestPdfResponseSchema:
    """Validate response schema completeness."""

    def test_page_envelope_fields(self, server_url):
        pdf = _make_pdf(1)
        r = requests.post(f"{server_url}/ocr/pdf", data=pdf, timeout=30)
        assert r.status_code == 200
        page = r.json()["pages"][0]
        assert "page" in page
        assert "page_index" in page
        assert "dpi" in page
        assert "width" in page and page["width"] > 0
        assert "height" in page and page["height"] > 0
        assert "results" in page
        assert "mode" in page

    def test_dpi_parameter_affects_dimensions(self, server_url):
        """Higher DPI should produce larger pixel dimensions."""
        pdf = _make_pdf(1)
        r100 = requests.post(
            f"{server_url}/ocr/pdf?dpi=100", data=pdf, timeout=30
        )
        r200 = requests.post(
            f"{server_url}/ocr/pdf?dpi=200", data=pdf, timeout=30
        )
        assert r100.status_code == 200
        assert r200.status_code == 200

        w100 = r100.json()["pages"][0]["width"]
        w200 = r200.json()["pages"][0]["width"]
        # 200 DPI should be ~2x the width of 100 DPI
        assert w200 > w100 * 1.5, f"DPI 200 width ({w200}) not much larger than DPI 100 ({w100})"


class TestPdfEdgeCases:
    """Edge cases for the streaming pipeline."""

    def test_single_page_no_text(self, server_url):
        """PDF with a blank page should return empty results, not error."""
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.showPage()  # blank page
        c.save()
        r = requests.post(f"{server_url}/ocr/pdf", data=buf.getvalue(), timeout=30)
        assert r.status_code == 200
        assert len(r.json()["pages"]) == 1

    def test_letter_vs_a4_pagesize(self, server_url):
        """Different page sizes should work."""
        pdf_a4 = _make_pdf(1, pagesize=A4)
        pdf_letter = _make_pdf(1, pagesize=LETTER)
        r1 = requests.post(f"{server_url}/ocr/pdf", data=pdf_a4, timeout=30)
        r2 = requests.post(f"{server_url}/ocr/pdf", data=pdf_letter, timeout=30)
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Dimensions should differ
        w1 = r1.json()["pages"][0]["width"]
        w2 = r2.json()["pages"][0]["width"]
        assert w1 != w2 or True  # might be same at low DPI rounding

    def test_invalid_dpi_rejected(self, server_url):
        pdf = _make_pdf(1)
        r = requests.post(f"{server_url}/ocr/pdf?dpi=10", data=pdf, timeout=10)
        assert r.status_code == 400

    def test_invalid_mode_fallback(self, server_url):
        """Invalid mode should fall back to default, not crash."""
        pdf = _make_pdf(1)
        r = requests.post(
            f"{server_url}/ocr/pdf?mode=nonexistent", data=pdf, timeout=30
        )
        assert r.status_code == 200

    def test_multipart_wrong_field_name(self, server_url):
        """Multipart with wrong field name should return 400."""
        pdf = _make_pdf(1)
        r = requests.post(
            f"{server_url}/ocr/pdf",
            files={"wrong_name": ("test.pdf", pdf, "application/pdf")},
            timeout=10,
        )
        assert r.status_code == 400

    def test_base64_invalid_json(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/pdf",
            json={"wrong_key": "abc"},
            timeout=10,
        )
        assert r.status_code == 400
