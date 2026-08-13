"""Shared fixtures for HTTP integration tests.

Note: --server-url CLI option and the server_url fixture are defined in the
root tests/conftest.py and inherited automatically by pytest.  This file
only adds integration-specific fixtures.
"""

import base64
import io

import pytest
from PIL import Image, ImageDraw

# ONE font resolver for the whole tree, in the root conftest. This file used to
# carry a byte-identical copy of `_FONT_PATHS` + `_get_font`, comment included —
# so the macOS bitmap-fallback trap documented there could be fixed in one
# conftest and still bite every test loaded through the other.
from conftest import _get_font


# ---------------------------------------------------------------------------
# Test image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_png_bytes():
    """A small 100x50 white PNG with 'Hello OCR' text, returned as raw bytes."""
    img = Image.new("RGB", (100, 50), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    draw.text((5, 10), "Hello OCR", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def test_png_base64(test_png_bytes):
    """Base64-encoded version of the test PNG image."""
    return base64.b64encode(test_png_bytes).decode("ascii")


@pytest.fixture(scope="session")
def test_image_pil():
    """PIL Image object of the test image (100x50, 'Hello OCR')."""
    img = Image.new("RGB", (100, 50), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    draw.text((5, 10), "Hello OCR", fill="black", font=font)
    return img


# ---------------------------------------------------------------------------
# Test PDF fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_pdf_bytes():
    """A small single-page PDF with 'Hello PDF' text, returned as bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica", 24)
    c.drawString(100, 700, "Hello PDF")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture(scope="session")
def test_pdf_base64(test_pdf_bytes):
    """Base64-encoded version of the test PDF."""
    return base64.b64encode(test_pdf_bytes).decode("ascii")


@pytest.fixture(scope="session")
def test_multipage_pdf_bytes():
    """A 3-page PDF for multi-page tests."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(3):
        c.setFont("Helvetica", 24)
        c.drawString(100, 700, f"Page {i + 1} content")
        c.showPage()
    c.save()
    return buf.getvalue()
