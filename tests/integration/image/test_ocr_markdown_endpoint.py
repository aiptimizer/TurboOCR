"""Integration tests for the /ocr/markdown faithful-export endpoint.

/ocr/markdown is GPU-only and requires the layout model. It had no integration
coverage. It returns a text/markdown body (not JSON), embeds assets as
self-contained data: URIs, and signals a degraded stage via the X-OCR-Degraded
header rather than silently dropping regions.
"""
import io

import pytest
import requests
from PIL import Image, ImageDraw

from conftest import make_text_image, pil_to_png_bytes, pil_to_jpeg_bytes


@pytest.fixture(scope="module")
def has_markdown(server_url):
    caps = requests.get(f"{server_url}/capabilities", timeout=5)
    if caps.status_code != 200 or "/ocr/markdown" not in caps.text:
        pytest.skip("/ocr/markdown not registered (GPU + layout required)")
    return True


def _post(server_url, data, ctype="image/png"):
    return requests.post(f"{server_url}/ocr/markdown", data=data,
                         headers={"Content-Type": ctype}, timeout=60)


class TestMarkdownEndpoint:
    def test_png_returns_markdown(self, server_url, has_markdown):
        img = make_text_image("MARKDOWN EXPORT 2026", width=500, height=120, font_size=32)
        r = _post(server_url, pil_to_png_bytes(img))
        if r.status_code == 400 and "LAYOUT_DISABLED" in r.text:
            pytest.skip("server started with DISABLE_LAYOUT=1")
        assert r.status_code == 200, r.text[:300]
        assert "markdown" in r.headers.get("Content-Type", "").lower()
        body = r.text.lower()
        assert any(tok in body for tok in ("markdown", "export", "2026")), \
            f"no expected token in markdown body: {r.text[:200]}"

    def test_jpeg_returns_markdown(self, server_url, has_markdown):
        img = make_text_image("JPEG MARKDOWN", width=500, height=120, font_size=32)
        r = _post(server_url, pil_to_jpeg_bytes(img), ctype="image/jpeg")
        if r.status_code == 400 and "LAYOUT_DISABLED" in r.text:
            pytest.skip("server started with DISABLE_LAYOUT=1")
        assert r.status_code == 200, r.text[:300]
        assert "markdown" in r.headers.get("Content-Type", "").lower()

    def test_empty_body_rejected(self, server_url, has_markdown):
        r = _post(server_url, b"")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "EMPTY_BODY"

    def test_garbage_body_rejected(self, server_url, has_markdown):
        r = _post(server_url, b"\x00\x01\x02not-an-image\xff\xd8")
        # Either decode fails (400) or, if layout is off, layout guard fires first.
        assert r.status_code == 400
        assert r.json()["error"]["code"] in ("IMAGE_DECODE_FAILED", "LAYOUT_DISABLED")

    def test_data_uri_is_self_contained(self, server_url, has_markdown):
        """A page with a figure should embed assets as data: URIs, never as
        file references the client can't fetch."""
        img = Image.new("RGB", (400, 300), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([40, 40, 360, 160], fill=(200, 200, 200))  # a "figure" block
        d.text((60, 200), "CAPTION BELOW FIGURE", fill="black")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        r = _post(server_url, buf.getvalue())
        if r.status_code == 400 and "LAYOUT_DISABLED" in r.text:
            pytest.skip("server started with DISABLE_LAYOUT=1")
        assert r.status_code == 200, r.text[:300]
        # If any image is embedded it must be a data: URI (self-contained).
        if "![" in r.text and "](" in r.text:
            assert "](data:" in r.text or "](http" not in r.text, \
                "markdown references a non-embedded asset the client cannot fetch"
