"""Integration tests for the /ocr/raw endpoint (raw image bytes)."""

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes, pil_to_jpeg_bytes, pil_to_base64


class TestOcrRawEndpoint:
    """Test /ocr/raw endpoint which accepts raw image bytes."""

    def test_png_raw(self, server_url, hello_image):
        """Sending PNG bytes directly should work."""
        png = pil_to_png_bytes(hello_image)
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=png,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_jpeg_raw(self, server_url, hello_image):
        """Sending JPEG bytes directly should work."""
        jpeg = pil_to_jpeg_bytes(hello_image)
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=jpeg,
            headers={"Content-Type": "image/jpeg"},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_raw_detects_same_as_base64(self, server_url, hello_image):
        """/ocr/raw and /ocr should produce the same text for the same image."""
        png = pil_to_png_bytes(hello_image)
        r_raw = requests.post(
            f"{server_url}/ocr/raw",
            data=png,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        b64 = pil_to_base64(hello_image)
        r_b64 = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)

        raw_text = " ".join(i["text"] for i in r_raw.json()["results"])
        b64_text = " ".join(i["text"] for i in r_b64.json()["results"])
        assert raw_text == b64_text, (
            f"Raw and base64 endpoints returned different text: "
            f"raw='{raw_text}', b64='{b64_text}'"
        )

    def test_raw_without_content_type(self, server_url, hello_image):
        """Raw bytes without explicit Content-Type should still decode
        (server uses magic bytes, not headers)."""
        png = pil_to_png_bytes(hello_image)
        r = requests.post(f"{server_url}/ocr/raw", data=png, timeout=10)
        assert r.status_code == 200

    def test_raw_empty_body(self, server_url):
        """Empty body should return 400."""
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=b"",
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_raw_large_image(self, server_url):
        """Large image (3000x2000) via raw should work."""
        img = make_text_image("BIG", width=3000, height=2000, font_size=80)
        png = pil_to_png_bytes(img)
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=png,
            headers={"Content-Type": "image/png"},
            timeout=30,
        )
        assert r.status_code == 200
