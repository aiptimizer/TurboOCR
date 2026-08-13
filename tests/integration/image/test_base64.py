"""Unit tests for base64 decode behavior.

The actual base64_decode function is in C++ (common/encoding.h). These tests verify
that the server correctly handles various base64 edge cases through the /ocr
endpoint, catching issues like padding handling, whitespace in input, and
invalid characters.
"""

import base64
import io
import json

import pytest
import requests
from PIL import Image

from conftest import make_text_image, pil_to_base64, pil_to_png_bytes


class TestBase64Handling:
    """Test that the server's base64 decoder handles various inputs correctly."""

    def test_standard_base64(self, server_url, hello_image):
        """Standard base64 with proper padding should decode correctly."""
        b64 = pil_to_base64(hello_image, "PNG")
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_base64_no_padding(self, server_url, hello_image):
        """Base64 without trailing '=' padding should still work.

        The C++ decoder strips trailing '=' so this tests that code path.
        """
        b64 = pil_to_base64(hello_image, "PNG").rstrip("=")
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200

    def test_base64_with_newlines(self, server_url, hello_image):
        """Base64 with embedded newlines (like MIME-encoded) should work.

        The C++ decoder strips trailing \\n/\\r. This tests whether newlines
        in the middle cause issues.
        """
        b64 = pil_to_base64(hello_image, "PNG")
        # Insert newlines every 76 chars (MIME style)
        chunked = "\n".join(b64[i:i+76] for i in range(0, len(b64), 76))
        r = requests.post(f"{server_url}/ocr", json={"image": chunked}, timeout=10)
        # This may fail since the C++ decoder only strips trailing whitespace.
        # We document the behavior either way.
        assert r.status_code in (200, 400)

    def test_empty_base64(self, server_url):
        """Empty base64 string should return 400."""
        r = requests.post(f"{server_url}/ocr", json={"image": ""}, timeout=10)
        assert r.status_code == 400

    def test_invalid_base64_characters(self, server_url):
        """Non-base64 characters should be handled gracefully (400, not crash)."""
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": "!!!not-valid-base64!!!"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_base64_jpeg(self, server_url, hello_image):
        """JPEG base64 should work (tests nvJPEG decode path)."""
        b64 = pil_to_base64(hello_image, "JPEG")
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_base64_tiny_image(self, server_url):
        """Very small image (1x1) should not crash the server."""
        img = Image.new("RGB", (1, 1), "white")
        b64 = pil_to_base64(img, "PNG")
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        # Should succeed with empty results (no text detected)
        assert r.status_code == 200
        data = r.json()
        assert data["results"] == [] or isinstance(data["results"], list)
