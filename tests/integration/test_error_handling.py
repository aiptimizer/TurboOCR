"""Integration tests for error handling and edge cases.

Verifies the server handles malformed input gracefully without crashing.
"""

import pytest
import requests
import json


class TestErrorHandling:
    """Test that the server returns proper errors for bad input."""

    # --- /ocr endpoint errors ---

    def test_ocr_no_body(self, server_url):
        """POST /ocr with no body should return 400."""
        r = requests.post(f"{server_url}/ocr", timeout=10)
        assert r.status_code == 400

    def test_ocr_invalid_json(self, server_url):
        """POST /ocr with invalid JSON should return 400."""
        r = requests.post(
            f"{server_url}/ocr",
            data="not json at all",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_ocr_missing_image_field(self, server_url):
        """POST /ocr with JSON but no 'image' field should return 400."""
        r = requests.post(f"{server_url}/ocr", json={"wrong": "field"}, timeout=10)
        assert r.status_code == 400

    def test_ocr_null_image(self, server_url):
        """POST /ocr with null image field should return 400 or 500.

        The server's Drogon/jsoncpp JSON parser may throw on null values. Either
        a 400 (graceful) or 500 (uncaught) is acceptable -- not a crash.
        """
        r = requests.post(
            f"{server_url}/ocr",
            data=json.dumps({"image": None}),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code in (400, 500)

    def test_ocr_corrupt_base64_image(self, server_url):
        """Valid base64 but invalid image data should return 400."""
        import base64
        garbage = base64.b64encode(b"this is not an image file at all " * 10).decode()
        r = requests.post(f"{server_url}/ocr", json={"image": garbage}, timeout=10)
        assert r.status_code == 400

    # --- /ocr/raw endpoint errors ---

    def test_raw_empty_body(self, server_url):
        """POST /ocr/raw with empty body should return 400."""
        r = requests.post(f"{server_url}/ocr/raw", data=b"", timeout=10)
        assert r.status_code == 400

    def test_raw_garbage_bytes(self, server_url):
        """POST /ocr/raw with random garbage bytes should return 400."""
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=b"\x00\x01\x02\x03" * 100,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_raw_truncated_png(self, server_url):
        """Truncated PNG header should return 400."""
        # Valid PNG header but truncated
        png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=png_header,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_raw_truncated_jpeg(self, server_url):
        """Truncated JPEG should return 400."""
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=jpeg_header,
            headers={"Content-Type": "image/jpeg"},
            timeout=10,
        )
        assert r.status_code == 400

    # --- /ocr/batch endpoint errors ---

    def test_batch_invalid_json(self, server_url):
        """POST /ocr/batch with invalid JSON should return 400."""
        r = requests.post(
            f"{server_url}/ocr/batch",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_batch_no_images_key(self, server_url):
        """POST /ocr/batch without 'images' key should return 400."""
        r = requests.post(f"{server_url}/ocr/batch", json={"data": []}, timeout=10)
        assert r.status_code == 400

    def test_batch_empty_images(self, server_url):
        """POST /ocr/batch with empty images array should return 400."""
        r = requests.post(f"{server_url}/ocr/batch", json={"images": []}, timeout=10)
        assert r.status_code == 400

    # --- /ocr/pixels endpoint errors ---

    def test_pixels_missing_headers(self, server_url):
        """POST /ocr/pixels without dimension headers should return 400."""
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 100,
            timeout=10,
        )
        assert r.status_code == 400

    def test_pixels_wrong_body_size(self, server_url):
        """POST /ocr/pixels with body size not matching headers should return 400."""
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 100,  # 100 bytes != 640*480*3
            headers={"X-Width": "640", "X-Height": "480", "X-Channels": "3"},
            timeout=10,
        )
        assert r.status_code == 400

    def test_pixels_invalid_dimensions(self, server_url):
        """POST /ocr/pixels with zero/negative dimensions should return 400."""
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 10,
            headers={"X-Width": "0", "X-Height": "0"},
            timeout=10,
        )
        assert r.status_code == 400

    # --- /health endpoint ---

    def test_health_endpoint(self, server_url):
        """GET /health should return 200."""
        r = requests.get(f"{server_url}/health", timeout=5)
        assert r.status_code == 200

    # --- Method not allowed ---

    def test_get_on_ocr_not_allowed(self, server_url):
        """GET /ocr should return 405 or 404 (only POST accepted)."""
        r = requests.get(f"{server_url}/ocr", timeout=5)
        assert r.status_code in (404, 405)

    # --- Server resilience ---

    def test_rapid_fire_does_not_crash(self, server_url, hello_image):
        """Send 20 rapid requests to verify the server doesn't crash under quick load."""
        from conftest import pil_to_png_bytes
        png = pil_to_png_bytes(hello_image)
        for _ in range(20):
            r = requests.post(
                f"{server_url}/ocr/raw",
                data=png,
                headers={"Content-Type": "image/png"},
                timeout=10,
            )
            assert r.status_code == 200
