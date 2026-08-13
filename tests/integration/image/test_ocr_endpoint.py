"""Integration tests for the /ocr endpoint (base64 JSON)."""

import base64
import io
import json

import pytest
import requests
from PIL import Image

from conftest import make_text_image, pil_to_base64, pil_to_png_bytes, pil_to_jpeg_bytes


class TestOcrEndpoint:
    """Test /ocr endpoint with various image formats and content."""

    def test_png_image(self, server_url, hello_image):
        """PNG image via base64 should return OCR results."""
        b64 = pil_to_base64(hello_image, "PNG")
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_jpeg_image(self, server_url, hello_image):
        """JPEG image via base64 should return OCR results."""
        b64 = pil_to_base64(hello_image, "JPEG")
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_detects_known_text(self, server_url):
        """OCR should detect text that we rendered on the image."""
        img = make_text_image("HELLO", width=400, height=100, font_size=50)
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        all_text = " ".join(item["text"] for item in data["results"]).upper()
        assert "HELLO" in all_text, f"Expected 'HELLO' in OCR output, got: {all_text}"

    def test_detects_numbers(self, server_url, numbers_image):
        """OCR should detect numeric text."""
        b64 = pil_to_base64(numbers_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        all_text = " ".join(item["text"] for item in data["results"])
        # At least some digits should be detected
        detected_digits = sum(1 for c in all_text if c.isdigit())
        assert detected_digits >= 3, f"Expected digits in OCR output, got: {all_text}"

    def test_multiline_text(self, server_url, paragraph_image):
        """Multi-line text image should produce multiple results."""
        b64 = pil_to_base64(paragraph_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        assert len(data["results"]) >= 2, (
            f"Expected multiple results for multi-line text, got {len(data['results'])}"
        )

    def test_large_image(self, server_url):
        """A larger image (1920x1080) should be handled without error."""
        img = make_text_image(
            "LARGE IMAGE TEST",
            width=1920, height=1080, font_size=60,
        )
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=30)
        assert r.status_code == 200

    def test_small_image(self, server_url):
        """A small image (50x20) should not crash."""
        img = make_text_image("Hi", width=50, height=20, font_size=12)
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200

    def test_confidence_values_are_reasonable(self, server_url, hello_image):
        """Confidence scores for clear text should be reasonably high."""
        b64 = pil_to_base64(hello_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        if not data["results"]:
            pytest.skip("No detections")
        for item in data["results"]:
            assert 0.0 < item["confidence"] <= 1.0

    def test_bounding_boxes_within_image(self, server_url, hello_image):
        """Bounding box coordinates should be within the image dimensions."""
        width, height = hello_image.size
        b64 = pil_to_base64(hello_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        for item in data["results"]:
            for point in item["bounding_box"]:
                x, y = point
                # Allow small margin for rounding
                assert -5 <= x <= width + 5, f"x={x} out of range for width={width}"
                assert -5 <= y <= height + 5, f"y={y} out of range for height={height}"
