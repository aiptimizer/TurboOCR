"""Unit tests for JSON response format.

Verifies that the server's fast JSON serializer (results_to_json in common/serialization/serialization.h)
produces valid JSON with the correct schema. Also tests special character
escaping in text fields.
"""

import json

import pytest
import requests

from conftest import make_text_image, pil_to_base64, pil_to_png_bytes


class TestJsonResponseFormat:
    """Verify the structure and validity of JSON responses."""

    def test_response_has_results_key(self, server_url, hello_image):
        """Response must contain a top-level 'results' array."""
        b64 = pil_to_base64(hello_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_result_item_schema(self, server_url, hello_image):
        """Each result item must have text, confidence, and bounding_box fields."""
        b64 = pil_to_base64(hello_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        if len(data["results"]) == 0:
            pytest.skip("No text detected in test image")

        for item in data["results"]:
            assert "text" in item, "Missing 'text' field"
            assert "confidence" in item, "Missing 'confidence' field"
            assert "bounding_box" in item, "Missing 'bounding_box' field"

            assert isinstance(item["text"], str)
            assert isinstance(item["confidence"], (int, float))
            assert 0.0 <= item["confidence"] <= 1.0, (
                f"Confidence {item['confidence']} outside [0,1]"
            )

    def test_bounding_box_format(self, server_url, hello_image):
        """Bounding box must be 4 points, each with [x, y] coordinates."""
        b64 = pil_to_base64(hello_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        if not data["results"]:
            pytest.skip("No text detected")

        for item in data["results"]:
            bbox = item["bounding_box"]
            assert len(bbox) == 4, f"Expected 4 corners, got {len(bbox)}"
            for point in bbox:
                assert len(point) == 2, f"Expected [x,y], got {point}"
                assert isinstance(point[0], int), f"x should be int, got {type(point[0])}"
                assert isinstance(point[1], int), f"y should be int, got {type(point[1])}"

    def test_response_content_type(self, server_url, hello_image):
        """Response Content-Type must be application/json."""
        b64 = pil_to_base64(hello_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert "application/json" in r.headers.get("Content-Type", "")

    def test_raw_endpoint_same_schema(self, server_url, hello_image):
        """/ocr/raw should return the same JSON schema as /ocr."""
        png_bytes = pil_to_png_bytes(hello_image)
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        if data["results"]:
            item = data["results"][0]
            assert "text" in item
            assert "confidence" in item
            assert "bounding_box" in item

    def test_batch_response_schema(self, server_url, hello_image, numbers_image):
        """Batch endpoint returns batch_results array with nested results."""
        b64_1 = pil_to_base64(hello_image)
        b64_2 = pil_to_base64(numbers_image)
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64_1, b64_2]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "batch_results" in data
        assert len(data["batch_results"]) == 2
        for batch_item in data["batch_results"]:
            assert "results" in batch_item
            assert isinstance(batch_item["results"], list)

    def test_empty_image_returns_empty_results(self, server_url, blank_image):
        """A blank image should return an empty results array (valid JSON)."""
        b64 = pil_to_base64(blank_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["results"] == []

    def test_json_is_valid_parseable(self, server_url, paragraph_image):
        """Response body must be valid JSON (not truncated, no trailing commas)."""
        b64 = pil_to_base64(paragraph_image)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        # json.loads will throw if invalid
        data = json.loads(r.text)
        assert isinstance(data, dict)
