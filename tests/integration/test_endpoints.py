"""Comprehensive HTTP integration tests for all OCR server endpoints.

Covers happy paths, error paths, structured error format, response headers,
layout query parameter, batch processing, PDF modes, and nginx 413 errors.

Run with:
    pytest tests/integration/test_endpoints.py -v
"""

import base64
import io
import json
import re
import uuid

import numpy as np
import pytest
import requests
from PIL import Image


# ===========================================================================
# Helpers
# ===========================================================================

UUID_V7_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _assert_success_headers(resp):
    """Verify X-Request-Id (UUID v7) and X-Inference-Time-Ms on success."""
    rid = resp.headers.get("X-Request-Id")
    assert rid is not None, "Missing X-Request-Id header"
    assert UUID_V7_RE.match(rid), f"X-Request-Id is not UUID v7: {rid}"

    inf_time = resp.headers.get("X-Inference-Time-Ms")
    assert inf_time is not None, "Missing X-Inference-Time-Ms header"
    assert float(inf_time) >= 0, f"X-Inference-Time-Ms should be >= 0, got {inf_time}"


def _assert_structured_error(resp):
    """Verify the response body has structured JSON error format."""
    data = resp.json()
    assert "error" in data, f"Expected 'error' key in response, got: {list(data.keys())}"
    err = data["error"]
    assert "code" in err, f"Expected 'error.code', got keys: {list(err.keys())}"
    assert "message" in err, f"Expected 'error.message', got keys: {list(err.keys())}"
    assert isinstance(err["code"], str) and len(err["code"]) > 0
    assert isinstance(err["message"], str) and len(err["message"]) > 0


def _pil_to_bgr_bytes(pil_img):
    """Convert PIL Image to raw BGR bytes for /ocr/pixels."""
    arr = np.asarray(pil_img.convert("RGB"))
    bgr = arr[:, :, ::-1].copy()
    return bgr.tobytes(), arr.shape[1], arr.shape[0], 3


# ===========================================================================
# 1. Health endpoints
# ===========================================================================


class TestHealthEndpoints:
    """Verify /health, /health/live, and /health/ready."""

    def test_health_returns_ok(self, server_url):
        r = requests.get(f"{server_url}/health", timeout=5)
        assert r.status_code == 200
        assert r.text.strip().strip('"') == "ok"

    def test_health_live_returns_ok(self, server_url):
        r = requests.get(f"{server_url}/health/live", timeout=5)
        assert r.status_code == 200
        assert r.text.strip().strip('"') == "ok"

    def test_health_ready_returns_ok(self, server_url):
        r = requests.get(f"{server_url}/health/ready", timeout=5)
        assert r.status_code == 200
        assert r.text.strip().strip('"') == "ok"


# ===========================================================================
# 2. POST /ocr  (base64 JSON)
# ===========================================================================


class TestOcrEndpoint:
    """Test /ocr endpoint happy and error paths."""

    # --- Happy path ---

    def test_ocr_happy_path(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": test_png_base64},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_ocr_response_headers(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": test_png_base64},
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_ocr_result_schema(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": test_png_base64},
            timeout=15,
        )
        data = r.json()
        if data["results"]:
            item = data["results"][0]
            assert "text" in item
            assert "confidence" in item
            assert "bounding_box" in item
            assert isinstance(item["bounding_box"], list)
            assert len(item["bounding_box"]) == 4
            assert 0.0 < item["confidence"] <= 1.0

    # --- Error paths ---

    def test_ocr_empty_body(self, server_url):
        r = requests.post(f"{server_url}/ocr", timeout=10)
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_ocr_invalid_json(self, server_url):
        r = requests.post(
            f"{server_url}/ocr",
            data="not valid json {{{",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_ocr_missing_image_field(self, server_url):
        r = requests.post(
            f"{server_url}/ocr",
            json={"wrong_field": "data"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_ocr_bad_base64(self, server_url):
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": "!!!not-base64!!!"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_ocr_valid_base64_invalid_image(self, server_url):
        garbage = base64.b64encode(b"this is not an image at all" * 5).decode()
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": garbage},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)


# ===========================================================================
# 3. POST /ocr/raw  (raw image bytes)
# ===========================================================================


class TestOcrRawEndpoint:
    """Test /ocr/raw endpoint happy and error paths."""

    def test_raw_happy_path_png(self, server_url, test_png_bytes):
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=test_png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_raw_response_headers(self, server_url, test_png_bytes):
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=test_png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_raw_empty_body(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=b"",
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_raw_garbage_bytes(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=b"\x00\x01\x02\x03" * 100,
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)


# ===========================================================================
# 4. POST /ocr/batch  (JSON array of base64 images)
# ===========================================================================


class TestOcrBatchEndpoint:
    """Test /ocr/batch endpoint happy and error paths."""

    def test_batch_happy_path(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [test_png_base64, test_png_base64]},
            timeout=20,
        )
        assert r.status_code == 200
        data = r.json()
        assert "batch_results" in data
        assert len(data["batch_results"]) == 2

    def test_batch_response_headers(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_batch_single_image(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 1
        assert "results" in data["batch_results"][0]

    def test_batch_result_schema(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        data = r.json()
        entry = data["batch_results"][0]
        assert "results" in entry
        if entry["results"]:
            item = entry["results"][0]
            assert "text" in item
            assert "confidence" in item
            assert "bounding_box" in item

    def test_batch_empty_images(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": []},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_batch_missing_images_key(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"data": []},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_batch_invalid_json(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/batch",
            data="not json",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    # --- Batch with layout ---

    def test_batch_with_layout_0(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch?layout=0",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        entry = data["batch_results"][0]
        assert "layout" not in entry or not entry.get("layout")

    def test_batch_with_layout_1(self, server_url, test_png_base64):
        """layout=1 should add layout array to batch results (if server has layout enabled)."""
        r = requests.post(
            f"{server_url}/ocr/batch?layout=1",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        # If layout is not enabled on the server, 400 is expected
        if r.status_code == 400 and "ENABLE_LAYOUT" in r.text:
            pytest.skip("Server does not have ENABLE_LAYOUT=1")
        assert r.status_code == 200
        data = r.json()
        entry = data["batch_results"][0]
        assert "layout" in entry, "Expected 'layout' key when layout=1"


# ===========================================================================
# 5. POST /ocr/pixels  (raw BGR bytes with dimension headers)
# ===========================================================================


class TestOcrPixelsEndpoint:
    """Test /ocr/pixels endpoint happy and error paths."""

    def test_pixels_happy_path(self, server_url, test_image_pil):
        raw, w, h, ch = _pil_to_bgr_bytes(test_image_pil)
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=raw,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
                "Content-Type": "application/octet-stream",
            },
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data

    def test_pixels_response_headers(self, server_url, test_image_pil):
        raw, w, h, ch = _pil_to_bgr_bytes(test_image_pil)
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=raw,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
            },
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_pixels_missing_headers(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 300,
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_pixels_wrong_body_size(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 100,
            headers={"X-Width": "640", "X-Height": "480", "X-Channels": "3"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_pixels_zero_dimensions(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 10,
            headers={"X-Width": "0", "X-Height": "0", "X-Channels": "3"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_pixels_oversized_dimensions(self, server_url):
        """Extremely large declared dimensions should be rejected (not OOM)."""
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 100,
            headers={"X-Width": "100000", "X-Height": "100000", "X-Channels": "3"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)


# ===========================================================================
# 6. POST /ocr/pdf  (PDF bytes)
# ===========================================================================


class TestOcrPdfEndpoint:
    """Test /ocr/pdf endpoint with various submission modes and PDF modes."""

    # --- Happy paths ---

    def test_pdf_raw_bytes(self, server_url, test_pdf_bytes):
        """Send raw PDF bytes with Content-Type: application/pdf."""
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "pages" in data
        assert len(data["pages"]) >= 1
        assert data["pages"][0]["page"] == 1

    def test_pdf_response_headers(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_pdf_multipart_upload(self, server_url, test_pdf_bytes):
        """Send PDF via multipart form upload."""
        r = requests.post(
            f"{server_url}/ocr/pdf",
            files={"file": ("test.pdf", test_pdf_bytes, "application/pdf")},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "pages" in data

    def test_pdf_base64_json(self, server_url, test_pdf_base64):
        """Send PDF as base64-encoded JSON body."""
        r = requests.post(
            f"{server_url}/ocr/pdf",
            json={"pdf": test_pdf_base64},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert "pages" in data

    def test_pdf_multipage(self, server_url, test_multipage_pdf_bytes):
        """Multi-page PDF should return results for each page in order."""
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=test_multipage_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == 3
        for i, page in enumerate(data["pages"]):
            assert page["page"] == i + 1

    def test_pdf_page_result_schema(self, server_url, test_pdf_bytes):
        """Each page should have the standard results schema."""
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        data = r.json()
        page = data["pages"][0]
        assert "results" in page
        if page["results"]:
            item = page["results"][0]
            assert "text" in item
            assert "confidence" in item
            assert "bounding_box" in item

    # --- PDF modes ---

    def test_pdf_mode_ocr(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf?mode=ocr",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        assert "pages" in r.json()

    def test_pdf_mode_geometric(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf?mode=geometric",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        assert "pages" in r.json()

    def test_pdf_mode_auto(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf?mode=auto",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        assert "pages" in r.json()

    def test_pdf_mode_auto_verified(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf?mode=auto_verified",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        assert "pages" in r.json()

    # --- Error paths ---

    def test_pdf_empty_body(self, server_url):
        r = requests.post(f"{server_url}/ocr/pdf", data=b"", timeout=10)
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_pdf_invalid_data(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=b"this is not a pdf file at all",
            headers={"Content-Type": "application/pdf"},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)


# ===========================================================================
# 7. Layout query parameter (?layout=0|1)
# ===========================================================================


def _skip_if_no_layout(resp):
    """Skip the test if server does not have ENABLE_LAYOUT=1."""
    if resp.status_code == 400 and "ENABLE_LAYOUT" in resp.text:
        pytest.skip("Server does not have ENABLE_LAYOUT=1")


def _assert_layout_region_schema(region):
    """Verify a single layout region has the expected fields and types."""
    assert "id" in region, f"Layout region missing 'id': {region}"
    assert isinstance(region["id"], int)
    assert "class" in region, f"Layout region missing 'class': {region}"
    assert isinstance(region["class"], str) and len(region["class"]) > 0
    assert "class_id" in region, f"Layout region missing 'class_id': {region}"
    assert isinstance(region["class_id"], int)
    assert "confidence" in region, f"Layout region missing 'confidence': {region}"
    assert isinstance(region["confidence"], (int, float))
    assert 0.0 < region["confidence"] <= 1.0
    assert "bounding_box" in region, f"Layout region missing 'bounding_box': {region}"
    assert isinstance(region["bounding_box"], list)
    assert len(region["bounding_box"]) == 4
    for point in region["bounding_box"]:
        assert isinstance(point, list) and len(point) == 2


class TestLayoutQueryParam:
    """Verify ?layout=1 adds layout array, ?layout=0 or absent does not."""

    # --- /ocr endpoint ---

    def test_ocr_no_layout_param(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": test_png_base64},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "layout" not in data or not data.get("layout")

    def test_ocr_layout_0(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr?layout=0",
            json={"image": test_png_base64},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "layout" not in data or not data.get("layout")

    def test_ocr_layout_1(self, server_url, test_png_base64):
        """layout=1 should return layout array (if server supports it)."""
        r = requests.post(
            f"{server_url}/ocr?layout=1",
            json={"image": test_png_base64},
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        assert "layout" in data, "Expected 'layout' key when layout=1"
        assert isinstance(data["layout"], list)

    # --- /ocr/raw endpoint ---

    def test_raw_no_layout_param(self, server_url, test_png_bytes):
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=test_png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "layout" not in data or not data.get("layout")

    def test_raw_layout_0(self, server_url, test_png_bytes):
        r = requests.post(
            f"{server_url}/ocr/raw?layout=0",
            data=test_png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "layout" not in data or not data.get("layout")

    def test_raw_layout_1(self, server_url, test_png_bytes):
        r = requests.post(
            f"{server_url}/ocr/raw?layout=1",
            data=test_png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        assert "layout" in data

    # --- /ocr/pixels endpoint ---

    def test_pixels_no_layout_param(self, server_url, test_image_pil):
        raw, w, h, ch = _pil_to_bgr_bytes(test_image_pil)
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=raw,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
            },
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "layout" not in data or not data.get("layout")

    def test_pixels_layout_0(self, server_url, test_image_pil):
        raw, w, h, ch = _pil_to_bgr_bytes(test_image_pil)
        r = requests.post(
            f"{server_url}/ocr/pixels?layout=0",
            data=raw,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
            },
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert "layout" not in data or not data.get("layout")

    def test_pixels_layout_1(self, server_url, test_image_pil):
        raw, w, h, ch = _pil_to_bgr_bytes(test_image_pil)
        r = requests.post(
            f"{server_url}/ocr/pixels?layout=1",
            data=raw,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
            },
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        assert "layout" in data

    # --- /ocr/batch endpoint ---

    def test_batch_no_layout_param(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        for entry in data["batch_results"]:
            assert "layout" not in entry or not entry.get("layout")

    def test_batch_layout_0(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch?layout=0",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        for entry in data["batch_results"]:
            assert "layout" not in entry or not entry.get("layout")

    def test_batch_layout_1(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch?layout=1",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        for entry in data["batch_results"]:
            assert "layout" in entry, "Expected 'layout' in each batch result"

    # --- /ocr/pdf endpoint ---

    def test_pdf_no_layout_param(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        for page in data["pages"]:
            assert "layout" not in page or not page.get("layout")

    def test_pdf_layout_0(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf?layout=0",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        for page in data["pages"]:
            assert "layout" not in page or not page.get("layout")

    def test_pdf_layout_1(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf?layout=1",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        for page in data["pages"]:
            assert "layout" in page


class TestLayoutRegionSchema:
    """Verify the schema of individual layout regions when layout=1."""

    def test_ocr_layout_region_schema(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr?layout=1",
            json={"image": test_png_base64},
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        if data.get("layout"):
            for region in data["layout"]:
                _assert_layout_region_schema(region)

    def test_raw_layout_region_schema(self, server_url, test_png_bytes):
        r = requests.post(
            f"{server_url}/ocr/raw?layout=1",
            data=test_png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        if data.get("layout"):
            for region in data["layout"]:
                _assert_layout_region_schema(region)

    def test_pixels_layout_region_schema(self, server_url, test_image_pil):
        raw, w, h, ch = _pil_to_bgr_bytes(test_image_pil)
        r = requests.post(
            f"{server_url}/ocr/pixels?layout=1",
            data=raw,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
            },
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        if data.get("layout"):
            for region in data["layout"]:
                _assert_layout_region_schema(region)

    def test_batch_layout_region_schema(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch?layout=1",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        for entry in data["batch_results"]:
            if entry.get("layout"):
                for region in entry["layout"]:
                    _assert_layout_region_schema(region)

    def test_pdf_layout_region_schema(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf?layout=1",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        _skip_if_no_layout(r)
        assert r.status_code == 200
        data = r.json()
        for page in data["pages"]:
            if page.get("layout"):
                for region in page["layout"]:
                    _assert_layout_region_schema(region)


# ===========================================================================
# 8. Structured JSON error format
# ===========================================================================


class TestStructuredErrors:
    """Verify all error responses use the structured JSON format:
    {"error": {"code": "ERROR_CODE", "message": "..."}}
    """

    def test_ocr_empty_body_structured_error(self, server_url):
        r = requests.post(f"{server_url}/ocr", timeout=10)
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_ocr_raw_empty_structured_error(self, server_url):
        r = requests.post(f"{server_url}/ocr/raw", data=b"", timeout=10)
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_batch_empty_array_structured_error(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": []},
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_pixels_missing_headers_structured_error(self, server_url):
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=b"\x00" * 100,
            timeout=10,
        )
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_pdf_empty_structured_error(self, server_url):
        r = requests.post(f"{server_url}/ocr/pdf", data=b"", timeout=10)
        assert r.status_code == 400
        _assert_structured_error(r)

    def test_error_code_is_uppercase_snake(self, server_url):
        """Error codes should follow UPPER_SNAKE_CASE convention."""
        r = requests.post(f"{server_url}/ocr", timeout=10)
        assert r.status_code == 400
        code = r.json()["error"]["code"]
        assert code == code.upper(), f"Error code should be uppercase: {code}"
        assert " " not in code, f"Error code should not contain spaces: {code}"


# ===========================================================================
# 9. Response headers on success
# ===========================================================================


class TestResponseHeaders:
    """Verify X-Request-Id and X-Inference-Time-Ms on successful responses."""

    def test_ocr_headers(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr",
            json={"image": test_png_base64},
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_raw_headers(self, server_url, test_png_bytes):
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=test_png_bytes,
            headers={"Content-Type": "image/png"},
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_batch_headers(self, server_url, test_png_base64):
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [test_png_base64]},
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_pdf_headers(self, server_url, test_pdf_bytes):
        r = requests.post(
            f"{server_url}/ocr/pdf",
            data=test_pdf_bytes,
            headers={"Content-Type": "application/pdf"},
            timeout=30,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_pixels_headers(self, server_url, test_image_pil):
        raw, w, h, ch = _pil_to_bgr_bytes(test_image_pil)
        r = requests.post(
            f"{server_url}/ocr/pixels",
            data=raw,
            headers={
                "X-Width": str(w),
                "X-Height": str(h),
                "X-Channels": str(ch),
            },
            timeout=15,
        )
        assert r.status_code == 200
        _assert_success_headers(r)

    def test_request_ids_are_unique(self, server_url, test_png_base64):
        """Two consecutive requests should get different X-Request-Id values."""
        ids = set()
        for _ in range(3):
            r = requests.post(
                f"{server_url}/ocr",
                json={"image": test_png_base64},
                timeout=15,
            )
            assert r.status_code == 200
            ids.add(r.headers.get("X-Request-Id"))
        assert len(ids) == 3, f"Expected 3 unique request IDs, got {len(ids)}: {ids}"


# ===========================================================================
# 10. Nginx 413 body too large
# ===========================================================================


class TestNginx413:
    """Verify that oversized requests (>100MB) are rejected with 413."""

    @pytest.mark.slow
    def test_ocr_raw_body_too_large(self, server_url):
        """Sending >100MB should trigger nginx 413 Request Entity Too Large."""
        # Generate ~101MB of data
        oversized = b"\x00" * (101 * 1024 * 1024)
        try:
            r = requests.post(
                f"{server_url}/ocr/raw",
                data=oversized,
                headers={"Content-Type": "image/png"},
                timeout=60,
            )
            assert r.status_code == 413, (
                f"Expected 413 for >100MB body, got {r.status_code}"
            )
        except requests.ConnectionError:
            # nginx may close the connection before reading all data;
            # this is acceptable behavior for an oversized request.
            pass
