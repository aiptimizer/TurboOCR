"""Happy-path integration tests for /ocr/pixels.

Covers the BGR raw-pixel upload path. Error cases live in test_error_handling.py.
"""

import numpy as np
import pytest
import requests
from PIL import Image

from conftest import make_text_image


def _pil_to_bgr_bytes(pil_img):
    arr = np.asarray(pil_img.convert("RGB"))  # HxWx3 RGB
    bgr = arr[:, :, ::-1].copy()  # RGB -> BGR
    return bgr.tobytes(), arr.shape[1], arr.shape[0], 3


def _pil_to_gray_bytes(pil_img):
    arr = np.asarray(pil_img.convert("L"))  # HxW
    return arr.tobytes(), arr.shape[1], arr.shape[0], 1


def _post_pixels(server_url, raw_bytes, w, h, ch, timeout=30):
    return requests.post(
        f"{server_url}/ocr/pixels",
        data=raw_bytes,
        headers={
            "X-Width": str(w),
            "X-Height": str(h),
            "X-Channels": str(ch),
            "Content-Type": "application/octet-stream",
        },
        timeout=timeout,
    )


class TestOcrPixelsHappyPath:
    def test_pixels_bgr_3channel(self, server_url):
        img = make_text_image("HELLO", width=400, height=100, font_size=40)
        raw, w, h, ch = _pil_to_bgr_bytes(img)
        r = _post_pixels(server_url, raw, w, h, ch)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        joined = " ".join(item["text"] for item in data["results"]).upper()
        assert "HELLO" in joined or "HELL" in joined

    def test_pixels_grayscale_1channel(self, server_url):
        # channels=1 is expanded to BGR in the handler (the pipeline is
        # BGR-only); grayscale input must OCR like its 3-channel twin.
        img = make_text_image("WORLD", width=400, height=100, font_size=40)
        raw, w, h, ch = _pil_to_gray_bytes(img)
        r = _post_pixels(server_url, raw, w, h, ch)
        assert r.status_code == 200, r.text
        assert "results" in r.json()
        joined = " ".join(item["text"] for item in r.json()["results"])
        assert "WORLD" in joined or "WORL" in joined

    def test_pixels_matches_raw_endpoint(self, server_url):
        """Same underlying image through /ocr/raw and /ocr/pixels should
        detect the same text (exact string match after normalization)."""
        img = make_text_image("MATCH", width=400, height=100, font_size=40)

        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        r_raw = requests.post(
            f"{server_url}/ocr/raw",
            data=buf.getvalue(),
            headers={"Content-Type": "image/png"},
            timeout=10,
        )
        assert r_raw.status_code == 200

        raw, w, h, ch = _pil_to_bgr_bytes(img)
        r_pix = _post_pixels(server_url, raw, w, h, ch)
        assert r_pix.status_code == 200

        def _joined(resp):
            return "".join(
                item["text"].upper().replace(" ", "")
                for item in resp.json()["results"]
            )

        assert _joined(r_raw) == _joined(r_pix), (
            f"raw={_joined(r_raw)!r} pixels={_joined(r_pix)!r}"
        )

    def test_pixels_4k_image(self, server_url, edge_cases_dir):
        big = Image.open(edge_cases_dir / "4k_text.png")
        raw, w, h, ch = _pil_to_bgr_bytes(big)
        r = _post_pixels(server_url, raw, w, h, ch, timeout=60)
        assert r.status_code == 200
        assert len(r.json().get("results", [])) > 0
