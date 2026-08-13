"""Image-format coverage across every image-accepting entry point.

The decode dispatch branches by container: JPEG -> nvJPEG (GPU), 8-bit PNG ->
wuffs fast path, 16-bit PNG -> OpenCV/libpng fallback, everything else
(WebP/BMP/TIFF) -> cv::imdecode. Each endpoint re-runs that dispatch, so a
format that decodes on /ocr/raw can still be broken on /ocr/batch or gRPC.
This exercises the real formats on each endpoint rather than a single canary.

The GT text is chosen to survive JPEG/WebP recompression and the 16->8-bit
downconvert so a decode regression shows up as missing text, not a crash only.
"""
import io

import numpy as np
import pytest
import requests
from PIL import Image

from conftest import make_text_image, pil_to_base64


GT = "FORMAT CHECK 2026"
GT_TOKENS = ("format", "check", "2026")


def _encode(fmt):
    """Return (bytes, content_type) for a text image in the given container.
    Skips the test if this Pillow build can't write the format."""
    img = make_text_image(GT, width=520, height=120, font_size=34)
    buf = io.BytesIO()
    ctype = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
             "BMP": "image/bmp", "TIFF": "image/tiff"}[fmt]
    try:
        if fmt == "JPEG":
            img.save(buf, format="JPEG", quality=95)
        else:
            img.save(buf, format=fmt)
    except (KeyError, OSError) as e:
        pytest.skip(f"Pillow cannot encode {fmt}: {e}")
    return buf.getvalue(), ctype


def _png16():
    """A genuine 16-bit grayscale PNG (routes to the OpenCV fallback branch)."""
    l8 = np.asarray(make_text_image(GT, width=520, height=120, font_size=34).convert("L"))
    arr = (l8.astype("<u2") * 257)  # little-endian uint16, matches PIL "I;16"
    img16 = Image.frombytes("I;16", (arr.shape[1], arr.shape[0]), arr.tobytes())
    buf = io.BytesIO()
    img16.save(buf, format="PNG")
    return buf.getvalue()


def _has_tokens(results):
    text = " ".join(r.get("text", "") for r in results).lower()
    return sum(tok in text for tok in GT_TOKENS)


IMAGE_FORMATS = ["PNG", "JPEG", "WEBP", "BMP", "TIFF"]


@pytest.mark.parametrize("fmt", IMAGE_FORMATS)
class TestFormatsPerEndpoint:
    def test_ocr_raw(self, server_url, fmt):
        data, ctype = _encode(fmt)
        r = requests.post(f"{server_url}/ocr/raw", data=data,
                          headers={"Content-Type": ctype}, timeout=30)
        assert r.status_code == 200, f"/ocr/raw {fmt}: {r.status_code} {r.text[:200]}"
        assert _has_tokens(r.json()["results"]) >= 2, \
            f"/ocr/raw {fmt}: text not recovered ({r.json()['results']})"

    def test_ocr_base64(self, server_url, fmt):
        data, _ = _encode(fmt)
        import base64
        b64 = base64.b64encode(data).decode("ascii")
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=30)
        assert r.status_code == 200, f"/ocr {fmt}: {r.status_code} {r.text[:200]}"
        assert _has_tokens(r.json()["results"]) >= 2, f"/ocr {fmt}: text not recovered"

    def test_ocr_batch(self, server_url, fmt):
        data, _ = _encode(fmt)
        import base64
        b64 = base64.b64encode(data).decode("ascii")
        r = requests.post(f"{server_url}/ocr/batch",
                          json={"images": [b64, b64]}, timeout=30)
        assert r.status_code == 200, f"/ocr/batch {fmt}: {r.status_code} {r.text[:200]}"
        results = r.json()["batch_results"]
        assert len(results) == 2
        assert _has_tokens(results[0]["results"]) >= 2, \
            f"/ocr/batch {fmt}: text not recovered"


class TestSixteenBitPng:
    """16-bit PNG is the one container that leaves the wuffs fast path for the
    OpenCV/libpng 16->8 downconvert. Regression guard for that branch."""

    def test_ocr_raw_16bit(self, server_url):
        r = requests.post(f"{server_url}/ocr/raw", data=_png16(),
                          headers={"Content-Type": "image/png"}, timeout=30)
        assert r.status_code == 200, r.text[:200]
        assert _has_tokens(r.json()["results"]) >= 2, "16-bit PNG text not recovered"

    def test_ocr_batch_16bit_mixed(self, server_url):
        """16-bit PNG batched next to an 8-bit JPEG: both decode branches in one
        request, results kept in order."""
        import base64
        b16 = base64.b64encode(_png16()).decode("ascii")
        jpeg, _ = _encode("JPEG")
        bj = base64.b64encode(jpeg).decode("ascii")
        r = requests.post(f"{server_url}/ocr/batch",
                          json={"images": [b16, bj]}, timeout=30)
        assert r.status_code == 200, r.text[:200]
        results = r.json()["batch_results"]
        assert len(results) == 2
        assert _has_tokens(results[0]["results"]) >= 2, "16-bit slot lost text"
        assert _has_tokens(results[1]["results"]) >= 2, "JPEG slot lost text"


class TestGrpcFormats:
    """gRPC Recognize path re-runs the same dispatch off the HTTP thread."""

    @pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "BMP", "TIFF"])
    def test_grpc_recognize(self, grpc_target, fmt):
        pytest.importorskip("grpc")
        import sys
        from pathlib import Path
        gen = Path(__file__).resolve().parents[2] / "_grpc_generated"
        sys.path.insert(0, str(gen))
        try:
            import grpc
            import ocr_pb2
            import ocr_pb2_grpc
        except ImportError:
            pytest.skip("gRPC stubs not available")
        channel = grpc.insecure_channel(grpc_target)
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
        except grpc.FutureTimeoutError:
            pytest.skip(f"gRPC server not reachable at {grpc_target}")
        stub = ocr_pb2_grpc.OCRServiceStub(channel)
        data, _ = _encode(fmt)
        resp = stub.Recognize(ocr_pb2.OCRRequest(image=data), timeout=30)
        # Default json_bytes mode carries text in json_response; results is empty.
        import json
        if resp.json_response:
            results = json.loads(resp.json_response).get("results", [])
        else:
            results = [{"text": r.text} for r in resp.results]
        text = " ".join(r.get("text", "") for r in results).lower()
        got = sum(tok in text for tok in GT_TOKENS)
        assert got >= 2, f"gRPC Recognize {fmt}: text not recovered ({text!r})"
