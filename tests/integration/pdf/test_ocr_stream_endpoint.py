"""Integration tests for /ocr/stream — the NDJSON streaming endpoint.

One request shape for PDFs and single images: the response is
application/x-ndjson, one JSON object per line, with a meta line first, one
page event per finished page (AS IT COMPLETES, out of order), page_error for
failed pages, and an end line last. Page events reuse the /ocr/pdf pages[]
element shape, so a consumer can share its parser.
"""
import io
import json

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes, pil_to_jpeg_bytes


def _pdf_bytes(pages=3):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        c.setFont("Helvetica", 24)
        c.drawString(100, 700, f"Stream page {i + 1} marker")
        c.showPage()
    c.save()
    return buf.getvalue()


def _stream(server_url, data, ctype, params=""):
    events = []
    with requests.post(f"{server_url}/ocr/stream{params}", data=data,
                       headers={"Content-Type": ctype}, stream=True,
                       timeout=180) as r:
        assert r.status_code == 200, r.text[:300]
        assert r.headers.get("Content-Type", "").startswith("application/x-ndjson")
        for line in r.iter_lines():
            if line:
                events.append(json.loads(line))
    return events


@pytest.fixture(scope="module")
def gpu_stream(server_url):
    caps = requests.get(f"{server_url}/capabilities", timeout=5)
    if "/ocr/stream" not in caps.text:
        pytest.skip("/ocr/stream not registered (GPU-only)")
    return True


class TestStreamPdf:
    def test_protocol_shape(self, server_url, gpu_stream):
        events = _stream(server_url, _pdf_bytes(3), "application/pdf")
        kinds = [e["event"] for e in events]
        assert kinds[0] == "meta" and kinds[-1] == "end", kinds
        assert kinds.count("page") == 3
        meta = events[0]
        assert meta["kind"] == "pdf" and meta["pages"] == 3
        end = events[-1]
        assert end["pages"] == 3 and end["failed"] == 0

    def test_page_events_carry_pdf_page_shape(self, server_url, gpu_stream):
        events = _stream(server_url, _pdf_bytes(2), "application/pdf")
        pages = [e for e in events if e["event"] == "page"]
        seen = set()
        for p in pages:
            for key in ("page", "page_index", "dpi", "width", "height",
                        "results", "mode", "text_layer_quality"):
                assert key in p, f"page event missing {key}"
            seen.add(p["page_index"])
            text = " ".join(x["text"] for x in p["results"]).lower()
            assert "stream" in text or "marker" in text or "page" in text
        assert seen == {0, 1}, "every page exactly once"

    def test_geometric_streams_instantly(self, server_url, gpu_stream):
        events = _stream(server_url, _pdf_bytes(2), "application/pdf",
                         params="?mode=geometric")
        pages = [e for e in events if e["event"] == "page"]
        assert len(pages) == 2
        assert all(p["mode"] == "geometric" for p in pages)
        assert all(p["results"] for p in pages), "text layer text expected"

    def test_layout_param(self, server_url, gpu_stream):
        events = _stream(server_url, _pdf_bytes(1), "application/pdf",
                         params="?layout=1")
        page = next(e for e in events if e["event"] == "page")
        assert "layout" in page

    def test_text0_images_inline(self, server_url, gpu_stream):
        events = _stream(server_url, _pdf_bytes(1), "application/pdf",
                         params="?text=0&images=inline&format=jpeg")
        page = next(e for e in events if e["event"] == "page")
        assert page["results"] == []
        assert "image_b64" in page


class TestStreamImage:
    def test_png(self, server_url, gpu_stream):
        img = make_text_image("STREAMED IMAGE 42", width=440, height=100, font_size=32)
        events = _stream(server_url, pil_to_png_bytes(img), "image/png")
        kinds = [e["event"] for e in events]
        assert kinds == ["meta", "page", "end"], kinds
        assert events[0]["kind"] == "image"
        page = events[1]
        assert page["page_index"] == 0 and page["width"] > 0
        text = " ".join(x["text"] for x in page["results"]).lower()
        assert "streamed" in text or "image" in text or "42" in text

    def test_jpeg(self, server_url, gpu_stream):
        img = make_text_image("JPEG STREAM", width=400, height=100, font_size=32)
        events = _stream(server_url, pil_to_jpeg_bytes(img), "image/jpeg")
        assert [e["event"] for e in events] == ["meta", "page", "end"]
        assert events[1]["results"]

    def test_garbage_image_streams_error_event(self, server_url, gpu_stream):
        events = _stream(server_url, b"\x00\x01not-an-image", "image/png")
        kinds = [e["event"] for e in events]
        assert "error" in kinds, kinds
        err = next(e for e in events if e["event"] == "error")
        assert err["code"] == "IMAGE_DECODE_FAILED"


class TestStreamValidation:
    def test_empty_body(self, server_url, gpu_stream):
        r = requests.post(f"{server_url}/ocr/stream", data=b"", timeout=15)
        assert r.status_code == 400

    def test_connection_close_fails_loud(self, server_url, gpu_stream, hello_image):
        """Drogon async streams need keep-alive; Connection: close would get an
        empty 200 body (silent failure) — must 400 instead."""
        r = requests.post(f"{server_url}/ocr/stream",
                          data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png",
                                   "Connection": "close"}, timeout=15)
        assert r.status_code == 400
        assert "keep-alive" in r.json()["error"]["message"]

    def test_structure_gates_still_apply(self, server_url, gpu_stream, hello_image):
        r = requests.post(f"{server_url}/ocr/stream?text=0&layout=1&tables=1",
                          data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png"}, timeout=15)
        assert r.status_code == 400
