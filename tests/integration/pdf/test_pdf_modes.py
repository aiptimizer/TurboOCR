"""Integration tests for /ocr/pdf mode matrix.

Covers the 4 extraction modes (ocr, geometric, auto, auto_verified) on
small real-world PDF fixtures. Accuracy floors live in tests/accuracy/.
"""

import pytest
import requests

MODES = ["ocr", "geometric", "auto", "auto_verified"]


def _post_pdf(server_url, pdf_path, mode, timeout=60):
    return requests.post(
        f"{server_url}/ocr/pdf?mode={mode}",
        data=pdf_path.read_bytes(),
        headers={"Content-Type": "application/pdf"},
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def small_pdfs(fixtures_dir):
    d = fixtures_dir / "pdf"
    fixtures = {
        "simple_letter": d / "simple_letter.pdf",
        "scanned": d / "scanned_document.pdf",
        "single_page_form": d / "single_page_form.pdf",
    }
    missing = [k for k, v in fixtures.items() if not v.exists()]
    if missing:
        pytest.skip(f"missing fixtures: {missing}")
    return fixtures


@pytest.mark.parametrize("mode", MODES)
class TestPdfModeMatrix:
    def test_returns_200_and_pages(self, server_url, small_pdfs, mode):
        r = _post_pdf(server_url, small_pdfs["simple_letter"], mode)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "pages" in data
        assert len(data["pages"]) >= 1

    def test_page_schema(self, server_url, small_pdfs, mode):
        r = _post_pdf(server_url, small_pdfs["single_page_form"], mode)
        assert r.status_code == 200
        for page in r.json()["pages"]:
            assert "page" in page or "page_index" in page
            assert "results" in page
            for item in page["results"]:
                assert "text" in item
                assert "confidence" in item
                assert "bounding_box" in item
                assert len(item["bounding_box"]) == 4

    def test_results_nonempty_for_digital_pdf(self, server_url, small_pdfs, mode):
        r = _post_pdf(server_url, small_pdfs["simple_letter"], mode)
        data = r.json()
        total = sum(len(p["results"]) for p in data["pages"])
        assert total > 0, f"mode={mode} returned zero results on digital PDF"


class TestPdfModeSemantics:
    def test_geometric_on_image_only_pdf_returns_empty(self, server_url, fixtures_dir):
        """geometric reads the text layer only; an image-only PDF has none."""
        image_only = fixtures_dir / "edge_cases" / "no_text_layer.pdf"
        if not image_only.exists():
            pytest.skip("edge_cases/no_text_layer.pdf not generated")
        r = _post_pdf(server_url, image_only, "geometric")
        assert r.status_code == 200
        total = sum(len(p["results"]) for p in r.json()["pages"])
        assert total == 0, (
            f"geometric should return 0 results on image-only PDF, got {total}"
        )

    def test_auto_falls_back_to_ocr_on_image_only(self, server_url, fixtures_dir):
        image_only = fixtures_dir / "edge_cases" / "no_text_layer.pdf"
        if not image_only.exists():
            pytest.skip()
        r = _post_pdf(server_url, image_only, "auto")
        assert r.status_code == 200
        total = sum(len(p["results"]) for p in r.json()["pages"])
        assert total > 0, "auto should fall back to OCR on image-only PDF"

    def test_ocr_mode_works_on_digital(self, server_url, small_pdfs):
        r = _post_pdf(server_url, small_pdfs["simple_letter"], "ocr")
        assert r.status_code == 200
        total = sum(len(p["results"]) for p in r.json()["pages"])
        assert total > 0

    def test_auto_verified_not_worse_than_ocr(self, server_url, small_pdfs):
        r_ocr = _post_pdf(server_url, small_pdfs["simple_letter"], "ocr")
        r_av = _post_pdf(server_url, small_pdfs["simple_letter"], "auto_verified")
        total_ocr = sum(len(p["results"]) for p in r_ocr.json()["pages"])
        total_av = sum(len(p["results"]) for p in r_av.json()["pages"])
        assert total_av >= total_ocr * 0.8, (
            f"auto_verified ({total_av}) should not be much worse than ocr ({total_ocr})"
        )


def test_geometric_reading_order_parity(server_url, small_pdfs):
    """Geometric pages must carry reading_order/blocks when requested, same
    as OCR pages — they are derived from the (rescaled) text + layout."""
    probe = requests.post(
        f"{server_url}/ocr/pdf?mode=ocr&layout=1",
        data=small_pdfs["simple_letter"].read_bytes(),
        headers={"Content-Type": "application/pdf"}, timeout=60)
    if probe.status_code != 200 or not any(
            p.get("layout") for p in probe.json().get("pages", [])):
        pytest.skip("server started without layout")
    r = requests.post(
        f"{server_url}/ocr/pdf?mode=geometric&layout=1&reading_order=1&as_blocks=1",
        data=small_pdfs["simple_letter"].read_bytes(),
        headers={"Content-Type": "application/pdf"}, timeout=60)
    assert r.status_code == 200, r.text
    pages = r.json()["pages"]
    # at least one page has text + layout → must also expose reading_order
    enriched = [p for p in pages if p.get("results") and p.get("layout")]
    assert enriched, "no geometric page produced both text and layout"
    for p in enriched:
        assert p.get("reading_order"), "geometric page missing reading_order"
        assert "blocks" in p, "geometric page missing blocks aggregate"


class TestPageImages:
    """/ocr/pdf?images=inline — per-page rendered image export. Everything is
    per-request (format/quality/lossless/max_side); default format is PNG."""

    def test_no_images_by_default(self, server_url, small_pdfs):
        r = _post_pdf(server_url, small_pdfs["simple_letter"], "ocr")
        assert r.status_code == 200
        assert all("image_b64" not in p for p in r.json()["pages"])

    def test_inline_default_png(self, server_url, small_pdfs):
        import base64
        r = requests.post(
            f"{server_url}/ocr/pdf?images=inline",
            data=small_pdfs["simple_letter"].read_bytes(),
            headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200, r.text
        for p in r.json()["pages"]:
            assert p.get("image_content_type") == "image/png", p.keys()
            raw = base64.b64decode(p["image_b64"])
            assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
            assert p["results"], "OCR text must still be present"

    def test_inline_jpeg_quality_max_side(self, server_url, small_pdfs):
        import base64, io
        r = requests.post(
            f"{server_url}/ocr/pdf?images=inline&format=jpeg&quality=80&max_side=480",
            data=small_pdfs["simple_letter"].read_bytes(),
            headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200, r.text
        p0 = r.json()["pages"][0]
        assert p0["image_content_type"] == "image/jpeg"
        raw = base64.b64decode(p0["image_b64"])
        assert raw[:2] == b"\xff\xd8"  # JPEG SOI
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(raw))
            assert max(img.size) <= 480, img.size
        except ImportError:
            pass

    def test_invalid_quality_rejected(self, server_url, small_pdfs):
        r = requests.post(
            f"{server_url}/ocr/pdf?images=inline&format=jpeg&quality=0",
            data=small_pdfs["simple_letter"].read_bytes(),
            headers={"Content-Type": "application/pdf"}, timeout=30)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_PARAMETER"


class TestAutorotate:
    """/ocr/pdf?autorotate=1 — de-rotate scanned pages upright via the
    doc-orientation model. Returned image, boxes and text end up upright;
    orientation_deg reports the detected clockwise rotation."""

    def _make_rotated_pdf(self, upright_png: bytes, transpose):
        import io
        try:
            import img2pdf
            from PIL import Image
        except ImportError:
            pytest.skip("img2pdf/PIL not available")
        img = Image.open(io.BytesIO(upright_png)).convert("RGB")
        if transpose is not None:
            img = img.transpose(transpose)
        b = io.BytesIO(); img.save(b, "PNG")
        return img2pdf.convert(b.getvalue())

    @pytest.fixture(scope="class")
    def upright_png(self, server_url, small_pdfs):
        import base64
        r = requests.post(
            f"{server_url}/ocr/pdf?images=inline&format=png&mode=ocr",
            data=small_pdfs["simple_letter"].read_bytes(),
            headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200
        return base64.b64decode(r.json()["pages"][0]["image_b64"])

    @pytest.fixture(scope="class")
    def _skip_if_no_autorotate(self, server_url, small_pdfs):
        r = requests.post(
            f"{server_url}/ocr/pdf?autorotate=1",
            data=small_pdfs["simple_letter"].read_bytes(),
            headers={"Content-Type": "application/pdf"}, timeout=60)
        if r.status_code == 400 and r.json().get("error", {}).get("code") == "AUTOROTATE_DISABLED":
            pytest.skip("server started without the doc-orientation model")

    @pytest.mark.parametrize("cw_deg,transpose_name", [
        (0, None), (90, "ROTATE_270"), (180, "ROTATE_180"), (270, "ROTATE_90")])
    def test_detects_and_uprights(self, server_url, upright_png,
                                   _skip_if_no_autorotate, cw_deg, transpose_name):
        import base64, io
        from PIL import Image
        transpose = getattr(Image, transpose_name) if transpose_name else None
        pdf = self._make_rotated_pdf(upright_png, transpose)
        r = requests.post(
            f"{server_url}/ocr/pdf?images=inline&format=png&mode=ocr&autorotate=1",
            data=pdf, headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200, r.text
        p0 = r.json()["pages"][0]
        assert p0["orientation_deg"] == cw_deg, f"detected {p0['orientation_deg']} != {cw_deg}"
        # Returned image is upright (portrait, like the original letter).
        ret = Image.open(io.BytesIO(base64.b64decode(p0["image_b64"])))
        assert ret.size[1] > ret.size[0], f"returned image not portrait: {ret.size}"
        # De-rotated BEFORE OCR, so text recovers (born-upright result count).
        assert len(p0["results"]) > 150

    def test_off_by_default(self, server_url, upright_png, _skip_if_no_autorotate):
        from PIL import Image
        pdf = self._make_rotated_pdf(upright_png, Image.ROTATE_270)
        r = requests.post(
            f"{server_url}/ocr/pdf?images=inline&format=png&mode=ocr",
            data=pdf, headers={"Content-Type": "application/pdf"}, timeout=120)
        assert r.status_code == 200
        # No autorotate -> no orientation_deg key, image stays rotated.
        assert "orientation_deg" not in r.json()["pages"][0]

    def test_invalid_value_rejected(self, server_url, small_pdfs):
        r = requests.post(
            f"{server_url}/ocr/pdf?autorotate=maybe",
            data=small_pdfs["simple_letter"].read_bytes(),
            headers={"Content-Type": "application/pdf"}, timeout=30)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_PARAMETER"
