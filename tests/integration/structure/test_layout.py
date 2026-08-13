"""Layout detection integration tests.

Layout is per-request opt-in: callers pass `?layout=1` to receive the
layout field. The server must have been started with `ENABLE_LAYOUT=1`
for the opt-in to succeed; otherwise the server returns a 400 with a
message explaining that the startup flag is missing.
"""

from functools import lru_cache

import pytest
import requests


@lru_cache(maxsize=1)
def _server_has_layout(server_url: str) -> bool:
    try:
        r = requests.post(
            f"{server_url}/ocr?layout=1",
            json={"image": ""},
            timeout=5,
        )
    except Exception:
        return False
    # A layout-less server rejects ?layout=1 with the stable LAYOUT_DISABLED
    # code (checked on the code, not a message substring, so it survives
    # message wording changes).
    if r.status_code == 400:
        try:
            if r.json().get("error", {}).get("code") == "LAYOUT_DISABLED":
                return False
        except Exception:
            pass
    return True


@pytest.fixture(scope="module")
def layout_enabled(server_url):
    return _server_has_layout(server_url)


def _post_pdf(server_url, pdf_path, mode="ocr", layout=None, timeout=60):
    url = f"{server_url}/ocr/pdf?mode={mode}"
    if layout is not None:
        url += f"&layout={layout}"
    return requests.post(
        url, data=pdf_path.read_bytes(),
        headers={"Content-Type": "application/pdf"}, timeout=timeout,
    )


def _post_image_raw(server_url, image_path, layout=None):
    url = f"{server_url}/ocr/raw"
    if layout is not None:
        url += f"?layout={layout}"
    return requests.post(
        url, data=image_path.read_bytes(),
        headers={"Content-Type": "image/png"}, timeout=30,
    )


def _post_image_json(server_url, image_path, layout=None):
    import base64
    url = f"{server_url}/ocr"
    if layout is not None:
        url += f"?layout={layout}"
    return requests.post(
        url, json={"image": base64.b64encode(image_path.read_bytes()).decode()},
        timeout=30,
    )


def _has_layout(resp_json) -> bool:
    if "layout" in resp_json and resp_json["layout"]:
        return True
    for page in resp_json.get("pages", []):
        if page.get("layout"):
            return True
    return False


class TestLayoutQueryParsing:
    def test_invalid_layout_value_returns_400(self, server_url, fixtures_dir):
        pdf = fixtures_dir / "pdf" / "simple_letter.pdf"
        if not pdf.exists():
            pytest.skip()
        r = _post_pdf(server_url, pdf, layout="maybe")
        assert r.status_code == 400
        assert "layout" in r.text.lower()

    @pytest.mark.parametrize("value", ["0", "false", "off", "no"])
    def test_layout_off_synonyms_accepted(self, server_url, fixtures_dir, value):
        pdf = fixtures_dir / "pdf" / "simple_letter.pdf"
        if not pdf.exists():
            pytest.skip()
        r = _post_pdf(server_url, pdf, layout=value)
        assert r.status_code == 200
        assert not _has_layout(r.json())

    def test_default_request_has_no_layout(self, server_url, fixtures_dir):
        pdf = fixtures_dir / "pdf" / "simple_letter.pdf"
        if not pdf.exists():
            pytest.skip()
        r = _post_pdf(server_url, pdf)
        assert r.status_code == 200
        assert not _has_layout(r.json())


class TestLayoutUnavailable:
    @pytest.fixture(autouse=True)
    def _skip_if_layout_server(self, layout_enabled):
        if layout_enabled:
            pytest.skip("server was started with ENABLE_LAYOUT=1")

    def test_pdf_layout_1_returns_400(self, server_url, fixtures_dir):
        pdf = fixtures_dir / "pdf" / "simple_letter.pdf"
        if not pdf.exists():
            pytest.skip()
        r = _post_pdf(server_url, pdf, layout="1")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "LAYOUT_DISABLED", r.text

    def test_ocr_raw_layout_1_returns_400(self, server_url, fixtures_dir):
        img = fixtures_dir / "images" / "png" / "business_letter.png"
        if not img.exists():
            pytest.skip()
        r = _post_image_raw(server_url, img, layout="1")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "LAYOUT_DISABLED", r.text

    def test_ocr_json_layout_1_returns_400(self, server_url, fixtures_dir):
        img = fixtures_dir / "images" / "png" / "business_letter.png"
        if not img.exists():
            pytest.skip()
        r = _post_image_json(server_url, img, layout="1")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "LAYOUT_DISABLED", r.text

    def test_ocr_raw_default_works(self, server_url, fixtures_dir):
        img = fixtures_dir / "images" / "png" / "business_letter.png"
        if not img.exists():
            pytest.skip()
        r = _post_image_raw(server_url, img)
        assert r.status_code == 200
        assert not _has_layout(r.json())

    @pytest.mark.parametrize("param", ["layout=1", "reading_order=1", "as_blocks=1"])
    def test_layout_unavailable_code_is_unified(self, server_url, fixtures_dir, param):
        # Every "layout feature unavailable" rejection returns one stable
        # code: LAYOUT_DISABLED (layout=1 used to leak INVALID_PARAMETER).
        img = fixtures_dir / "images" / "png" / "business_letter.png"
        if not img.exists():
            pytest.skip()
        r = requests.post(
            f"{server_url}/ocr/raw?{param}", data=img.read_bytes(),
            headers={"Content-Type": "image/png"}, timeout=30)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "LAYOUT_DISABLED", r.text

    def test_malformed_layout_value_is_invalid_parameter(self, server_url, fixtures_dir):
        # A malformed bool stays INVALID_PARAMETER, not LAYOUT_DISABLED.
        img = fixtures_dir / "images" / "png" / "business_letter.png"
        if not img.exists():
            pytest.skip()
        r = requests.post(
            f"{server_url}/ocr/raw?layout=garbage", data=img.read_bytes(),
            headers={"Content-Type": "image/png"}, timeout=30)
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_PARAMETER", r.text


class TestLayoutRequested:
    @pytest.fixture(autouse=True)
    def _skip_if_no_layout_server(self, layout_enabled):
        if not layout_enabled:
            pytest.skip("server was not started with ENABLE_LAYOUT=1")

    def test_pdf_layout_1_returns_layout(self, server_url, fixtures_dir):
        pdf = fixtures_dir / "pdf" / "academic_paper.pdf"
        if not pdf.exists():
            pytest.skip()
        r = _post_pdf(server_url, pdf, layout="1")
        assert r.status_code == 200
        assert _has_layout(r.json()), "expected layout field when layout=1"

    def test_pdf_layout_0_returns_no_layout(self, server_url, fixtures_dir):
        pdf = fixtures_dir / "pdf" / "academic_paper.pdf"
        if not pdf.exists():
            pytest.skip()
        r = _post_pdf(server_url, pdf, layout="0")
        assert r.status_code == 200
        assert not _has_layout(r.json())

    @pytest.mark.parametrize("mode", ["ocr", "geometric", "auto", "auto_verified"])
    def test_layout_1_works_for_all_modes(self, server_url, fixtures_dir, mode):
        pdf = fixtures_dir / "pdf" / "simple_letter.pdf"
        if not pdf.exists():
            pytest.skip()
        r = _post_pdf(server_url, pdf, mode=mode, layout="1")
        assert r.status_code == 200
        for page in r.json()["pages"]:
            assert "layout" in page

    def test_ocr_raw_layout_1_returns_layout(self, server_url, fixtures_dir):
        img = fixtures_dir / "images" / "png" / "business_letter.png"
        if not img.exists():
            pytest.skip()
        r = _post_image_raw(server_url, img, layout="1")
        assert r.status_code == 200
        assert _has_layout(r.json())

    def test_ocr_json_layout_1_returns_layout(self, server_url, fixtures_dir):
        img = fixtures_dir / "images" / "png" / "business_letter.png"
        if not img.exists():
            pytest.skip()
        r = _post_image_json(server_url, img, layout="1")
        assert r.status_code == 200
        assert _has_layout(r.json())

    def test_layout_0_faster_than_layout_1(self, server_url, fixtures_dir):
        import time
        pdf = fixtures_dir / "pdf" / "academic_paper.pdf"
        if not pdf.exists():
            pytest.skip()

        def median(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2]

        off, on = [], []
        for _ in range(7):
            t0 = time.perf_counter()
            _post_pdf(server_url, pdf, layout="0")
            off.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _post_pdf(server_url, pdf, layout="1")
            on.append(time.perf_counter() - t0)

        off_ms = median(off) * 1000
        on_ms = median(on) * 1000
        assert off_ms < on_ms, (
            f"layout=0 ({off_ms:.0f}ms) should be faster than layout=1 ({on_ms:.0f}ms)"
        )
