"""Integration tests for /metrics (Prometheus exposition) and the
observability middleware (X-Request-Id / X-Inference-Time-Ms headers).

Neither had any test coverage. The exporter is hand-rolled, so the tests pin
the exposition-format invariants a Prometheus scraper depends on:
  - histogram le-buckets are cumulative and monotonically non-decreasing,
  - bucket{le="+Inf"} == _count,
  - counters only ever increase,
  - every registered OCR endpoint reports under its own route label
    (regression: /ocr/markdown and /infer used to be lumped into "other").
"""
import re
import uuid

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes


def _scrape(server_url):
    r = requests.get(f"{server_url}/metrics", timeout=10)
    assert r.status_code == 200
    return r.text


def _counter(body, metric, **labels):
    """Value of a counter/gauge sample with the given label set (0 if absent)."""
    if labels:
        lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
        pat = rf'^{re.escape(metric)}\{{{re.escape(lab)}\}} (\d+(?:\.\d+)?)$'
    else:
        pat = rf'^{re.escape(metric)} (\d+(?:\.\d+)?)$'
    m = re.search(pat, body, re.MULTILINE)
    return float(m.group(1)) if m else 0.0


class TestMetricsExposition:
    def test_scrape_ok_and_content_type(self, server_url):
        r = requests.get(f"{server_url}/metrics", timeout=10)
        assert r.status_code == 200
        ct = r.headers.get("Content-Type", "")
        assert ct.startswith("text/plain"), f"unexpected content type: {ct}"
        assert "version=0.0.4" in ct  # Prometheus text exposition format
        assert "# TYPE turbo_ocr_requests_total counter" in r.text
        assert "# TYPE turbo_ocr_request_duration_seconds histogram" in r.text

    def test_2xx_counter_increments(self, server_url, hello_image):
        before = _counter(_scrape(server_url), "turbo_ocr_requests_total",
                          route="/ocr/raw", status="2xx")
        r = requests.post(f"{server_url}/ocr/raw", data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png"}, timeout=30)
        assert r.status_code == 200
        after = _counter(_scrape(server_url), "turbo_ocr_requests_total",
                         route="/ocr/raw", status="2xx")
        assert after >= before + 1, f"2xx counter did not increment ({before} -> {after})"

    def test_4xx_counter_increments(self, server_url):
        before = _counter(_scrape(server_url), "turbo_ocr_requests_total",
                          route="/ocr/raw", status="4xx")
        r = requests.post(f"{server_url}/ocr/raw", data=b"not an image",
                          headers={"Content-Type": "image/png"}, timeout=15)
        assert r.status_code == 400
        after = _counter(_scrape(server_url), "turbo_ocr_requests_total",
                         route="/ocr/raw", status="4xx")
        assert after >= before + 1, f"4xx counter did not increment ({before} -> {after})"

    def test_histogram_invariants(self, server_url, hello_image):
        """Cumulative le-buckets must be non-decreasing and +Inf must equal
        _count — a scraper computes quantiles from exactly this structure."""
        # Ensure at least one observation exists.
        requests.post(f"{server_url}/ocr/raw", data=pil_to_png_bytes(hello_image),
                      headers={"Content-Type": "image/png"}, timeout=30)
        body = _scrape(server_url)
        buckets = {}
        for m in re.finditer(
                r'turbo_ocr_request_duration_seconds_bucket\{route="([^"]+)",le="([^"]+)"\} (\d+)',
                body):
            buckets.setdefault(m.group(1), []).append((m.group(2), int(m.group(3))))
        counts = dict(re.findall(
            r'turbo_ocr_request_duration_seconds_count\{route="([^"]+)"\} (\d+)', body))
        assert buckets, "no histogram buckets exported"
        for route, series in buckets.items():
            values = [v for _, v in series]  # exposition order = ascending le
            assert values == sorted(values), \
                f"{route}: cumulative buckets not monotonic: {series}"
            inf = dict(series).get("+Inf")
            assert inf is not None, f"{route}: missing le=+Inf bucket"
            assert inf == int(counts[route]), \
                f"{route}: +Inf bucket {inf} != _count {counts[route]}"
            assert values[-2] <= inf if len(values) > 1 else True

    def test_markdown_and_infer_have_own_route_labels(self, server_url, hello_image):
        """Regression: /ocr/markdown and /infer used to be bucketed under
        route="other", making two real endpoints invisible in monitoring."""
        caps = requests.get(f"{server_url}/capabilities", timeout=5).text
        if "/ocr/markdown" not in caps:
            pytest.skip("GPU-only routes not registered on this server")
        requests.post(f"{server_url}/ocr/markdown", data=pil_to_png_bytes(hello_image),
                      headers={"Content-Type": "image/png"}, timeout=60)
        requests.post(f"{server_url}/infer", json={"bad": "payload"}, timeout=15)
        body = _scrape(server_url)
        assert 'route="/ocr/markdown"' in body, "/ocr/markdown missing from metrics"
        assert 'route="/infer"' in body, "/infer missing from metrics"
        # the bad /infer payload above must have landed in ITS 4xx counter
        assert _counter(body, "turbo_ocr_requests_total",
                        route="/infer", status="4xx") >= 1

    def test_gauges_present_and_sane(self, server_url):
        body = _scrape(server_url)
        assert _counter(body, "turbo_ocr_pipeline_pool_size") >= 1
        assert "turbo_ocr_workpool_queue_depth" in body
        assert "turbo_ocr_pool_exhaustions_total" in body

    def test_scrape_not_self_counted(self, server_url):
        """/metrics must not count its own scrapes (scrape noise would drown
        the 'other' bucket and grow unboundedly with monitoring frequency)."""
        b1 = _counter(_scrape(server_url), "turbo_ocr_requests_total",
                      route="other", status="2xx")
        for _ in range(3):
            _scrape(server_url)
        b2 = _counter(_scrape(server_url), "turbo_ocr_requests_total",
                      route="other", status="2xx")
        assert b2 == b1, f"scrapes were self-counted ({b1} -> {b2})"


class TestObservabilityHeaders:
    def test_request_id_generated(self, server_url, hello_image):
        r = requests.post(f"{server_url}/ocr/raw", data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png"}, timeout=30)
        assert r.headers.get("X-Request-Id"), "no X-Request-Id on response"

    def test_request_id_echoed(self, server_url, hello_image):
        rid = str(uuid.uuid4())
        r = requests.post(f"{server_url}/ocr/raw", data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png", "X-Request-Id": rid},
                          timeout=30)
        assert r.headers.get("X-Request-Id") == rid, "client X-Request-Id not echoed"

    def test_inference_time_header(self, server_url, hello_image):
        r = requests.post(f"{server_url}/ocr/raw", data=pil_to_png_bytes(hello_image),
                          headers={"Content-Type": "image/png"}, timeout=30)
        t = r.headers.get("X-Inference-Time-Ms")
        assert t is not None and t.isdigit(), f"bad X-Inference-Time-Ms: {t!r}"
        assert 0 <= int(t) < 120_000
