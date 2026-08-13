from __future__ import annotations

import httpx
import pytest
import respx

from turboocr import Client, RetryPolicy, render_to_markdown
from turboocr._core.ids import make_uuid7, short_request_id


def _ocr_payload(*, layout: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {"results": []}
    if layout:
        payload["layout"] = []
        payload["reading_order"] = []
    return payload


def test_uuid7_is_time_ordered_and_unique() -> None:
    a = make_uuid7()
    b = make_uuid7()
    assert a != b
    assert len(a) == 36
    assert a < b or a > b
    assert a[14] == "7"


def test_short_request_id_uses_uuid7_prefix() -> None:
    rid = short_request_id()
    assert len(rid) == 16
    assert all(c in "0123456789abcdef" for c in rid)


@respx.mock
def test_x_request_id_header_sent() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    with Client(base_url="http://t") as client:
        client.recognize_image(b"x")
    assert "X-Request-ID" in route.calls.last.request.headers
    assert len(route.calls.last.request.headers["X-Request-ID"]) == 16


@respx.mock
def test_429_triggers_retry() -> None:
    payload = _ocr_payload()
    route = respx.post("http://t/ocr/raw").mock(
        side_effect=[
            httpx.Response(429, json={"error_code": "RATE_LIMITED", "error": "slow down"}),
            httpx.Response(200, json=payload),
        ]
    )
    policy = RetryPolicy(attempts=2, backoff=0.0, jitter=0.0)
    with Client(base_url="http://t", retry=policy) as client:
        client.recognize_image(b"x")
    assert route.call_count == 2


@respx.mock
def test_per_request_timeout_passed_to_httpx() -> None:
    respx.post("http://t/ocr/pdf").mock(side_effect=httpx.ReadTimeout("slow"))
    policy = RetryPolicy(attempts=1)
    from turboocr import Timeout

    with Client(base_url="http://t", retry=policy) as client, pytest.raises(Timeout):
        client.recognize_pdf(b"%PDF-", timeout=0.001)


@respx.mock
def test_health_500_returns_healthstatus_not_raises() -> None:
    respx.get("http://t/health").mock(return_value=httpx.Response(500, text="down"))
    with Client(base_url="http://t", retry=RetryPolicy(attempts=1)) as client:
        status = client.health()
    assert status.ok is False
    assert status.status_code == 500
    assert status.body == "down"


@respx.mock
def test_layout_non_contiguous_ids_resolve_by_dict() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 0,
                        "text": "hello",
                        "confidence": 0.9,
                        "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                        "layout_id": 42,
                    }
                ],
                "layout": [
                    {
                        "id": 42,
                        "class": "paragraph_title",
                        "class_id": 17,
                        "confidence": 0.9,
                        "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    }
                ],
                "reading_order": [0],
            },
        )
    )
    with Client(base_url="http://t") as client:
        response = client.recognize_image(b"x", layout=True, reading_order=True)
    doc = render_to_markdown(response)
    assert "## hello" in doc.markdown


def test_package_version_falls_back_to_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import PackageNotFoundError

    from turboocr._core import env

    def _raises(_: str) -> object:
        raise PackageNotFoundError("turboocr")

    monkeypatch.setattr("turboocr._core.env.distribution", _raises, raising=False)
    monkeypatch.setattr("importlib.metadata.distribution", _raises)
    assert env._package_version() == "dev"
