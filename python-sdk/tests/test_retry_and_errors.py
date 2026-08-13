from __future__ import annotations

import httpx
import pytest
import respx

from turboocr import Client, NetworkError, RetryPolicy, Timeout


@respx.mock
def test_timeout_raises_timeout_subclass() -> None:
    respx.post("http://t/ocr/raw").mock(side_effect=httpx.ReadTimeout("slow"))
    with Client(base_url="http://t", retry=RetryPolicy(attempts=1)) as client, pytest.raises(
        Timeout
    ):
        client.recognize_image(b"x")


@respx.mock
def test_connect_error_raises_network_error_subclass() -> None:
    respx.post("http://t/ocr/raw").mock(side_effect=httpx.ConnectError("no route"))
    with Client(base_url="http://t", retry=RetryPolicy(attempts=1)) as client, pytest.raises(
        NetworkError
    ):
        client.recognize_image(b"x")


@respx.mock
def test_retries_on_503_then_succeeds() -> None:
    payload = {"results": []}
    route = respx.post("http://t/ocr/raw").mock(
        side_effect=[
            httpx.Response(503, json={"error_code": "NOT_READY", "error": "warming up"}),
            httpx.Response(503, json={"error_code": "NOT_READY", "error": "warming up"}),
            httpx.Response(200, json=payload),
        ]
    )
    policy = RetryPolicy(attempts=3, backoff=0.0, jitter=0.0)
    with Client(base_url="http://t", retry=policy) as client:
        response = client.recognize_image(b"x")
    assert response.results == []
    assert route.call_count == 3


@respx.mock
def test_retry_honors_integer_retry_after() -> None:
    import time

    payload = {"results": []}
    respx.post("http://t/ocr/raw").mock(
        side_effect=[
            httpx.Response(
                503,
                headers={"Retry-After": "1"},
                json={"error_code": "NOT_READY", "error": "warming"},
            ),
            httpx.Response(200, json=payload),
        ]
    )
    policy = RetryPolicy(attempts=2, backoff=0.0, jitter=0.0)
    with Client(base_url="http://t", retry=policy) as client:
        start = time.monotonic()
        client.recognize_image(b"x")
        elapsed = time.monotonic() - start
    assert elapsed >= 1.0


@respx.mock
def test_retries_exhausted_propagates_server_error() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(503, json={"error_code": "NOT_READY", "error": "down"})
    )
    policy = RetryPolicy(attempts=2, backoff=0.0, jitter=0.0)
    with Client(base_url="http://t", retry=policy) as client, pytest.raises(Exception) as ei:
        client.recognize_image(b"x")
    assert ei.value.status_code == 503  # type: ignore[attr-defined]
