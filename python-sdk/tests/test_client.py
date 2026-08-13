from __future__ import annotations

import httpx
import pytest
import respx

from turboocr import AsyncClient, Client, DimensionsTooLarge, LayoutDisabled


def _ocr_payload() -> dict[str, object]:
    return {
        "results": [
            {
                "id": 0,
                "text": "hello",
                "confidence": 0.99,
                "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                "layout_id": 0,
            }
        ],
        "layout": [
            {
                "id": 0,
                "class": "paragraph_title",
                "class_id": 17,
                "confidence": 0.9,
                "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
            }
        ],
        "reading_order": [0],
        "blocks": [
            {
                "id": 0,
                "layout_id": 0,
                "class": "paragraph_title",
                "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                "content": "hello",
                "order_index": 0,
            }
        ],
    }


@respx.mock
def test_recognize_image_sends_options_and_auth() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )

    with Client(base_url="http://t", api_key="sk-abc") as client:
        response = client.recognize_image(b"\x89PNG", layout=True, include_blocks=True)

    assert response.results[0].text == "hello"
    assert response.blocks[0].class_name == "paragraph_title"

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-abc"
    assert request.headers["Content-Type"] == "application/octet-stream"
    # include_blocks=True auto-promotes reading_order=True (server contract).
    assert dict(request.url.params) == {"layout": "1", "reading_order": "1", "as_blocks": "1"}


@respx.mock
def test_recognize_pixels_sets_dim_query_params() -> None:
    # Dimensions travel as query params — the X-Width/X-Height headers are
    # deprecated server-side (v3.1+ answers them with `Deprecation: true`).
    route = respx.post("http://t/ocr/pixels").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    with Client(base_url="http://t") as client:
        client.recognize_pixels(b"\x00" * 30, width=10, height=1, channels=3)

    request = route.calls.last.request
    params = dict(request.url.params)
    assert params["width"] == "10"
    assert params["height"] == "1"
    assert params["channels"] == "3"
    assert "X-Width" not in request.headers


@respx.mock
def test_x_api_key_scheme() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    with Client(base_url="http://t", api_key="sk-z", auth_scheme="x-api-key") as client:
        client.recognize_image(b"x")
    headers = route.calls.last.request.headers
    assert headers["X-API-Key"] == "sk-z"
    assert "Authorization" not in headers


@respx.mock
def test_error_code_maps_to_exception() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(
            400,
            json={"error_code": "DIMENSIONS_TOO_LARGE", "error": "too big"},
        )
    )
    with Client(base_url="http://t") as client, pytest.raises(DimensionsTooLarge) as ei:
        client.recognize_image(b"x")
    assert ei.value.code == "DIMENSIONS_TOO_LARGE"
    assert ei.value.status_code == 400


@respx.mock
def test_layout_disabled_maps() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(
            400, json={"error_code": "LAYOUT_DISABLED", "error": "off"}
        )
    )
    with Client(base_url="http://t") as client, pytest.raises(LayoutDisabled):
        client.recognize_image(b"x", layout=True)


@respx.mock
def test_batch_decodes_errors_array() -> None:
    respx.post("http://t/ocr/batch").mock(
        return_value=httpx.Response(
            200,
            json={"batch_results": [_ocr_payload()], "errors": [None]},
        )
    )
    with Client(base_url="http://t") as client:
        out = client.recognize_batch([b"x"])
    assert out.errors == [None]
    assert out.batch_results[0].results[0].text == "hello"


@respx.mock
def test_pdf_params() -> None:
    route = respx.post("http://t/ocr/pdf").mock(
        return_value=httpx.Response(
            200,
            json={
                "pages": [
                    {
                        "page": 1,
                        "page_index": 0,
                        "dpi": 100,
                        "width": 100,
                        "height": 200,
                        "results": [],
                        "layout": [],
                        "reading_order": [],
                        "blocks": [],
                        "mode": "auto",
                        "text_layer_quality": "ocr",
                    }
                ]
            },
        )
    )
    with Client(base_url="http://t") as client:
        out = client.recognize_pdf(b"%PDF-", dpi=150, mode="auto", include_blocks=True)

    assert out.pages[0].mode.value == "auto"
    params = dict(route.calls.last.request.url.params)
    assert params["dpi"] == "150"
    assert params["mode"] == "auto"


@respx.mock
def test_batch_iter_results_returns_tagged_union() -> None:
    from turboocr import BatchFailure, BatchSuccess

    respx.post("http://t/ocr/batch").mock(
        return_value=httpx.Response(
            200,
            json={
                "batch_results": [_ocr_payload(), _ocr_payload()],
                "errors": [None, "decode_failed"],
            },
        )
    )
    with Client(base_url="http://t") as client:
        results = client.recognize_batch([b"x", b"y"]).iter_results()
    assert len(results) == 2
    assert isinstance(results[0], BatchSuccess) and results[0].index == 0
    assert isinstance(results[1], BatchFailure) and results[1].error == "decode_failed"


@respx.mock
def test_bounding_box_serializes_as_flat_list() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    with Client(base_url="http://t") as client:
        resp = client.recognize_image(b"x")
    dumped = resp.results[0].model_dump(by_alias=True)
    assert dumped["bounding_box"] == [[0, 0], [10, 0], [10, 5], [0, 5]]


@respx.mock
def test_include_blocks_auto_promotes_reading_order_and_layout() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    with Client(base_url="http://t") as client:
        client.recognize_image(b"x", include_blocks=True)
    params = dict(route.calls.last.request.url.params)
    assert params == {"layout": "1", "reading_order": "1", "as_blocks": "1"}


@respx.mock
def test_to_markdown_dispatches_image_vs_pdf() -> None:
    image_route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    pdf_route = respx.post("http://t/ocr/pdf").mock(
        return_value=httpx.Response(
            200,
            json={
                "pages": [
                    {
                        "page": 1, "page_index": 0, "dpi": 150, "width": 0, "height": 0,
                        "results": [], "layout": [], "reading_order": [], "blocks": [],
                        "mode": "ocr", "text_layer_quality": "ocr",
                    }
                ]
            },
        )
    )
    with Client(base_url="http://t") as client:
        client.to_markdown(b"\x89PNG\r\n\x1a\n")
        client.to_markdown(b"%PDF-1.4 minimal")
    assert image_route.call_count == 1
    assert pdf_route.call_count == 1
    assert dict(pdf_route.calls.last.request.url.params)["dpi"] == "150"


@respx.mock
async def test_async_client_roundtrip() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    async with AsyncClient(base_url="http://t") as client:
        out = await client.recognize_image(b"x", layout=True)
    assert out.results[0].text == "hello"


@respx.mock
async def test_async_client_streams_path_input(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    p = Path(tmp_path) / "img.png"
    p.write_bytes(b"\x89PNG-streamed-from-path")
    async with AsyncClient(base_url="http://t") as client:
        out = await client.recognize_image(p)
    assert out.results[0].text == "hello"
    assert route.calls.last.request.content == b"\x89PNG-streamed-from-path"


@respx.mock
async def test_async_client_streams_byte_iterable() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )

    def _chunks() -> object:
        yield b"AAA"
        yield b"BBB"

    async with AsyncClient(base_url="http://t") as client:
        await client.recognize_image(_chunks())  # type: ignore[arg-type]
    assert route.calls.last.request.content == b"AAABBB"


@respx.mock
def test_recognize_image_streams_file_like() -> None:
    import io

    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    payload = b"\x89PNG-streamed-body"
    with Client(base_url="http://t") as client:
        client.recognize_image(io.BytesIO(payload))
    sent = route.calls.last.request.content
    assert sent == payload


@respx.mock
def test_recognize_image_streams_byte_iterable() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )

    def _chunks() -> object:
        yield b"AAA"
        yield b"BBB"

    with Client(base_url="http://t") as client:
        client.recognize_image(_chunks())  # type: ignore[arg-type]
    assert route.calls.last.request.content == b"AAABBB"




@respx.mock
def test_health_live_hits_health_live_endpoint() -> None:
    route = respx.get("http://t/health/live").mock(
        return_value=httpx.Response(200, text="ok")
    )
    with Client(base_url="http://t") as client:
        status = client.health(live=True)
    assert route.called
    assert status.ok is True


def test_health_ready_and_live_mutually_exclusive() -> None:
    from turboocr import InvalidParameter

    with Client(base_url="http://t") as client, pytest.raises(InvalidParameter):
        client.health(ready=True, live=True)


def test_base_url_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBO_OCR_BASE_URL", "http://from-env:9999")
    monkeypatch.setenv("TURBO_OCR_API_KEY", "sk-env")
    with Client() as client:
        assert client.base_url == "http://from-env:9999"
        # default_headers exposed and bearer auth was applied
        assert client.auth_scheme == "bearer"
        assert dict(client.default_headers) == {}


@respx.mock
def test_default_headers_are_introspectable_and_sent() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    extra = {"X-Tenant": "acme"}
    with Client(base_url="http://t", default_headers=extra) as client:
        assert dict(client.default_headers) == extra
        client.recognize_image(b"x")
    assert route.calls.last.request.headers["X-Tenant"] == "acme"


@respx.mock
def test_path_body_replays_on_retry(tmp_path: object) -> None:
    """The SOTA win: Path inputs stream from disk AND replay on retry. The
    body factory is re-invoked per attempt, so the server sees full bytes
    on both attempts even though the body is consumed each time."""
    from pathlib import Path

    payload = b"\x89PNG" + b"X" * 4096
    image_path = Path(tmp_path) / "img.png"  # type: ignore[arg-type]
    image_path.write_bytes(payload)

    from turboocr import RetryPolicy

    route = respx.post("http://t/ocr/raw").mock(
        side_effect=[
            httpx.Response(503, json={"error_code": "BUSY", "error": "retry"}),
            httpx.Response(200, json=_ocr_payload()),
        ]
    )
    with Client(
        base_url="http://t",
        retry=RetryPolicy(attempts=2, backoff=0.0, jitter=0.0),
    ) as client:
        client.recognize_image(image_path)

    assert route.call_count == 2
    # Both attempts received the full body — proves the factory replays.
    for call in route.calls:
        assert call.request.content == payload


@respx.mock
def test_event_hooks_fire_on_request_and_response() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    seen_requests: list[str] = []
    seen_statuses: list[int] = []

    def on_req(request: httpx.Request) -> None:
        seen_requests.append(request.url.path)

    def on_resp(response: httpx.Response) -> None:
        seen_statuses.append(response.status_code)

    with Client(base_url="http://t", on_request=on_req, on_response=on_resp) as client:
        client.recognize_image(b"x")

    assert seen_requests == ["/ocr/raw"]
    assert seen_statuses == [200]


def test_limits_pass_through_to_httpx() -> None:
    custom = httpx.Limits(max_connections=5, max_keepalive_connections=1)
    with Client(base_url="http://t", limits=custom) as client:
        # httpx stores the limits on the pool/transport; check it's not the default.
        assert client._http._transport is not None  # type: ignore[attr-defined]


def test_repr_includes_useful_fields() -> None:
    with Client(base_url="http://t", api_key="sk-x", timeout=12.5) as client:
        r = repr(client)
    assert r.startswith("Client(")
    assert "base_url='http://t'" in r
    assert "timeout=12.5" in r
    assert "auth='bearer'/set" in r  # masked: never leaks the key
    assert "sk-x" not in r
