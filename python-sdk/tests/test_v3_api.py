"""Unit tests for the v3.x server API surface (wire shapes captured from a
live TurboOCR v3.4.0 server)."""

from __future__ import annotations

import httpx
import pytest
import respx

from turboocr import (
    BackendDisabled,
    Capabilities,
    Client,
    Formula,
    InferenceTimeout,
    OcrResponse,
    PdfMode,
    StreamEvent,
    Table,
)

QUAD = [[0, 0], [10, 0], [10, 10], [0, 10]]


def _ocr_payload() -> dict[str, object]:
    return {"results": [{"text": "hi", "confidence": 0.9, "bounding_box": QUAD}]}


# --- first-class tables / formulas (server wire shape) ---


def test_server_tables_and_formulas_parse() -> None:
    payload = {
        "results": [],
        "layout": [],
        "tables": [
            {
                "layout_id": 5,
                "html": "<html><body><table><tr><td>p</td></tr></table></body></html>",
                "confidence": 0.99996,
                "bounding_box": QUAD,
            }
        ],
        "formulas": [
            {
                "layout_id": 4,
                "latex": "a \\equiv g^{k}",
                "confidence": 0.896,
                "bounding_box": QUAD,
            }
        ],
    }
    response = OcrResponse.model_validate(payload)
    assert response.tables[0].html is not None and "<table>" in response.tables[0].html
    assert response.tables[0].layout_id == 5
    assert response.formulas[0].latex == "a \\equiv g^{k}"
    # `.text` mirrors latex for server formulas so both sources read the same.
    assert response.formulas[0].text == "a \\equiv g^{k}"


def test_tables_synthesized_from_blocks_when_server_omits() -> None:
    payload = {
        "results": [],
        "blocks": [
            {
                "id": 0,
                "layout_id": 1,
                "class": "table",
                "bounding_box": QUAD,
                "content": "a | b",
                "order_index": 0,
            },
            {
                "id": 1,
                "layout_id": 2,
                "class": "display_formula",
                "bounding_box": QUAD,
                "content": "E=mc^2",
                "order_index": 1,
            },
        ],
    }
    response = OcrResponse.model_validate(payload)
    assert len(response.tables) == 1
    assert response.tables[0].text == "a | b"
    assert response.tables[0].html is None
    assert len(response.formulas) == 1
    assert response.formulas[0].text == "E=mc^2"


def test_server_fields_win_over_synthesis() -> None:
    payload = {
        "results": [],
        "blocks": [
            {
                "id": 0,
                "layout_id": 1,
                "class": "table",
                "bounding_box": QUAD,
                "content": "block-table",
                "order_index": 0,
            }
        ],
        "tables": [
            {"layout_id": 1, "html": "<table></table>", "confidence": 1.0, "bounding_box": QUAD}
        ],
        "formulas": [],
    }
    response = OcrResponse.model_validate(payload)
    assert len(response.tables) == 1
    assert response.tables[0].html == "<table></table>"
    assert response.formulas == []  # server said none; blocks must not overwrite


# --- query params ---


@respx.mock
def test_tables_formulas_params_sent_and_layout_promoted() -> None:
    route = respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(200, json=_ocr_payload())
    )
    with Client(base_url="http://t") as client:
        client.recognize_image(b"x", tables=True, formulas=True)
    params = dict(route.calls.last.request.url.params)
    assert params == {"layout": "1", "tables": "1", "formulas": "1"}


@respx.mock
def test_pdf_autorotate_param() -> None:
    route = respx.post("http://t/ocr/pdf").mock(
        return_value=httpx.Response(200, json={"pages": []})
    )
    with Client(base_url="http://t") as client:
        client.recognize_pdf(b"%PDF-x", autorotate=True)
    assert dict(route.calls.last.request.url.params)["autorotate"] == "1"


def test_pdf_mode_text_is_gone() -> None:
    # Server v3 removed mode=text (silently falls back to ocr); the enum
    # must not offer it.
    assert not hasattr(PdfMode, "text")
    assert {m.value for m in PdfMode} == {"ocr", "geometric", "auto", "auto_verified"}


# --- error envelope ---


@respx.mock
def test_nested_error_envelope_maps_to_backend_disabled() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "TABLE_BACKEND_DISABLED",
                    "message": "no table backend configured",
                }
            },
        )
    )
    with Client(base_url="http://t", retry=None) as client, pytest.raises(BackendDisabled):
        client.recognize_image(b"x", tables=True)


@respx.mock
def test_inference_timeout_maps() -> None:
    respx.post("http://t/ocr/raw").mock(
        return_value=httpx.Response(
            504, json={"error": {"code": "INFERENCE_TIMEOUT", "message": "deadline"}}
        )
    )
    from turboocr import RetryPolicy

    with (
        Client(base_url="http://t", retry=RetryPolicy(attempts=1)) as client,
        pytest.raises(InferenceTimeout),
    ):
        client.recognize_image(b"x")


# --- capabilities ---


def test_capabilities_parses_live_shape() -> None:
    caps = Capabilities.model_validate(
        {
            "build": "gpu",
            "features": {
                "layout": True,
                "tables": True,
                "formulas": True,
                "autorotate": True,
                "profile_endpoint": False,
                "grpc_response_mode": "json_bytes",
            },
            "pdf": {
                "modes": ["ocr", "geometric", "auto", "auto_verified"],
                "default_dpi": 100,
                "max_pages": 2000,
            },
            "limits": {"max_body_mb": 100, "max_image_dim": 16384, "max_batch_images": 1024},
            "endpoints": ["/health", "/ocr/raw"],
        }
    )
    assert caps.features.tables is True
    assert "geometric" in caps.pdf.modes
    assert caps.limits.max_image_dim == 16384


# --- markdown endpoints ---


@respx.mock
def test_page_markdown_returns_text() -> None:
    respx.post("http://t/ocr/markdown").mock(
        return_value=httpx.Response(
            200, text="# Title\n\nBody", headers={"Content-Type": "text/markdown"}
        )
    )
    with Client(base_url="http://t") as client:
        md = client.page_markdown(b"img")
    assert md.startswith("# Title")


@respx.mock
def test_pdf_markdown_as_pages() -> None:
    route = respx.post("http://t/ocr/pdf").mock(
        return_value=httpx.Response(
            200,
            json={
                "pages": [
                    {"page_index": 0, "markdown": "# p1"},
                    {"page_index": 1, "markdown": "# p2", "table_degraded": True},
                ]
            },
        )
    )
    with Client(base_url="http://t") as client:
        result = client.pdf_markdown(b"%PDF-x", as_pages=True)
    params = dict(route.calls.last.request.url.params)
    assert params["markdown"] == "1" and params["as_pages"] == "1"
    assert result.pages[1].table_degraded is True
    assert result.markdown == "# p1\n\n# p2"


# --- streaming ---


@respx.mock
def test_stream_yields_typed_events() -> None:
    ndjson = "\n".join(
        [
            '{"event":"meta","kind":"pdf","pages":2,"dpi":100,"mode":"ocr"}',
            '{"event":"page","page":2,"page_index":1,"dpi":100,"width":100,"height":100,'
            '"results":[{"text":"x","confidence":0.9,"bounding_box":[[0,0],[1,0],[1,1],[0,1]]}],'
            '"mode":"ocr","text_layer_quality":"absent"}',
            '{"event":"end","pages":2,"failed":0}',
        ]
    )
    respx.post("http://t/ocr/stream").mock(
        return_value=httpx.Response(
            200, text=ndjson, headers={"Content-Type": "application/x-ndjson"}
        )
    )
    with Client(base_url="http://t") as client:
        events = list(client.stream(b"%PDF-x", layout=True))
    assert [e.event for e in events] == ["meta", "page", "end"]
    page = events[1].page
    assert page is not None and page.page_index == 1
    assert page.results[0].text == "x"
    assert events[2].failed == 0


# --- degraded flags ---


def test_degraded_flags_parse() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [],
            "table_degraded": True,
            "table_warning": "table backend produced no output",
        }
    )
    assert response.table_degraded is True
    assert response.table_warning is not None


def test_model_exports() -> None:
    assert isinstance(Table.model_fields, dict)
    assert isinstance(Formula.model_fields, dict)
    assert StreamEvent.model_fields["event"].is_required()
