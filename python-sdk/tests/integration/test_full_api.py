"""Live per-function coverage of the whole public surface against a running
TurboOCR v3.x server.

Run with a full-featured server for full coverage:

    docker run --gpus all -p 8000:8000 -p 50051:50051 \
      -e TABLE_BACKEND=slanext -e FORMULA_BACKEND=ppformulanet_s \
      ghcr.io/aiptimizer/turboocr:latest
    TURBO_OCR_BASE_URL=http://localhost:8000 uv run pytest tests/integration -q

Feature-gated tests (tables, formulas, autorotate, gRPC) skip themselves when
the server lacks the stage.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO

import pypdf
import pytest

from turboocr import (
    AsyncClient,
    BatchResponse,
    Capabilities,
    Client,
    MarkdownPagesResponse,
    OcrResponse,
    PdfResponse,
)


@pytest.fixture(scope="session")
def caps(server_url: str) -> Capabilities:
    with Client(base_url=server_url) as c:
        return c.capabilities()


def _needs(caps: Capabilities, feature: str) -> None:
    if not getattr(caps.features, feature):
        pytest.skip(f"server has no {feature} backend loaded")


# --- capabilities / health ---


def test_capabilities(client: Client) -> None:
    caps = client.capabilities()
    assert caps.features.layout in (True, False)
    assert "ocr" in caps.pdf.modes
    assert caps.limits.max_image_dim


def test_health_all_variants(client: Client) -> None:
    assert client.health().ok
    assert client.health(live=True).ok
    assert client.health(ready=True).ok


# --- recognize_* ---


def test_recognize_image_plain(client: Client, sample_image: bytes) -> None:
    response = client.recognize_image(sample_image)
    assert isinstance(response, OcrResponse)
    assert response.results
    assert "quick brown fox" in response.text.lower()


def test_recognize_base64(client: Client, sample_image: bytes) -> None:
    response = client.recognize_base64(base64.b64encode(sample_image).decode())
    assert response.results


def test_recognize_pixels(client: Client, sample_image: bytes) -> None:
    from PIL import Image

    img = Image.open(BytesIO(sample_image)).convert("RGB")
    # Server expects BGR channel order.
    bgr = Image.merge("RGB", img.split()[::-1])
    response = client.recognize_pixels(
        bgr.tobytes(), width=img.width, height=img.height, channels=3
    )
    assert response.results


def test_recognize_batch(client: Client, sample_image: bytes) -> None:
    batch = client.recognize_batch([sample_image, sample_image])
    assert isinstance(batch, BatchResponse)
    assert len(batch.batch_results) == 2
    assert batch.errors == [None, None]


def test_recognize_pdf_modes(client: Client, sample_pdf: bytes) -> None:
    ocr = client.recognize_pdf(sample_pdf, mode="ocr")
    assert isinstance(ocr, PdfResponse) and ocr.pages
    geometric = client.recognize_pdf(sample_pdf, mode="geometric")
    # Born-digital reportlab PDF has a trusted text layer.
    assert geometric.pages[0].text_layer_quality in {"trusted", "rejected", "absent"}
    assert "quick brown fox" in geometric.text.lower()


def test_recognize_pdf_autorotate(client: Client, caps: Capabilities, sample_pdf: bytes) -> None:
    _needs(caps, "autorotate")
    response = client.recognize_pdf(sample_pdf, autorotate=True)
    assert response.pages


# --- tables & formulas (strict opt-in) ---


def test_tables_and_formulas_first_class(
    client: Client, caps: Capabilities, sample_image: bytes
) -> None:
    _needs(caps, "tables")
    _needs(caps, "formulas")
    response = client.recognize_image(sample_image, tables=True, formulas=True)
    # The synthetic sample has no tables/formulas — the point is the request
    # is accepted and the fields exist (empty lists, not errors).
    assert response.tables == [] or response.tables[0].html is not None
    assert response.formulas == [] or response.formulas[0].latex is not None


def test_backend_disabled_raises(client: Client, caps: Capabilities, sample_image: bytes) -> None:
    if caps.features.tables:
        pytest.skip("server has a table backend; disabled-error not testable")
    from turboocr import BackendDisabled

    with pytest.raises(BackendDisabled):
        client.recognize_image(sample_image, tables=True)


# --- markdown ---


def test_page_markdown_server_side(client: Client, sample_image: bytes) -> None:
    md = client.page_markdown(sample_image)
    assert isinstance(md, str)
    assert "quick brown fox" in md.lower()


def test_pdf_markdown_whole_document(client: Client, sample_pdf: bytes) -> None:
    md = client.pdf_markdown(sample_pdf)
    assert isinstance(md, str)
    assert "<!-- page" in md
    assert "quick brown fox" in md.lower()


def test_pdf_markdown_as_pages(client: Client, sample_pdf: bytes) -> None:
    result = client.pdf_markdown(sample_pdf, as_pages=True)
    assert isinstance(result, MarkdownPagesResponse)
    assert result.pages[0].page_index == 0
    assert "quick brown fox" in result.markdown.lower()


def test_client_side_to_markdown(client: Client, sample_image: bytes) -> None:
    doc = client.to_markdown(sample_image)
    assert "quick brown fox" in doc.markdown.lower()


# --- streaming ---


def test_stream_pdf(client: Client, sample_pdf: bytes) -> None:
    events = list(client.stream(sample_pdf, layout=True))
    kinds = [e.event for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "end"
    pages = [e.page for e in events if e.event == "page"]
    assert pages and pages[0] is not None
    assert pages[0].results


def test_stream_single_image(client: Client, sample_image: bytes) -> None:
    events = list(client.stream(sample_image))
    assert [e.event for e in events] == ["meta", "page", "end"]


# --- searchable pdf ---


def test_make_searchable_pdf(client: Client, sample_pdf: bytes) -> None:
    out = client.make_searchable_pdf(sample_pdf, dpi=150)
    reader = pypdf.PdfReader(BytesIO(out))
    text = reader.pages[0].extract_text() or ""
    assert "quick brown fox" in text.lower()


# --- async client parity ---


@pytest.mark.asyncio
async def test_async_full_surface(server_url: str, sample_image: bytes, sample_pdf: bytes) -> None:
    async with AsyncClient(base_url=server_url) as client:
        caps = await client.capabilities()
        assert caps.limits.max_image_dim
        assert (await client.health()).ok
        response = await client.recognize_image(sample_image, layout=True)
        assert response.results
        b64 = base64.b64encode(sample_image).decode()
        assert (await client.recognize_base64(b64)).results
        batch = await client.recognize_batch([sample_image])
        assert batch.errors == [None]
        pdf = await client.recognize_pdf(sample_pdf)
        assert pdf.pages
        md = await client.page_markdown(sample_image)
        assert "quick brown fox" in md.lower()
        pages = await client.pdf_markdown(sample_pdf, as_pages=True)
        assert isinstance(pages, MarkdownPagesResponse) and pages.pages
        events = []
        async for event in await client.stream(sample_pdf, layout=True):
            events.append(event.event)
        assert events[0] == "meta" and events[-1] == "end"
        overlay = await client.make_searchable_pdf(sample_pdf, dpi=150)
        assert overlay.startswith(b"%PDF-")
        doc = await client.to_markdown(sample_image)
        assert doc.markdown


# --- gRPC (needs the grpc extra + a mapped 50051) ---


def _grpc_target() -> str | None:
    return os.environ.get("TURBO_OCR_GRPC_TARGET")


def test_grpc_full_surface(sample_image: bytes, sample_pdf: bytes, caps: Capabilities) -> None:
    target = _grpc_target()
    if target is None:
        pytest.skip("set TURBO_OCR_GRPC_TARGET=host:port to run gRPC integration tests")
    pytest.importorskip("grpc")
    from turboocr import GrpcClient

    with GrpcClient(target=target) as client:
        response = client.recognize_image(sample_image, layout=True)
        assert response.results
        batch = client.recognize_batch([sample_image])
        assert len(batch.batch_results) == 1
        pdf = client.recognize_pdf(sample_pdf)
        assert pdf.pages
        if caps.features.tables and caps.features.formulas:
            rich = client.recognize_image(sample_image, tables=True, formulas=True)
            assert rich.tables == [] or rich.tables[0].html is not None
