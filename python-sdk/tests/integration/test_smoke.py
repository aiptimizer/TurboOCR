from __future__ import annotations

from io import BytesIO

import pypdf

from turboocr import (
    Client,
    OcrResponse,
    PdfResponse,
    render_to_markdown,
)


def test_health_responds(client: Client) -> None:
    status = client.health()
    assert status.ok


def test_image_round_trip_with_layout(client: Client, sample_image: bytes) -> None:
    response = client.recognize_image(
        sample_image, layout=True, reading_order=True, include_blocks=True
    )
    assert isinstance(response, OcrResponse)
    assert len(response.results) > 0


def test_image_to_markdown_renders(client: Client, sample_image: bytes) -> None:
    response = client.recognize_image(
        sample_image, layout=True, reading_order=True, include_blocks=True
    )
    doc = render_to_markdown(response)
    assert doc.markdown.strip()
    assert len(doc.nodes) > 0


def test_blocks_dump(client: Client, sample_image: bytes) -> None:
    response = client.recognize_image(sample_image, include_blocks=True)
    assert len(response.blocks) > 0
    dumped = response.blocks[0].model_dump_json(by_alias=True)
    assert dumped.startswith("{") and '"class"' in dumped


def test_pdf_round_trip(client: Client, sample_pdf: bytes) -> None:
    response = client.recognize_pdf(sample_pdf, dpi=150)
    assert isinstance(response, PdfResponse)
    assert len(response.pages) == 1


def test_searchable_pdf_extracts_recognized_text(
    client: Client, sample_pdf: bytes
) -> None:
    out = client.make_searchable_pdf(sample_pdf, dpi=200)
    text = pypdf.PdfReader(BytesIO(out)).pages[0].extract_text()
    assert text.strip(), "expected non-empty text layer"
