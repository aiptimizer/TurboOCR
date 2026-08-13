from __future__ import annotations

import pytest

from turboocr import OcrResponse, PdfResponse, ProtocolError
from turboocr.markdown import NodeKind, render_ocr_to_markdown, render_pdf_to_markdown


def _bbox(x1: int = 10, y1: int = 10) -> list[list[int]]:
    return [[0, 0], [x1, 0], [x1, y1], [0, y1]]


def test_blocks_drive_markdown_when_present() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [],
            "layout": [],
            "reading_order": [],
            "blocks": [
                {
                    "id": 0,
                    "layout_id": 0,
                    "class": "doc_title",
                    "bounding_box": _bbox(),
                    "content": "The Title",
                    "order_index": 0,
                },
                {
                    "id": 1,
                    "layout_id": 1,
                    "class": "paragraph_title",
                    "bounding_box": _bbox(),
                    "content": "Section",
                    "order_index": 1,
                },
                {
                    "id": 2,
                    "layout_id": 2,
                    "class": "text",
                    "bounding_box": _bbox(),
                    "content": "Body line one\nBody line two",
                    "order_index": 2,
                },
                {
                    "id": 3,
                    "layout_id": 3,
                    "class": "image",
                    "bounding_box": _bbox(),
                    "content": "",
                    "order_index": 3,
                },
                {
                    "id": 4,
                    "layout_id": 4,
                    "class": "footer",
                    "bounding_box": _bbox(),
                    "content": "footer note",
                    "order_index": 4,
                },
            ],
        }
    )
    doc = render_ocr_to_markdown(response)
    md = doc.markdown
    assert "# The Title" in md
    assert "## Section" in md
    assert "Body line one" in md
    assert "![image]()" in md
    assert "<!-- footer:" in md

    kinds = [n.kind for n in doc.nodes]
    assert kinds == [
        NodeKind.heading,
        NodeKind.heading,
        NodeKind.paragraph,
        NodeKind.figure,
        NodeKind.footer,
    ]


def test_layout_plus_reading_order_without_blocks() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [
                {
                    "id": 0,
                    "text": "Hello",
                    "confidence": 0.9,
                    "bounding_box": _bbox(),
                    "layout_id": 0,
                },
                {
                    "id": 1,
                    "text": "world",
                    "confidence": 0.9,
                    "bounding_box": _bbox(),
                    "layout_id": 0,
                },
            ],
            "layout": [
                {
                    "id": 0,
                    "class": "paragraph_title",
                    "class_id": 17,
                    "confidence": 0.95,
                    "bounding_box": _bbox(),
                }
            ],
            "reading_order": [0, 1],
        }
    )
    doc = render_ocr_to_markdown(response)
    assert "## Hello world" in doc.markdown


def test_falls_back_to_plain_paragraphs_with_no_layout() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [
                {
                    "text": "alpha",
                    "confidence": 0.9,
                    "bounding_box": _bbox(),
                },
                {
                    "text": "beta",
                    "confidence": 0.9,
                    "bounding_box": _bbox(),
                },
            ]
        }
    )
    doc = render_ocr_to_markdown(response)
    assert "alpha\n\nbeta" in doc.markdown


def test_layout_enabled_response_with_missing_layout_id_raises() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [
                {
                    "id": 0,
                    "text": "orphan",
                    "confidence": 0.9,
                    "bounding_box": _bbox(),
                }
            ],
            "layout": [
                {
                    "id": 0,
                    "class": "paragraph_title",
                    "class_id": 17,
                    "confidence": 0.95,
                    "bounding_box": _bbox(),
                }
            ],
            "reading_order": [0],
        }
    )
    with pytest.raises(ProtocolError):
        render_ocr_to_markdown(response)


def test_pdf_rendering_inserts_page_break() -> None:
    page_payload = {
        "page": 1,
        "page_index": 0,
        "dpi": 100,
        "width": 100,
        "height": 100,
        "results": [],
        "layout": [],
        "reading_order": [],
        "blocks": [
            {
                "id": 0,
                "layout_id": 0,
                "class": "text",
                "bounding_box": _bbox(),
                "content": "page one",
                "order_index": 0,
            }
        ],
        "mode": "ocr",
        "text_layer_quality": "ocr",
    }
    page_two = {**page_payload, "page": 2, "page_index": 1}
    page_two["blocks"] = [
        {
            "id": 0,
            "layout_id": 0,
            "class": "text",
            "bounding_box": _bbox(),
            "content": "page two",
            "order_index": 0,
        }
    ]
    response = PdfResponse.model_validate({"pages": [page_payload, page_two]})

    doc = render_pdf_to_markdown(response)
    assert doc.source == "pdf"
    assert doc.pages == 2
    assert "page one" in doc.markdown
    assert "page two" in doc.markdown
    assert "---" in doc.markdown
