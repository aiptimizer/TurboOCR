from __future__ import annotations

from turboocr import BoundingBox, LayoutBox, OcrResponse, TextItem


def test_bounding_box_aabb_and_center() -> None:
    bb = BoundingBox(points=((10, 20), (50, 20), (50, 60), (10, 60)))
    assert bb.aabb == (10, 20, 50, 60)
    assert bb.center == (30, 40)
    assert bb.width == 40
    assert bb.height == 40


def test_text_item_accepts_raw_box_array() -> None:
    item = TextItem.model_validate(
        {
            "text": "hello",
            "confidence": 0.9,
            "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
        }
    )
    assert item.bounding_box.width == 10


def test_layout_box_uses_class_alias() -> None:
    layout = LayoutBox.model_validate(
        {
            "class": "doc_title",
            "class_id": 6,
            "confidence": 0.95,
            "bounding_box": [[0, 0], [100, 0], [100, 30], [0, 30]],
        }
    )
    assert layout.class_name == "doc_title"


def test_ocr_response_text_falls_back_through_reading_order() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [
                {"text": "B", "confidence": 0.9, "bounding_box": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                {"text": "A", "confidence": 0.9, "bounding_box": [[0, 0], [1, 0], [1, 1], [0, 1]]},
            ],
            "reading_order": [1, 0],
        }
    )
    assert response.text == "A\nB"


def test_extra_server_fields_survive_on_model_extra() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [],
            "future_field": {"version": "2.3"},
        }
    )
    assert response.model_extra == {"future_field": {"version": "2.3"}}


def test_required_field_rename_raises() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as ei:
        OcrResponse.model_validate({"items": []})  # server renamed results → items
    assert any(err["type"] == "missing" and err["loc"] == ("results",) for err in ei.value.errors())


def test_batch_response_required_fields_rename_raises() -> None:
    import pytest
    from pydantic import ValidationError

    from turboocr import BatchResponse

    with pytest.raises(ValidationError):
        BatchResponse.model_validate({"results_renamed": [], "errors": []})
    with pytest.raises(ValidationError):
        BatchResponse.model_validate({"batch_results": []})


def test_pdf_response_required_fields_rename_raises() -> None:
    import pytest
    from pydantic import ValidationError

    from turboocr import PdfResponse

    with pytest.raises(ValidationError):
        PdfResponse.model_validate({"documents": []})


def test_synthesized_tables_from_blocks() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [],
            "blocks": [
                {
                    "id": 0, "layout_id": 0, "class": "table",
                    "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "content": "row1col1 row1col2",
                    "order_index": 0,
                },
                {
                    "id": 1, "layout_id": 1, "class": "paragraph_title",
                    "bounding_box": [[0, 6], [10, 6], [10, 10], [0, 10]],
                    "content": "Section",
                    "order_index": 1,
                },
            ],
        }
    )
    assert len(response.tables) == 1
    assert response.tables[0].text == "row1col1 row1col2"
    assert response.tables[0].html is None
    assert response.formulas == []


def test_synthesized_formulas_from_blocks() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [],
            "blocks": [
                {
                    "id": 0, "layout_id": 0, "class": "display_formula",
                    "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "content": "E = mc^2",
                    "order_index": 0,
                },
                {
                    "id": 1, "layout_id": 1, "class": "inline_formula",
                    "bounding_box": [[0, 6], [10, 6], [10, 10], [0, 10]],
                    "content": "x_i",
                    "order_index": 1,
                },
            ],
        }
    )
    assert len(response.formulas) == 2
    assert response.formulas[0].text == "E = mc^2"
    assert response.formulas[0].is_inline is False
    assert response.formulas[1].is_inline is True
    assert response.formulas[0].latex is None


def test_pdf_response_tables_aggregates_across_pages() -> None:
    from turboocr import PdfResponse

    payload = {
        "pages": [
            {
                "page": 1, "page_index": 0, "dpi": 100, "width": 100, "height": 100,
                "results": [], "layout": [], "reading_order": [],
                "blocks": [
                    {
                        "id": 0, "layout_id": 0, "class": "table",
                        "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                        "content": "page1 table", "order_index": 0,
                    }
                ],
                "mode": "ocr", "text_layer_quality": "ocr",
            },
            {
                "page": 2, "page_index": 1, "dpi": 100, "width": 100, "height": 100,
                "results": [], "layout": [], "reading_order": [],
                "blocks": [
                    {
                        "id": 0, "layout_id": 0, "class": "table",
                        "bounding_box": [[0, 0], [10, 0], [10, 5], [0, 5]],
                        "content": "page2 table a", "order_index": 0,
                    },
                    {
                        "id": 1, "layout_id": 1, "class": "table",
                        "bounding_box": [[0, 6], [10, 6], [10, 10], [0, 10]],
                        "content": "page2 table b", "order_index": 1,
                    },
                ],
                "mode": "ocr", "text_layer_quality": "ocr",
            },
        ]
    }
    response = PdfResponse.model_validate(payload)
    assert len(response.pages[0].tables) == 1
    assert len(response.pages[1].tables) == 2
    assert len(response.tables) == 3
    assert [t.text for t in response.tables] == [
        "page1 table", "page2 table a", "page2 table b"
    ]
