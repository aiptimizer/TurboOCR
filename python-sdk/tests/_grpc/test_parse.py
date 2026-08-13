from __future__ import annotations

import pytest

pytest.importorskip("grpc")

from turboocr._grpc.parse import parse_ocr_response

from .conftest import build_ocr_response_proto


def test_fast_path_uses_json_response() -> None:
    """When json_response is populated, the SDK validates straight from
    those bytes — same shape as the HTTP body, no field-by-field copy."""
    proto = build_ocr_response_proto(with_json=True, with_results=False)
    parsed = parse_ocr_response(proto)
    assert parsed.results[0].text == "hello"
    # Layout/blocks come from the JSON, not from the repeated results field.
    assert parsed.layout[0].class_name == "paragraph_title"
    assert parsed.blocks[0].class_name == "paragraph_title"


def test_fallback_path_uses_results_field() -> None:
    """Response with empty json_response: structural translation kicks in."""
    proto = build_ocr_response_proto(with_json=False, with_results=True)
    parsed = parse_ocr_response(proto)
    assert parsed.results[0].text == "hello"
    assert parsed.results[0].confidence == pytest.approx(0.99)
    # Quad has 4 points
    assert len(parsed.results[0].bounding_box.points) == 4
    # Reading-order from proto repeated field carries through
    assert parsed.reading_order == [0]
    # Layout / blocks are not produced by the fallback path
    assert parsed.layout == []
    assert parsed.blocks == []
