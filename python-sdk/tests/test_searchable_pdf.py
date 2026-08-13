from __future__ import annotations

from io import BytesIO

import pypdf
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

from turboocr import OcrResponse, PdfResponse, ProtocolError
from turboocr.markdown import MarkdownStyle, NodeKind
from turboocr.searchable_pdf import (
    PDF_POINTS_PER_INCH,
    make_searchable_pdf,
)


def _blank_pdf(
    *, width_pt: float = letter[0], height_pt: float = letter[1], pages: int = 1
) -> bytes:
    buf = BytesIO()
    canvas = Canvas(buf, pagesize=(width_pt, height_pt))
    for _ in range(pages):
        canvas.showPage()
    canvas.save()
    return buf.getvalue()


def _pixel_box(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _pdf_response_with_text(text: str, *, dpi: int = 200) -> PdfResponse:
    width_pt, height_pt = letter
    page_w_px = int(width_pt * dpi / PDF_POINTS_PER_INCH)
    page_h_px = int(height_pt * dpi / PDF_POINTS_PER_INCH)
    box = _pixel_box(page_w_px // 4, page_h_px // 4, 3 * page_w_px // 4, page_h_px // 4 + 60)
    return PdfResponse.model_validate(
        {
            "pages": [
                {
                    "page": 1,
                    "page_index": 0,
                    "dpi": dpi,
                    "width": page_w_px,
                    "height": page_h_px,
                    "results": [
                        {"id": 0, "text": text, "confidence": 0.99, "bounding_box": box}
                    ],
                    "layout": [],
                    "reading_order": [0],
                    "blocks": [],
                    "mode": "ocr",
                    "text_layer_quality": "ocr",
                }
            ]
        }
    )


def test_make_searchable_pdf_round_trips_latin_text() -> None:
    pdf = _blank_pdf()
    response = _pdf_response_with_text("hello world")
    out = make_searchable_pdf(pdf, response)
    extracted = pypdf.PdfReader(BytesIO(out)).pages[0].extract_text()
    assert "hello world" in extracted


@pytest.mark.parametrize(
    "fmt,save_kwargs",
    [
        ("PNG", {}),
        ("JPEG", {"quality": 85}),
        ("BMP", {}),
        ("TIFF", {"compression": "tiff_lzw"}),
        ("GIF", {}),
        ("WEBP", {"quality": 80}),
    ],
)
def test_make_searchable_pdf_auto_detects_image_inputs(
    fmt: str, save_kwargs: dict[str, object]
) -> None:
    """Image inputs (PNG/JPEG/BMP/TIFF/GIF/WebP) wrap to PDF automatically."""
    from PIL import Image

    img = Image.new("RGB", (800, 600), color=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, fmt, **save_kwargs)
    image_bytes = buf.getvalue()

    response = PdfResponse.model_validate(
        {
            "pages": [
                {
                    "page": 1,
                    "page_index": 0,
                    "dpi": 200,
                    "width": 800,
                    "height": 600,
                    "results": [
                        {
                            "id": 0,
                            "text": "abc",
                            "confidence": 0.9,
                            "bounding_box": _pixel_box(10, 10, 60, 30),
                        }
                    ],
                    "layout": [],
                    "reading_order": [0],
                    "blocks": [],
                    "mode": "ocr",
                    "text_layer_quality": "ocr",
                }
            ]
        }
    )

    out = make_searchable_pdf(image_bytes, response, dpi=200)
    assert out.startswith(b"%PDF-"), f"{fmt} did not produce a PDF"
    extracted = pypdf.PdfReader(BytesIO(out)).pages[0].extract_text()
    assert "abc" in extracted, f"{fmt} text did not round-trip"


def test_make_searchable_pdf_rejects_page_count_mismatch() -> None:
    pdf = _blank_pdf(pages=2)
    response = _pdf_response_with_text("only one")
    with pytest.raises(ValueError, match="2 pages but OCR response has 1"):
        make_searchable_pdf(pdf, response)


def test_make_searchable_pdf_from_ocr_response_requires_dpi() -> None:
    pdf = _blank_pdf()
    ocr = OcrResponse.model_validate(
        {
            "results": [
                {
                    "text": "x",
                    "confidence": 0.9,
                    "bounding_box": _pixel_box(0, 0, 50, 20),
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="dpi must be provided"):
        make_searchable_pdf(pdf, ocr)
    out = make_searchable_pdf(pdf, ocr, dpi=200)
    assert out.startswith(b"%PDF-")


def test_glyphless_font_handles_non_latin_out_of_the_box() -> None:
    """Bundled GlyphLessFont covers all of BMP — no setup needed."""
    pdf = _blank_pdf()
    response = _pdf_response_with_text("héllo wörld 北京 مرحبا")
    out = make_searchable_pdf(pdf, response)
    assert out.startswith(b"%PDF-")
    extracted = pypdf.PdfReader(BytesIO(out)).pages[0].extract_text()
    # The text layer round-trips through the cmap; we don't assert exact
    # equality (pypdf normalisation can reflow whitespace) but at least
    # each script should produce something extractable.
    assert extracted.strip()


def test_block_model_dump_round_trip() -> None:
    response = OcrResponse.model_validate(
        {
            "results": [],
            "blocks": [
                {
                    "id": 0,
                    "layout_id": 0,
                    "class": "paragraph_title",
                    "bounding_box": _pixel_box(0, 0, 10, 10),
                    "content": "Heading",
                    "order_index": 0,
                }
            ],
        }
    )
    payload = response.blocks[0].model_dump_json(by_alias=True)
    assert '"content":"Heading"' in payload
    assert '"class":"paragraph_title"' in payload


def test_degenerate_bbox_raises_protocol_error() -> None:
    pdf = _blank_pdf()
    response = PdfResponse.model_validate(
        {
            "pages": [
                {
                    "page": 1, "page_index": 0, "dpi": 200, "width": 100, "height": 100,
                    "results": [
                        {
                            "id": 0,
                            "text": "x",
                            "confidence": 0.9,
                            "bounding_box": [[10, 10], [10, 10], [10, 10], [10, 10]],
                        }
                    ],
                    "layout": [], "reading_order": [0], "blocks": [],
                    "mode": "ocr", "text_layer_quality": "ocr",
                }
            ]
        }
    )
    with pytest.raises(ProtocolError):
        make_searchable_pdf(pdf, response)


def test_markdown_style_registers_custom_label() -> None:
    style = MarkdownStyle()
    style.register("invoice_total", NodeKind.heading, level=2)
    assert style.classify("invoice_total").kind is NodeKind.heading
    assert style.classify("invoice_total").level == 2
    assert style.classify("unknown_label").kind is NodeKind.paragraph
