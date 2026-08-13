from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from importlib.resources import as_file, files
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pypdf
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas

from .errors import ProtocolError
from .models import OcrResponse, PdfPage, PdfResponse, TextItem

if TYPE_CHECKING:
    from reportlab.pdfgen.canvas import Canvas


logger = logging.getLogger("turboocr.searchable_pdf")

PDF_POINTS_PER_INCH: Final[float] = 72.0
INVISIBLE_TEXT_MODE: Final[int] = 3

# Bundled glyphless font: one zero-mark glyph that every BMP codepoint
# (U+0001..U+FFFF) maps to. Same trick Tesseract's GlyphLessFont uses —
# the visible page is the original scan, so the font is only needed so
# PDF readers can compute text-selection bboxes. ~760 bytes; ships in
# the wheel via package-data.
GLYPHLESS_FONT_NAME: Final[str] = "TurboOcrGlyphless"
_GLYPHLESS_FONT_FILE: Final[str] = "glyphless.ttf"

# reportlab's pdfmetrics holds a process-wide global font registry. Multiple
# threads calling make_searchable_pdf concurrently would race the check +
# register pair. Lock + idempotent check protects against double registration.
_REGISTRATION_LOCK: Final[threading.Lock] = threading.Lock()


_PDF_MAGIC: Final[bytes] = b"%PDF-"


@dataclass(frozen=True, slots=True)
class _OverlayPage:
    width_pt: float
    height_pt: float
    dpi: int
    items: list[TextItem]


def _wrap_image_as_pdf(image_bytes: bytes, *, dpi: int) -> bytes:
    """Wrap a single image (JPEG/PNG/TIFF/BMP/…) as a one-page PDF.

    The page is sized so that the image fills it at the given DPI:
    `page_width_pt = pixels * 72 / dpi`. The image is drawn at full
    bleed so the OCR bounding boxes (which are in pixel coordinates at
    the source resolution) land in the right place.
    """
    reader = ImageReader(BytesIO(image_bytes))
    width_px, height_px = reader.getSize()
    width_pt = width_px * PDF_POINTS_PER_INCH / dpi
    height_pt = height_px * PDF_POINTS_PER_INCH / dpi
    buf = BytesIO()
    canvas = rl_canvas.Canvas(buf, pagesize=(width_pt, height_pt))
    canvas.drawImage(reader, 0, 0, width=width_pt, height=height_pt)
    canvas.showPage()
    canvas.save()
    return buf.getvalue()


class FontError(RuntimeError):
    """Raised when a caller-supplied font cannot render the OCR text.

    The default code path never raises this — the bundled glyphless font
    covers every Basic Multilingual Plane codepoint. You can only hit it
    by passing `font_path=<my.ttf>` to a font that lacks glyphs the OCR
    text needs.
    """


class FontGlyphMissing(FontError):
    """A user-supplied font has no glyph for some OCR character."""


def _register_glyphless_font() -> str:
    """Register the bundled glyphless TTF with reportlab. Idempotent."""
    with _REGISTRATION_LOCK:
        if GLYPHLESS_FONT_NAME in pdfmetrics.getRegisteredFontNames():
            return GLYPHLESS_FONT_NAME
        font_resource = files("turboocr._data").joinpath(_GLYPHLESS_FONT_FILE)
        with as_file(font_resource) as font_path:
            pdfmetrics.registerFont(TTFont(GLYPHLESS_FONT_NAME, str(font_path)))
    return GLYPHLESS_FONT_NAME


def _register_custom_font(font_path: str) -> str:
    """Register a user-supplied TTF under its file stem."""
    name = Path(font_path).stem or "TurboOcrCustomFont"
    with _REGISTRATION_LOCK:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, font_path))
    return name


def _resolve_font(font_path: str | None) -> str:
    if font_path:
        logger.debug("turbo-ocr searchable_pdf using custom font %s", font_path)
        return _register_custom_font(font_path)
    return _register_glyphless_font()


def _draw_invisible_item(
    canvas: Canvas, item: TextItem, *, font_name: str, dpi: int, page_height_pt: float
) -> None:
    if not item.text.strip():
        logger.debug("skipping whitespace-only OCR item id=%s", item.id)
        return
    x0, y0, x1, y1 = item.bounding_box.aabb
    box_w_pt = _px_to_pt(x1 - x0, dpi)
    box_h_pt = _px_to_pt(y1 - y0, dpi)
    if box_w_pt <= 0 or box_h_pt <= 0:
        raise ProtocolError(
            f"degenerate bbox {item.bounding_box.aabb} for OCR item id={item.id}"
        )

    text_pt_x = _px_to_pt(x0, dpi)
    text_pt_y = page_height_pt - _px_to_pt(y1, dpi)
    font_size = max(1.0, box_h_pt * 0.9)
    canvas.setFont(font_name, font_size)
    text_width = canvas.stringWidth(item.text, font_name, font_size)
    if text_width <= 0:
        raise FontGlyphMissing(
            f"font {font_name!r} cannot render OCR item id={item.id} "
            f"(text={item.text!r}); drop the font_path= override to fall "
            "back to the bundled glyphless font"
        )

    canvas.saveState()
    canvas.translate(text_pt_x, text_pt_y)
    canvas.scale(box_w_pt / text_width, 1.0)
    text_obj = canvas.beginText(0, 0)
    text_obj.setTextRenderMode(INVISIBLE_TEXT_MODE)
    text_obj.textOut(item.text)
    canvas.drawText(text_obj)
    canvas.restoreState()


def _px_to_pt(px: float, dpi: int) -> float:
    return px * PDF_POINTS_PER_INCH / dpi


def _build_overlay_pdf(pages: list[_OverlayPage], font_name: str) -> bytes:
    buffer = BytesIO()
    pdf = rl_canvas.Canvas(buffer)
    for page in pages:
        pdf.setPageSize((page.width_pt, page.height_pt))
        for item in page.items:
            _draw_invisible_item(
                pdf, item, font_name=font_name, dpi=page.dpi, page_height_pt=page.height_pt
            )
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _items_for_page(page: PdfPage) -> list[TextItem]:
    """Order a page's text items by reading_order, falling back to natural order.

    Validates that reading_order is a complete permutation of result indices
    (every index present exactly once). On any drift — out-of-range index,
    duplicate, or missing index — logs a warning and returns the unordered
    `page.results` list. Silently using a partial reading_order would lose or
    duplicate text in the invisible PDF layer.
    """
    if not page.reading_order:
        return list(page.results)
    n = len(page.results)
    if sorted(page.reading_order) != list(range(n)):
        logger.warning(
            "page %d reading_order is not a permutation of results "
            "(len=%d, results=%d) — falling back to natural order to avoid "
            "lost/duplicated text",
            page.page, len(page.reading_order), n,
        )
        return list(page.results)
    return [page.results[i] for i in page.reading_order]


def _coerce_to_pdf_response(
    response: PdfResponse | OcrResponse,
    *,
    dpi: int | None,
    page_width_pt: float | None = None,
    page_height_pt: float | None = None,
) -> PdfResponse:
    if isinstance(response, PdfResponse):
        return response
    if dpi is None:
        raise ValueError("dpi must be provided when overlaying an OcrResponse")
    # Pixel dimensions at the rendered DPI. When called from make_searchable_pdf
    # we pass the source PDF's mediabox so introspection of page.width/height
    # returns truthful values; when called standalone we fall back to 0 — the
    # overlay loop uses mediabox directly and doesn't read these fields.
    width_px = int(page_width_pt * dpi / PDF_POINTS_PER_INCH) if page_width_pt else 0
    height_px = int(page_height_pt * dpi / PDF_POINTS_PER_INCH) if page_height_pt else 0
    return PdfResponse(
        pages=[
            PdfPage(
                page=1,
                page_index=0,
                dpi=dpi,
                width=width_px,
                height=height_px,
                results=response.results,
                layout=response.layout,
                reading_order=response.reading_order,
                blocks=response.blocks,
                mode="ocr",  # type: ignore[arg-type]
                text_layer_quality="ocr",
            )
        ]
    )


def make_searchable_pdf(
    original: bytes,
    response: PdfResponse | OcrResponse,
    *,
    dpi: int | None = None,
    font_path: str | None = None,
) -> bytes:
    """Overlay an invisible OCR text layer on the input.

    Accepts a PDF or any single-page image. Tested input formats: PDF,
    PNG, JPEG, BMP, TIFF, GIF, WebP. Image inputs are wrapped into a
    single-page PDF first, sized to the image's pixel dimensions at
    `dpi`. The detection is by magic bytes, so the caller does not
    have to tell the function which format the input is in.

    By default uses a bundled glyphless font that covers every Basic
    Multilingual Plane codepoint, so non-Latin scans (CJK, Arabic, Cyrillic,
    …) work out of the box with zero configuration.

    Pass `font_path=<my.ttf>` only if you have a specific reason to embed a
    real visible font instead.
    """
    if not original.startswith(_PDF_MAGIC):
        if dpi is None:
            raise ValueError(
                "dpi must be provided when overlaying an image input"
            )
        original = _wrap_image_as_pdf(original, dpi=dpi)
    reader = pypdf.PdfReader(BytesIO(original))
    if isinstance(response, OcrResponse) and reader.pages:
        media = reader.pages[0].mediabox
        pdf_response = _coerce_to_pdf_response(
            response, dpi=dpi,
            page_width_pt=float(media.width),
            page_height_pt=float(media.height),
        )
    else:
        pdf_response = _coerce_to_pdf_response(response, dpi=dpi)
    if len(reader.pages) != len(pdf_response.pages):
        hint = ""
        if isinstance(response, OcrResponse) and len(reader.pages) > 1:
            hint = (
                " — you passed an OcrResponse (single-image OCR) for a "
                "multi-page PDF; call client.recognize_pdf() to get a "
                "PdfResponse instead"
            )
        raise ValueError(
            f"PDF has {len(reader.pages)} pages but OCR response has "
            f"{len(pdf_response.pages)}{hint}"
        )

    items_per_page = [_items_for_page(p) for p in pdf_response.pages]
    font_name = _resolve_font(font_path)

    overlay_pages: list[_OverlayPage] = []
    for original_page, ocr_page, items in zip(
        reader.pages, pdf_response.pages, items_per_page, strict=True
    ):
        media = original_page.mediabox
        overlay_pages.append(
            _OverlayPage(
                width_pt=float(media.width),
                height_pt=float(media.height),
                dpi=ocr_page.dpi,
                items=items,
            )
        )

    overlay_bytes = _build_overlay_pdf(overlay_pages, font_name)
    overlay_reader = pypdf.PdfReader(BytesIO(overlay_bytes))
    writer = pypdf.PdfWriter(clone_from=reader)
    for writer_page, overlay_page in zip(writer.pages, overlay_reader.pages, strict=True):
        writer_page.merge_page(overlay_page)

    out = BytesIO()
    writer.write(out)
    return out.getvalue()
