"""Build a searchable PDF. Two page forms, decided per page:

* a page WITH a raster becomes the image with an INVISIBLE OCR text layer on
  top (render mode 3) — looks identical to the scan, text selectable and
  searchable: the ocrmypdf/Acrobat "OCR a PDF" deliverable;
* a page WITHOUT a raster (a text-layer page from ``mode="auto"``/``"text"``,
  or an OCR'd page read with ``keep_image=False``) becomes a VISIBLE
  text-only page — a re-typeset rendering, NOT a facsimile of the source.
  For guaranteed image+invisible-text output use ``pdf_to_searchable()``
  (which pins ``mode="ocr"`` and keeps rasters) or read with
  ``keep_image=True`` and ``mode="ocr"``.

Invisible runs are positioned by each line's bounding box and horizontally
scaled to the box width so search highlights track the glyphs. The page media
box is sized in real points from the render DPI. Assumes axis-aligned /
near-horizontal lines (rotated lines use their AABB); a later upgrade can
overlay onto the original vector page instead of a raster.

Not PDF/A-safe: the text layer uses reportlab's non-embedded ``STSong-Light``
CID font (viewer-substituted). For archival/PDF-A conformance, embed a glyphless
Unicode font instead. reportlab ships with the engine wheel (4.0.0a6+).
"""

from __future__ import annotations

import io
import os
import tempfile
import warnings
from typing import Iterable, Optional

import cv2


def _reportlab():
    try:
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas

        return canvas, ImageReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "searchable PDF needs reportlab — it ships with the engine wheel "
            "since 4.0.0a6, so this environment is missing it unusually: "
            "`pip install reportlab` restores it."
        ) from exc


_TEXT_FONT: Optional[str] = None


def _text_font() -> str:
    """Font for the invisible text layer.

    Helvetica only encodes Latin-1, so it mangles CJK/Korean/Cyrillic/Greek/Thai
    into unsearchable boxes. reportlab's bundled ``STSong-Light`` UnicodeCIDFont
    round-trips all of those correctly (verified via text re-extraction) — so we
    use it for every script. (Arabic and other RTL scripts extract in visual
    order, a known limitation.) Falls back to Helvetica if the CID font is
    unavailable in this reportlab build."""
    global _TEXT_FONT
    if _TEXT_FONT is not None:
        return _TEXT_FONT
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        _TEXT_FONT = "STSong-Light"
    except Exception:
        _TEXT_FONT = "Helvetica"
    return _TEXT_FONT


def build_searchable_pdf(pages: Iterable, out_path: Optional[str] = None) -> Optional[bytes]:
    """Render ``pages`` (PageResults carrying ``.image``) into a searchable PDF.

    ``pages`` may be a list OR a lazy generator — pages are consumed one at a
    time, so a streaming OCR run never has to hold every page raster at once.
    Writes to ``out_path`` if given (returns None), else returns the PDF bytes.

    A page WITH a raster becomes image + invisible text; a page WITHOUT one
    (a text-layer page from ``mode="auto"``/``"text"``, or a page contained
    by ``on_error="skip"``) becomes a visible text-only page. Only when NO
    page carries a raster while OCR-sourced lines exist — the signature of a
    read that dropped its rasters (``keep_image=False`` is the PDF/batch
    default) — does this refuse, naming the fix."""
    canvas, ImageReader = _reportlab()
    font = _text_font()

    # Write straight to a temp file next to out_path (atomic os.replace on
    # success — a mid-run crash never leaves a truncated .pdf), or to memory
    # when returning bytes. Either way reportlab buffers the whole doc once.
    tmp_path = None
    if out_path:
        d = os.path.dirname(os.path.abspath(out_path)) or "."
        fd, tmp_path = tempfile.mkstemp(dir=d, suffix=".pdf.part")
        os.close(fd)
        dest: object = tmp_path
    else:
        dest = io.BytesIO()

    c = None
    n = 0
    saw_image = False
    saw_ocr_lines = False
    any_lines = False
    warned_blank = False
    last_size_px = None  # (w, h): fallback for degenerate (failed-page) dims
    try:
        for page in pages:
            if getattr(page, "lines", None) is None:
                continue
            n += 1
            img = getattr(page, "image", None)
            saw_image = saw_image or img is not None
            saw_ocr_lines = saw_ocr_lines or any(
                getattr(ln, "source", "") != "pdf" for ln in page.lines
            )
            any_lines = any_lines or bool(page.lines)
            # px -> pt using the page's render DPI (1pt = 1/72"), so a 150-DPI
            # letter scan makes an 8.5x11" page, not a 17x22" one.
            dpi = float(getattr(page, "dpi", None) or 72.0)
            s = 72.0 / dpi
            if img is not None:
                h_px, w_px = img.shape[:2]
                raster = img
            else:
                if not warned_blank and any(
                    getattr(ln, "source", "") != "pdf" for ln in page.lines
                ):
                    # Only OCR'd-but-rasterless pages get the warning; a
                    # text-layer page never had a raster and a text-only page
                    # is its correct rendering, not a degradation.
                    warnings.warn(
                        "searchable PDF: an OCR'd page has no image (read "
                        "with keep_image=True). Writing it as a text-only "
                        "page.",
                        stacklevel=3,
                    )
                    warned_blank = True
                h_px, w_px = int(page.height or 0), int(page.width or 0)
                if h_px <= 1 or w_px <= 1:
                    # A page contained by on_error="skip" carries 0x0 dims —
                    # rendered literally that was a 0.48pt speck of a page.
                    # Reuse the document's running page size, else US-letter.
                    if last_size_px is not None:
                        w_px, h_px = last_size_px
                    else:
                        w_px = round(612 * dpi / 72.0)
                        h_px = round(792 * dpi / 72.0)
                raster = None
            last_size_px = (w_px, h_px)

            w_pt, h_pt = w_px * s, h_px * s
            if c is None:
                c = canvas.Canvas(dest, pagesize=(w_pt, h_pt))
            else:
                c.setPageSize((w_pt, h_pt))

            if raster is not None:
                ok, png = cv2.imencode(".png", raster)  # BGR in, correct colors
                if ok:
                    c.drawImage(
                        ImageReader(io.BytesIO(png.tobytes())),
                        0, 0, width=w_pt, height=h_pt,
                    )

            # Invisible text layer. PDF origin is bottom-left, so flip y. Font
            # size ~0.85x box height reads truer; horizontal scale matches the
            # invisible run's width to the box so search highlights track glyphs.
            render_mode = 3 if raster is not None else 0  # visible if no raster
            for ln in page.lines:
                text = ln.text
                if not text.strip():
                    continue
                # UNLIKE the C++ writer (pdf_searchable_encoding.cpp keep()),
                # source=="pdf" lines are NOT skipped here: the C++ path
                # overlays onto the ORIGINAL page, whose own text layer stays
                # selectable — this writer builds a NEW document from rasters
                # (or text-only pages), so the original layer is gone and
                # skipping its lines silently produced BLANK pages for every
                # text-layer page of a mode="auto" document.
                x0, y0, x1, y1 = ln.bbox
                box_w_pt = max(1.0, (x1 - x0) * s)
                font_size = max(1.0, (y1 - y0) * s * 0.85)
                try:
                    tw = c.stringWidth(text, font, font_size)
                    scale = (box_w_pt / tw * 100.0) if tw > 0 else 100.0
                    to = c.beginText()
                    to.setTextRenderMode(render_mode)
                    to.setFont(font, font_size)
                    to.setHorizScale(scale)
                    to.setTextOrigin(x0 * s, (h_px - y1) * s)
                    to.textLine(text)
                    c.drawText(to)
                except Exception:
                    # A pathological glyph must not sink the whole document.
                    continue

            c.showPage()

        if c is None or n == 0:
            raise ValueError("no pages to write")
        if not saw_image and not any_lines:
            # No rasters AND no text anywhere: N blank white pages would be
            # the output — a failed/blank scan read with the keep_image=False
            # default. Nothing meaningful to write; say why.
            raise ValueError(
                f"searchable PDF: nothing to write — none of the {n} pages "
                "carries a raster or any recognized text. Check "
                "PageResult.warnings, and read with keep_image=True if you "
                "want the page images embedded regardless."
            )
        if not saw_image and saw_ocr_lines:
            # The dropped-raster signature: OCR produced lines but no page
            # kept its pixels — since 4.0.0a6 read_pdf/read_batch default to
            # keep_image=False. (An all-text-layer document is NOT an error:
            # it was written above as text-only pages — there never were
            # rasters to embed.)
            raise ValueError(
                f"searchable PDF: none of the {n} pages carries its raster — "
                "re-read with keep_image=True AND mode=\"ocr\" (mode=\"auto\" "
                "serves text-layer pages without rendering), or use "
                "pdf_to_searchable(), which does both automatically."
            )
        c.save()
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if out_path:
        assert tmp_path is not None  # set whenever out_path is (narrow for mypy)
        os.replace(tmp_path, out_path)  # atomic
        return None
    return dest.getvalue()  # type: ignore[attr-defined]
