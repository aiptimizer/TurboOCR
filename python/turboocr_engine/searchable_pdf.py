"""Build a searchable PDF: the rendered page image with an INVISIBLE OCR text
layer on top, so the output looks identical to the scan but the text can be
selected, searched, and copied — the ocrmypdf/Acrobat "OCR a PDF" deliverable.

v1 rasterizes each page (from the kept page image) and overlays invisible text
(PDF text render mode 3), positioned by each line's bounding box and
horizontally scaled to the box width so search highlights track the glyphs.
The page media box is sized in real points from the render DPI. Assumes
axis-aligned / near-horizontal lines (rotated lines use their AABB); a later
upgrade can overlay onto the original vector page instead of a raster.

Not PDF/A-safe: the text layer uses reportlab's non-embedded ``STSong-Light``
CID font (viewer-substituted). For archival/PDF-A conformance, embed a glyphless
Unicode font instead. Requires reportlab (``pip install "turboocr[cpu,pdf]"``).
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
            "searchable PDF needs reportlab — `pip install \"turboocr[cpu,pdf]\"` "
            "(or `pip install reportlab`)."
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
    Each page should carry its source image (``keep_image=True``, the default)."""
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
    warned_blank = False
    try:
        for page in pages:
            if getattr(page, "lines", None) is None:
                continue
            n += 1
            img = getattr(page, "image", None)
            # px -> pt using the page's render DPI (1pt = 1/72"), so a 150-DPI
            # letter scan makes an 8.5x11" page, not a 17x22" one.
            dpi = float(getattr(page, "dpi", None) or 72.0)
            s = 72.0 / dpi
            if img is not None:
                h_px, w_px = img.shape[:2]
                raster = img
            else:
                if not warned_blank:
                    warnings.warn(
                        "searchable PDF: a page has no image (build the OCR with "
                        "keep_image=True). Emitting a text-only page.",
                        stacklevel=3,
                    )
                    warned_blank = True
                h_px, w_px = int(page.height or 1), int(page.width or 1)
                raster = None

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
                # Same keep-predicate as the C++ writer
                # (pdf_searchable_encoding.cpp keep()): a line whose source is
                # the PDF's OWN text layer must NOT be stamped again — the
                # original text is already selectable, so re-stamping doubled
                # every search hit and copy-pasted every word twice on any PDF
                # that arrived with a text layer.
                if getattr(ln, "source", "") == "pdf":
                    continue
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
