"""PDF rendering via PDFium (pypdfium2).

The C++ engine renders PDF pages with a pooled ``fastpdf2png``/PDFium daemon;
the Python bindings use ``pypdfium2`` — the maintained Python binding to the
same PDFium library, shipped as a self-contained wheel (the PDFium shared
library is bundled, nothing to build). Pages are rendered to BGR arrays at the
requested DPI and handed straight to the OCR pipeline.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .imaging import ImageInput

_PDF_DEFAULT_DPI = 150


def _import_pdfium():
    try:
        import pypdfium2 as pdfium  # type: ignore

        return pdfium
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PDF support needs pypdfium2. Install it with "
            "`pip install \"turboocr[cpu,pdf]\"` (or `pip install pypdfium2`)."
        ) from exc


def _open_document(pdf: ImageInput):
    pdfium = _import_pdfium()
    if isinstance(pdf, (bytes, bytearray, memoryview)):
        return pdfium.PdfDocument(bytes(pdf))
    if isinstance(pdf, (str, os.PathLike)):
        return pdfium.PdfDocument(os.fspath(pdf))
    raise TypeError(f"unsupported PDF input type: {type(pdf)!r}")


def render_pdf(
    pdf: ImageInput,
    *,
    dpi: int = _PDF_DEFAULT_DPI,
    pages: Optional[List[int]] = None,
    max_pages: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(page_number, bgr_array)`` for each requested page.

    ``pages`` is a 1-based list (``None`` => all pages); ``max_pages`` caps how
    many are rendered. DPI maps to a PDFium scale of ``dpi/72``.
    """
    doc = _open_document(pdf)
    try:
        n = len(doc)
        if pages is None:
            wanted = list(range(1, n + 1))
        else:
            wanted = [p for p in pages if 1 <= p <= n]
        if max_pages is not None:
            wanted = wanted[:max_pages]

        scale = dpi / 72.0
        for page_no in wanted:
            page = doc[page_no - 1]
            try:
                bitmap = page.render(
                    scale=scale,
                    rev_byteorder=False,  # PDFium native order is BGR(A)
                    fill_color=(255, 255, 255, 255),  # flatten transparency to white
                )
                arr = bitmap.to_numpy()
                bgr = _to_bgr(arr)
                # MUST copy: `bgr` may be a view into the PDFium bitmap buffer,
                # which is freed by bitmap.close() below as the generator
                # advances. Own the pixels.
                yield page_no, np.array(bgr, copy=True)
            finally:
                try:
                    bitmap.close()  # type: ignore[has-type]
                except Exception:
                    pass
                page.close()
    finally:
        doc.close()


def _to_bgr(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:  # grayscale
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:  # BGRA -> BGR
        return arr[:, :, :3]
    if arr.shape[2] == 3:  # BGR already
        return arr
    raise ValueError(f"unexpected PDFium bitmap shape {arr.shape}")


def pdf_page_count(pdf: ImageInput) -> int:
    doc = _open_document(pdf)
    try:
        return len(doc)
    finally:
        doc.close()
