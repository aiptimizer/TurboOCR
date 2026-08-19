"""PDF rendering via PDFium (pypdfium2).

The C++ engine renders PDF pages with a pooled ``fastpdf2png``/PDFium daemon;
the Python bindings use ``pypdfium2`` — the maintained Python binding to the
same PDFium library, shipped as a self-contained wheel (the PDFium shared
library is bundled, nothing to build). Pages are rendered to BGR arrays at the
requested DPI and handed straight to the OCR pipeline.

Text-layer extraction ports the C++ extractor's design (char-flow line
grouping, control-character stripping, /Rotate + MediaBox-origin handling —
src/pdf/text/pdf_text_extract.cpp + pdf_text_internal.h). Known, deliberate
divergences from the server, all in the stricter direction here:

* trust threshold ~50 visible chars vs the server's 10 — the gate here only
  runs when a caller explicitly opts into mode="auto" (the default on both
  sides is OCR), and a thin layer is exactly the case worth re-OCR'ing
  (_AUTO_TRUST_MIN_CHARS);
* char_count semantics: this side counts visible CODE POINTS (\r\n
  excluded, a surrogate pair = 1) where the server's gate still counts raw
  UTF-16 units — so the ratio gates bite slightly earlier here;
* nonprint counting reads the GetUnicode stream (stricter: wrap-hyphen
  control glyphs count) where the server reads GetText's normalized one.

The whitespace box-union rule is shared (both sides exclude space/tab/NBSP
from the line bbox). Line SETS are identical across the corpus; keep the
check STRUCTURE aligned when either side changes.
"""

from __future__ import annotations

import math
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .imaging import ImageInput, _PDF_SNIFF_WINDOW, looks_like_pdf
from .options import DEFAULT_DPI as _PDF_DEFAULT_DPI
from .options import OnError, PdfMode, check_max_pages, check_on_error, check_pdf_mode

# PDFium is NOT thread-safe — not per document but GLOBALLY: concurrent calls
# on DIFFERENT documents from different threads crash (reproduced: a
# one-document-per-worker text extractor segfaulted the interpreter; the
# pypdfium2 FAQ says the same). Every pdfium touchpoint below therefore
# serializes behind this one process-wide lock. That is not the performance
# loss it sounds like: text extraction is pure C at hundreds of pages/s, page
# RENDERING (~5 ms) already happens on the single producer thread of the
# read_pdf fan-out, and OCR — the actual cost — runs outside the lock. It is
# also what makes `asyncio.gather` over several `aread_pdf` calls safe, which
# previously raced pdfium across worker threads.
import threading

_PDFIUM_LOCK = threading.RLock()


def _import_pdfium():
    try:
        import pypdfium2 as pdfium  # type: ignore

        return pdfium
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PDF support needs pypdfium2 — it ships with the engine wheel "
            "since 4.0.0a6, so this environment is missing it unusually: "
            "`pip install pypdfium2` restores it."
        ) from exc


def _require_pdf(head: bytes, what: str) -> None:
    """The mirror of load_image's PDF sniff (same shared magic check): a
    non-PDF handed to the PDF reader gets pointed at read()/read_batch()
    instead of whatever opaque parse error PDFium would raise."""
    if not looks_like_pdf(head):
        raise ValueError(
            f"{what} is not a PDF (no %PDF- header) — read_pdf() renders PDF "
            "documents. For images use read() / read_batch()."
        )


def _open_document(pdf: ImageInput, password: Optional[str] = None):
    """Open with the sniff guard. CALLER HOLDS _PDFIUM_LOCK (or is the only
    pdfium user, e.g. single-threaded scripts). ``password`` unlocks an
    encrypted document (PDFium accepts the user or the owner password).

    PDFium's own open errors are wrapped into ValueError: the raw
    ``PdfiumError`` names its GLOBAL last-error slot, which is stale across
    calls — the same broken document reported "(PDFium: Success)" in one run
    and "(PDFium: Incorrect password error)" in the next. Our message carries
    the actionable hint instead of trusting that slot."""
    if password is not None and not isinstance(password, str):
        raise ValueError(
            f"password must be a str, got {type(password).__name__}"
        )
    pdfium = _import_pdfium()

    def _open(target, what: str):
        try:
            doc = pdfium.PdfDocument(target, password=password)
        except pdfium.PdfiumError as exc:
            hint = (
                " (wrong password?)" if password is not None
                else " — if the PDF is encrypted, pass password="
            )
            raise ValueError(
                f"could not open {what}: {exc}{hint}"
            ) from exc
        if len(doc) == 0:
            doc.close()
            raise ValueError(f"{what}: the PDF contains no pages")
        return doc

    if isinstance(pdf, (bytes, bytearray, memoryview)):
        data = bytes(pdf)
        _require_pdf(data[:_PDF_SNIFF_WINDOW], "these bytes")
        return _open(data, "these bytes")
    if isinstance(pdf, (str, os.PathLike)):
        path = os.fspath(pdf)
        # Reading the head ourselves also turns a missing file into a plain
        # FileNotFoundError instead of PDFium's opaque load failure.
        with open(path, "rb") as fh:
            _require_pdf(fh.read(_PDF_SNIFF_WINDOW), path)
        return _open(path, path)
    raise TypeError(f"unsupported PDF input type: {type(pdf)!r}")


def _wanted_pages(doc, pages: Optional[List[int]], max_pages: Optional[int]) -> List[int]:
    """CALLER HOLDS _PDFIUM_LOCK: ``len(doc)`` lazily parses the page tree in
    PDFium (it is not a cached attribute), so it is a pdfium touchpoint like
    any other."""
    check_max_pages(max_pages)
    n = len(doc)
    wanted = list(range(1, n + 1)) if pages is None else [p for p in pages if 1 <= p <= n]
    if pages is not None and pages and not wanted:
        # An explicit page selection that matches NOTHING would return an
        # empty DocumentResult indistinguishable from a blank document.
        raise ValueError(
            f"requested pages {pages!r} are all outside this document "
            f"(it has {n} page{'s' if n != 1 else ''}; pages are 1-based)"
        )
    return wanted[:max_pages] if max_pages is not None else wanted


def _render_page(page, scale: float) -> np.ndarray:
    bitmap = None
    try:
        bitmap = page.render(
            scale=scale,
            rev_byteorder=False,  # PDFium native order is BGR(A)
            fill_color=(255, 255, 255, 255),  # flatten transparency to white
        )
        arr = bitmap.to_numpy()
        # MUST copy: the array may be a view into the PDFium bitmap buffer,
        # which is freed by bitmap.close(). Own the pixels.
        return np.array(_to_bgr(arr), copy=True)
    finally:
        try:
            if bitmap is not None:
                bitmap.close()
        except Exception:
            pass


#: One extracted text-layer line: (text, quad in rendered-pixel space).
TextLayerLine = Tuple[str, Tuple[Tuple[int, int], ...]]

#: Per-page text-layer stats feeding the quality gate:
#: (visible_char_count, fffd_count, nonprint_count, rotation_deg).
#: visible = code points excluding the generated line breaks; a surrogate
#: pair counts once.
LayerStats = Tuple[int, int, int, int]


def _extract_page_text(
    page, scale: float
) -> Tuple[int, int, List[TextLayerLine], LayerStats]:
    """One page's EMBEDDED text layer as ``(width_px, height_px, lines,
    stats)`` in the same pixel space a render at this dpi would produce.

    Port of the C++ extractor, with its three deliberate choices:

    * lines come from PDFium's own char-flow segmentation (the generated
      ``\\r\\n`` in the char stream), NOT ``FPDFText_CountRects`` — rects are
      per same-font/style RUN, so a mid-line font change fragments one visual
      line into several, and nested run rects DUPLICATE text;
    * control characters (< U+0020 except tab) are dropped: soft/wrap hyphens
      and unmapped ligature glyphs report low control codes — dropping them
      rejoins the word instead of embedding a NUL or tofu;
    * boxes are transformed from PDFium's pre-rotation, y-up, origin-offset
      space to visual top-left pixel space (all four corners, AABB), so
      ``/Rotate`` pages and trimmed MediaBox origins land where the RENDER
      puts the ink — the naive ``h - y`` flip was wrong for all of 90/180/270
      and for any non-zero origin.

    ``stats`` feeds :func:`_layer_quality`. Page pixel dims use ``ceil`` to
    match how pypdfium2 sizes the render bitmap, so text pages and rendered
    pages of one document report identical sizes.
    """
    import ctypes

    import pypdfium2.raw as C

    # get_size() (FPDF_GetPageWidthF/HeightF) reports POST-rotation extents;
    # char boxes are in PRE-rotation space, so the transform needs the pre
    # dims (swap back for 90/270). The visual page size IS the post-rotation
    # size. The mapping below is derived EMPIRICALLY against rendered ink for
    # all four /Rotate values (FPDFPage_GetRotation=1 -> (y, x); =3 ->
    # (Hpre-y, Wpre-x)) and MATCHES the C++ header's pre_to_visual
    # (pdf_text_internal.h), whose 90/270 cases were ink-verified and fixed
    # to the same assignment — keep the two in lockstep.
    vis_w_pt, vis_h_pt = (float(v) for v in page.get_size())
    rot = (int(C.FPDFPage_GetRotation(page)) % 4) * 90
    if rot % 180 == 0:
        Wp, Hp = vis_w_pt, vis_h_pt
    else:
        Wp, Hp = vis_h_pt, vis_w_pt
    ox = oy = 0.0
    rect = C.FS_RECTF()
    if C.FPDF_GetPageBoundingBox(page, rect):
        ox, oy = float(rect.left), float(rect.bottom)

    def pre_to_visual(x: float, y: float) -> Tuple[float, float]:
        x -= ox
        y -= oy
        if rot == 90:
            return y, x
        if rot == 180:
            return Wp - x, y
        if rot == 270:
            return Hp - y, Wp - x
        return x, Hp - y

    tp = page.get_textpage()
    try:
        n = int(C.FPDFText_CountChars(tp))
        fffd = nonprint = 0
        lines: List[TextLayerLine] = []
        buf: List[str] = []
        cl = ctypes.c_double()
        cr = ctypes.c_double()
        cb = ctypes.c_double()
        ct = ctypes.c_double()
        bx0 = by0 = bx1 = by1 = 0.0
        have_box = False

        def flush() -> None:
            nonlocal have_box
            text = "".join(buf).rstrip()
            buf.clear()
            had_box, have_box = have_box, False
            if not text or not had_box:
                return
            xs: List[float] = []
            ys: List[float] = []
            for px, py in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)):
                vx, vy = pre_to_visual(px, py)
                xs.append(vx)
                ys.append(vy)
            x0 = round(min(xs) * scale)
            x1 = round(max(xs) * scale)
            y0 = round(min(ys) * scale)
            y1 = round(max(ys) * scale)
            lines.append((text, ((x0, y0), (x1, y0), (x1, y1), (x0, y1))))

        def union_box(idx: int) -> None:
            nonlocal bx0, bx1, by0, by1, have_box
            if C.FPDFText_GetCharBox(tp, idx, cl, cr, cb, ct):
                if have_box:
                    bx0 = min(bx0, cl.value)
                    bx1 = max(bx1, cr.value)
                    by0 = min(by0, cb.value)
                    by1 = max(by1, ct.value)
                else:
                    bx0, bx1, by0, by1 = cl.value, cr.value, cb.value, ct.value
                    have_box = True

        # `visible` counts CODE POINTS (a surrogate pair is one; the
        # generated \r\n separators are none) — CountChars counts UTF-16
        # units incl. line breaks, which inflated the trust threshold by 2
        # per line and let a multi-line stamp block sneak past the gate.
        visible = 0
        pending_hi = 0   # high surrogate awaiting its low half
        pending_hi_idx = -1

        def abandon_pending() -> None:
            nonlocal pending_hi, visible, nonprint
            if pending_hi:
                visible += 1
                nonprint += 1  # a lone half is an unmappable glyph
                pending_hi = 0

        for i in range(n):
            u = int(C.FPDFText_GetUnicode(tp, i))
            if 0xD800 <= u <= 0xDBFF:
                # PDFium reports astral code points from surrogate-pair
                # CMaps as TWO char indices. chr() on a lone half builds an
                # unencodable str — every UTF-8 file write then raised
                # UnicodeEncodeError, and the searchable writer silently
                # dropped the whole line. Defer until the low half arrives.
                abandon_pending()
                pending_hi, pending_hi_idx = u, i
                continue
            if 0xDC00 <= u <= 0xDFFF:
                if pending_hi:
                    cp = 0x10000 + ((pending_hi - 0xD800) << 10) + (u - 0xDC00)
                    union_box(pending_hi_idx)
                    union_box(i)
                    buf.append(chr(cp))
                    pending_hi = 0
                    visible += 1
                else:
                    visible += 1
                    nonprint += 1  # lone low half
                continue
            abandon_pending()
            if u in (0x0D, 0x0A):
                flush()
                continue
            visible += 1
            if u == 0xFFFD:
                fffd += 1
            elif u < 0x20 and u != 0x09:
                nonprint += 1
                continue  # dropped: rejoins wrap-hyphenated words, no tofu
            if u not in (0x20, 0x09, 0xA0):
                # Whitespace keeps its place in the TEXT but must not
                # stretch the BOX: a positioned trailing space run (tab
                # leaders, empty right-hand table cells) used to inflate the
                # line box ~2x past the ink — and the searchable writer then
                # stretched its invisible run to the inflated width.
                union_box(i)
            buf.append(chr(u))
        abandon_pending()
        flush()
        w_px = math.ceil(vis_w_pt * scale)
        h_px = math.ceil(vis_h_pt * scale)
        return w_px, h_px, lines, (visible, fffd, nonprint, rot)
    finally:
        tp.close()


# The server's gate trusts any page with >= 10 clean chars — enough for a
# Bates stamp ("BATES 000123", 12 chars) on a SCANNED page to clear the bar
# and silently replace the whole page's OCR with one stamp line. The bar is
# higher here: a caller who asked for mode="auto" wants the SPEED of a real
# text layer, and a page carrying a dozen characters offers none of it while
# risking the whole page body. Below the bar the page renders and OCRs. For
# a genuinely sparse digital page ("Exhibit A" separators, chapter titles)
# that costs one OCR pass and returns the same text; for a stamped or
# fax-headed scan it saves the document body.
_AUTO_TRUST_MIN_CHARS = 50


def _layer_quality(stats: LayerStats, n_lines: int) -> str:
    """``"absent"`` | ``"rejected"`` | ``"trusted"``. ``"absent"`` covers
    both no-layer-at-all and a layer too THIN to trust (below the char
    threshold) — either way the page renders and OCRs. Mirrors the server's
    ``text_layer_quality_for`` (pdf_job_pages.cpp), plus the stricter
    default-mode char threshold above. Anything not trusted falls through to
    render+OCR in ``mode="auto"``; ``mode="text"`` is exempt (an explicit
    layer request returns the layer, whatever its quality)."""
    char_count, fffd, nonprint, _rotation_deg = stats
    if char_count == 0 or n_lines == 0:
        return "absent"
    # rotation is no longer rejected (parity with the server's gate): the
    # /Rotate transform is ink-verified on both sides, so a born-digital
    # rotated page (landscape reports) serves its layer like any other.
    if char_count < _AUTO_TRUST_MIN_CHARS:
        return "absent"
    if fffd * 20 > char_count:
        return "rejected"  # broken ToUnicode CMap: mostly U+FFFD
    if nonprint * 10 > char_count:
        return "rejected"
    return "trusted"


def render_pdf(
    pdf: ImageInput,
    *,
    dpi: int = _PDF_DEFAULT_DPI,
    pages: Optional[List[int]] = None,
    max_pages: Optional[int] = None,
    password: Optional[str] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(page_number, bgr_array)`` for each requested page.

    ``pages`` is a 1-based list (``None`` => all pages); ``max_pages`` caps how
    many are rendered. DPI maps to a PDFium scale of ``dpi/72``.
    """
    for kind, page_no, *rest in iter_pdf_pages(
        pdf, dpi=dpi, pages=pages, max_pages=max_pages, mode="ocr",
        password=password,
    ):
        if kind != "img":  # pragma: no cover — mode="ocr" yields img only
            # A raise, not an assert: under python -O an assert vanishes and
            # this would yield a non-array where callers expect pixels.
            raise RuntimeError(f"unexpected payload kind {kind!r} in render_pdf")
        yield page_no, rest[0]


def iter_pdf_pages(
    pdf: ImageInput,
    *,
    dpi: int = _PDF_DEFAULT_DPI,
    pages: Optional[List[int]] = None,
    max_pages: Optional[int] = None,
    mode: PdfMode = "ocr",
    password: Optional[str] = None,
    text_with_raster: bool = False,
    on_error: OnError = "raise",
) -> Iterator[tuple]:
    """The per-page payload stream under ``read_pdf_stream``:

    * ``("img", page_no, bgr_array)`` — a page that needs OCR;
    * ``("text", page_no, width_px, height_px, lines, arr, warnings)`` — a
      page served by its embedded text layer (``lines`` =
      :data:`TextLayerLine` list). ``arr`` is the rendered raster when
      ``text_with_raster=True`` (the caller wants structure stages or
      keep_image on text pages), else None; ``warnings`` is a list of
      page-level notes (e.g. a render failure that cost the structure pass
      but NOT the text, or a text-mode page with no layer at all);
    * ``("error", page_no, message)`` — only with ``on_error="skip"``: the
      page failed entirely and was contained. With the default
      ``on_error="raise"`` the exception propagates instead.

    Modes: ``"ocr"`` renders every page. ``"auto"`` extracts each page's
    text layer first and serves it ONLY when the quality gate trusts it
    (:func:`_layer_quality` — absent/thin/garbled/rotated layers fall
    through to render+OCR, and so does a layer whose EXTRACTION throws: an
    unusable layer is exactly what falls through). ``"text"`` serves the
    layer only — no gate, no rendering; a page without a layer yields empty
    lines plus a ``no_text_layer`` warning so blank output is detectable.

    A document that cannot be OPENED always raises. Every pdfium touchpoint
    holds the process-wide ``_PDFIUM_LOCK`` (released between pages) — this
    is the path under read_pdf/aread_pdf, so without it two concurrent
    documents would race pdfium and crash the interpreter."""
    check_pdf_mode(mode)
    check_on_error(on_error)
    with _PDFIUM_LOCK:
        doc = _open_document(pdf, password)
        try:
            wanted = _wanted_pages(doc, pages, max_pages)
        except Exception:
            doc.close()
            raise
    try:
        scale = dpi / 72.0
        for page_no in wanted:
            try:
                with _PDFIUM_LOCK:
                    page = doc[page_no - 1]
                    try:
                        payload: Optional[tuple] = None
                        lines = None
                        stats: Optional[LayerStats] = None
                        if mode in ("auto", "text"):
                            try:
                                w, h, lines, stats = _extract_page_text(page, scale)
                            except Exception:
                                if mode == "text":
                                    raise
                                lines = None  # unusable layer: render+OCR
                        if mode == "text":
                            warns = []
                            if not lines:
                                warns.append(
                                    "no_text_layer: page has no embedded "
                                    "text (a scan?) — mode='auto' or 'ocr' "
                                    "OCRs it")
                            payload = ("text", page_no, w, h, lines or [],
                                       None, warns)
                        elif (lines is not None and stats is not None
                              and _layer_quality(stats, len(lines)) == "trusted"):
                            arr = None
                            warns = []
                            if text_with_raster:
                                try:
                                    arr = _render_page(page, scale)
                                except Exception as exc:
                                    # The TEXT is already in hand; a render
                                    # failure must not discard it. The
                                    # structure pass / keep_image are what's
                                    # lost — reported, never silent.
                                    warns.append(
                                        "page_render_failed: structure "
                                        "stages and keep_image unavailable "
                                        f"for this page "
                                        f"({type(exc).__name__}: {exc})")
                            payload = ("text", page_no, w, h, lines, arr,
                                       warns)
                        if payload is None:
                            payload = ("img", page_no, _render_page(page, scale))
                    finally:
                        page.close()
            except Exception as exc:
                if on_error == "raise":
                    raise
                payload = ("error", page_no, f"{type(exc).__name__}: {exc}")
            yield payload
    finally:
        with _PDFIUM_LOCK:
            doc.close()


def extract_pdf_text(
    pdf: ImageInput,
    *,
    dpi: int = _PDF_DEFAULT_DPI,
    pages: Optional[List[int]] = None,
    max_pages: Optional[int] = None,
    password: Optional[str] = None,
) -> Iterator[Tuple[int, int, int, List[TextLayerLine]]]:
    """Yield ``(page_no, width_px, height_px, lines)`` from the EMBEDDED text
    layer only — no rasterization, no OCR, no models — which is why this runs
    at hundreds of pages per second. A page without a text layer (a scan)
    yields an empty ``lines`` list. No quality gate: an explicit layer
    request returns the layer as-is (rotated pages included — boxes are
    transformed to the visual space).

    All pdfium work serializes behind the process-wide lock (see
    ``_PDFIUM_LOCK`` — pdfium is globally thread-hostile), held per page so
    concurrent callers interleave rather than block for whole documents."""
    scale = dpi / 72.0
    with _PDFIUM_LOCK:
        doc = _open_document(pdf, password)
        try:
            wanted = _wanted_pages(doc, pages, max_pages)
        except Exception:
            doc.close()
            raise
    try:
        for page_no in wanted:
            with _PDFIUM_LOCK:
                page = doc[page_no - 1]
                try:
                    w, h, lines, _stats = _extract_page_text(page, scale)
                finally:
                    page.close()
            yield page_no, w, h, lines
    finally:
        with _PDFIUM_LOCK:
            doc.close()


def _to_bgr(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:  # grayscale
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:  # BGRA -> BGR
        return arr[:, :, :3]
    if arr.shape[2] == 3:  # BGR already
        return arr
    raise ValueError(f"unexpected PDFium bitmap shape {arr.shape}")


def pdf_page_count(pdf: ImageInput, password: Optional[str] = None) -> int:
    with _PDFIUM_LOCK:
        doc = _open_document(pdf, password)
        try:
            return len(doc)
        finally:
            doc.close()
