"""Shared option constants and validators — the leaf module both the
pipeline and the pdf layer import, so a default or an error message exists
exactly once. Imports nothing from the package (it sits below everything)."""

from __future__ import annotations

from typing import Literal, Optional

#: Render DPI default for every PDF entry point (pipeline, pdf, CLI).
DEFAULT_DPI = 150

#: kDropScore in the C++ engine (applied there too; a safety net here).
DROP_SCORE = 0.5

#: The CLI's output formats — one tuple drives argparse choices AND _emit.
OUTPUT_FORMATS = ("text", "json", "markdown", "tsv", "hocr")

#: The pipeline stages a page can carry, spelled with the SAME names the
#: capability registry and the HTTP query parameters use, so Python, HTTP and
#: gRPC never drift into synonyms for one concept.
#:
#: "autorotate" is in the RECORD vocabulary but is not a per-call request flag:
#: it is input preparation (it rotates the pixels every other stage then sees)
#: rather than an output stage, it never reaches the shared request gate
#: (python_options.h excludes DocOrientation from `acts_on`), and it stays
#: inherited from the constructor. PageResult.stages records whether it ran so
#: the fact is auditable rather than invisible.
Stage = Literal["text", "layout", "reading_order", "tables", "formulas",
                "autorotate"]
STAGES = ("text", "layout", "reading_order", "tables", "formulas", "autorotate")

#: How a PDF page's text is obtained. Spelled as a Literal so editors and
#: type checkers OFFER the three values at the call site instead of showing
#: a bare `str` — the options are discoverable without opening the docs.
#:
#: * ``"ocr"`` (default) — render every page and OCR it. This is an OCR
#:   library: the default runs the recognizer, and a page's text never
#:   silently comes from somewhere else.
#: * ``"auto"``  — serve a page's embedded text layer when a quality gate
#:   trusts it (born-digital pages: ~10x faster and byte-exact), OCR the
#:   rest. Opt in when throughput matters more than a uniform code path.
#: * ``"text"``  — the embedded layer only: no rendering, no OCR, no models
#:   (so combining it with layout/tables/formulas/reading_order raises —
#:   ``"auto"`` is the mode that serves the layer AND runs structure).
PdfMode = Literal["ocr", "auto", "text"]

#: Valid :data:`PdfMode` values, for argparse choices and validation.
PDF_MODES = ("ocr", "auto", "text")

#: What a per-item failure does: propagate, or become a placeholder result
#: carrying a ``page_failed`` warning.
OnError = Literal["raise", "skip"]

#: The ENGINE execution path (unrelated to :data:`PdfMode` — this one picks
#: how models run, not where a page's text comes from).
EngineMode = Literal["auto", "native", "ultra", "onnx", "fast"]

#: Valid :data:`EngineMode` values.
ENGINE_MODES = ("auto", "native", "ultra", "onnx", "fast")


def check_drop_score(drop_score: float) -> None:
    """One validator for EVERY entry point that takes ``drop_score`` — read,
    read_batch, read_pdf/read_pdf_stream, pdf_to_searchable. It used to live
    only in read(), so read_batch(drop_score=nan) silently disabled filtering
    (every NaN comparison is False, so nothing was ever dropped) and
    read_pdf(drop_score=2.0) returned every page empty — indistinguishable
    from a blank document.

    The floor exists because the C++ engine applies its own hard kDropScore
    (0.5) BEFORE Python sees any item, so a lower value here can only pretend:
    it returned output byte-identical to 0.5 with no warning. Refused until
    the floor is plumbed through RunFlags."""
    if drop_score != drop_score:  # NaN slips past BOTH range checks below
        raise ValueError("drop_score must be a number, got nan")
    if drop_score < DROP_SCORE:
        raise ValueError(
            f"drop_score={drop_score} is below the engine's hard floor "
            f"({DROP_SCORE}): the C++ pipeline filters at "
            f"{DROP_SCORE} before Python sees results, so lower values "
            "have no effect. Use drop_score >= 0.5."
        )
    if drop_score > 1.0:
        # Confidences are 0..1, so anything above 1 can only ever return an
        # empty page with NO warning — indistinguishable from a blank scan.
        raise ValueError(
            f"drop_score={drop_score} is above the maximum confidence "
            "(1.0), so every line would be dropped and the page would "
            "come back empty. Use a value in [0.5, 1.0]."
        )


def check_on_error(on_error: str) -> None:
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")


def check_pdf_mode(mode: str) -> None:
    if mode not in PDF_MODES:
        raise ValueError(
            f"mode must be one of {PDF_MODES}, got {mode!r}"
        )


def check_pages(pages) -> None:
    """``pages`` is a 1-based LIST of ints. A bare int (``pages=3``) or a range
    STRING (``pages="1-6"``, the CLI's spelling) used to survive this call and
    blow up much later as a TypeError raised inside a worker thread — a
    traceback with no connection to the caller's mistake."""
    if pages is None:
        return
    if isinstance(pages, (str, bytes)) or not isinstance(pages, (list, tuple)):
        raise ValueError(
            f"pages must be a list of 1-based page numbers, got "
            f"{type(pages).__name__}. Use pages=[1, 3, 5] (and note the CLI's "
            '"1,3,5-8" string form is parsed by the CLI, not accepted here).'
        )
    for p in pages:
        if isinstance(p, bool) or not isinstance(p, int):
            raise ValueError(
                f"pages entries must be ints, got {type(p).__name__}: {p!r}"
            )
        if p < 1:
            raise ValueError(f"pages are 1-based, got {p}")


def check_max_pages(max_pages: Optional[int]) -> None:
    if max_pages is not None and max_pages < 1:
        # max_pages=0/-1 used to silently truncate (or empty) the result —
        # indistinguishable from a blank document.
        raise ValueError(f"max_pages must be >= 1, got {max_pages!r}")
