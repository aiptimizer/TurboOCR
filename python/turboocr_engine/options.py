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
#: * ``"text"``  — the embedded layer only: no rendering, no OCR, no models.
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


def check_on_error(on_error: str) -> None:
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")


def check_pdf_mode(mode: str) -> None:
    if mode not in PDF_MODES:
        raise ValueError(
            f"mode must be one of {PDF_MODES}, got {mode!r}"
        )


def check_max_pages(max_pages: Optional[int]) -> None:
    if max_pages is not None and max_pages < 1:
        # max_pages=0/-1 used to silently truncate (or empty) the result —
        # indistinguishable from a blank document.
        raise ValueError(f"max_pages must be >= 1, got {max_pages!r}")
