"""Shared option constants and validators — the leaf module both the
pipeline and the pdf layer import, so a default or an error message exists
exactly once. Imports nothing from the package (it sits below everything)."""

from __future__ import annotations

from typing import Optional

#: Render DPI default for every PDF entry point (pipeline, pdf, CLI).
DEFAULT_DPI = 150

#: kDropScore in the C++ engine (applied there too; a safety net here).
DROP_SCORE = 0.5

#: The CLI's output formats — one tuple drives argparse choices AND _emit.
OUTPUT_FORMATS = ("text", "json", "markdown", "tsv", "hocr")


def check_on_error(on_error: str) -> None:
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")


def check_max_pages(max_pages: Optional[int]) -> None:
    if max_pages is not None and max_pages < 1:
        # max_pages=0/-1 used to silently truncate (or empty) the result —
        # indistinguishable from a blank document.
        raise ValueError(f"max_pages must be >= 1, got {max_pages!r}")
