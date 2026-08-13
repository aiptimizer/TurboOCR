"""Markdown rendering for OCR/PDF responses.

Public API (re-exported here so external imports use `from turboocr.markdown
import …` regardless of internal split):

  - `NodeKind`           — enum of markdown construct kinds
  - `MarkdownNode`       — single typed node (kind + text + optional metadata)
  - `MarkdownDocument`   — rendered document (nodes + joined `.markdown`)
  - `StyleRule`          — (kind, level) tuple stored in the rules dict
  - `MarkdownStyle`      — registry mapping layout-label → kind, plus renderers
  - `render_to_markdown` — dispatcher (OcrResponse or PdfResponse)
  - `render_ocr_to_markdown`
  - `render_pdf_to_markdown`

Internal split:
  - `style.py`  — node/document models, default rules, default renderers, registry
  - `render.py` — walker that turns a response into a `MarkdownDocument`
"""
from .render import (
    render_ocr_to_markdown,
    render_pdf_to_markdown,
    render_to_markdown,
)
from .style import (
    MarkdownDocument,
    MarkdownNode,
    MarkdownStyle,
    NodeKind,
    StyleRule,
)

__all__ = [
    "MarkdownDocument",
    "MarkdownNode",
    "MarkdownStyle",
    "NodeKind",
    "StyleRule",
    "render_ocr_to_markdown",
    "render_pdf_to_markdown",
    "render_to_markdown",
]
