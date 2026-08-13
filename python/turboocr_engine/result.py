"""Result types for TurboOCR.

Mirrors the C++ ``OCRResultItem`` / ``OcrPipelineResult`` so the Python
bindings and the C++ engine describe a page the same way. Everything here is a
plain dataclass with ``to_dict`` / ``to_json`` helpers — no numpy in the public
surface, so results pickle and JSON-serialize cleanly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

# A detection quad, always ordered [top-left, top-right, bottom-right, bottom-left].
Quad = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Layout labels (PP-DocLayoutV3) → structured-output roles.
_TITLE_LEVELS = {"doc_title": 1, "abstract": 2, "paragraph_title": 2,
                 "figure_title": 3, "chart_title": 3, "algorithm": 3}
_SKIP_LABELS = {"image", "chart", "seal", "header_image", "footer_image",
                "figure", "vision_footnote", "number"}
_FORMULA_LABELS = {"display_formula", "inline_formula", "formula_number"}

# Block markers that mint document structure when they lead a line.
_MD_BLOCK_MARKERS = "#-*+`|=>"


def _md_escape(s: str) -> str:
    """Escape untrusted document text for Markdown output.

    Mirrors the C++ writer's ``escape_md_text`` (markdown_latex.cpp): OCR'd and
    PDF-extracted text is UNTRUSTED — in ``mode=auto`` a PDF's text layer
    reaches us byte-exact, so a raw ``<img onerror=...>`` in a scan becomes live
    inline HTML in any consumer that renders the Markdown (CommonMark and GFM
    pass inline HTML through by default). Entity-escape the HTML metacharacters
    (``&`` FIRST, or the other replacements get double-escaped), then neutralize
    a leading block marker on EVERY line — not just the first — because a
    heading or fence may interrupt a paragraph. Rendered output is unchanged:
    ``&amp;`` renders as ``&``.

    ``to_html`` escapes every text branch already; this is the Markdown half.
    """
    out = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = []
    for line in out.split("\n"):
        stripped = line.lstrip(" ")
        if stripped and stripped[0] in _MD_BLOCK_MARKERS:
            pad = len(line) - len(stripped)
            line = line[:pad] + "\\" + stripped
        lines.append(line)
    return "\n".join(lines)


def _html_table_to_markdown(html: str) -> str:
    """Convert a simple ``<table>`` to a Markdown table; keep raw HTML (valid in
    Markdown) for ragged tables with colspan/rowspan."""
    import html as _htmlmod
    import re

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    parsed: List[List[str]] = []
    for r in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S | re.I)
        parsed.append([_htmlmod.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells])
    if not parsed or any(len(r) != len(parsed[0]) for r in parsed) or not parsed[0]:
        return html  # ragged / empty -> raw HTML is still valid Markdown
    ncol = len(parsed[0])

    def _row(cells: List[str]) -> str:
        # A literal | in ANY cell (header included) breaks the column grid.
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    out = [_row(parsed[0]), "| " + " | ".join(["---"] * ncol) + " |"]
    out.extend(_row(r) for r in parsed[1:])
    return "\n".join(out)


def _hocr_document(page_divs: List[str]) -> str:
    """Wrap ocr_page divs in a complete hOCR document (hocr-tools/hocr-pdf
    consumable)."""
    body = "\n".join(page_divs)
    return (
        "<!DOCTYPE html>\n"
        '<html xmlns="http://www.w3.org/1999/xhtml">\n<head>\n'
        '  <meta charset="utf-8" />\n'
        '  <meta name="ocr-system" content="turboocr" />\n'
        '  <meta name="ocr-capabilities" content="ocr_page ocr_line ocrx_word" />\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>"
    )


def _quad_to_list(box: Quad) -> List[List[int]]:
    return [[int(x), int(y)] for (x, y) in box]


@dataclass
class TextLine:
    """One recognized text line: the transcript, its confidence, and the quad
    (four corner points) it was read from, in original-image pixel coordinates.
    """

    text: str
    confidence: float
    box: Quad
    #: Provenance: "" (implicit OCR) or "pdf" (from a PDF's embedded text layer).
    source: str = ""
    #: Reading-order index, or -1 when reading order was not requested.
    id: int = -1
    #: Owning layout region index, or -1 when layout was not requested.
    layout_id: int = -1

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Axis-aligned (x0, y0, x1, y1) over the four corners."""
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def order(self) -> int:
        """Reading-order index (alias for ``id``; -1 if unset)."""
        return self.id

    def crop(self, image) -> "Any":
        """Return the pixel strip of this line from ``image`` (BGR numpy array),
        rectified upright via a perspective warp of the quad."""
        import cv2
        import numpy as np

        src = np.array(self.box, dtype=np.float32)
        w = int(max(_dist(src[0], src[1]), _dist(src[3], src[2])))
        h = int(max(_dist(src[0], src[3]), _dist(src[1], src[2])))
        w, h = max(w, 1), max(h, 1)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, m, (w, h))

    def to_dict(self) -> dict:
        d: dict = {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "box": _quad_to_list(self.box),
        }
        if self.source:
            d["source"] = self.source
        if self.id >= 0:
            d["id"] = self.id
        if self.layout_id >= 0:
            d["layout_id"] = self.layout_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TextLine":
        box = tuple((int(p[0]), int(p[1])) for p in d["box"])
        return cls(
            text=d.get("text", ""),
            confidence=float(d.get("confidence", 0.0)),
            box=box,  # type: ignore
            source=d.get("source", ""),
            id=int(d.get("id", -1)),
            layout_id=int(d.get("layout_id", -1)),
        )


@dataclass
class LayoutBox:
    """A layout region (title / text / table / figure / formula ...)."""

    label: str
    confidence: float
    box: Quad
    id: int = -1

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": _quad_to_list(self.box),
            "id": self.id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutBox":
        box = tuple((int(p[0]), int(p[1])) for p in d["box"])
        return cls(
            label=d.get("label", ""),
            confidence=float(d.get("confidence", 0.0)),
            box=box,  # type: ignore
            id=int(d.get("id", -1)),
        )


@dataclass
class TableRegion:
    """A recognized table region: reconstructed HTML + its quad."""

    html: str
    score: float
    box: Quad
    layout_id: int = -1

    def to_dict(self) -> dict:
        return {
            "html": self.html,
            "score": round(float(self.score), 4),
            "box": _quad_to_list(self.box),
            "layout_id": self.layout_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TableRegion":
        box = tuple((int(p[0]), int(p[1])) for p in d["box"])
        return cls(html=d.get("html", ""), score=float(d.get("score", 0.0)),
                   box=box, layout_id=int(d.get("layout_id", -1)))  # type: ignore


@dataclass
class FormulaRegion:
    """A recognized formula region: LaTeX + its quad."""

    latex: str
    score: float
    box: Quad
    layout_id: int = -1

    def to_dict(self) -> dict:
        return {
            "latex": self.latex,
            "score": round(float(self.score), 4),
            "box": _quad_to_list(self.box),
            "layout_id": self.layout_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FormulaRegion":
        box = tuple((int(p[0]), int(p[1])) for p in d["box"])
        return cls(latex=d.get("latex", ""), score=float(d.get("score", 0.0)),
                   box=box, layout_id=int(d.get("layout_id", -1)))  # type: ignore


@dataclass
class PageResult:
    """Everything read from a single image / PDF page."""

    lines: List[TextLine] = field(default_factory=list)
    width: int = 0
    height: int = 0
    #: 1-based page number for PDF pages; None for standalone images.
    page: Optional[int] = None
    #: Render DPI (PDF pages) so a searchable PDF sizes its media box in real
    #: points; None => treat the image as 72 DPI (1px = 1pt).
    dpi: Optional[int] = None
    #: Detected page rotation applied before OCR (0/90/180/270), when autorotate ran.
    orientation: int = 0
    layout: List[LayoutBox] = field(default_factory=list)
    #: Recognized tables (HTML), when tables=True and the backend is loaded.
    tables: List[TableRegion] = field(default_factory=list)
    #: Recognized formulas (LaTeX), when formulas=True and the backend is loaded.
    formulas: List[FormulaRegion] = field(default_factory=list)
    #: Reading-order indices emitted by the engine when reading_order=True.
    reading_order: List[int] = field(default_factory=list)
    #: Additive degradation warnings (recognition produced boxes but no text, etc.).
    warnings: List[str] = field(default_factory=list)
    #: The BGR source image this page was read from (kept for draw()/crop();
    #: set by the pipeline unless keep_image=False). Not serialized.
    image: Any = field(default=None, repr=False, compare=False)

    # -- convenience -------------------------------------------------------
    def __iter__(self) -> Iterator[TextLine]:
        return iter(self.lines)

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, i: int) -> TextLine:
        return self.lines[i]

    @property
    def results(self) -> List[TextLine]:
        """Alias for ``lines`` (matches the ``\"results\"`` JSON key)."""
        return self.lines

    @property
    def text(self) -> str:
        """All recognized lines joined by newlines, in reading order."""
        return "\n".join(line.text for line in self.lines)

    # -- filtering ---------------------------------------------------------
    def filter(
        self,
        *,
        min_confidence: float = 0.0,
        contains: Optional[str] = None,
        predicate: Optional[Callable[[TextLine], bool]] = None,
    ) -> "PageResult":
        """Return a new PageResult keeping only lines that pass the filters."""
        keep = []
        for ln in self.lines:
            if ln.confidence < min_confidence:
                continue
            if contains is not None and contains not in ln.text:
                continue
            if predicate is not None and not predicate(ln):
                continue
            keep.append(ln)
        # Carry page-level context through. reading_order is intentionally NOT
        # carried: it indexes into the original line list, so filtering lines
        # would leave stale indices — better empty than wrong.
        # dpi/tables/formulas ARE carried: filtering is a predicate over text
        # LINES, so dropping them silently changed page geometry (a filtered
        # page saved via save_searchable_pdf came out at the wrong size,
        # because dpi=None means "treat as 72 DPI") and lost recognized
        # tables/formulas that no filter had rejected.
        return PageResult(
            lines=keep, width=self.width, height=self.height, page=self.page,
            dpi=self.dpi, orientation=self.orientation, layout=list(self.layout),
            tables=list(self.tables), formulas=list(self.formulas),
            warnings=list(self.warnings), image=self.image,
        )

    # -- visualization -----------------------------------------------------
    def draw(
        self,
        image: Any = None,
        *,
        color: Tuple[int, int, int] = (0, 200, 0),
        thickness: int = 2,
        show_text: bool = False,
    ) -> Any:
        """Return a copy of the source image with detected quads drawn on it.

        ``image`` defaults to the page's stored source image (BGR numpy). Set
        ``show_text=True`` to also render the recognized text above each box.
        Returns a **BGR** array (OpenCV order); convert with ``arr[..., ::-1]``
        before handing it to PIL/matplotlib."""
        import cv2
        import numpy as np

        base = image if image is not None else self.image
        if base is None:
            raise ValueError(
                "no image to draw on — pass image=..., or construct the OCR with "
                "keep_image=True (the default)."
            )
        canvas = base.copy()
        for ln in self.lines:
            pts = np.array(ln.box, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, color, thickness)
            if show_text:
                x, y = ln.box[0]
                cv2.putText(canvas, ln.text, (int(x), int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return canvas

    def save_overlay(self, path: str, *, image: Any = None, **kw: Any) -> str:
        """Draw the boxes and write the result to ``path``. Returns the path."""
        import cv2

        cv2.imwrite(path, self.draw(image, **kw))
        return path

    def save_searchable_pdf(self, path: str) -> str:
        """Write a single-page searchable PDF (image + invisible OCR text layer).
        Needs the page image (``keep_image=True``, the default) and reportlab."""
        from .searchable_pdf import build_searchable_pdf

        build_searchable_pdf([self], out_path=path)
        return path

    def to_pdf_bytes(self) -> bytes:
        """Return this page as a searchable PDF (bytes) — e.g. for a web
        response. Needs the page image (``keep_image=True``) and reportlab."""
        from .searchable_pdf import build_searchable_pdf

        return build_searchable_pdf([self])  # type: ignore[return-value]

    # -- exports -----------------------------------------------------------
    def to_tsv(self) -> str:
        """Tab-separated: index, confidence, x0, y0, x1, y1, text (bbox coords)."""
        rows = ["index\tconfidence\tx0\ty0\tx1\ty1\ttext"]
        for i, ln in enumerate(self.lines):
            x0, y0, x1, y1 = ln.bbox
            text = ln.text.replace("\t", " ").replace("\n", " ")
            rows.append(f"{i}\t{ln.confidence:.4f}\t{x0}\t{y0}\t{x1}\t{y1}\t{text}")
        return "\n".join(rows)

    def _hocr_page_div(self, page_id: int = 1) -> str:
        import html

        lines = []
        for i, ln in enumerate(self.lines):
            x0, y0, x1, y1 = ln.bbox
            conf = int(round(ln.confidence * 100))
            lines.append(
                f'   <span class="ocr_line" id="line_{page_id}_{i}" '
                f'title="bbox {x0} {y0} {x1} {y1}">'
                f'<span class="ocrx_word" title="bbox {x0} {y0} {x1} {y1}; '
                f'x_wconf {conf}">{html.escape(ln.text)}</span></span>'
            )
        body = "\n".join(lines)
        return (
            f'<div class="ocr_page" id="page_{page_id}" '
            f'title="bbox 0 0 {self.width} {self.height}">\n{body}\n</div>'
        )

    def to_hocr(self, *, page_id: int = 1, full: bool = False) -> str:
        """hOCR for this page. ``full=True`` wraps it in a complete hOCR
        document (with the ``ocr-system`` meta) that hocr-tools / hocr-pdf
        consume directly; the default emits just the ``ocr_page`` div."""
        div = self._hocr_page_div(page_id)
        return _hocr_document([div]) if full else div

    def to_pandas(self):
        """Return a pandas DataFrame of the lines (requires ``pip install
        "turboocr[cpu,pandas]"``)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_pandas() needs pandas — `pip install \"turboocr[cpu,pandas]\"`."
            ) from exc

        return pd.DataFrame(
            [
                {
                    "text": ln.text,
                    "confidence": ln.confidence,
                    "x0": ln.bbox[0], "y0": ln.bbox[1],
                    "x1": ln.bbox[2], "y1": ln.bbox[3],
                    "box": [list(p) for p in ln.box],
                }
                for ln in self.lines
            ]
        )

    def to_dict(self) -> dict:
        d: dict = {
            "width": self.width,
            "height": self.height,
            "results": [ln.to_dict() for ln in self.lines],
        }
        if self.page is not None:
            d["page"] = self.page
        if self.orientation:
            d["orientation"] = self.orientation
        if self.layout:
            d["layout"] = [lb.to_dict() for lb in self.layout]
        if self.tables:
            d["tables"] = [t.to_dict() for t in self.tables]
        if self.formulas:
            d["formulas"] = [f.to_dict() for f in self.formulas]
        if self.reading_order:
            d["reading_order"] = list(self.reading_order)
        if self.warnings:
            d["warnings"] = list(self.warnings)
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "PageResult":
        return cls(
            lines=[TextLine.from_dict(x) for x in d.get("results", [])],
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            page=d.get("page"),
            orientation=int(d.get("orientation", 0)),
            layout=[LayoutBox.from_dict(x) for x in d.get("layout", [])],
            tables=[TableRegion.from_dict(x) for x in d.get("tables", [])],
            formulas=[FormulaRegion.from_dict(x) for x in d.get("formulas", [])],
            reading_order=list(d.get("reading_order", [])),
            warnings=list(d.get("warnings", [])),
        )

    @classmethod
    def from_json(cls, s: str) -> "PageResult":
        return cls.from_dict(json.loads(s))

    def _structured_blocks(self) -> List[Tuple[str, str]]:
        """Reading-order (kind, payload) blocks from the layout regions.

        kind ∈ {h1,h2,h3,p,table,formula}. Regions are ordered top-to-bottom /
        left-to-right by their box; a region's text is the OCR lines whose
        layout_id points at it; tables/formulas come from the recognized
        structure; figures/images/seals are skipped."""
        from collections import defaultdict

        by_region: dict = defaultdict(list)
        unassigned: List[TextLine] = []
        for ln in self.lines:
            if ln.layout_id >= 0:
                by_region[ln.layout_id].append(ln)
            else:
                unassigned.append(ln)
        tables = {t.layout_id: t for t in self.tables}
        formulas = {f.layout_id: f for f in self.formulas}

        def _key(r: LayoutBox):
            xs = [p[0] for p in r.box]
            ys = [p[1] for p in r.box]
            return (min(ys), min(xs))

        blocks: List[Tuple[str, str]] = []
        for r in sorted(self.layout, key=_key):
            label = r.label
            if label in _SKIP_LABELS:
                continue
            if label == "table" and r.id in tables:
                blocks.append(("table", tables[r.id].html))
                continue
            if label in _FORMULA_LABELS and r.id in formulas:
                blocks.append(("formula", formulas[r.id].latex))
                continue
            text = " ".join(l.text for l in by_region.get(r.id, []) if l.text.strip()).strip()
            if not text:
                continue
            lvl = _TITLE_LEVELS.get(label)
            blocks.append((f"h{lvl}", text) if lvl else ("p", text))

        for ln in unassigned:
            if ln.text.strip():
                blocks.append(("p", ln.text))
        return blocks

    def to_markdown(self, *, structured: Optional[bool] = None) -> str:
        """Markdown for the page.

        When layout regions are present (``OCR(layout=True)``) this is
        layout-aware by default: title regions become headings, tables become
        Markdown tables, formulas become ``$$…$$``, in reading order. Without
        layout it falls back to reading-order paragraphs. ``structured=False``
        forces the flat form; ``structured=True`` forces the structured form."""
        want = self.layout if structured is None else structured
        if structured and not self.layout:
            import warnings as _w

            _w.warn("structured=True but no layout regions — run OCR(layout=True).",
                    stacklevel=2)
            want = False
        if not want:
            return "\n\n".join(
                _md_escape(ln.text) for ln in self.lines if ln.text.strip()
            )

        parts: List[str] = []
        for kind, payload in self._structured_blocks():
            if kind == "table":
                # Cell text is pipe-escaped inside; the wrapper is our own HTML.
                parts.append(_html_table_to_markdown(payload))
            elif kind == "formula":
                # NOT escaped: this is recognizer LaTeX, and entity-escaping it
                # would break rendering ($$a < b$$ must stay a comparison, not
                # &lt;). The delimiters are the trust boundary here.
                parts.append(f"$$\n{payload}\n$$")
            elif kind.startswith("h"):
                parts.append("#" * int(kind[1]) + " " + _md_escape(payload))
            else:
                parts.append(_md_escape(payload))
        return "\n\n".join(parts)

    def to_html(self) -> str:
        """Structured HTML body: headings, paragraphs, and tables (as HTML), in
        reading order when layout is present; else reading-order paragraphs."""
        import html as _h

        blocks = (
            self._structured_blocks()
            if self.layout
            else [("p", ln.text) for ln in self.lines if ln.text.strip()]
        )
        out: List[str] = []
        for kind, payload in blocks:
            if kind == "table":
                out.append(payload)  # already HTML
            elif kind == "formula":
                out.append(f"<p>\\[{_h.escape(payload)}\\]</p>")
            elif kind.startswith("h"):
                out.append(f"<{kind}>{_h.escape(payload)}</{kind}>")
            else:
                out.append(f"<p>{_h.escape(payload)}</p>")
        return "\n".join(out)


@dataclass
class DocumentResult:
    """OCR over a whole document (a multi-page PDF, or a batch of images)."""

    pages: List[PageResult] = field(default_factory=list)
    source: str = ""

    def __iter__(self) -> Iterator[PageResult]:
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, i: int) -> PageResult:
        return self.pages[i]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def to_dict(self) -> dict:
        d: dict = {"pages": [p.to_dict() for p in self.pages]}
        if self.source:
            d["source"] = self.source
        return d

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "DocumentResult":
        return cls(
            pages=[PageResult.from_dict(p) for p in d.get("pages", [])],
            source=d.get("source", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "DocumentResult":
        return cls.from_dict(json.loads(s))

    def to_markdown(self) -> str:
        parts: List[str] = []
        for p in self.pages:
            if p.page is not None:
                parts.append(f"<!-- page {p.page} -->")
            parts.append(p.to_markdown())
        return "\n\n".join(x for x in parts if x)

    def to_tsv(self) -> str:
        """Tab-separated across all pages, with a leading ``page`` column."""
        rows = ["page\tindex\tconfidence\tx0\ty0\tx1\ty1\ttext"]
        for p in self.pages:
            pg = p.page if p.page is not None else 0
            for i, ln in enumerate(p.lines):
                x0, y0, x1, y1 = ln.bbox
                text = ln.text.replace("\t", " ").replace("\n", " ")
                rows.append(
                    f"{pg}\t{i}\t{ln.confidence:.4f}\t{x0}\t{y0}\t{x1}\t{y1}\t{text}"
                )
        return "\n".join(rows)

    def to_hocr(self) -> str:
        """A single multi-page hOCR document over all pages."""
        divs = [p._hocr_page_div(page_id=(p.page or i + 1)) for i, p in enumerate(self.pages)]
        return _hocr_document(divs)

    def to_pandas(self):
        """One DataFrame across all pages, with a ``page`` column (needs pandas)."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_pandas() needs pandas — `pip install \"turboocr[cpu,pandas]\"`."
            ) from exc

        # Build the rows once and hand pandas a single list, instead of
        # constructing a DataFrame per page and concatenating: for a
        # several-hundred-page scan that was N frame allocations plus an N-way
        # concat to produce exactly the same table.
        rows = [
            {
                "page": p.page if p.page is not None else 0,
                "text": ln.text,
                "confidence": ln.confidence,
                "x0": ln.bbox[0], "y0": ln.bbox[1],
                "x1": ln.bbox[2], "y1": ln.bbox[3],
                "box": [list(pt) for pt in ln.box],
            }
            for p in self.pages
            for ln in p.lines
        ]
        return pd.DataFrame(
            rows,
            columns=["page", "text", "confidence", "x0", "y0", "x1", "y1", "box"],
        )

    def filter(self, **kw) -> "DocumentResult":
        """Filter every page (see :meth:`PageResult.filter`)."""
        return DocumentResult(
            pages=[p.filter(**kw) for p in self.pages], source=self.source
        )

    def to_html(self, *, full: bool = False) -> str:
        """Structured HTML for the whole document (per-page bodies concatenated).
        ``full=True`` wraps it in a minimal ``<html>`` document."""
        body = "\n".join(p.to_html() for p in self.pages)
        if not full:
            return body
        return (
            "<!DOCTYPE html>\n<html>\n<head><meta charset=\"utf-8\">"
            "<meta name=\"generator\" content=\"turboocr\"></head>\n"
            f"<body>\n{body}\n</body>\n</html>"
        )

    def to_pdf_bytes(self) -> bytes:
        """Return a searchable PDF (page rasters + invisible OCR text) as bytes."""
        from .searchable_pdf import build_searchable_pdf

        return build_searchable_pdf(self.pages)  # type: ignore[return-value]

    def save_searchable_pdf(self, path: str) -> str:
        """Write a searchable PDF (image + invisible OCR text layer) to ``path``.
        Needs page images (``keep_image=True``, the default) and reportlab."""
        from .searchable_pdf import build_searchable_pdf

        build_searchable_pdf(self.pages, out_path=path)
        return path
