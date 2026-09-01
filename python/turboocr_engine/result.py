"""Result types for TurboOCR.

Mirrors the C++ ``OCRResultItem`` / ``OcrPipelineResult`` so the Python
bindings and the C++ engine describe a page the same way. Everything here is a
plain dataclass with ``to_dict`` / ``to_json`` helpers — no numpy in the public
surface, so results pickle and JSON-serialize cleanly.

Key-name note: this library's ``to_dict`` uses ``box``/``label``/``score``;
the HTTP server's JSON spells the same fields ``bounding_box``/``class``/
``confidence``. Every ``from_dict`` here accepts BOTH spellings, so a server
response parses into these types directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional, Tuple

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


def _table_element(html: str) -> str:
    """The ``<table>...</table>`` element out of a table region's HTML.

    Falls back to the input unchanged when no <table> is present, so a producer
    that already returns a bare fragment is passed through untouched."""
    import re as _re

    m = _re.search(r"<table\b.*?</table\s*>", html, _re.S | _re.I)
    return m.group(0) if m else html


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
        # Ragged (colspan/rowspan) tables pass through as raw HTML — valid
        # in Markdown. TRUST NOTE: the native pipeline entity-escapes cell
        # text at the source (html_reconstruct.cpp), so engine-produced HTML
        # is safe; a TableRegion built from UNTRUSTED JSON carries whatever
        # its author wrote — sanitize `html` yourself before rendering.
        return html
    ncol = len(parsed[0])

    def _cell(c: str) -> str:
        # Cell text was UNescaped above (to strip markup cleanly), so it must
        # be re-entity-escaped before landing in Markdown — a scanned cell
        # reading "<img onerror=...>" is untrusted text, and CommonMark/GFM
        # pass inline HTML through. This is the same hole _md_escape closes
        # for every other text sink. A literal | additionally breaks the grid.
        c = c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Backslash FIRST: escaping | on text that contains \| would emit
        # \\| — an escaped backslash followed by an ACTIVE separator.
        # Newlines inside a cell would split the Markdown row outright.
        c = c.replace("\\", "\\\\").replace("|", "\\|")
        return c.replace("\n", " ").replace("\r", " ")

    def _row(cells: List[str]) -> str:
        return "| " + " | ".join(_cell(c) for c in cells) + " |"

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
        '  <meta name="ocr-capabilities" content="ocr_page ocr_line" />\n'
        "</head>\n<body>\n" + body + "\n</body>\n</html>"
    )


# Distinct BGR colors for layout-region overlays, assigned per label via a
# stable crc32 (see PageResult.draw).
_LAYOUT_PALETTE = [
    (255, 120, 0), (0, 120, 255), (0, 200, 255), (255, 0, 200),
    (120, 220, 0), (200, 0, 255), (0, 255, 160), (0, 80, 255),
]


# Shared by every pandas entry point — the hint and the parse-failure
# message exist once (they were spelled three times each).
_PANDAS_HINT = (
    "to_pandas() needs pandas — `pip install \"turboocr[cpu,pandas]\"`."
)
_NO_TABLE_MSG = "no table could be parsed from this region's HTML"
#: pandas.read_html needs an HTML parser, which pandas does NOT depend on.
#: The `[pandas]` extra ships lxml for exactly this reason; installing bare
#: pandas leaves TableRegion.to_pandas() one import short.
_LXML_HINT = (
    "TableRegion.to_pandas() needs an HTML parser for pandas.read_html, and "
    "lxml is not installed (pandas does not require it on its own). Install "
    "`pip install \"turboocr[cpu,pandas]\"`, which ships it, or "
    "`pip install lxml`. The table HTML is available without any extra as "
    "TableRegion.html."
)


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_PANDAS_HINT) from exc
    return pd


def _line_row(ln: "TextLine", page=None) -> dict:
    """One TextLine as the flat row both to_pandas() shapes share (the page
    column leads only when a page number applies)."""
    row = {} if page is None else {"page": page}
    x0, y0, x1, y1 = ln.bbox
    row.update({
        "text": ln.text, "confidence": ln.confidence,
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "box": [list(pt) for pt in ln.box],
    })
    return row


def _tsv_row(ln: "TextLine", index: int, page=None) -> str:
    x0, y0, x1, y1 = ln.bbox
    text = ln.text.replace("\t", " ").replace("\n", " ")
    lead = f"{page}\t" if page is not None else ""
    return f"{lead}{index}\t{ln.confidence:.4f}\t{x0}\t{y0}\t{x1}\t{y1}\t{text}"


def _quad_to_list(box: Quad) -> List[List[int]]:
    return [[int(x), int(y)] for (x, y) in box]


def _quad_from_dict(d: dict) -> Quad:
    """Read a quad under either spelling: this library serializes ``box``;
    the C++ server emits ``bounding_box`` (serialization_items.h). from_dict
    must accept both, or PageResult.from_dict(server_response) dies on a
    KeyError — the parity these types exist to provide. Malformed shapes
    fail HERE with a message naming the field, not three calls later as
    ``min() arg is an empty sequence``."""
    pts = d.get("box")
    if pts is None:
        pts = d.get("bounding_box")
    if pts is None:
        raise ValueError("missing 'box'/'bounding_box' in region dict")
    try:
        quad = tuple((int(p[0]), int(p[1])) for p in pts)
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError(
            f"'box'/'bounding_box' must be four [x, y] points, got {pts!r}"
        ) from exc
    if len(quad) != 4:
        raise ValueError(
            f"'box'/'bounding_box' must have exactly 4 points, got {len(quad)}"
        )
    return quad  # type: ignore[return-value]


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
        box = _quad_from_dict(d)
        return cls(
            text=d.get("text", ""),
            confidence=float(d.get("confidence", d.get("score", 0.0))),
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
    #: Containing region's id (nested regions), -1 for a top-level region.
    parent_id: int = -1

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": _quad_to_list(self.box),
            "id": self.id,
        }
        if self.parent_id >= 0:
            d["parent_id"] = self.parent_id
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayoutBox":
        box = _quad_from_dict(d)
        return cls(
            # The server spells the label "class" (layout regions) — accept
            # both, same rationale as _quad_from_dict. `or`-chained so an
            # explicit null under one key falls through to the other.
            label=d.get("label") or d.get("class") or "",
            confidence=float(d.get("confidence", d.get("score", 0.0))),
            box=box,  # type: ignore
            id=int(d.get("id", -1)),
            parent_id=int(d.get("parent_id", -1)),
        )


@dataclass
class TableRegion:
    """A recognized table region: reconstructed HTML + its quad."""

    html: str
    score: float
    box: Quad
    layout_id: int = -1
    #: Server-shape passthrough: the HTTP API also emits per-cell geometry
    #: (``cells: [{text, bounding_box, row, col, rowspan, colspan}]``) — the
    #: only machine-usable form of a merged-cell table. Kept as plain dicts,
    #: populated by from_dict, re-emitted by to_dict; the native pipeline
    #: does not fill it (its cell geometry lives in the HTML).
    cells: Optional[List[dict]] = None

    @property
    def confidence(self) -> float:
        """Alias for ``score`` — TextLine/LayoutBox call the same number
        ``confidence``, so both spellings work everywhere. ``score`` stays
        canonical in this library's ``to_dict()``; the SERVER's own JSON
        spells it ``confidence`` (from_dict accepts both)."""
        return self.score

    def to_dict(self) -> dict:
        d = {
            "html": self.html,
            "score": round(float(self.score), 4),
            "box": _quad_to_list(self.box),
            "layout_id": self.layout_id,
        }
        if self.cells is not None:
            d["cells"] = self.cells
        return d

    def to_pandas(self):
        """The table as a pandas DataFrame, parsed from the reconstructed
        HTML (needs the ``[pandas]`` extra, which includes the lxml parser
        ``pandas.read_html`` uses). Merged cells expand the way read_html
        expands them: the spanned value repeats into each covered cell. A
        ``<thead>`` row becomes the column header; without one, columns are
        numbered. If the region's HTML holds several ``<table>`` elements,
        the FIRST one is returned.

        >>> page = ocr.read("invoice.png", tables=True)
        >>> dfs = [t.to_pandas() for t in page.tables]
        """
        pd = _require_pandas()
        from io import StringIO

        try:
            # keep_default_na/na_values: recognized CELL TEXT is data, not
            # missing-value markers — "NA"/"None" in an invoice must stay the
            # strings the engine read, and "1,234" must not be silently
            # rewritten (no thousands parsing).
            frames = pd.read_html(
                StringIO(self.html), keep_default_na=False, na_values=[],
                thousands=None,
            )
        except ImportError as exc:
            try:
                import lxml  # noqa: F401
            except ImportError:
                # Name the missing piece and the fix. Re-raising the original
                # gave a bare "ModuleNotFoundError: No module named 'lxml'"
                # from inside pandas — a dependency the caller never asked
                # for, with nothing saying which extra supplies it.
                raise ImportError(_LXML_HINT) from exc
            # lxml IS installed: pandas only reaches its html5lib-fallback
            # ImportError when lxml parsed the HTML and found no table — the
            # no-table case wearing a dependency error's clothes.
            raise ValueError(_NO_TABLE_MSG) from exc
        except Exception as exc:
            # read_html raises different types per failure (XMLSyntaxError on
            # empty input, its own ValueError, even ImportError fallbacks on
            # tables lxml can't see). One stable, documented type instead.
            raise ValueError(
                f"{_NO_TABLE_MSG} ({type(exc).__name__}: {exc})"
            ) from exc
        if not frames:  # belt and braces
            raise ValueError(_NO_TABLE_MSG)
        df = frames[0]
        # Provenance rides in pandas' metadata slot, not in data columns.
        df.attrs.update({"score": float(self.score), "box": _quad_to_list(self.box)})
        return df

    @classmethod
    def from_dict(cls, d: dict) -> "TableRegion":
        box = _quad_from_dict(d)
        # This library serializes "score"; the server emits "confidence" for
        # structure regions — accept both (score wins when both appear).
        score = float(d.get("score", d.get("confidence", 0.0)))
        return cls(html=d.get("html", ""), score=score,
                   box=box, layout_id=int(d.get("layout_id", -1)),
                   cells=d.get("cells"))  # type: ignore


@dataclass
class FormulaRegion:
    """A recognized formula region: LaTeX + its quad."""

    latex: str
    score: float
    box: Quad
    layout_id: int = -1

    @property
    def confidence(self) -> float:
        """Alias for ``score`` (see :attr:`TableRegion.confidence`)."""
        return self.score

    def to_dict(self) -> dict:
        return {
            "latex": self.latex,
            "score": round(float(self.score), 4),
            "box": _quad_to_list(self.box),
            "layout_id": self.layout_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FormulaRegion":
        box = _quad_from_dict(d)
        score = float(d.get("score", d.get("confidence", 0.0)))
        return cls(latex=d.get("latex", ""), score=score,
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
    #: Which pipeline stages actually RAN for this page — not which produced
    #: output. A layout pass that found zero regions leaves ``layout == []`` but
    #: still reports ``"layout"``, so "never ran" is never confused with "ran and
    #: found nothing" (a blank scan legitimately yields zero of everything, and
    #: ``to_dict`` omits empty lists, so emptiness survives serialization as an
    #: ambiguity). Recorded, never inferred.
    stages: Tuple[str, ...] = field(default_factory=tuple, compare=False)
    #: The BGR source image this page was read from (kept for draw()/crop()).
    #: read() stores it by default; read_pdf/read_batch drop it unless
    #: keep_image=True. Not serialized.
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
            # reading_order is dropped above (its indices point into the ORIGINAL
            # line list), so the stage record has to agree — otherwise the page
            # would claim a stage whose output was just discarded.
            stages=tuple(s for s in self.stages if s != "reading_order"),
        )

    # -- visualization -----------------------------------------------------
    def draw(
        self,
        image: Any = None,
        *,
        color: Tuple[int, int, int] = (0, 200, 0),
        thickness: int = 2,
        show_text: bool = False,
        layout: bool = False,
        lines: bool = True,
    ) -> Any:
        """Return a copy of the source image with detected quads drawn on it.

        ``image`` defaults to the page's stored source image (BGR numpy). Set
        ``show_text=True`` to also render the recognized text above each box.
        ``layout=True`` additionally draws the LAYOUT REGIONS (run with
        ``layout=True`` at read time): each region in a stable per-label color
        with a ``label score`` caption — pass ``lines=False`` for a
        layout-only overlay. Returns a **BGR** array (OpenCV order); convert
        with ``arr[..., ::-1]`` before handing it to PIL/matplotlib."""
        import cv2
        import numpy as np

        base = image if image is not None else self.image
        if base is None:
            raise ValueError(
                "no image to draw on — pass image=..., or read with "
                "keep_image=True (read() keeps rasters by default; "
                "read_pdf/read_batch drop them)."
            )
        canvas = base.copy()
        if lines:
            for ln in self.lines:
                pts = np.array(ln.box, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, color, thickness)
                if show_text:
                    x, y = ln.box[0]
                    cv2.putText(canvas, ln.text, (int(x), int(y) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if layout:
            import zlib

            # Stable per-label colors: crc32, not hash() — hash() is salted
            # per process, and a label changing color between two runs of the
            # same script reads as a different detection.
            for lb in self.layout:
                c = _LAYOUT_PALETTE[zlib.crc32(lb.label.encode()) % len(_LAYOUT_PALETTE)]
                pts = np.array(lb.box, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, c, thickness + 1)
                x, y = lb.box[0]
                caption = f"{lb.label} {lb.confidence:.2f}"
                cv2.putText(canvas, caption, (int(x) + 2, max(int(y) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        return canvas

    def save_overlay(self, path: str, *, image: Any = None, **kw: Any) -> str:
        """Draw the boxes and write the result to ``path``. Returns the path."""
        import cv2

        cv2.imwrite(path, self.draw(image, **kw))
        return path

    def save_searchable_pdf(self, path: str) -> str:
        """Write a single-page searchable PDF (image + invisible OCR text layer).
        Needs the page image — read() keeps it by default; on the PDF/batch
        paths pass ``keep_image=True``."""
        from .searchable_pdf import build_searchable_pdf

        build_searchable_pdf([self], out_path=path)
        return path

    def to_pdf_bytes(self) -> bytes:
        """Return this page as a searchable PDF (bytes) — e.g. for a web
        response. Needs the page image (``keep_image=True`` on the PDF/batch
        paths)."""
        from .searchable_pdf import build_searchable_pdf

        return build_searchable_pdf([self])  # type: ignore[return-value]

    # -- exports -----------------------------------------------------------
    def to_tsv(self) -> str:
        """Tab-separated: index, confidence, x0, y0, x1, y1, text (bbox coords)."""
        rows = ["index\tconfidence\tx0\ty0\tx1\ty1\ttext"]
        rows.extend(_tsv_row(ln, i) for i, ln in enumerate(self.lines))
        return "\n".join(rows)

    def _hocr_page_div(self, page_id: int = 1) -> str:
        import html
        import re as _re

        # XML 1.0 forbids C0 controls (except \t \n \r): one stray control
        # glyph from a PDF text layer or recognizer output would make the
        # whole hOCR document unparseable to every consumer.
        _ctrl = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

        # LINE granularity, honestly: the engine recognizes whole lines, so
        # there are no word boxes to report. The old output wrapped each full
        # line in a fake single ocrx_word span, which made word-level
        # consumers (hocr-pdf word highlighting, layout analysis) treat every
        # line as one giant word. x_wconf on the line is a common extension;
        # the capabilities meta above claims only ocr_page/ocr_line.
        lines = []
        for i, ln in enumerate(self.lines):
            x0, y0, x1, y1 = ln.bbox
            conf = round(ln.confidence * 100)
            lines.append(
                f'   <span class="ocr_line" id="line_{page_id}_{i}" '
                f'title="bbox {x0} {y0} {x1} {y1}; x_wconf {conf}">'
                f'{html.escape(_ctrl.sub("", ln.text))}</span>'
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

    def tables_to_pandas(self):
        """The RECOGNIZED TABLES as DataFrames — one per table region, in
        region order (run with ``tables=True``; needs the ``[pandas]``
        extra). This — not :meth:`to_pandas` — is the tabular view of the
        page's tables: :meth:`to_pandas` is the page's text LINES as one
        tidy frame. Each frame carries provenance in ``DataFrame.attrs``:
        ``box``, ``score``, and ``page`` when known.

        >>> page = ocr.read("invoice.png", tables=True)
        >>> for df in page.tables_to_pandas():
        ...     print(df.attrs["box"], df.shape)
        """
        frames = [t.to_pandas() for t in self.tables]
        if self.page is not None:
            for df in frames:
                df.attrs["page"] = self.page
        return frames

    def to_pandas(self):
        """The page's text LINES as one tidy DataFrame — text, confidence and
        geometry per row (requires ``pip install "turboocr[cpu,pandas]"``).
        For the recognized tables themselves use :meth:`tables_to_pandas`,
        which turns each table region into its own DataFrame."""
        pd = _require_pandas()
        return pd.DataFrame([_line_row(ln) for ln in self.lines])

    def to_dict(self) -> dict:
        d: dict = {
            "width": self.width,
            "height": self.height,
            "results": [ln.to_dict() for ln in self.lines],
        }
        if self.page is not None:
            d["page"] = self.page
        if self.stages:
            d["stages"] = list(self.stages)
        if self.dpi is not None:
            # Without this, a page serialized and restored lost its render
            # DPI, and a searchable PDF built from the restored page came out
            # at the wrong physical size (dpi=None means "treat as 72 DPI").
            d["dpi"] = self.dpi
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
        # Server-key acceptance, same rationale as _quad_from_dict:
        # * lines live under "results" (this library AND the server); a
        #   hand-built {"lines": [...]} used to parse SILENTLY into an
        #   empty page — accepted as an alias instead;
        # * the server spells orientation "orientation_deg";
        # * the server reports degradation as flags + message pairs
        #   (text_/table_/formula_degraded), which the native path turns
        #   into warnings strings — do the same here, or a degraded server
        #   page parses as clean.
        warnings_ = list(d.get("warnings", []))
        for kind in ("text", "table", "formula"):
            if d.get(f"{kind}_degraded"):
                msg = f"{kind}_degraded: {d.get(f'{kind}_warning') or 'no detail'}"
                if msg not in warnings_:  # flags + a warnings array: no dupes
                    warnings_.append(msg)
        return cls(
            lines=[TextLine.from_dict(x)
                   for x in (d.get("results") or d.get("lines") or [])],
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            page=d.get("page"),
            dpi=d.get("dpi"),
            orientation=int(d.get("orientation",
                                  d.get("orientation_deg", 0)) or 0),
            layout=[LayoutBox.from_dict(x) for x in d.get("layout", [])],
            tables=[TableRegion.from_dict(x) for x in d.get("tables", [])],
            formulas=[FormulaRegion.from_dict(x) for x in d.get("formulas", [])],
            reading_order=list(d.get("reading_order", [])),
            warnings=warnings_,
            stages=tuple(d.get("stages", ())),
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

        # Tables/formulas whose layout_id points at no region we emitted are
        # appended rather than dropped. A fresh read can no longer produce
        # this (tables/formulas imply layout in the RESULT, and layout=False
        # with tables=True is refused), but a PageResult does not only come
        # from a fresh read: from_dict() of output serialized before that
        # implication existed, or a hand-built page, can still carry a table
        # whose region is absent — and silently losing content the engine
        # already paid to recognize is the wrong failure mode for an export.
        emitted_t = {r.id for r in self.layout if r.label == "table"}
        emitted_f = {r.id for r in self.layout if r.label in _FORMULA_LABELS}
        for t in self.tables:
            if t.layout_id not in emitted_t:
                blocks.append(("table", t.html))
        for f in self.formulas:
            if f.layout_id not in emitted_f:
                blocks.append(("formula", f.latex))
        return blocks

    def to_markdown(self, *, structured: Optional[bool] = None) -> str:
        """Markdown for the page.

        When layout regions are present (``OCR(layout=True)``) this is
        layout-aware by default: title regions become headings, tables become
        Markdown tables, formulas become ``$$…$$``, in reading order. Without
        layout it falls back to reading-order paragraphs. ``structured=False``
        forces the flat form; ``structured=True`` forces the structured form."""
        # "Is there structure to lay out?" is not the same question as "are
        # there layout REGIONS?": OCR(tables=True).read(img, layout=False)
        # recognizes a table and leaves self.layout empty, and keying off
        # layout alone silently dropped it from this export.
        has_structure = bool(self.layout or self.tables or self.formulas)
        want = has_structure if structured is None else structured
        if structured and not has_structure:
            import warnings as _w

            _w.warn("structured=True but no layout regions, tables or formulas "
                    "— run OCR(layout=True).", stacklevel=2)
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
            if (self.layout or self.tables or self.formulas)
            else [("p", ln.text) for ln in self.lines if ln.text.strip()]
        )
        out: List[str] = []
        for kind, payload in blocks:
            if kind == "table":
                # Splice the <table> ELEMENT, not the document around it. The
                # table stage returns a complete "<html><body><table>...</table>
                # </body></html>", so appending it verbatim nested one full HTML
                # document per table inside the body — a 54-table run produced 55
                # <html> and 55 <body> tags. Browsers recover; strict/XHTML
                # parsers and pandoc do not.
                out.append(_table_element(payload))
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
        # The PDF route wraps pages in {"pages": [...]}; the /ocr/batch route
        # uses {"batch_results": [...]} — accept both. A dict with NEITHER
        # used to return a silently empty document; refuse it instead.
        pages = d.get("pages")
        if pages is None:
            pages = d.get("batch_results")
        if pages is None and d:
            raise ValueError(
                "not a serialized DocumentResult: expected 'pages' (PDF "
                f"route) or 'batch_results' (/ocr/batch), got keys "
                f"{sorted(d.keys())[:8]}"
            )
        return cls(
            pages=[PageResult.from_dict(x) for x in (pages or [])],
            source=d.get("source", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "DocumentResult":
        return cls.from_dict(json.loads(s))

    def to_markdown(self, *, structured: Optional[bool] = None) -> str:
        parts: List[str] = []
        for p in self.pages:
            if p.page is not None:
                parts.append(f"<!-- page {p.page} -->")
            parts.append(p.to_markdown(structured=structured))
        return "\n\n".join(x for x in parts if x)

    def to_tsv(self) -> str:
        """Tab-separated across all pages, with a leading ``page`` column."""
        rows = ["page\tindex\tconfidence\tx0\ty0\tx1\ty1\ttext"]
        for p in self.pages:
            pg = p.page if p.page is not None else 0
            rows.extend(_tsv_row(ln, i, page=pg)
                        for i, ln in enumerate(p.lines))
        return "\n".join(rows)

    def to_hocr(self) -> str:
        """A single multi-page hOCR document over all pages."""
        divs = [p._hocr_page_div(page_id=(p.page or i + 1)) for i, p in enumerate(self.pages)]
        return _hocr_document(divs)

    def tables_to_pandas(self):
        """Every recognized table in the document as its own DataFrame, in
        page order (see :meth:`PageResult.tables_to_pandas`); each frame's
        ``attrs["page"]`` says which page it came from."""
        return [df for p in self.pages for df in p.tables_to_pandas()]

    def to_pandas(self):
        """The text LINES of all pages as one DataFrame with a ``page`` column
        (needs pandas). For the recognized tables use
        :meth:`tables_to_pandas`."""
        pd = _require_pandas()

        # Build the rows once and hand pandas a single list, instead of
        # constructing a DataFrame per page and concatenating: for a
        # several-hundred-page scan that was N frame allocations plus an N-way
        # concat to produce exactly the same table.
        rows = [
            _line_row(ln, page=p.page if p.page is not None else 0)
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
        Needs page images — pass ``keep_image=True`` to read_pdf/read_batch
        (they drop rasters by default), or use ``pdf_to_searchable()``."""
        from .searchable_pdf import build_searchable_pdf

        build_searchable_pdf(self.pages, out_path=path)
        return path
