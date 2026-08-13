from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from typing import Annotated, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

type Point = tuple[int, int]
type Quad = tuple[Point, Point, Point, Point]


class LayoutLabel(StrEnum):
    abstract = "abstract"
    algorithm = "algorithm"
    aside_text = "aside_text"
    chart = "chart"
    content = "content"
    display_formula = "display_formula"
    doc_title = "doc_title"
    figure_title = "figure_title"
    footer = "footer"
    footer_image = "footer_image"
    footnote = "footnote"
    formula_number = "formula_number"
    header = "header"
    header_image = "header_image"
    image = "image"
    inline_formula = "inline_formula"
    number = "number"
    paragraph_title = "paragraph_title"
    reference = "reference"
    reference_content = "reference_content"
    seal = "seal"
    table = "table"
    text = "text"
    vertical_text = "vertical_text"
    vision_footnote = "vision_footnote"
    supplementary_region = "SupplementaryRegion"


class PdfMode(StrEnum):
    ocr = "ocr"
    auto = "auto"
    auto_verified = "auto_verified"
    geometric = "geometric"


# extra="allow" preserves new server fields on .model_extra so an additive
# server change (e.g. a new "request_id" key) parses cleanly without code
# changes here. A *rename* of a required field still raises
# ValidationError because all required fields have no default — only
# conditionally-emitted fields (layout / reading_order / blocks) carry
# default_factory=list, matching the server's "omit when not requested"
# contract.
class _Frozen(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        populate_by_name=True,
        validate_assignment=False,
        ser_json_inf_nan="constants",
    )


class BoundingBox(_Frozen):
    points: Annotated[Quad, Field(description="4-corner polygon")]

    @field_validator("points", mode="before")
    @classmethod
    def _coerce(cls, v: object) -> object:
        if isinstance(v, list):
            return tuple((int(p[0]), int(p[1])) for p in v)
        return v

    @model_serializer
    def _serialize(self) -> list[list[int]]:
        # Match the server wire format: `[[x,y],[x,y],[x,y],[x,y]]`.
        return [list(p) for p in self.points]

    @cached_property
    def aabb(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    @cached_property
    def center(self) -> Point:
        x0, y0, x1, y1 = self.aabb
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    @cached_property
    def width(self) -> int:
        x0, _, x1, _ = self.aabb
        return x1 - x0

    @cached_property
    def height(self) -> int:
        _, y0, _, y1 = self.aabb
        return y1 - y0


class TextItem(_Frozen):
    text: str
    confidence: float
    bounding_box: BoundingBox
    id: int | None = None
    layout_id: int | None = None
    source: Literal["ocr", "pdf", "geometric", "auto", "auto_verified"] | None = None

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _wrap_bbox(cls, v: object) -> object:
        if isinstance(v, list):
            return {"points": v}
        return v


class LayoutBox(_Frozen):
    class_name: Annotated[str, Field(alias="class")]
    class_id: int
    confidence: float
    bounding_box: BoundingBox
    id: int | None = None

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _wrap_bbox(cls, v: object) -> object:
        if isinstance(v, list):
            return {"points": v}
        return v


class Block(_Frozen):
    id: int
    layout_id: int
    class_name: Annotated[str, Field(alias="class")]
    bounding_box: BoundingBox
    content: str
    order_index: int

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _wrap_bbox(cls, v: object) -> object:
        if isinstance(v, list):
            return {"points": v}
        return v


_TABLE_LABELS: Final[frozenset[str]] = frozenset({LayoutLabel.table.value})
_FORMULA_LABELS: Final[frozenset[str]] = frozenset(
    {
        LayoutLabel.display_formula.value,
        LayoutLabel.formula_number.value,
        LayoutLabel.inline_formula.value,
    }
)


class Table(_Frozen):
    """A recognized table.

    Server-recognized tables (`tables=True`, SLANet-Plus) carry `html` (the
    full `<table>` markup), `layout_id`, and the recognizer `confidence`.
    Tables synthesized from layout blocks (no table backend requested) carry
    only the region `text` and `id`.
    """

    bounding_box: BoundingBox
    html: str | None = None
    layout_id: int | None = None
    confidence: float | None = None
    # Synthesized-from-blocks fields (legacy shape, kept for compatibility).
    id: int | None = None
    text: str = ""

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _wrap_bbox(cls, v: object) -> object:
        if isinstance(v, list):
            return {"points": v}
        return v


class Formula(_Frozen):
    """A recognized formula.

    Server-recognized formulas (`formulas=True`, PP-FormulaNet) carry
    `latex`, `layout_id`, and the recognizer `confidence`; `text` mirrors
    `latex` so downstream code can treat both sources uniformly. Formulas
    synthesized from layout blocks carry only the region `text` and `id`.
    """

    bounding_box: BoundingBox
    latex: str | None = None
    layout_id: int | None = None
    confidence: float | None = None
    id: int | None = None
    text: str = ""
    is_inline: bool = False

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _wrap_bbox(cls, v: object) -> object:
        if isinstance(v, list):
            return {"points": v}
        return v

    def model_post_init(self, __context: object) -> None:
        if not self.text and self.latex:
            object.__setattr__(self, "text", self.latex)


def _synthesize_tables_raw(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": b.get("id"),
            "bounding_box": b.get("bounding_box"),
            "text": b.get("content", ""),
            "layout_id": b.get("layout_id"),
        }
        for b in blocks
        if b.get("class") in _TABLE_LABELS
    ]


def _synthesize_formulas_raw(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": b.get("id"),
            "bounding_box": b.get("bounding_box"),
            "text": b.get("content", ""),
            "layout_id": b.get("layout_id"),
            "is_inline": b.get("class") == LayoutLabel.inline_formula.value,
        }
        for b in blocks
        if b.get("class") in _FORMULA_LABELS
    ]


def _inject_synthesized(data: object) -> object:
    """When the server omitted `tables` / `formulas` (backend not requested),
    synthesize them from block regions so `.tables` / `.formulas` keep
    working. First-class server fields always win."""
    if not isinstance(data, dict):
        return data
    blocks = data.get("blocks")
    if isinstance(blocks, list) and blocks and isinstance(blocks[0], dict):
        if "tables" not in data:
            data["tables"] = _synthesize_tables_raw(blocks)
        if "formulas" not in data:
            data["formulas"] = _synthesize_formulas_raw(blocks)
    return data


class OcrResponse(_Frozen):
    """Single-image OCR result returned by `recognize_image` / `recognize_pixels`.

    Attributes:
        results: Token-level text items in detection order. Always
            populated.
        layout: Region boxes (titles, paragraphs, tables, figures, …).
            Populated only when the call was made with `layout=True`;
            empty list otherwise.
        reading_order: Indices into `results` giving human reading order.
            Populated only when `reading_order=True` was requested.
        blocks: Paragraph-level groupings with their own reading order.
            Populated only when `include_blocks=True` was requested.
        text: Cached property — the full text joined in reading order
            (blocks > reading_order > raw `results`).
        tables: Computed view of `blocks` filtered to table-like labels.
        formulas: Computed view of `blocks` filtered to formula-like
            labels (display, numbered, and inline).
    """

    results: list[TextItem]
    layout: list[LayoutBox] = Field(default_factory=list)
    reading_order: list[int] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    # First-class server fields when `tables=True` / `formulas=True` was
    # requested (SLANet-Plus HTML / PP-FormulaNet LaTeX). When the server
    # omitted them, the model_validator below synthesizes entries from the
    # table/formula-labelled blocks so `.tables` / `.formulas` always work.
    tables: list[Table] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    # Fail-loud degradation contract (server v3.1+): a configured stage that
    # produced nothing flags itself instead of returning a silent empty.
    text_degraded: bool = False
    table_degraded: bool = False
    formula_degraded: bool = False
    text_warning: str | None = None
    table_warning: str | None = None
    formula_warning: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _synthesize(cls, data: object) -> object:
        return _inject_synthesized(data)

    @cached_property
    def text(self) -> str:
        if self.blocks:
            return "\n\n".join(b.content for b in self.blocks)
        if self.reading_order:
            return "\n".join(self.results[i].text for i in self.reading_order)
        return "\n".join(r.text for r in self.results)


class BatchSuccess(_Frozen):
    index: int
    response: OcrResponse


class BatchFailure(_Frozen):
    index: int
    error: str


type BatchResult = BatchSuccess | BatchFailure


class BatchResponse(_Frozen):
    """Multi-image OCR result returned by `recognize_batch`.

    `batch_results` and `errors` are *parallel* lists of equal length:
    slot `i` is either a valid [`OcrResponse`][turboocr.OcrResponse] with
    `errors[i] is None`, or a failure where `errors[i]` carries the
    server's error message and `batch_results[i]` is an empty placeholder
    `OcrResponse(results=[])`. Per-slot failures never raise — they land
    in `errors` so one bad input cannot fail the whole batch.

    Prefer [`iter_results`][turboocr.BatchResponse.iter_results] for a
    tagged-union iteration instead of zipping the two lists manually.

    Attributes:
        batch_results: One `OcrResponse` per input, in submission order.
        errors: Parallel `str | None` list — `None` for successes, the
            server's error message otherwise.
    """

    batch_results: list[OcrResponse]
    errors: list[str | None]

    def iter_results(self) -> list[BatchResult]:
        """Pair `batch_results` and `errors` into a tagged-union list.

        Pythonic iteration; saves users from zipping two parallel lists.
        """
        if len(self.batch_results) != len(self.errors):
            raise ValueError(
                f"length mismatch: batch_results={len(self.batch_results)} "
                f"errors={len(self.errors)}"
            )
        out: list[BatchResult] = []
        for i, (r, e) in enumerate(zip(self.batch_results, self.errors, strict=True)):
            if e is None:
                out.append(BatchSuccess(index=i, response=r))
            else:
                out.append(BatchFailure(index=i, error=e))
        return out


class PdfPage(_Frozen):
    page: int
    page_index: int
    dpi: int
    width: int
    height: int
    results: list[TextItem]
    layout: list[LayoutBox] = Field(default_factory=list)
    reading_order: list[int] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    formulas: list[Formula] = Field(default_factory=list)
    mode: PdfMode
    text_layer_quality: str = "absent"
    # `?autorotate=1`: detected clockwise page rotation (only on de-rotated pages).
    orientation_deg: int | None = None
    # `?images=inline`: the rendered page shipped back alongside the OCR result.
    image_b64: str | None = None
    image_content_type: str | None = None
    text_degraded: bool = False
    table_degraded: bool = False
    formula_degraded: bool = False
    text_warning: str | None = None
    table_warning: str | None = None
    formula_warning: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _synthesize(cls, data: object) -> object:
        return _inject_synthesized(data)

    def _as_ocr_response(self) -> OcrResponse:
        return OcrResponse(
            results=self.results,
            layout=self.layout,
            reading_order=self.reading_order,
            blocks=self.blocks,
            tables=self.tables,
            formulas=self.formulas,
        )

    @cached_property
    def image_bytes(self) -> bytes | None:
        """Decoded page image when the request used `images="inline"`."""
        if self.image_b64 is None:
            return None
        import base64

        return base64.b64decode(self.image_b64)


class PdfResponse(_Frozen):
    """Multi-page PDF OCR result returned by `recognize_pdf`.

    Attributes:
        pages: One [`PdfPage`][turboocr.PdfPage] per input page, in order.
            Each page carries its own `results`, optional `layout`,
            `reading_order`, `blocks`, plus per-page metadata (`page`,
            `page_index`, `dpi`, `width`, `height`, `mode`,
            `text_layer_quality`).
        text: Cached property — full document text joined across pages
            with blank-line separators.
        tables: Flattened view of `tables` across every page.
        formulas: Flattened view of `formulas` across every page.
    """

    pages: list[PdfPage]

    @cached_property
    def text(self) -> str:
        return "\n\n".join(p._as_ocr_response().text for p in self.pages)

    @property
    def tables(self) -> list[Table]:
        return [t for p in self.pages for t in p.tables]

    @property
    def formulas(self) -> list[Formula]:
        return [f for p in self.pages for f in p.formulas]


class HealthStatus(_Frozen):
    ok: bool
    status_code: int
    body: str
    body_json: dict[str, object] | None = None


class CapabilityFeatures(_Frozen):
    layout: bool = False
    tables: bool = False
    formulas: bool = False
    autorotate: bool = False


class CapabilityPdf(_Frozen):
    modes: list[str] = Field(default_factory=list)
    default_dpi: int | None = None
    max_pages: int | None = None


class CapabilityLimits(_Frozen):
    max_body_mb: int | None = None
    max_image_dim: int | None = None
    max_batch_images: int | None = None


class Capabilities(_Frozen):
    """`GET /capabilities` — what the running server actually loaded.

    Check `features.tables` / `features.formulas` before sending
    `tables=True` / `formulas=True`: those are strict opt-ins and return
    `400 TABLE_BACKEND_DISABLED` / `FORMULA_BACKEND_DISABLED` when the
    backend was not configured at server startup.
    """

    build: str | None = None
    features: CapabilityFeatures = Field(default_factory=CapabilityFeatures)
    pdf: CapabilityPdf = Field(default_factory=CapabilityPdf)
    limits: CapabilityLimits = Field(default_factory=CapabilityLimits)
    endpoints: list[str] = Field(default_factory=list)


class MarkdownPage(_Frozen):
    """One page of a server-side PDF → Markdown conversion (`as_pages=True`)."""

    page_index: int
    markdown: str
    text_degraded: bool = False
    table_degraded: bool = False
    formula_degraded: bool = False


class MarkdownPagesResponse(_Frozen):
    pages: list[MarkdownPage]

    @cached_property
    def markdown(self) -> str:
        """All pages joined in order, separated by blank lines."""
        return "\n\n".join(p.markdown for p in self.pages)


class StreamEvent(_Frozen):
    """One NDJSON line from `POST /ocr/stream`.

    `event` is one of `"meta"`, `"page"`, `"page_error"`, `"error"`,
    `"end"`. Page events arrive as each page completes — out of order by
    design; use `page.page_index` to reorder client-side if needed.
    """

    event: Literal["meta", "page", "page_error", "error", "end"]
    # meta
    kind: str | None = None
    pages: int | None = None
    dpi: int | None = None
    mode: str | None = None
    # page_error / error
    page_index: int | None = None
    code: str | None = None
    # end
    failed: int | None = None

    @cached_property
    def page(self) -> PdfPage | None:
        """The parsed page for `event == "page"`, else `None`."""
        if self.event != "page":
            return None
        extra = self.model_extra or {}
        data = {k: v for k, v in extra.items()}
        data["page_index"] = self.page_index
        if self.dpi is not None:
            data["dpi"] = self.dpi
        if self.mode is not None:
            data["mode"] = self.mode
        return PdfPage.model_validate(data)
