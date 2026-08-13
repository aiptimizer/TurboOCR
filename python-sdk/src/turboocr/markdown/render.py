"""Markdown render walker: turn an OCR/PDF response into a `MarkdownDocument`.

The style/registry logic lives in `style.py`; this module is the transformation.
"""
from __future__ import annotations

from ..errors import ProtocolError
from ..models import (
    Block,
    LayoutBox,
    OcrResponse,
    PdfResponse,
    TextItem,
)
from .style import (
    DEFAULT_STYLE,
    MarkdownDocument,
    MarkdownNode,
    MarkdownStyle,
    NodeKind,
)


def _node_from_block(block: Block, style: MarkdownStyle) -> MarkdownNode:
    rule = style.classify(block.class_name)
    return MarkdownNode(
        kind=rule.kind,
        text=block.content.strip(),
        level=rule.level,
        label=block.class_name,
        bounding_box=block.bounding_box,
        layout_id=block.layout_id,
        order_index=block.order_index,
    )


def _node_from_layout_and_texts(
    layout: LayoutBox, texts: list[TextItem], order_index: int, style: MarkdownStyle
) -> MarkdownNode:
    rule = style.classify(layout.class_name)
    joined = " ".join(t.text for t in texts).strip()
    return MarkdownNode(
        kind=rule.kind,
        text=joined,
        level=rule.level,
        label=layout.class_name,
        bounding_box=layout.bounding_box,
        layout_id=layout.id,
        order_index=order_index,
    )


def _index_layout_by_id(layout: list[LayoutBox]) -> dict[int, LayoutBox]:
    indexed: dict[int, LayoutBox] = {}
    for box in layout:
        if box.id is None:
            raise ProtocolError("layout box missing id in layout-enabled response")
        indexed[box.id] = box
    return indexed


def _group_texts_by_layout_required(response: OcrResponse) -> dict[int, list[TextItem]]:
    buckets: dict[int, list[TextItem]] = {}
    order = response.reading_order or list(range(len(response.results)))
    for idx in order:
        item = response.results[idx]
        if item.layout_id is None:
            raise ProtocolError(
                f"text item {idx} has no layout_id in a layout-enabled response"
            )
        buckets.setdefault(item.layout_id, []).append(item)
    return buckets


def _nodes_from_response(response: OcrResponse, style: MarkdownStyle) -> list[MarkdownNode]:
    if response.blocks:
        return [_node_from_block(b, style) for b in response.blocks]
    if response.layout and response.reading_order:
        buckets = _group_texts_by_layout_required(response)
        layout_by_id = _index_layout_by_id(response.layout)
        nodes: list[MarkdownNode] = []
        seen_layouts: set[int] = set()
        for order_index, text_idx in enumerate(response.reading_order):
            lid = response.results[text_idx].layout_id
            if lid is None:
                raise ProtocolError(
                    f"text item {text_idx} has no layout_id in a layout-enabled response"
                )
            if lid in seen_layouts:
                continue
            seen_layouts.add(lid)
            layout_box = layout_by_id.get(lid)
            if layout_box is None:
                raise ProtocolError(
                    f"text item {text_idx} references layout_id={lid} not present in layout"
                )
            nodes.append(
                _node_from_layout_and_texts(
                    layout_box, buckets.get(lid, []), order_index, style
                )
            )
        return nodes
    return [
        MarkdownNode(
            kind=NodeKind.paragraph,
            text=item.text,
            label=None,
            bounding_box=item.bounding_box,
            order_index=i,
        )
        for i, item in enumerate(response.results)
    ]


def _join_markdown(nodes: list[MarkdownNode], style: MarkdownStyle) -> str:
    rendered = [r for n in nodes if (r := style.render(n))]
    return "\n\n".join(rendered) + ("\n" if rendered else "")


def render_ocr_to_markdown(
    response: OcrResponse, *, style: MarkdownStyle | None = None
) -> MarkdownDocument:
    style = style or DEFAULT_STYLE
    nodes = _nodes_from_response(response, style)
    return MarkdownDocument(
        source="image", pages=1, nodes=nodes, markdown=_join_markdown(nodes, style)
    )


def render_pdf_to_markdown(
    response: PdfResponse, *, style: MarkdownStyle | None = None
) -> MarkdownDocument:
    style = style or DEFAULT_STYLE
    all_nodes: list[MarkdownNode] = []
    for page_num, page in enumerate(response.pages, start=1):
        page_nodes = _nodes_from_response(page._as_ocr_response(), style)
        all_nodes.extend(page_nodes)
        if page_num < len(response.pages):
            all_nodes.append(MarkdownNode(kind=NodeKind.page_break, text=""))
    return MarkdownDocument(
        source="pdf",
        pages=len(response.pages),
        nodes=all_nodes,
        markdown=_join_markdown(all_nodes, style),
    )


def render_to_markdown(
    response: OcrResponse | PdfResponse, *, style: MarkdownStyle | None = None
) -> MarkdownDocument:
    """Render an OCR or PDF response to a [`MarkdownDocument`][turboocr.MarkdownDocument].

    Dispatches on the response type: [`PdfResponse`][turboocr.PdfResponse]
    is rendered page-by-page with `---` page-break separators;
    [`OcrResponse`][turboocr.OcrResponse] is rendered as a single page.

    Args:
        response: Either an `OcrResponse` (single image) or a
            `PdfResponse` (multi-page). Must have been produced with
            `include_blocks=True` for richest output; bare token-only
            responses fall back to one paragraph per token line.
        style: [`MarkdownStyle`][turboocr.MarkdownStyle] controlling
            label-to-node classification and per-kind renderers. `None`
            uses `DEFAULT_STYLE`.

    Returns:
        A `MarkdownDocument` with the rendered `.markdown` string, the
        structured `.nodes` list, the `.source` discriminator
        (`"image"` / `"pdf"`), and `.pages`.

    Raises:
        ProtocolError: A layout-enabled response is internally
            inconsistent — e.g. a text item without `layout_id`, or a
            `reading_order` index pointing at a layout box that is not in
            the response.
    """
    if isinstance(response, PdfResponse):
        return render_pdf_to_markdown(response, style=style)
    return render_ocr_to_markdown(response, style=style)
