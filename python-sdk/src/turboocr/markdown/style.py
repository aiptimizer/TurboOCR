"""Markdown style configuration: node kinds, default label→kind rules,
default renderers, and the `MarkdownStyle` registry users override.

The render walker that traverses an OCR response and emits Markdown lives
in `render.py`; this module is configuration only.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from ..models import BoundingBox, LayoutLabel


class NodeKind(StrEnum):
    heading = "heading"
    paragraph = "paragraph"
    list_item = "list_item"
    table = "table"
    figure = "figure"
    formula = "formula"
    inline_formula = "inline_formula"
    code = "code"
    footer = "footer"
    header = "header"
    aside = "aside"
    page_break = "page_break"


class MarkdownNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: NodeKind
    text: str
    level: int = 0
    label: str | None = None
    bounding_box: BoundingBox | None = None
    layout_id: int | None = None
    order_index: int | None = None


class MarkdownDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["image", "pdf"] = "image"
    pages: int = 1
    nodes: list[MarkdownNode]
    markdown: str

    def structured(self) -> list[dict[str, object]]:
        """Return the parsed nodes as plain dicts, for programmatic inspection.

        Equivalent to `[n.model_dump(exclude_none=True) for n in self.nodes]`.
        Useful when you want to walk the document tree without depending on
        the [`MarkdownNode`][turboocr.MarkdownNode] type — e.g. when handing
        the structure to a templating engine or another language runtime.
        """
        return [n.model_dump(exclude_none=True) for n in self.nodes]


@dataclass(frozen=True, slots=True)
class StyleRule:
    kind: NodeKind
    level: int = 0


_DEFAULT_RULES: Final[Mapping[str, StyleRule]] = {
    LayoutLabel.doc_title.value: StyleRule(NodeKind.heading, 1),
    LayoutLabel.abstract.value: StyleRule(NodeKind.heading, 2),
    LayoutLabel.paragraph_title.value: StyleRule(NodeKind.heading, 2),
    LayoutLabel.figure_title.value: StyleRule(NodeKind.heading, 3),
    LayoutLabel.image.value: StyleRule(NodeKind.figure),
    LayoutLabel.chart.value: StyleRule(NodeKind.figure),
    LayoutLabel.seal.value: StyleRule(NodeKind.figure),
    LayoutLabel.table.value: StyleRule(NodeKind.table),
    LayoutLabel.algorithm.value: StyleRule(NodeKind.code),
    LayoutLabel.display_formula.value: StyleRule(NodeKind.formula),
    LayoutLabel.formula_number.value: StyleRule(NodeKind.formula),
    LayoutLabel.inline_formula.value: StyleRule(NodeKind.inline_formula),
    LayoutLabel.footer.value: StyleRule(NodeKind.footer),
    LayoutLabel.footer_image.value: StyleRule(NodeKind.footer),
    LayoutLabel.footnote.value: StyleRule(NodeKind.footer),
    LayoutLabel.vision_footnote.value: StyleRule(NodeKind.footer),
    LayoutLabel.header.value: StyleRule(NodeKind.header),
    LayoutLabel.header_image.value: StyleRule(NodeKind.header),
    LayoutLabel.aside_text.value: StyleRule(NodeKind.aside),
    LayoutLabel.reference.value: StyleRule(NodeKind.list_item),
    LayoutLabel.reference_content.value: StyleRule(NodeKind.list_item),
}


type RenderFn = Callable[[MarkdownNode], str]


class MarkdownStyle:
    """Configurable label-to-Markdown classifier plus per-kind renderers.

    Holds two mappings used by
    [`render_to_markdown`][turboocr.render_to_markdown]: a label →
    `StyleRule` dict and a
    `NodeKind` → renderer-function dict.
    Override either via `register` / `register_renderer` to customise the
    output for project-specific layouts.
    """

    def __init__(
        self,
        *,
        rules: Mapping[str, StyleRule] | None = None,
        default: StyleRule | None = None,
        renderers: Mapping[NodeKind, RenderFn] | None = None,
    ) -> None:
        """Build a `MarkdownStyle` from optional rule/renderer overrides.

        Args:
            rules: Replacement label → `StyleRule` mapping. When `None`,
                the SDK's built-in defaults are used (doc_title → heading
                level 1, paragraph_title → heading level 2, table →
                table, …).
            default: Fallback rule for labels not present in `rules`.
                Defaults to `StyleRule(NodeKind.paragraph)`.
            renderers: Replacement `NodeKind` → render-function mapping.
                When `None`, the SDK's built-in renderers are used.

        Returns:
            A new `MarkdownStyle` instance.
        """
        self._rules: dict[str, StyleRule] = (
            dict(rules) if rules is not None else dict(_DEFAULT_RULES)
        )
        self._default: StyleRule = default or StyleRule(NodeKind.paragraph)
        self._renderers: dict[NodeKind, RenderFn] = (
            dict(renderers) if renderers is not None else dict(_DEFAULT_RENDERERS)
        )

    def register(self, label: str, kind: NodeKind, level: int = 0) -> None:
        """Register or override a label-to-kind classification rule.

        Args:
            label: Layout label as it appears on `LayoutBox.class_name`
                or `Block.class_name` (e.g. `"doc_title"`, `"table"`, or
                a custom server-emitted label).
            kind: Target `NodeKind`.
            level: Heading level (1-6) when `kind=NodeKind.heading`.
                Ignored for other kinds. Default `0`.

        Returns:
            None. Mutates the style in place.
        """
        self._rules[label] = StyleRule(kind, level)

    def register_renderer(self, kind: NodeKind, renderer: RenderFn) -> None:
        """Register or override the render function for a node kind.

        Args:
            kind: The `NodeKind` to
                customise.
            renderer: A callable `(MarkdownNode) -> str` that produces the
                Markdown for one node. Returning an empty string skips the
                node.

        Returns:
            None. Mutates the style in place.
        """
        self._renderers[kind] = renderer

    def classify(self, label: str) -> StyleRule:
        return self._rules.get(label, self._default)

    def render(self, node: MarkdownNode) -> str:
        renderer = self._renderers.get(node.kind)
        return renderer(node) if renderer else ""

    def copy(self) -> MarkdownStyle:
        return MarkdownStyle(
            rules=dict(self._rules), default=self._default, renderers=dict(self._renderers)
        )


def _render_heading(node: MarkdownNode) -> str:
    return f"{'#' * max(1, min(6, node.level))} {node.text}" if node.text else ""


def _render_list_item(node: MarkdownNode) -> str:
    return f"- {node.text}" if node.text else ""


def _render_formula(node: MarkdownNode) -> str:
    return f"$$\n{node.text}\n$$" if node.text else ""


def _render_inline_formula(node: MarkdownNode) -> str:
    return f"${node.text}$" if node.text else ""


def _render_code(node: MarkdownNode) -> str:
    return f"```\n{node.text}\n```" if node.text else ""


def _render_table(node: MarkdownNode) -> str:
    if not node.text:
        return ""
    return node.text if node.text.lstrip().startswith("<table") else f"```\n{node.text}\n```"


def _render_figure(node: MarkdownNode) -> str:
    alt = node.text or (node.label or "figure")
    return f"![{alt}]()"


def _render_comment(node: MarkdownNode) -> str:
    return f"<!-- {node.kind.value}: {node.text} -->" if node.text else ""


def _render_page_break(_: MarkdownNode) -> str:
    return "\n---\n"


def _render_paragraph(node: MarkdownNode) -> str:
    return node.text


_DEFAULT_RENDERERS: Final[Mapping[NodeKind, RenderFn]] = {
    NodeKind.heading: _render_heading,
    NodeKind.list_item: _render_list_item,
    NodeKind.formula: _render_formula,
    NodeKind.inline_formula: _render_inline_formula,
    NodeKind.code: _render_code,
    NodeKind.table: _render_table,
    NodeKind.figure: _render_figure,
    NodeKind.footer: _render_comment,
    NodeKind.header: _render_comment,
    NodeKind.aside: _render_comment,
    NodeKind.page_break: _render_page_break,
    NodeKind.paragraph: _render_paragraph,
}


DEFAULT_STYLE: Final[MarkdownStyle] = MarkdownStyle()
