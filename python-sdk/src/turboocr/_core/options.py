from __future__ import annotations

from dataclasses import dataclass

# Server contract couples the flags: include_blocks ⇒ reading_order ⇒ layout,
# and tables / formulas ⇒ layout. We auto-promote so users get the richer
# output by default when they ask for any downstream feature, instead of
# getting a server 400 for missing the prereq.
# Users who want a flag explicitly OFF pass `False` (not None).


@dataclass(frozen=True, slots=True)
class OcrOptions:
    layout: bool | None = None
    reading_order: bool | None = None
    include_blocks: bool | None = None
    tables: bool | None = None
    formulas: bool | None = None

    def __post_init__(self) -> None:
        layout, reading_order, include_blocks = self.layout, self.reading_order, self.include_blocks
        if include_blocks and reading_order is None:
            reading_order = True
        if (reading_order or self.tables or self.formulas) and layout is None:
            layout = True
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "reading_order", reading_order)
        object.__setattr__(self, "include_blocks", include_blocks)

    def to_query_params(self) -> dict[str, str]:
        # SDK kwarg `include_blocks` maps to server query param `as_blocks`
        # — server contract uses the older name.
        params: dict[str, str] = {}
        if self.layout is not None:
            params["layout"] = "1" if self.layout else "0"
        if self.reading_order is not None:
            params["reading_order"] = "1" if self.reading_order else "0"
        if self.include_blocks is not None:
            params["as_blocks"] = "1" if self.include_blocks else "0"
        if self.tables is not None:
            params["tables"] = "1" if self.tables else "0"
        if self.formulas is not None:
            params["formulas"] = "1" if self.formulas else "0"
        return params
