#!/usr/bin/env python3
"""Convert per-image OCR JSON into OmniDocBench-style markdown.

Mapping per internal engineering notes. Tables and formulas
are emitted as v1 placeholders since the pipeline's `tables[]` /
`formulas[]` outputs are empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# class label → emitter token.
SKIP = {"footer", "footer_image", "header", "header_image",
        "image", "chart", "seal"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path,
                   default=Path("/tmp/omnidoc_predictions/json"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("/tmp/omnidoc_predictions/md"))
    return p.parse_args()


def _bbox_minmax(box) -> tuple[float, float, float, float]:
    """box is a 4-point polygon [[x,y]*4]; return (xmin, ymin, xmax, ymax)."""
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def _center(box) -> tuple[float, float]:
    x0, y0, x1, y1 = _bbox_minmax(box)
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def _inside(pt: tuple[float, float], box) -> bool:
    x, y = pt
    x0, y0, x1, y1 = _bbox_minmax(box)
    return x0 <= x <= x1 and y0 <= y <= y1


def _layout_sort_key(blk) -> tuple[float, float]:
    x0, y0, _x1, _y1 = _bbox_minmax(blk["bounding_box"])
    return y0, x0


def gather_text(layout_blk, results) -> str:
    """Pick OCR lines belonging to this layout block.

    Primary: results whose `layout_id` matches the block's `id`.
    Fallback: results whose bbox center falls inside the layout bbox
    (covers cases where the server didn't tag a result).
    """
    lid = layout_blk["id"]
    lines = []
    used = set()

    for r in results:
        if r.get("layout_id") == lid:
            text = (r.get("text") or "").strip()
            if text:
                lines.append((r, text))
                used.add(id(r))

    if not lines:
        for r in results:
            if id(r) in used:
                continue
            if r.get("layout_id") not in (None, -1):
                continue
            box = r.get("bounding_box")
            if not box:
                continue
            if _inside(_center(box), layout_blk["bounding_box"]):
                text = (r.get("text") or "").strip()
                if text:
                    lines.append((r, text))

    lines.sort(key=lambda rt: (_center(rt[0]["bounding_box"])[1],
                               _center(rt[0]["bounding_box"])[0]))
    return " ".join(t for _, t in lines).strip()


def render(payload: dict) -> str:
    layout = payload.get("layout") or []
    results = payload.get("results") or []
    order = payload.get("reading_order")
    # Formula regions are recognized by the formula backend (FORMULA_BACKEND);
    # prefer its LaTeX (keyed by layout_id) over the raw text-rec read.
    formula_by_lid = {
        f["layout_id"]: (f.get("latex") or "").strip()
        for f in (payload.get("formulas") or [])
        if f.get("layout_id") is not None
    }
    # Table regions are recognized by the table-structure backend; prefer its
    # reconstructed HTML (keyed by layout_id) over the empty placeholder.
    table_by_lid = {
        t["layout_id"]: (t.get("html") or "").strip()
        for t in (payload.get("tables") or [])
        if t.get("layout_id") is not None
    }

    if order:
        # Fallback: the reading_order array indexes RESULTS (OCR lines) in XY-cut order,
        # not layout ids. Derive layout-block order = first-encounter layout_id walking
        # results; append any block with no text line in geometric order so nothing drops.
        seen: set = set()
        ordered_ids: list = []
        for ri in order:
            if 0 <= ri < len(results):
                lid = results[ri].get("layout_id")
                if isinstance(lid, int) and lid not in seen:
                    seen.add(lid)
                    ordered_ids.append(lid)
        by_id = {b["id"]: b for b in layout}
        blocks = [by_id[i] for i in ordered_ids if i in by_id]
        blocks += sorted((b for b in layout if b["id"] not in seen),
                         key=_layout_sort_key)
    else:
        blocks = sorted(layout, key=_layout_sort_key)

    refs_open = False
    out: list[str] = []

    def emit(*chunks: str) -> None:
        for c in chunks:
            out.append(c)
        out.append("")

    for blk in blocks:
        cls = blk.get("class") or ""
        if cls in SKIP:
            continue
        text = gather_text(blk, results)

        if cls == "doc_title":
            if text:
                emit(f"# {text}")
        elif cls == "paragraph_title":
            if text:
                emit(f"## {text}")
        elif cls == "figure_title":
            if text:
                emit(f"### {text}")
        elif cls == "abstract":
            if text:
                emit(f"**Abstract** {text}")
        elif cls == "table":
            html = table_by_lid.get(blk.get("id"))
            emit(html if html else "<table></table>")
        elif cls == "display_formula":
            latex = formula_by_lid.get(blk.get("id")) or text
            emit(f"$${latex}$$" if latex else "$$ $$")
        elif cls == "inline_formula":
            latex = formula_by_lid.get(blk.get("id")) or text
            emit(f"${latex}$" if latex else "$ $")
        elif cls in ("reference", "reference_content"):
            if not refs_open:
                emit("### References")
                refs_open = True
            if text:
                emit(f"- {text}")
        elif cls == "content":
            if text:
                for line in text.splitlines():
                    line = line.strip()
                    if line:
                        out.append(f"- {line}")
                out.append("")
        elif cls == "aside_text":
            if text:
                emit(f"> {text}")
        elif cls == "algorithm":
            if text:
                emit("```algorithm", text, "```")
        else:
            # text, vertical_text, footnote, vision_footnote, formula_number,
            # number, SupplementaryRegion, and any unknown labels
            if text:
                emit(text)

    md = "\n".join(out).rstrip() + "\n"
    return md


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(args.in_dir.glob("*.json"))
    total = len(inputs)
    print(f"[omnidoc_to_md] inputs={total} in={args.in_dir} out={args.out_dir}",
          flush=True)

    n_ok = 0
    n_err = 0
    err_path = args.out_dir / "_errors.txt"

    for i, jp in enumerate(inputs, 1):
        out = args.out_dir / (jp.stem + ".md")
        try:
            payload = json.loads(jp.read_text())
            md = render(payload)
            tmp = out.with_suffix(".md.tmp")
            tmp.write_text(md)
            tmp.replace(out)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            with err_path.open("a") as f:
                f.write(f"{jp.name}\t{type(e).__name__}: {e}\n")
        if i % 200 == 0 or i == total:
            print(f"[omnidoc_to_md] {i}/{total} ok={n_ok} err={n_err}",
                  flush=True)

    print(f"[omnidoc_to_md] done ok={n_ok} err={n_err}", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
