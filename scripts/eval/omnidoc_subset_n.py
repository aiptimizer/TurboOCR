#!/usr/bin/env python3
"""Emit an N-doc stratified OmniDocBench subset and a filtered ground-truth JSON.

Selection priority (so the table + formula paths are exercised hard while the
set stays representative):

  1. Every doc that has BOTH a table and an equation_isolated block.
  2. Round-robin top-up from the table-bearing and formula-bearing pools,
     each pick chosen to flatten the running (data_source, language, layout)
     histogram of the already-selected set (greedy max-coverage stratification).
  3. General text docs (text_block but neither table nor formula), same
     stratified greedy fill, until N is reached.

Selection is deterministic (sorted inputs, stable tie-breaks) so reruns
produce an identical subset. Every selected image_path is verified to exist
under data/images.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

_OB = Path(os.environ.get("OMNIDOCBENCH",
                          Path(__file__).resolve().parents[3] / "omnidocbench"))
GT_PATH = _OB / "data" / "OmniDocBench.json"
IMAGES_DIR = _OB / "data" / "images"
OUT_DIR = Path("/tmp/omnidoc_subset125")

STRATA_KEYS = ("data_source", "language", "layout")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, default=GT_PATH)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--n", type=int, default=125)
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    return p.parse_args()


def _cats(doc: dict) -> set[str]:
    return {b.get("category_type") for b in doc.get("layout_dets", [])}


def _strata(doc: dict) -> tuple:
    attr = doc["page_info"].get("page_attribute", {})
    return tuple(attr.get(k) for k in STRATA_KEYS)


def _strata_cost(running: Counter, doc: dict) -> tuple:
    """Lower is better: prefer the doc whose (per-dimension) attribute values
    are least represented so far. Tie-break on image_path for determinism."""
    attr = doc["page_info"].get("page_attribute", {})
    cost = sum(running[(k, attr.get(k))] for k in STRATA_KEYS)
    return (cost, doc["page_info"]["image_path"])


def _bump(running: Counter, doc: dict) -> None:
    attr = doc["page_info"].get("page_attribute", {})
    for k in STRATA_KEYS:
        running[(k, attr.get(k))] += 1


def _greedy_fill(pool: list[dict], running: Counter, chosen_paths: set[str],
                 selected: list[dict], limit: int) -> None:
    """Greedily add docs from `pool` (not already chosen) that most flatten the
    running strata histogram, until `selected` reaches `limit`."""
    avail = [d for d in pool if d["page_info"]["image_path"] not in chosen_paths]
    while avail and len(selected) < limit:
        best = min(avail, key=lambda d: _strata_cost(running, d))
        selected.append(best)
        path = best["page_info"]["image_path"]
        chosen_paths.add(path)
        _bump(running, best)
        avail = [d for d in avail if d["page_info"]["image_path"] not in chosen_paths]


def main() -> int:
    args = parse_args()
    with args.gt.open() as f:
        full = json.load(f)

    # Deterministic order for all downstream selection.
    full = sorted(full, key=lambda d: d["page_info"]["image_path"])

    has_table = [d for d in full if "table" in _cats(d)]
    has_formula = [d for d in full if "equation_isolated" in _cats(d)]
    both = [d for d in full
            if "table" in _cats(d) and "equation_isolated" in _cats(d)]
    table_only = [d for d in has_table
                  if "equation_isolated" not in _cats(d)]
    formula_only = [d for d in has_formula
                    if "table" not in _cats(d)]
    text_only = [d for d in full
                 if "text_block" in _cats(d)
                 and "table" not in _cats(d)
                 and "equation_isolated" not in _cats(d)]

    selected: list[dict] = []
    chosen: set[str] = set()
    running: Counter = Counter()

    # 1. all table+formula docs
    for d in both:
        selected.append(d)
        chosen.add(d["page_info"]["image_path"])
        _bump(running, d)

    # 2. balance the remaining table/formula budget. Target ~ split the rest
    #    evenly between table-only and formula-only, but never overshoot N.
    remaining = args.n - len(selected)
    tf_budget = min(remaining, len(table_only) + len(formula_only))
    # round-robin pick: alternate the pool that is currently least represented
    # among selected docs, each pick stratified.
    tbl_left = list(table_only)
    fml_left = list(formula_only)
    n_tbl_sel = sum(1 for d in selected if "table" in _cats(d))
    n_fml_sel = sum(1 for d in selected if "equation_isolated" in _cats(d))
    added_tf = 0
    while added_tf < tf_budget and (tbl_left or fml_left):
        # choose pool with fewer currently-selected representatives
        take_table = None
        if tbl_left and fml_left:
            take_table = n_tbl_sel <= n_fml_sel
        elif tbl_left:
            take_table = True
        else:
            take_table = False
        pool = tbl_left if take_table else fml_left
        pool[:] = [d for d in pool if d["page_info"]["image_path"] not in chosen]
        if not pool:
            continue
        best = min(pool, key=lambda d: _strata_cost(running, d))
        selected.append(best)
        chosen.add(best["page_info"]["image_path"])
        _bump(running, best)
        pool.remove(best)
        if take_table:
            n_tbl_sel += 1
        else:
            n_fml_sel += 1
        added_tf += 1

    # 3. top up with general text docs, stratified.
    _greedy_fill(text_only, running, chosen, selected, args.n)
    # 3b. if still short (tiny GT), backfill from anything remaining.
    if len(selected) < args.n:
        _greedy_fill(full, running, chosen, selected, args.n)

    # Stable output ordering by image_path.
    selected = sorted(selected, key=lambda d: d["page_info"]["image_path"])

    # Verify every image exists.
    missing = [d["page_info"]["image_path"] for d in selected
               if not (args.images_dir / d["page_info"]["image_path"]).exists()]
    if missing:
        print(f"[omnidoc_subset_n] MISSING images on disk: {missing}",
              file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"OmniDocBench_subset{args.n}.json"
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(selected, ensure_ascii=False))
    tmp.replace(out_json)

    # Composition report.
    n_table_sel = sum(1 for d in selected if "table" in _cats(d))
    n_formula_sel = sum(1 for d in selected if "equation_isolated" in _cats(d))
    n_both_sel = sum(1 for d in selected
                     if "table" in _cats(d) and "equation_isolated" in _cats(d))
    n_text_sel = sum(1 for d in selected if "text_block" in _cats(d))

    for d in selected:
        print(d["page_info"]["image_path"])
    print(f"[omnidoc_subset_n] wrote {out_json} (n={len(selected)})",
          file=sys.stderr)
    print(f"[omnidoc_subset_n] composition: table={n_table_sel} "
          f"formula={n_formula_sel} both={n_both_sel} text_block={n_text_sel}",
          file=sys.stderr)
    ds = Counter(d["page_info"]["page_attribute"].get("data_source")
                 for d in selected)
    lang = Counter(d["page_info"]["page_attribute"].get("language")
                   for d in selected)
    lay = Counter(d["page_info"]["page_attribute"].get("layout")
                  for d in selected)
    print(f"[omnidoc_subset_n] data_source={dict(ds)}", file=sys.stderr)
    print(f"[omnidoc_subset_n] language={dict(lang)}", file=sys.stderr)
    print(f"[omnidoc_subset_n] layout={dict(lay)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
