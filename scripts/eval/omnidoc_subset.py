#!/usr/bin/env python3
"""Emit a 5-doc OmniDocBench subset and a filtered ground-truth JSON.

The 5 images were hand-picked from the universe of docs that contain
text_block + equation_isolated + table simultaneously, stratified across
data_source / language / layout so a small regression suite still touches
the regimes that exercise text OCR, formula recognition, and table HTML.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GT_PATH = Path(__file__).resolve().parents[3] / "omnidocbench/data/OmniDocBench.json"
OUT_DIR = Path("/tmp/omnidoc_subset5")
OUT_JSON = OUT_DIR / "OmniDocBench_subset5.json"

SUBSET = [
    "docstructbench_llm-raw-scihub-o.O-ceat.200600410.pdf_5.jpg",
    "page-28657a85-95cc-4bfa-8156-cc98518a0b4c.png",
    "book_en_国外数学教材-数论-Melvyn B. Nathanson—Elementary Methods in Number Theory_0100.png",
    "book_zh_CNASGL0262018_extracted_page_82.png",
    "jiaocaineedrop_33137159.pdf_3.jpg",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, default=GT_PATH)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with args.gt.open() as f:
        full = json.load(f)

    wanted = set(SUBSET)
    by_path = {d["page_info"]["image_path"]: d for d in full}
    missing = sorted(wanted - by_path.keys())
    if missing:
        print(f"[omnidoc_subset] MISSING from GT: {missing}", file=sys.stderr)
        return 1

    subset = [by_path[p] for p in SUBSET]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "OmniDocBench_subset5.json"
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(subset, ensure_ascii=False))
    tmp.replace(out_json)

    for p in SUBSET:
        print(p)
    print(f"[omnidoc_subset] wrote {out_json} (n={len(subset)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
