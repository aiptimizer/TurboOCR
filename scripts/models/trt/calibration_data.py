#!/usr/bin/env python3
"""Deterministic 200-image stratified sample from OmniDocBench for INT8 PTQ.

The same sample is consumed by every model's quantization run so a single
calibration corpus drives det/layout/nemotron/formula_encoder INT8 builds. The
distribution covers the OmniDocBench layout/source mix that production sees.

Usage as library:
    from calibration_data import calibration_image_paths
    paths = calibration_image_paths()           # 200 absolute PNG paths
    paths = calibration_image_paths(n=500)      # override count

Usage as CLI:
    python3 calibration_data.py                 # prints the 200 paths, one per line
    python3 calibration_data.py --copy /tmp/calib  # also stages a copy at /tmp/calib
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

OMNIDOC_IMAGES = Path(__file__).resolve().parents[4] / "omnidocbench/data/images"
DEFAULT_N = 200


def _stratum_key(name: str) -> str:
    # OmniDocBench filenames embed source class as the first underscore token
    # (book_en_*, magazine_*, newspaper_*, paper_*, exam_*, …). Falling back
    # to "other" keeps unknown patterns from breaking the stratifier.
    head = name.split("_", 1)[0] if "_" in name else "other"
    return head


def _deterministic_rank(name: str) -> int:
    # Hash-based stable ordering — same input always picks the same sample.
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16)


def calibration_image_paths(n: int = DEFAULT_N,
                            root: Path = OMNIDOC_IMAGES) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"OmniDocBench images dir not found: {root}")

    all_imgs = sorted(p for p in root.iterdir()
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not all_imgs:
        raise RuntimeError(f"No images under {root}")

    # Bucket by source class.
    buckets: dict[str, list[Path]] = {}
    for p in all_imgs:
        buckets.setdefault(_stratum_key(p.name), []).append(p)

    # Sort each bucket by deterministic hash so we don't always pull the
    # alphabetically-first files.
    for k in buckets:
        buckets[k].sort(key=lambda p: _deterministic_rank(p.name))

    # Round-robin pick across buckets until we hit n.
    picked: list[Path] = []
    cursors = {k: 0 for k in buckets}
    keys = sorted(buckets.keys())  # stable order
    while len(picked) < n:
        progressed = False
        for k in keys:
            if len(picked) >= n:
                break
            if cursors[k] < len(buckets[k]):
                picked.append(buckets[k][cursors[k]])
                cursors[k] += 1
                progressed = True
        if not progressed:
            # Exhausted every bucket before reaching n.
            break

    return picked[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--copy", type=Path, default=None,
                    help="Stage a copy of the sample into this directory")
    args = ap.parse_args()

    paths = calibration_image_paths(n=args.n)
    if args.copy is not None:
        args.copy.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, args.copy / p.name)
        print(f"# staged {len(paths)} images under {args.copy}", file=sys.stderr)

    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
