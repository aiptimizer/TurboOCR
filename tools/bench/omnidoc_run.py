#!/usr/bin/env python3
"""Walk omnidocbench images and POST each to /ocr/raw?layout=1.

Saves raw JSON response per image into the output directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

IMAGES_DIR = Path(__file__).resolve().parents[2] / "omnidocbench/data/images"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N images (0 = all).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip images whose output JSON already exists.")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out-dir", type=Path,
                   default=Path("/tmp/omnidoc_predictions/json"))
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    return p.parse_args()


_CONTENT_TYPES = {".png": "image/png", ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg"}


def process_one(session: requests.Session, server: str, img: Path,
                out_dir: Path) -> tuple[Path, str | None]:
    out = out_dir / (img.stem + ".json")
    try:
        with img.open("rb") as f:
            data = f.read()
        ct = _CONTENT_TYPES.get(img.suffix.lower(), "application/octet-stream")
        r = session.post(
            f"{server}/ocr/raw?layout=1&reading_order=1&tables=1&formulas=1",
            data=data,
            headers={"Content-Type": ct},
            timeout=120,
        )
        r.raise_for_status()
        payload = r.json()
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False))
        tmp.replace(out)
        return img, None
    except Exception as e:  # noqa: BLE001
        return img, f"{type(e).__name__}: {e}"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = args.out_dir / "_errors.txt"

    images = sorted([p for p in args.images_dir.iterdir()
                     if p.suffix.lower() in _CONTENT_TYPES])
    if args.limit > 0:
        images = images[: args.limit]

    if args.skip_existing:
        images = [p for p in images
                  if not (args.out_dir / (p.stem + ".json")).exists()]

    total = len(images)
    print(f"[omnidoc_run] images={total} concurrency={args.concurrency} "
          f"server={args.server} out={args.out_dir}", flush=True)
    if total == 0:
        return 0

    session = requests.Session()
    err_lock = Lock()
    n_ok = 0
    n_err = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(process_one, session, args.server, p, args.out_dir)
                for p in images]
        for i, fut in enumerate(as_completed(futs), 1):
            img, err = fut.result()
            if err:
                n_err += 1
                with err_lock, errors_path.open("a") as f:
                    f.write(f"{img.name}\t{err}\n")
            else:
                n_ok += 1
            if i % 100 == 0 or i == total:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0.0
                print(f"[omnidoc_run] {i}/{total} ok={n_ok} err={n_err} "
                      f"rate={rate:.2f}/s", flush=True)

    print(f"[omnidoc_run] done ok={n_ok} err={n_err} "
          f"elapsed={time.time() - t0:.1f}s", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
