#!/usr/bin/env python3
"""Drive PaddleOCRVL v1.5 pipeline against OmniDocBench images.

Uses the official `paddleocr.PaddleOCRVL` pipeline pointed at a vLLM
OpenAI-compatible server. Saves per-image markdown to <out-dir>/md.

Run with the compare-ocrs venv (Python 3.12 + paddle + paddleocr 3.4):
  /path/to/compare-ocrs/.venv/bin/python tools/bench/omnidoc_run_paddlevl.py ...

Layout detection (PP-DocLayoutV3) runs locally on this host; only the
VL element-level OCR is delegated to vLLM. `vl_rec_max_concurrency`
controls how many in-flight VL requests vLLM sees in parallel
(continuous batching handles them server-side).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "omnidocbench/data/images")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:8003/v1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--vl-concurrency", type=int, default=16,
                   help="vLLM in-flight VL element requests")
    p.add_argument("--doc-workers", type=int, default=2,
                   help="parallel document pipelines (layout det runs on this host)")
    p.add_argument("--pipeline-version", default="v1.5")
    return p.parse_args()


_EXTS = {".png", ".jpg", ".jpeg"}


def _build_pipeline(args):
    from paddleocr import PaddleOCRVL
    return PaddleOCRVL(
        pipeline_version=args.pipeline_version,
        vl_rec_backend="vllm-server",
        vl_rec_server_url=args.server,
        vl_rec_max_concurrency=args.vl_concurrency,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


def _extract_markdown(results) -> str:
    chunks: list[str] = []
    for r in results:
        md = None
        try:
            m = r.markdown
            if isinstance(m, str):
                md = m
            elif isinstance(m, dict):
                md = m.get("markdown_texts") or ""
        except Exception:
            md = None
        if md is None:
            try:
                j = r.json
                if isinstance(j, dict):
                    md = j.get("markdown") or j.get("markdown_texts") or ""
            except Exception:
                md = ""
        chunks.append(md if isinstance(md, str) else str(md))
    return ("\n\n".join(chunks).strip() + "\n") if chunks else "\n"


def _run_one(pipeline, img: Path, md_dir: Path) -> tuple[Path, str | None]:
    out = md_dir / (img.stem + ".md")
    try:
        results = pipeline.predict(str(img))
        md = _extract_markdown(list(results))
        tmp = out.with_suffix(".md.tmp")
        tmp.write_text(md)
        tmp.replace(out)
        return img, None
    except Exception as e:  # noqa: BLE001
        return img, f"{type(e).__name__}: {e}"


def main() -> int:
    args = parse_args()
    md_dir = args.out_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)
    errors_path = args.out_dir / "_errors.txt"

    images = sorted([p for p in args.images_dir.iterdir()
                     if p.suffix.lower() in _EXTS])
    if args.limit > 0:
        images = images[: args.limit]
    if args.skip_existing:
        images = [p for p in images if not (md_dir / (p.stem + ".md")).exists()]

    total = len(images)
    print(f"[paddlevl] images={total} doc_workers={args.doc_workers} "
          f"vl_concurrency={args.vl_concurrency} server={args.server} "
          f"md_out={md_dir}", flush=True)
    if total == 0:
        return 0

    pipeline = _build_pipeline(args)

    n_ok = 0
    n_err = 0
    t0 = time.time()
    err_lock = Lock()

    if args.doc_workers <= 1:
        for i, img in enumerate(images, 1):
            _, err = _run_one(pipeline, img, md_dir)
            if err:
                n_err += 1
                with errors_path.open("a") as f:
                    f.write(f"{img.name}\t{err}\n")
            else:
                n_ok += 1
            if i % 25 == 0 or i == total:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0.0
                print(f"[paddlevl] {i}/{total} ok={n_ok} err={n_err} "
                      f"rate={rate:.2f}/s elapsed={dt:.0f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.doc_workers) as pool:
            futs = [pool.submit(_run_one, pipeline, p, md_dir)
                    for p in images]
            for i, fut in enumerate(as_completed(futs), 1):
                img, err = fut.result()
                if err:
                    n_err += 1
                    with err_lock, errors_path.open("a") as f:
                        f.write(f"{img.name}\t{err}\n")
                else:
                    n_ok += 1
                if i % 25 == 0 or i == total:
                    dt = time.time() - t0
                    rate = i / dt if dt > 0 else 0.0
                    print(f"[paddlevl] {i}/{total} ok={n_ok} err={n_err} "
                          f"rate={rate:.2f}/s elapsed={dt:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"[paddlevl] done ok={n_ok} err={n_err} elapsed={dt:.1f}s "
          f"rate={total/dt:.2f}/s", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
