#!/usr/bin/env python3
"""FUNSD bench against a running turbo-ocr server — no HuggingFace dependency.

Images: compare-ocrs/funsd_cache/funsd_{i:03d}.png (50 test pages)
GT:     tests/benchmark/funsd_gt_words.json (extracted from the HF arrow cache;
        kept one level up because the C++ FUNSD gate hardcodes that path)
Metric: Counter-based bag-of-words F1 (same as tests/benchmark/comparison/bench_turbo_ocr.py)
"""
import concurrent.futures
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests

URL = os.environ.get("TURBO_OCR_URL", "http://localhost:8080")
N = 50
CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "16"))
THROUGHPUT_ITERS = int(os.environ.get("BENCH_ITERS", "600"))

def _find_cache() -> Path:
    """Locate compare-ocrs/funsd_cache without assuming where the repo lives.

    This was `parents[4] / "compare-ocrs/funsd_cache"`, which silently encoded
    "the repo sits directly in $HOME". Cloned anywhere else — e.g.
    ~/Documents/turbo — parents[4] walks one level too far up and the script
    dies with FileNotFoundError on funsd_000.png, so the repo's own headline
    benchmark did not run as shipped. Search the plausible roots instead, and
    let FUNSD_CACHE win outright for a cache kept somewhere else entirely.
    """
    if env := os.environ.get("FUNSD_CACHE"):
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root.parent / "compare-ocrs/funsd_cache",   # sibling of the repo
        repo_root / "compare-ocrs/funsd_cache",          # inside the repo
        Path.home() / "compare-ocrs/funsd_cache",        # the historical spot
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit(
        "FUNSD cache not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nFetch it with scripts/models/fetch/fetch_funsd_cache.sh, or point "
          "FUNSD_CACHE at an existing copy."
    )


CACHE = _find_cache()
# The GT list stays at tests/benchmark/ (not next to this script): the compiled
# FUNSD gate hardcodes that path (tests/cpp/backends/harness.h default_gt_path).
GT_PATH = Path(__file__).resolve().parents[1] / "funsd_gt_words.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower()) if text else []


def word_f1(gt_tokens: list[str], pred_tokens: list[str]) -> dict:
    if not gt_tokens and not pred_tokens:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
    if not gt_tokens or not pred_tokens:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    gt_bag, pred_bag = Counter(gt_tokens), Counter(pred_tokens)
    tp = sum((gt_bag & pred_bag).values())
    r = tp / sum(gt_bag.values())
    p = tp / sum(pred_bag.values())
    f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return {"recall": r, "precision": p, "f1": f1}


def main():
    images_png = [(CACHE / f"funsd_{i:03d}.png").read_bytes() for i in range(N)]
    ground_truths = [tokenize(" ".join(w)) for w in json.load(open(GT_PATH))]

    session_pool = [requests.Session() for _ in range(CONCURRENCY)]

    def run_one(i: int, sess=None) -> list[str]:
        s = sess or session_pool[0]
        r = s.post(f"{URL}/ocr/raw", data=images_png[i],
                   headers={"Content-Type": "image/png"}, timeout=60)
        r.raise_for_status()
        return [item["text"] for item in r.json().get("results", [])]

    run_one(0); run_one(0)  # warmup

    accs, lats = [], []
    for i in range(N):
        t0 = time.perf_counter()
        preds = run_one(i)
        lats.append((time.perf_counter() - t0) * 1000)
        accs.append(word_f1(ground_truths[i], tokenize(" ".join(preds))))

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        def worker(k):
            return run_one(k % N, session_pool[k % CONCURRENCY])
        list(pool.map(worker, range(THROUGHPUT_ITERS)))
    throughput = THROUGHPUT_ITERS / (time.perf_counter() - t0)

    f1 = np.mean([a["f1"] for a in accs])
    p = np.mean([a["precision"] for a in accs])
    r = np.mean([a["recall"] for a in accs])
    print(f"F1={f1:.2%} P={p:.2%} R={r:.2%} | {throughput:.1f} img/s "
          f"| p50={np.median(lats):.0f}ms p95={np.percentile(lats, 95):.0f}ms")
    out = {"f1": f1, "precision": p, "recall": r,
           "throughput_img_per_sec": throughput,
           "p50_ms": float(np.median(lats)), "lats_ms": lats}
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    with open(f"/tmp/funsd_bench_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
