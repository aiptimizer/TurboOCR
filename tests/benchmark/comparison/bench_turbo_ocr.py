#!/usr/bin/env python3
"""Benchmark Turbo-OCR (C++/TRT) via HTTP on FUNSD — matches bench_paddleocr_latin.py scoring."""
import concurrent.futures
import io
import json
import os
import re
import tempfile
import time

import numpy as np
import requests
from datasets import load_dataset

URL = "http://localhost:8000"
N = 50
CONCURRENCY = 16
THROUGHPUT_ITERS = 200  # cycle the 50 FUNSD images 4× for a stable c=16 measurement


def tokens(text: str) -> set:
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", text) if len(t) >= 2}


def metrics(pred: set, gt: set) -> dict:
    if not gt:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    tp = len(pred & gt)
    r = tp / len(gt)
    p = tp / len(pred) if pred else 0.0
    f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return {"recall": r, "precision": p, "f1": f1}


def main():
    print("Loading FUNSD…")
    ds = load_dataset("nielsr/funsd", split="test").select(range(N))

    images_png, ground_truths = [], []
    for sample in ds:
        buf = io.BytesIO()
        sample["image"].convert("RGB").save(buf, format="PNG")
        images_png.append(buf.getvalue())
        ground_truths.append(tokens(" ".join(sample["words"])))

    def run_one(i: int) -> list[str]:
        r = requests.post(f"{URL}/ocr/raw", data=images_png[i],
                          headers={"Content-Type": "image/png"}, timeout=30)
        return [item["text"] for item in r.json().get("results", [])]

    print("Warmup…")
    run_one(0); run_one(0)

    print(f"Accuracy + latency on {N} images…")
    accs, lats = [], []
    for i in range(N):
        t0 = time.perf_counter()
        preds = run_one(i)
        lat = (time.perf_counter() - t0) * 1000
        lats.append(lat)
        accs.append(metrics(tokens(" ".join(preds)), ground_truths[i]))
        if (i + 1) % 10 == 0:
            mf1 = np.mean([a["f1"] for a in accs])
            ml = np.mean(lats)
            print(f"  [{i+1}/{N}] F1={mf1:.1%} lat={ml:.0f}ms")

    print(f"Concurrent throughput (c={CONCURRENCY}, {THROUGHPUT_ITERS} requests)…")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(lambda k: run_one(k % N), range(THROUGHPUT_ITERS)))
    tp_total = time.perf_counter() - t0
    throughput = THROUGHPUT_ITERS / tp_total

    result = {
        "name": "Turbo-OCR (C++/TRT)",
        "accuracy": accs,
        "latencies_ms": lats,
        "throughput_img_per_sec": throughput,
        "errors": 0,
        "total_images": N,
    }
    f1 = np.mean([a["f1"] for a in accs])
    print(f"\nTurbo-OCR: F1={f1:.1%} | {throughput:.2f} img/s | p50={np.median(lats):.0f}ms")

    with open("vlm_result_turbo_ocr.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved to vlm_result_turbo_ocr.json")


if __name__ == "__main__":
    main()
