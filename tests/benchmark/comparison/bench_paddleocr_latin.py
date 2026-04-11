#!/usr/bin/env python3
"""Benchmark PaddleOCR Python with the same PP-OCRv5 mobile latin model as Turbo-OCR."""

import concurrent.futures
import io
import json
import os
import re
import tempfile
import time

import numpy as np
from datasets import load_dataset

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


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
    ds = load_dataset("nielsr/funsd", split="test").select(range(50))

    tmpdir = tempfile.mkdtemp(prefix="funsd_latin_")
    image_paths, ground_truths = [], []
    for i, sample in enumerate(ds):
        p = os.path.join(tmpdir, f"img_{i:03d}.png")
        sample["image"].convert("RGB").save(p)
        image_paths.append(p)
        ground_truths.append(tokens(" ".join(sample["words"])))

    print("Creating PaddleOCR with PP-OCRv5 mobile latin (same as Turbo-OCR)…")
    from paddleocr import PaddleOCR
    engine = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="latin_PP-OCRv5_mobile_rec",
        device="gpu",
    )

    def run_one(p: str) -> list[str]:
        out = engine.predict(p)
        texts = []
        for r in out:
            texts.extend(r.get("rec_texts", []))
        return texts

    # Warmup
    print("Warmup…")
    run_one(image_paths[0])
    run_one(image_paths[0])

    # Accuracy + latency
    print(f"Accuracy + latency on {len(image_paths)} images…")
    accs, lats = [], []
    for i, p in enumerate(image_paths):
        t0 = time.perf_counter()
        preds = run_one(p)
        lat = (time.perf_counter() - t0) * 1000
        lats.append(lat)
        accs.append(metrics(tokens(" ".join(preds)), ground_truths[i]))
        if (i + 1) % 10 == 0:
            mf1 = np.mean([a["f1"] for a in accs])
            ml = np.mean(lats)
            print(f"  [{i+1}/{len(image_paths)}] F1={mf1:.1%} lat={ml:.0f}ms")

    # Throughput: sequential (PaddleOCR predict() is not thread-safe)
    print(f"Sequential throughput on {len(image_paths)} images…")
    t0 = time.perf_counter()
    for p in image_paths:
        run_one(p)
    tp_total = time.perf_counter() - t0
    throughput = len(image_paths) / tp_total

    result = {
        "name": "PaddleOCR mobile latin (Python)",
        "accuracy": accs,
        "latencies_ms": lats,
        "throughput_img_per_sec": throughput,
        "errors": 0,
        "total_images": len(image_paths),
    }
    f1 = np.mean([a["f1"] for a in accs])
    print(f"\n{result['name']}: F1={f1:.1%} | {throughput:.2f} img/s | p50={np.median(lats):.0f}ms")

    with open("vlm_result_paddleocr_latin.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
