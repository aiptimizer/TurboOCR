#!/usr/bin/env python3
"""Benchmark PaddleOCR-VL via the proper 2-stage pipeline.

Stage 1 (layout detection) runs locally in Python.
Stage 2 (VLM recognition) calls a running `vllm serve` via OpenAI-compatible API.
"""
import asyncio
import io
import json
import os
import re
import tempfile
import time

import numpy as np
from datasets import load_dataset

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

VLLM_URL = "http://localhost:8077/v1"
SERVED_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"
N_IMAGES = 50


def tokens(text):
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", text) if len(t) >= 2}


def metrics(pred: set, gt: set) -> dict:
    if not gt:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    tp = len(pred & gt)
    r = tp / len(gt)
    p = tp / len(pred) if pred else 0.0
    f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return {"recall": r, "precision": p, "f1": f1}


def extract_text_from_result(r) -> str:
    """Pull all text content out of a PaddleOCRVLResult."""
    md = getattr(r, "markdown", None)
    if isinstance(md, dict):
        t = md.get("markdown_texts") or ""
        if t:
            return t
    # Fallback: walk the blocks
    try:
        parsing = r["parsing_res_list"]
        return "\n".join(block.get("block_content", "") for block in parsing if block.get("block_content"))
    except Exception:
        pass
    return str(md or "")


def main():
    print("Loading FUNSD…")
    ds = load_dataset("nielsr/funsd", split="test")
    ds = ds.select(range(min(N_IMAGES, len(ds))))

    # Prepare images + ground truth
    image_paths = []
    ground_truths = []
    word_counts = []
    tmpdir = tempfile.mkdtemp(prefix="funsd_paddle_vl_")
    for i, sample in enumerate(ds):
        img = sample["image"].convert("RGB")
        p = os.path.join(tmpdir, f"img_{i:03d}.png")
        img.save(p)
        image_paths.append(p)
        ground_truths.append(tokens(" ".join(sample["words"])))
        word_counts.append(len(sample["words"]))
    print(f"Dataset: {len(image_paths)} images · words/img: "
          f"min={min(word_counts)} max={max(word_counts)} mean={np.mean(word_counts):.0f}")

    print("Creating PaddleOCRVL pipeline (layout on GPU + vLLM backend)…")
    from paddleocr import PaddleOCRVL

    pipe = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=VLLM_URL,
        vl_rec_api_model_name=SERVED_MODEL_NAME,
        vl_rec_max_concurrency=16,
        device="gpu",
    )

    # Verify the layout detector is on GPU
    try:
        sub = pipe.paddlex_pipeline
        for attr_name in ("layout_det_model", "layout_detection_model", "doc_layout_model"):
            m = getattr(sub, attr_name, None)
            if m is not None and hasattr(m, "model"):
                print(f"  layout model [{attr_name}] device: {getattr(m, '_device', 'unknown')}")
                break
    except Exception:
        pass

    # Warmup
    print("Warmup…")
    try:
        list(pipe.predict(image_paths[0]))
    except Exception as e:
        print(f"Warmup error: {e}")
        raise

    # Accuracy + latency
    print(f"Accuracy + latency on {len(image_paths)} images…")
    accs, lats = [], []
    for i, p in enumerate(image_paths):
        t0 = time.perf_counter()
        results = list(pipe.predict(p))
        lat = (time.perf_counter() - t0) * 1000
        lats.append(lat)
        pred_text = "\n".join(extract_text_from_result(r) for r in results)
        acc = metrics(tokens(pred_text), ground_truths[i])
        accs.append(acc)
        if (i + 1) % 5 == 0:
            mf1 = np.mean([a["f1"] for a in accs])
            ml = np.mean(lats)
            print(f"  [{i+1}/{len(image_paths)}] F1={mf1:.1%} lat={ml:.0f}ms")

    # Concurrent throughput — dispatch all images in parallel threads
    # Each thread does its own layout + VLM calls; VLM calls share the vllm server.
    import concurrent.futures
    print(f"Concurrent throughput on {len(image_paths)} images (threads=8)…")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lambda p=p: list(pipe.predict(p))) for p in image_paths]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    tp_total = time.perf_counter() - t0
    throughput = len(image_paths) / tp_total

    result = {
        "name": "PaddleOCR-VL (pipeline)",
        "accuracy": accs,
        "latencies_ms": lats,
        "throughput_img_per_sec": throughput,
        "errors": 0,
        "total_images": len(image_paths),
    }
    f1 = np.mean([a["f1"] for a in accs])
    print(f"\nPaddleOCR-VL (pipeline): F1={f1:.1%} | {throughput:.2f} img/s | p50={np.median(lats):.0f}ms")

    with open("vlm_result_PaddleOCR-VL-pipeline.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved to vlm_result_PaddleOCR-VL-pipeline.json")


if __name__ == "__main__":
    main()
