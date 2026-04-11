#!/usr/bin/env python3
"""Benchmark a VLM served via vLLM OpenAI-compatible endpoint.

Uses async OpenAI SDK for true concurrent throughput measurement.
Usage: bench_vlm_async.py <model_id> <display_name> <port> [prompt]
"""

import asyncio
import base64
import io
import json
import sys
import time

import numpy as np
from datasets import load_dataset
from openai import AsyncOpenAI

MODEL_ID = sys.argv[1]
MODEL_NAME = sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8077
PROMPT = sys.argv[4] if len(sys.argv) > 4 else "OCR:"
CONCURRENCY = 8

client = AsyncOpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{PORT}/v1",
    timeout=300,
    max_retries=0,
)

print("Loading FUNSD…")
dataset = load_dataset("nielsr/funsd", split="test")

images_uri: list[str] = []
ground_truths: list[set[str]] = []
for sample in dataset:
    img = sample["image"].convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    images_uri.append(f"data:image/png;base64,{b64}")
    words = set()
    for w in sample["words"]:
        w_clean = w.lower().strip(".,;:!?\"'()[]{}-/\\")
        if len(w_clean) >= 2:
            words.add(w_clean)
    ground_truths.append(words)


def extract_words(text: str) -> set[str]:
    words = set()
    for w in text.lower().split():
        w_clean = w.strip(".,;:!?\"'()[]{}-/\\")
        if len(w_clean) >= 2:
            words.add(w_clean)
    return words


def compute_metrics(pred: set[str], gt: set[str]) -> dict:
    if not gt:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    tp = len(gt & pred)
    r = tp / len(gt)
    p = tp / len(pred) if pred else 0.0
    f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return {"recall": r, "precision": p, "f1": f1}


async def call_vlm(idx: int) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": images_uri[idx]}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=2048,
        temperature=0.0,
    )
    return resp.choices[0].message.content, (time.perf_counter() - t0) * 1000


async def run_sequential() -> tuple[list[dict], list[float]]:
    """Sequential accuracy + latency measurement."""
    accs, lats = [], []
    for i in range(len(images_uri)):
        text, lat = await call_vlm(i)
        lats.append(lat)
        accs.append(compute_metrics(extract_words(text), ground_truths[i]))
        if (i + 1) % 10 == 0:
            mean_f1 = np.mean([a["f1"] for a in accs])
            mean_lat = np.mean(lats)
            print(f"  [{i+1}/{len(images_uri)}] F1={mean_f1:.1%} lat={mean_lat:.0f}ms")
    return accs, lats


async def run_concurrent_throughput(n: int, concurrency: int) -> float:
    """Fire n concurrent requests, bounded by a semaphore."""
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int):
        async with sem:
            await call_vlm(i % len(images_uri))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    return time.perf_counter() - t0


async def main():
    # Warmup
    print("Warming up…")
    await call_vlm(0)
    await call_vlm(0)

    print(f"Accuracy + latency ({len(images_uri)} images, sequential)…")
    accs, lats = await run_sequential()

    print(f"Throughput ({len(images_uri)} images, concurrency={CONCURRENCY})…")
    total = await run_concurrent_throughput(len(images_uri), CONCURRENCY)
    throughput = len(images_uri) / total

    result = {
        "name": MODEL_NAME,
        "accuracy": accs,
        "latencies_ms": lats,
        "throughput_img_per_sec": throughput,
        "errors": 0,
        "total_images": len(images_uri),
    }

    f1 = np.mean([a["f1"] for a in accs])
    print(f"\n{MODEL_NAME}: F1={f1:.1%} | {throughput:.2f} img/s | p50={np.median(lats):.0f}ms")

    out = f"vlm_result_{MODEL_ID.replace('/', '_')}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
