#!/usr/bin/env python3
"""Validate OCR accuracy against a HuggingFace dataset (CORD-v2 receipts)."""

import argparse
import base64
import io
import json
import requests
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets Pillow")
    raise


def image_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def extract_ground_truth_text(sample):
    """Extract text from CORD-v2 ground truth annotation."""
    gt = sample.get("ground_truth", "")
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            return []

    texts = []
    if isinstance(gt, dict):
        # CORD-v2 format: gt_parse → nested dicts with text_sequence
        def walk(obj):
            if isinstance(obj, dict):
                if "text_sequence" in obj:
                    texts.append(obj["text_sequence"])
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)
        walk(gt)
    return texts


def run_ocr(image_b64, endpoint):
    resp = requests.post(endpoint, json={"image": image_b64}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [item["text"] for item in data.get("results", [])]


def normalize(text):
    """Normalize for comparison: lowercase, strip whitespace."""
    return " ".join(text.lower().split())


def main():
    parser = argparse.ArgumentParser(description="Validate OCR accuracy with CORD-v2")
    parser.add_argument("--endpoint", default="http://localhost:8000/ocr",
                        help="OCR API endpoint")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max validation samples")
    parser.add_argument("--dataset", default="naver-clova-ix/cord-v2",
                        help="HuggingFace dataset name")
    parser.add_argument("--split", default="validation", help="Dataset split")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, split=args.split)

    total = min(args.max_samples, len(ds))
    stats = defaultdict(int)

    for i in range(total):
        sample = ds[i]
        img = sample["image"]
        gt_texts = extract_ground_truth_text(sample)
        gt_combined = normalize(" ".join(gt_texts))

        if not gt_combined:
            stats["skipped_no_gt"] += 1
            continue

        try:
            image_b64 = image_to_base64(img)
            ocr_texts = run_ocr(image_b64, args.endpoint)
            ocr_combined = normalize(" ".join(ocr_texts))
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR: {e}")
            stats["errors"] += 1
            continue

        # Simple word-level overlap metric
        gt_words = set(gt_combined.split())
        ocr_words = set(ocr_combined.split())

        if gt_words:
            recall = len(gt_words & ocr_words) / len(gt_words)
            precision = len(gt_words & ocr_words) / len(ocr_words) if ocr_words else 0
        else:
            recall = precision = 0

        stats["total"] += 1
        stats["recall_sum"] += recall
        stats["precision_sum"] += precision

        detected = len(ocr_texts) > 0
        if detected:
            stats["detected"] += 1

        print(f"  [{i+1}/{total}] recall={recall:.2f} precision={precision:.2f} "
              f"gt_words={len(gt_words)} ocr_words={len(ocr_words)}")

    print("\n=== Results ===")
    n = stats["total"] or 1
    print(f"Samples processed: {stats['total']}")
    print(f"Detection rate: {stats['detected']}/{stats['total']} "
          f"({100*stats['detected']/n:.1f}%)")
    print(f"Avg word recall: {stats['recall_sum']/n:.3f}")
    print(f"Avg word precision: {stats['precision_sum']/n:.3f}")
    if stats["errors"]:
        print(f"Errors: {stats['errors']}")
    if stats["skipped_no_gt"]:
        print(f"Skipped (no GT): {stats['skipped_no_gt']}")


if __name__ == "__main__":
    main()
