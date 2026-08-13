#!/usr/bin/env python3
"""OOD smoke check for the v2 script_id model on OmniDocBench page crops.

Sample text-line bounding boxes from `/tmp/omnidoc_predictions/json/*.json`,
crop them out of the source page PNG under `omnidocbench/data/images/`,
classify with `models/script_id/script_id.onnx`, and report per-script
accuracy using the page's `OmniDocBench.json` `language` tag as the label.

Languages mapped to canonical ScriptId labels:
    english          -> latin
    simplified_chinese / traditional_chinese -> chinese
    en_ch_mixed      -> skipped (ambiguous)
    other            -> skipped

Output: models/script_id/ood_smoke.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

OMNI = Path(__file__).resolve().parents[4] / "omnidocbench" / "data"
PREDS = Path("/tmp/omnidoc_predictions/json")
ONNX = Path(__file__).resolve().parents[3] / "models/script_id/script_id.onnx"
OUT = Path(__file__).resolve().parents[3] / "models/script_id/ood_smoke.json"

CANON = ["latin", "chinese", "arabic", "eslav", "greek", "korean", "thai"]
H, W = 32, 128

LANG_TO_CANON = {
    "english": "latin",
    "simplified_chinese": "chinese",
    "traditional_chinese": "chinese",
}

SAMPLES_PER_LANG = 100  # ~50 each side; total ≤ 200 crops
MIN_W, MIN_H = 24, 12


def preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    if crop_rgb.shape[:2] != (H, W):
        img = Image.fromarray(crop_rgb).resize((W, H), Image.BILINEAR)
        crop_rgb = np.array(img)
    arr = crop_rgb.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None]  # 1,3,32,128


def main():
    with open(OMNI / "OmniDocBench.json") as f:
        manifest = json.load(f)
    # map image_path -> lang
    image_lang = {p["page_info"]["image_path"]: p["page_info"]["page_attribute"].get("language", "?") for p in manifest}

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = random.Random(0)
    # collect pages by canon lang
    by_lang = {"latin": [], "chinese": []}
    for img_name, lang in image_lang.items():
        canon = LANG_TO_CANON.get(lang)
        if canon is None:
            continue
        by_lang[canon].append(img_name)
    for k in by_lang:
        rng.shuffle(by_lang[k])

    crops_per_class = {k: [] for k in CANON}
    crops_per_class.setdefault("latin", [])
    crops_per_class.setdefault("chinese", [])

    for canon_lang, pages in by_lang.items():
        collected = 0
        for img_name in pages:
            if collected >= SAMPLES_PER_LANG:
                break
            pred_path = PREDS / (Path(img_name).stem + ".json")
            if not pred_path.exists():
                continue
            try:
                with open(pred_path) as f:
                    pred = json.load(f)
            except Exception:
                continue
            img_path = OMNI / "images" / img_name
            if not img_path.exists():
                continue
            try:
                page = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
            for r in pred.get("results", []):
                bb = r.get("bounding_box")
                if not bb or len(bb) != 4:
                    continue
                xs = [p[0] for p in bb]
                ys = [p[1] for p in bb]
                x0, x1 = max(0, min(xs)), min(page.shape[1], max(xs))
                y0, y1 = max(0, min(ys)), min(page.shape[0], max(ys))
                if x1 - x0 < MIN_W or y1 - y0 < MIN_H:
                    continue
                if (x1 - x0) / max(1, y1 - y0) < 1.2:
                    continue  # skip non-line crops
                crop = page[y0:y1, x0:x1]
                crops_per_class[canon_lang].append(crop)
                collected += 1
                if collected >= SAMPLES_PER_LANG:
                    break

    # classify
    correct = {k: 0 for k in CANON}
    total = {k: 0 for k in CANON}
    per_class_pred_dist = {k: {c: 0 for c in CANON} for k in CANON}
    for true_canon, crops in crops_per_class.items():
        if not crops:
            continue
        for crop in crops:
            arr = preprocess(crop)
            logits = sess.run(None, {in_name: arr})[0][0]
            pred_idx = int(np.argmax(logits))
            pred_canon = CANON[pred_idx]
            total[true_canon] += 1
            per_class_pred_dist[true_canon][pred_canon] += 1
            if pred_canon == true_canon:
                correct[true_canon] += 1

    out = {
        "n_per_class": total,
        "correct_per_class": correct,
        "acc_per_class": {k: (correct[k] / total[k] if total[k] else None) for k in CANON},
        "pred_distribution": per_class_pred_dist,
        "notes": (
            "Auto-labeled from OmniDocBench page-level `language` tag. "
            "Only `latin` and `chinese` are populated (the dataset's available labels). "
            "Other canonical classes report None — no OOD ground truth in this dataset."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2))
    for c in CANON:
        if total[c]:
            print(f"OOD {c:8s}: {correct[c]}/{total[c]} = {correct[c]/total[c]:.3f}", flush=True)
    print(f"[ood] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
