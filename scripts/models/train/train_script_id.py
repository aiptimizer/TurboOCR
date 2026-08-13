#!/usr/bin/env python3
"""Train a 7-class script_id classifier on 32x128 RGB text-line crops.

Locked-in contract (matches C++ ScriptId stub in include/turbo_ocr/models/lang_cls/script_id.h):

    Input  : [N, 3, 32, 128] RGB
    Normalize: (pixel/255 - 0.5) / 0.5    i.e. mean=std=0.5 across channels
    Output : [N, 7] raw logits — softmax applied at inference time
    Class order (load-bearing — argmax → enum):
        Latin=0, Han=1, Cyrillic=2, Arabic=3, Greek=4, Hangul=5, Thai=6
    Backbone: torchvision.models.mobilenet_v3_small(weights="DEFAULT")
              classifier[3] = nn.Linear(in_features, 7)
    Training: Adam lr=4e-3 wd=1e-4, CosineAnnealing T_max=80,
              label_smoothing=0.1, batch 256, 80 epochs,
              10% per-class held-out val split.
    Acceptance: per-class top-1 ≥ 95% on val.

Output:
    models/script_id/script_id.onnx           (opset 17, dynamic batch)
    models/script_id/meta.json                (mirrors lang_cls/meta.json)
    models/script_id/training_log.json        (per-epoch metrics)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

CLASS_ORDER = ["latin", "han", "cyrillic", "arabic", "greek", "hangul", "thai"]
NUM_CLASSES = 7
H, W = 32, 128


class CropDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # match C++ stub norm exactly
        # HWC -> CHW
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        return tensor, label


def build_split(data_root: Path, val_frac: float, seed: int):
    rng = random.Random(seed)
    train, val = [], []
    for label, klass in enumerate(CLASS_ORDER):
        files = sorted((data_root / klass).glob("*.png"))
        rng.shuffle(files)
        cut = int(round(len(files) * val_frac))
        val.extend([(p, label) for p in files[:cut]])
        train.extend([(p, label) for p in files[cut:]])
    rng.shuffle(train)
    return train, val


def per_class_accuracy(model, loader, device):
    model.eval()
    correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    total = np.zeros(NUM_CLASSES, dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            preds = model(x).argmax(dim=1).cpu().numpy()
            ytrue = y.numpy()
            for c in range(NUM_CLASSES):
                mask = ytrue == c
                correct[c] += int((preds[mask] == c).sum())
                total[c] += int(mask.sum())
    acc = correct / np.maximum(total, 1)
    overall = correct.sum() / max(total.sum(), 1)
    return acc, float(overall), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/script_id_data")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "models" / "script_id"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} torch={torch.__version__}", flush=True)

    data_root = Path(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_items, val_items = build_split(data_root, args.val_frac, args.seed)
    print(f"[train] train={len(train_items)} val={len(val_items)}", flush=True)

    train_ds = CropDataset(train_items)
    val_ds = CropDataset(val_items)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = models.mobilenet_v3_small(weights="DEFAULT")
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log = []
    best_min_acc = -1.0
    best_state = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * x.size(0)
            n += x.size(0)
        scheduler.step()
        train_loss = loss_sum / max(n, 1)

        acc, overall, totals = per_class_accuracy(model, val_loader, device)
        min_acc = float(acc.min())
        elapsed = time.time() - t0
        per_class = {CLASS_ORDER[c]: float(acc[c]) for c in range(NUM_CLASSES)}
        msg = (
            f"epoch={epoch+1:02d}/{args.epochs} "
            f"loss={train_loss:.4f} overall_acc={overall:.4f} min_acc={min_acc:.4f} "
            f"elapsed_s={elapsed:.1f} "
            f"per_class={per_class}"
        )
        print(msg, flush=True)
        log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_overall_acc": overall,
            "val_min_acc": min_acc,
            "val_per_class_acc": per_class,
            "elapsed_s": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if min_acc > best_min_acc:
            best_min_acc = min_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval with best
    acc, overall, totals = per_class_accuracy(model, val_loader, device)
    final_per_class = {CLASS_ORDER[c]: float(acc[c]) for c in range(NUM_CLASSES)}
    print(f"[train] final best overall_acc={overall:.4f} per_class={final_per_class}", flush=True)

    # ---- export ONNX ----
    model.eval()
    dummy = torch.randn(1, 3, H, W, device=device)
    onnx_path = out_root / "script_id.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[train] wrote {onnx_path} ({onnx_path.stat().st_size} bytes)", flush=True)

    # ---- meta.json ----
    meta = {
        "model": "MobileNetV3-Small (torchvision, classifier[3]=Linear(in,7))",
        "task": "text-line crop script classifier (7-class)",
        "input_layout": "NCHW",
        "input_shape": [1, 3, H, W],
        "input_dtype": "float32",
        "input_color_order": "RGB",
        "normalization": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "scale": "divide pixel uint8 by 255 before mean/std",
        },
        "resize": {
            "method": "PIL bilinear, force resize to (W=128, H=32)",
            "note": "Aspect-ratio not preserved — classifier reads stroke statistics, not character identity.",
        },
        "classes": {str(i): name for i, name in enumerate(CLASS_ORDER)},
        "class_names": CLASS_ORDER,
        "output": "logits over 7 classes; apply softmax for probabilities",
        "training_data": {
            "source": "Synthetic PIL renders of Unicode text-line crops, per-script Noto/DejaVu/Free fonts.",
            "per_class": int(np.mean(totals + (np.maximum(totals, 1)))) if False else None,
            "split": "per-class shuffle then 10% val",
            "val_frac": args.val_frac,
            "seed": args.seed,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": args.wd,
            "schedule": "CosineAnnealingLR(T_max=epochs)",
            "label_smoothing": args.label_smoothing,
            "amp": "fp16 autocast (GradScaler)",
        },
        "metrics": {
            "best_val_overall_acc": float(overall),
            "best_val_per_class_acc": final_per_class,
            "best_val_min_acc": float(acc.min()),
        },
        "onnx": {
            "opset": 17,
            "input_name": "input",
            "output_name": "logits",
            "input_dim_order": ["batch", "channels=3", f"height={H}", f"width={W}"],
            "output_dim_order": ["batch", f"num_classes={NUM_CLASSES}"],
            "dynamic_axes": {"input": {"0": "batch"}, "logits": {"0": "batch"}},
            "file": str(onnx_path),
            "file_size_bytes": onnx_path.stat().st_size,
        },
    }
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] wrote {out_root/'meta.json'}", flush=True)

    with open(out_root / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train] wrote {out_root/'training_log.json'}", flush=True)

    # acceptance gate
    if min(final_per_class.values()) >= 0.95:
        print(f"[train] PASS acceptance gate (all classes >= 95%)", flush=True)
    else:
        print(f"[train] BELOW acceptance gate min_acc={min(final_per_class.values()):.4f}", flush=True)


if __name__ == "__main__":
    main()
