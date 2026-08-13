#!/usr/bin/env python3
"""v2 training for the 7-class script_id classifier.

Recipe (locked-in contract):
    Input  [N,3,32,128] RGB
    Norm   (pixel/255 - 0.5) / 0.5
    Output [N,7] raw logits
    Trainer class order (pre-permute):
        0=latin, 1=han, 2=cyrillic, 3=arabic, 4=greek, 5=hangul, 6=thai
    Canonical C++ ScriptId order (post-permute):
        0=Latin, 1=Chinese, 2=Arabic, 3=ESlav, 4=Greek, 5=Korean, 6=Thai
    Permutation PERM=[0,1,3,2,4,5,6] applied to final Linear before export.

    Backbone     : torchvision.models.mobilenet_v3_small(weights="DEFAULT")
                   classifier[3] = Linear(in_features, 7)
    Aug (train)  : albumentations — rotate ±5°, perspective, JPEG Q=30-95,
                   blur, brightness/contrast, coarse dropout, channel shuffle
    Mix          : MixUp α=0.2 + CutMix α=1.0 alternated 50/50 per batch
    Optimizer    : AdamW lr=3e-3 wd=0.05
    Schedule     : LinearWarmup(3) → CosineAnnealing(27)
    Loss         : CE(label_smoothing=0.1) over (y, y_perm) mix
    EMA          : decay=0.999, eval both plain + ema each epoch
    Batch        : 256
    Epochs       : 30
    AMP          : fp16
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import albumentations as A
import cv2
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

# trainer -> canonical permutation:  new_row[i] = old_row[PERM[i]]
# trainer:  [latin, han, cyrillic, arabic, greek, hangul, thai]
# canon  :  [latin, chinese, arabic, eslav, greek, korean, thai]
PERM = [0, 1, 3, 2, 4, 5, 6]
CANON_NAMES = ["latin", "chinese", "arabic", "eslav", "greek", "korean", "thai"]

TRAIN_TF = A.Compose([
    A.Rotate(limit=5, p=0.3, border_mode=cv2.BORDER_CONSTANT, fill=255),
    A.Perspective(scale=(0.02, 0.05), p=0.2, pad_mode=cv2.BORDER_CONSTANT, pad_val=255),
    A.RandomBrightnessContrast(0.15, 0.15, p=0.3),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=3),
    ], p=0.3),
    A.ImageCompression(quality_range=(30, 95), p=0.5),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(2, 6),
                    hole_width_range=(4, 12), fill=255, p=0.2),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(2, 6),
                    hole_width_range=(4, 12), fill=0, p=0.2),
    A.ChannelShuffle(p=0.05),
    A.Resize(H, W, interpolation=cv2.INTER_LINEAR),
])


class CropDataset(Dataset):
    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        if self.augment:
            img = TRAIN_TF(image=img)["image"]
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        arr = img.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())
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
            yt = y.numpy()
            for c in range(NUM_CLASSES):
                m = yt == c
                correct[c] += int((preds[m] == c).sum())
                total[c] += int(m.sum())
    acc = correct / np.maximum(total, 1)
    overall = correct.sum() / max(total.sum(), 1)
    return acc, float(overall)


def mixup_batch(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cutmix_batch(x, y, alpha=1.0):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H_, W_ = x.shape
    cut_h = int(H_ * np.sqrt(1.0 - lam))
    cut_w = int(W_ * np.sqrt(1.0 - lam))
    cy = np.random.randint(H_)
    cx = np.random.randint(W_)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H_, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W_, cx + cut_w // 2)
    x_out = x.clone()
    x_out[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # adjust lam to reflect actual area
    lam_eff = 1.0 - ((y2 - y1) * (x2 - x1)) / float(H_ * W_)
    return x_out, y, y[idx], lam_eff


def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), model.parameters()):
            pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for be, b in zip(ema.buffers(), model.buffers()):
            be.copy_(b)


def export_onnx_permuted(model: nn.Module, out_path: Path):
    """Permute final Linear rows to canonical order, then export single-file ONNX."""
    final = model.classifier[3]
    perm_idx = torch.tensor(PERM, dtype=torch.long, device=final.weight.device)
    with torch.no_grad():
        new_w = final.weight.detach()[perm_idx].clone()
        new_b = final.bias.detach()[perm_idx].clone()
        final.weight.copy_(new_w)
        final.bias.copy_(new_b)

    model_eval = model.eval()
    dummy = torch.randn(1, 3, H, W, device=final.weight.device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # use legacy exporter; default writes inline weights; single-file fallback below
    torch.onnx.export(
        model_eval,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    # If the torch exporter emitted external data, fold it back into a single file.
    sidecar = out_path.with_suffix(".onnx.data")
    if sidecar.exists():
        import onnx
        m = onnx.load(str(out_path), load_external_data=True)
        for init in m.graph.initializer:
            if init.data_location == onnx.TensorProto.EXTERNAL:
                init.data_location = onnx.TensorProto.DEFAULT
            while init.external_data:
                init.external_data.pop()
        onnx.save_model(m, str(out_path), save_as_external_data=False)
        sidecar.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/script_id_data_v2")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "models" / "script_id"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--cutmix-alpha", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} torch={torch.__version__} alb={A.__version__}", flush=True)

    data_root = Path(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    train_items, val_items = build_split(data_root, args.val_frac, args.seed)
    print(f"[train] train={len(train_items)} val={len(val_items)}", flush=True)

    train_ds = CropDataset(train_items, augment=True)
    val_ds = CropDataset(val_items, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(2, args.workers // 2), pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = models.mobilenet_v3_small(weights="DEFAULT")
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log = []
    best_ema_min = -1.0
    best_ema_state = None
    best_ema_per_class = None
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # alternate mixup / cutmix per batch
            if np.random.rand() < 0.5:
                x_in, y_a, y_b, lam = mixup_batch(x, y, args.mixup_alpha)
            else:
                x_in, y_a, y_b, lam = cutmix_batch(x, y, args.cutmix_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x_in)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, decay=args.ema)
            loss_sum += float(loss.detach()) * x.size(0)
            n += x.size(0)
        scheduler.step()
        train_loss = loss_sum / max(n, 1)

        acc_p, overall_p = per_class_accuracy(model, val_loader, device)
        acc_e, overall_e = per_class_accuracy(ema_model, val_loader, device)
        min_p, min_e = float(acc_p.min()), float(acc_e.min())
        per_class_p = {CLASS_ORDER[c]: float(acc_p[c]) for c in range(NUM_CLASSES)}
        per_class_e = {CLASS_ORDER[c]: float(acc_e[c]) for c in range(NUM_CLASSES)}
        elapsed = time.time() - t0
        msg = (
            f"epoch={epoch+1:02d}/{args.epochs} loss={train_loss:.4f} "
            f"plain_acc={overall_p:.4f}/min={min_p:.4f} "
            f"ema_acc={overall_e:.4f}/min={min_e:.4f} "
            f"elapsed_s={elapsed:.1f} lr={optimizer.param_groups[0]['lr']:.5f} "
            f"ema_per_class={per_class_e}"
        )
        print(msg, flush=True)
        log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "plain_overall_acc": overall_p,
            "plain_min_acc": min_p,
            "plain_per_class_acc": per_class_p,
            "ema_overall_acc": overall_e,
            "ema_min_acc": min_e,
            "ema_per_class_acc": per_class_e,
            "elapsed_s": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if min_e > best_ema_min:
            best_ema_min = min_e
            best_ema_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
            best_ema_per_class = per_class_e

    if best_ema_state is None:
        best_ema_state = ema_model.state_dict()
        best_ema_per_class = per_class_e

    # restore best EMA
    ema_model.load_state_dict(best_ema_state)
    final_acc, final_overall = per_class_accuracy(ema_model, val_loader, device)
    final_per_class = {CLASS_ORDER[c]: float(final_acc[c]) for c in range(NUM_CLASSES)}
    print(f"[train] best EMA overall={final_overall:.4f} per_class={final_per_class}", flush=True)

    # Save state_dict for diagnostics + future re-export
    torch.save(best_ema_state, out_root / "script_id_v2.pt")

    # Export ONNX with permuted final Linear → canonical enum order
    onnx_path = out_root / "script_id.onnx"
    export_onnx_permuted(ema_model, onnx_path)
    print(f"[train] wrote {onnx_path} ({onnx_path.stat().st_size} bytes)", flush=True)

    # Map trainer per-class to canonical names for meta.json
    name_remap = {"han": "chinese", "cyrillic": "eslav", "hangul": "korean"}
    final_per_class_canon = {name_remap.get(k, k): v for k, v in final_per_class.items()}

    meta = {
        "model": "MobileNetV3-Small (torchvision, EMA, v2 recipe)",
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
            "note": "aspect-ratio not preserved",
        },
        "classes": {str(i): n for i, n in enumerate(CANON_NAMES)},
        "class_names": CANON_NAMES,
        "enum_contract": "canonical C++ turbo_ocr::script_id::ScriptId",
        "trainer_class_order": CLASS_ORDER,
        "trainer_to_canonical_perm": PERM,
        "output": "logits over 7 classes (canonical order); apply softmax for probabilities",
        "training_data": {
            "source": "Synthetic PIL renders, 4+ fonts per script, varied size/contrast/color",
            "per_class": int(len(train_items) / NUM_CLASSES + len(val_items) / NUM_CLASSES),
            "split": "per-class shuffle then 10% val",
            "val_frac": args.val_frac,
            "seed": args.seed,
        },
        "training": {
            "epochs": args.epochs,
            "warmup_epochs": args.warmup,
            "batch_size": args.batch,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.wd,
            "schedule": "LinearLR(start=0.01, iters=warmup) -> CosineAnnealingLR",
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "ema_decay": args.ema,
            "amp": "fp16 autocast",
            "augmentations": "albumentations: rotate±5°, perspective, jpeg Q30-95, blur, brightness, coarse dropout, channel shuffle",
        },
        "metrics": {
            "best_val_ema_overall_acc": float(final_overall),
            "best_val_ema_per_class_acc": final_per_class_canon,
            "best_val_ema_min_acc": float(final_acc.min()),
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
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2))
    (out_root / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"[train] wrote meta.json + training_log.json", flush=True)

    gate_pass = all(v >= 0.98 for v in final_per_class_canon.values())
    print(f"[train] {'PASS' if gate_pass else 'BELOW'} synthetic gate (≥98% per class) "
          f"min={min(final_per_class_canon.values()):.4f}", flush=True)


if __name__ == "__main__":
    main()
