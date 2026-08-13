#!/usr/bin/env python3
"""Permute the final Linear layer of script_id.onnx into the canonical C++ enum order.

Trainer class order (this S13 run):
    0=latin, 1=han, 2=cyrillic, 3=arabic, 4=greek, 5=hangul, 6=thai

Canonical C++ ScriptId enum (turbo_ocr::script_id):
    0=Latin, 1=Chinese, 2=Arabic, 3=ESlav, 4=Greek, 5=Korean, 6=Thai

Names map: han→chinese, cyrillic→eslav, hangul→korean. Index slots 2 and 3
are swapped between trainer and C++. Fix by permuting the rows of the final
classifier Linear's weight + bias and re-saving the ONNX.

Why operate on the ONNX directly: the trainer didn't snapshot a .pt
state_dict (gap in v1 driver — fixed for v2). Going via the ONNX initializers
is bulletproof and avoids a 4-minute retrain.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[3] / "models" / "script_id"
SRC = ROOT / "script_id.onnx"
DST = ROOT / "script_id.onnx"  # overwrite — old data file will be orphaned + deleted at end

# new_row[i] = old_row[PERM[i]]
PERM = [0, 1, 3, 2, 4, 5, 6]

CANON_NAMES = ["latin", "chinese", "arabic", "eslav", "greek", "korean", "thai"]
TRAINER_NAMES = ["latin", "han", "cyrillic", "arabic", "greek", "hangul", "thai"]


def main():
    model = onnx.load(str(SRC), load_external_data=True)
    g = model.graph

    # Find every initializer with first dim == 7 — those are the final-layer
    # weight (7×in_features) and bias (7,). There should be exactly two.
    candidates = []
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        if arr.shape and arr.shape[0] == 7:
            candidates.append((init, arr))
    if len(candidates) < 2:
        sys.exit(f"expected ≥2 initializers with first-dim 7 (weight + bias), got {len(candidates)}")

    permuted = 0
    for init, arr in candidates:
        if arr.ndim == 2:
            # weight: shape (7, in_features). permute rows
            new_arr = arr[PERM, :].copy()
        elif arr.ndim == 1:
            # bias: shape (7,)
            new_arr = arr[PERM].copy()
        else:
            print(f"  skipping {init.name} (shape={arr.shape}) — not a final-layer tensor", flush=True)
            continue
        new_init = numpy_helper.from_array(new_arr, name=init.name)
        # in-place swap inside the graph
        idx = list(g.initializer).index(init)
        g.initializer.remove(init)
        g.initializer.insert(idx, new_init)
        permuted += 1
        print(f"  permuted {init.name}  shape={arr.shape}", flush=True)

    if permuted != 2:
        sys.exit(f"expected to permute exactly 2 tensors (weight+bias), got {permuted}")

    # Strip any leftover external_data references so we save self-contained.
    for init in g.initializer:
        # external_data + data_location set the .data file path
        if init.data_location == onnx.TensorProto.EXTERNAL:
            init.data_location = onnx.TensorProto.DEFAULT
        while init.external_data:
            init.external_data.pop()

    # Single-file save — no external data sidecar.
    onnx.save_model(model, str(DST), save_as_external_data=False)
    sidecar = DST.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()
        print(f"  removed orphaned sidecar {sidecar}", flush=True)
    print(f"wrote permuted {DST} ({DST.stat().st_size} bytes)", flush=True)

    # Update meta.json
    meta_path = ROOT / "meta.json"
    meta = json.loads(meta_path.read_text())
    # Rewrite class fields
    meta["classes"] = {str(i): name for i, name in enumerate(CANON_NAMES)}
    meta["class_names"] = CANON_NAMES
    meta["enum_contract"] = "canonical C++ turbo_ocr::script_id::ScriptId"
    meta["trainer_class_order"] = TRAINER_NAMES
    meta["trainer_to_canonical_perm"] = PERM
    # Remap per-class metrics dict keys to canonical names
    name_remap = {"han": "chinese", "cyrillic": "eslav", "hangul": "korean"}
    if "metrics" in meta and "best_val_per_class_acc" in meta["metrics"]:
        old = meta["metrics"]["best_val_per_class_acc"]
        meta["metrics"]["best_val_per_class_acc"] = {
            name_remap.get(k, k): v for k, v in old.items()
        }
    meta["onnx"]["file_size_bytes"] = DST.stat().st_size
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"updated {meta_path}", flush=True)


if __name__ == "__main__":
    main()
