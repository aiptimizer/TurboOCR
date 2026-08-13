#!/usr/bin/env python3
"""Build models/forms/ffdetr.onnx from the published FFDetr checkpoint.

FFDetr is the form-field detector behind ?fields=1 — an RF-DETR trained on
CommonForms (~450k pages of real fillable PDFs) that predicts Text, CheckBox
and Signature widgets from a page raster. It is OPTIONAL: without this file the
server falls back to the four geometry detectors in field_detector.h.

    python3 tools/modelgen/export_ffdetr.py --out models/forms/ffdetr.onnx

Needs torch + rfdetr, which are NOT runtime dependencies — install them in a
throwaway environment and delete it afterwards:

    uv venv /tmp/ffdetr && VIRTUAL_ENV=/tmp/ffdetr \
        uv pip install rfdetr onnx onnxruntime onnxconverter-common huggingface_hub
    /tmp/ffdetr/bin/python tools/modelgen/export_ffdetr.py --out models/forms/ffdetr.onnx

LICENCE — the whole chain is Apache-2.0, and that is why this model and not the
better-known one:
    jbarrow/FFDetr weights ......... Apache-2.0
    jbarrow/CommonForms (data) ..... Apache-2.0
    roboflow/rf-detr (arch+init) ... Apache-2.0
    facebookresearch/dinov2 ........ Apache-2.0
The paper's headline model, FFDNet, is YOLO11 via Ultralytics and therefore
AGPL-3.0. Do not substitute it here — it would relicense the server.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

FP32, FP16 = TensorProto.FLOAT, TensorProto.FLOAT16

# Measured on the CommonForms test split, 150 pages, IoU 0.5, conf 0.40:
#   1024  text F1 0.910 | checkbox F1 0.551 | signature F1 0.444 | overall 0.859
#   1280  text F1 0.923 | checkbox F1 0.534 | signature F1 0.000 | overall 0.867
# Higher resolution buys text accuracy and LOSES checkbox and signature, the
# two classes that exist to beat what commercial preparers can do. 1024 is also
# the only resolution the checkpoint has been evaluated at upstream.
DEFAULT_RESOLUTION = 1024


def export_onnx(resolution: int, workdir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRMedium

    ckpt = hf_hub_download(repo_id="jbarrow/FFDetr", filename="FFDetr.pth")
    print(f"[1/3] checkpoint {ckpt}", flush=True)

    # NO num_classes= here. The checkpoint carries a 90-slot COCO-shaped head
    # and trains only its first three slots; passing a class count makes rfdetr
    # REINITIALISE that head at random. The export then still succeeds, still
    # loads, and still emits 300 boxes — of pure noise.
    print(f"[2/3] loading RFDETRMedium @ {resolution}", flush=True)
    model = RFDETRMedium(pretrain_weights=ckpt, device="cpu",
                         resolution=resolution)

    workdir.mkdir(parents=True, exist_ok=True)
    model.export(output_dir=str(workdir), opset_version=17, batch_size=1,
                 shape=(resolution, resolution))
    produced = sorted(workdir.rglob("*.onnx"))
    if not produced:
        raise RuntimeError("rfdetr produced no .onnx")
    return max(produced, key=lambda p: p.stat().st_size)


def to_fp16(src: Path, dst: Path) -> None:
    """fp16 weights behind an fp32 interface — smaller, but NOT faster.

    Accuracy is unaffected (overall F1 0.859 either way, per class within
    0.003), so this is purely 139 MB -> 77 MB. It costs latency on the CPU
    provider at every thread count, because ORT has no native fp16 kernels for
    this graph and casts back to fp32 internally: 651 vs 520 ms at 8 threads.

    Two things the stock converter gets wrong on this graph, both of which make
    ORT refuse to load it at the first patch-embedding Conv:
      * keep_io_types=True does not insert the input Cast, so an fp32 graph
        input feeds fp16 weights;
      * RF-DETR's exporter emits 35 explicit Cast(to=FLOAT) nodes, which the
        converter leaves alone — fp32 islands wired into fp16 consumers.
    So: convert everything, retarget those casts, then re-attach an fp32
    boundary by hand.
    """
    from onnxconverter_common import float16

    m = float16.convert_float_to_float16(onnx.load(src), keep_io_types=False,
                                         disable_shape_infer=True)

    retargeted = 0
    for node in m.graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == FP32:
                attr.i = FP16
                retargeted += 1

    g, casts = m.graph, []
    for io, is_input in [(i, True) for i in g.input] + [(o, False) for o in g.output]:
        if io.type.tensor_type.elem_type != FP16:
            continue
        inner = io.name + "_fp16"
        for node in g.node:
            names = node.input if is_input else node.output
            for i, name in enumerate(names):
                if name == io.name:
                    names[i] = inner
        casts.append((is_input, helper.make_node(
            "Cast", [io.name if is_input else inner],
            [inner if is_input else io.name],
            to=FP16 if is_input else FP32, name=f"cast_{io.name}")))
        io.type.tensor_type.elem_type = FP32

    nodes = list(g.node)
    del g.node[:]
    g.node.extend(n for is_in, n in casts if is_in)
    g.node.extend(nodes)
    g.node.extend(n for is_in, n in casts if not is_in)
    del g.value_info[:]  # stale fp16 shapes would contradict the new casts

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"[3/3] fp16: retargeted {retargeted} internal casts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("models/forms/ffdetr.onnx"))
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    # fp32 by default. fp16 halves the file (139 -> 77 MB) at identical
    # accuracy (overall F1 0.859 either way), but it is SLOWER on the CPU
    # provider at every thread count — 651 vs 520 ms at 8 threads — because ORT
    # has no native fp16 kernels for this graph and casts back internally.
    # Choose it only when 62 MB matters more than 25% of the page latency.
    ap.add_argument("--fp16", action="store_true",
                    help="half the file size, ~25%% slower on CPU, same accuracy")
    args = ap.parse_args()

    work = args.out.parent / ".ffdetr_export"
    try:
        raw = export_onnx(args.resolution, work)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.fp16:
            to_fp16(raw, args.out)
        else:
            shutil.copy2(raw, args.out)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    import numpy as np
    import onnxruntime as ort

    s = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    outs = s.run(None, {s.get_inputs()[0].name:
                        np.zeros((1, 3, args.resolution, args.resolution), np.float32)})
    ok = all(bool(np.isfinite(o).all()) for o in outs)
    print(f"\n{args.out}  {os.path.getsize(args.out) / 1e6:.1f} MB")
    print(f"  in  {s.get_inputs()[0].name} {s.get_inputs()[0].shape}")
    for o in s.get_outputs():
        print(f"  out {o.name} {o.type}")
    print(f"  smoke run: {'OK' if ok else 'NON-FINITE OUTPUT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
