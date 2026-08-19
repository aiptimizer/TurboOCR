#!/usr/bin/env python3
"""Produce the Apple NATIVE-mode artefact bundle for one model tier.

The apple backend has two modes (src/backends/apple/): `onnx` (CoreML EP —
what a bare model download gets) and `native` (Metal + MPSGraph on the GPU
with the ANE recognition lane — the measured ~5x mode). Native loads from
export artefacts this script generates; the engine's discovery is automatic:
if the artefacts sit next to the ONNX files, `mode="auto"` selects native.

Layout produced (drop it INTO a models directory, or ship it as the
`apple_native_<tier>.tar.gz` release asset):

    det_<tier>/det_c<H>x<W>/  graph.json+weights.bin  MPSGraph detector export.
                                                   ONE canvas: the runtime
                                                   re-specializes it per page
                                                   shape (fully-convolutional
                                                   graph; shared 128-grid
                                                   snap, bounded cache), so
                                                   extra canvases would be
                                                   duplicate weights, not
                                                   coverage
    rec_<tier>/rec_b<W>/  ...                      MPSGraph recognizer, one
                                                   static graph per width of
                                                   the SHARED 9-bucket ladder
    cls/                ...                        MPSGraph 0/180 classifier
    coreml/<tier>/rec_ane_<W>.mlpackage            ANE lane packages for the
                                                   narrow buckets (<= the
                                                   TURBO_APPLE_ANE_MAXW
                                                   default of 800)

Naming rules are the ENGINE's, not this script's: `mps_export_dir()` in
apple_backend.mm maps models/<stem>.onnx -> models/<stem>/, MpsRecognizer
walks rec_b<W> subdirs restricted to recognition::kRecWidthBuckets, and the
ANE loader probes coreml/<tier>/rec_ane_<W>.mlpackage first. Medium is the
un-infixed stem (det.onnx/rec.onnx -> det/, rec/) — historical, matched here.

The 992x768 det canvas is the conformance-validated one (a 1280x1280 export
measured 98.2% -> red on the det-canvas conformance check; 992x768 passes).
The rec widths are recognition::kRecWidthBuckets — keep LADDER below equal
to that table (rec_geometry.h); the recognizer ignores any other width.

MPSGraph exports need only onnx+numpy+onnxruntime. The ANE step (--ane)
additionally needs torch, onnx2torch and coremltools (heavy); it is skipped
with a notice when they are missing. Both artefact kinds are verified against
an ONNX Runtime golden output before being accepted.

Usage:
    python tools/modelgen/apple/export_apple_native.py --tier small \\
        --models models --out build-apple-native [--ane] [--pack]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
MPS_EXPORT = os.path.join(HERE, "..", "mps_export_rec.py")

# == recognition::kRecWidthBuckets (rec_geometry.h). The engine restricts
# discovery to this table, so exporting anything else is dead weight.
LADDER = [320, 480, 800, 1200, 1600, 2000, 2500, 3200, 4000]
#: ANE lane widths: buckets at or below the TURBO_APPLE_ANE_MAXW default (800).
ANE_WIDTHS = [320, 480, 800]
#: Det canvas export (H, W). ONE canvas is the whole story now: the engine
#: builds its MPSGraph from graph.json AT RUNTIME and the det graph is fully
#: convolutional (zero Reshape ops, scale-based Resize), so MpsDetector
#: re-specializes this one export per page shape (shared policy:
#: detection::compute_det_resize -> snap_det_canvas_grid, letterboxed content,
#: LRU-bounded canvas cache — see mps_stages.h). Exporting more canvases would
#: ship byte-identical weights again without covering anything the runtime
#: doesn't already cover. 992x768 stays as the template because it is the
#: conformance-validated portrait-document canvas: it serves warmup, and it is
#: the fallback if a runtime specialization ever fails (and the single fixed
#: canvas under TURBO_APPLE_DET_JIT=0).
DET_CANVASES = [(992, 768)]
#: cls canvas (H, W). cls.onnx declares STATIC 80x160 spatial dims, and the
#: export tool defers to a model's static dims — these values only matter if a
#: future cls export ships with dynamic spatial dims.
CLS_CANVAS = (80, 160)

#: models/<file> per tier. Medium uses the historical un-infixed stems.
TIER_FILES = {
    "tiny": {"det": "det_tiny.onnx", "rec": "rec_tiny.onnx"},
    "small": {"det": "det_small.onnx", "rec": "rec_small.onnx"},
    "medium": {"det": "det.onnx", "rec": "rec.onnx"},
}
CLS_FILE = "cls.onnx"


def run_mps_export(src: str, out: str, h: int, w: int) -> None:
    """One static-shape MPSGraph export (graph.json + weights.bin + golden)."""
    os.makedirs(out, exist_ok=True)
    subprocess.run([sys.executable, MPS_EXPORT, src, out, str(h), str(w)],
                   check=True)
    for required in ("graph.json", "weights.bin"):
        p = os.path.join(out, required)
        if not os.path.isfile(p):
            raise RuntimeError(f"export produced no {required}: {out}")


def export_mpsgraph(models: str, tier: str, out: str) -> None:
    files = TIER_FILES[tier]
    det_stem = os.path.splitext(files["det"])[0]
    rec_stem = os.path.splitext(files["rec"])[0]

    for dh, dw in DET_CANVASES:
        sub = f"det_c{dh}x{dw}"
        print(f"[det] {files['det']} @ {dh}x{dw} -> {det_stem}/{sub}/")
        run_mps_export(os.path.join(models, files["det"]),
                       os.path.join(out, det_stem, sub), dh, dw)

    for w in LADDER:
        print(f"[rec] {files['rec']} @ 48x{w} -> {rec_stem}/rec_b{w}/")
        run_mps_export(os.path.join(models, files["rec"]),
                       os.path.join(out, rec_stem, f"rec_b{w}"), 48, w)

    cls_src = os.path.join(models, CLS_FILE)
    if os.path.isfile(cls_src):
        ch, cw = CLS_CANVAS
        print(f"[cls] {CLS_FILE} @ {ch}x{cw} -> cls/")
        run_mps_export(cls_src, os.path.join(out, "cls"), ch, cw)
    else:
        print(f"[cls] {cls_src} not present — skipping (cls falls back to ONNX)")


def export_ane(models: str, tier: str, out: str) -> bool:
    """CoreML mlprogram packages for the ANE recognition lane.

    FIXED batch-1 mlprogram, the argmax+max head baked in (the engine reads
    [B,T] int32 indices + fp32 scores), compute units CPU_AND_NE — the exact
    recipe of the measured tiered packages (convert_tiers). NOT
    EnumeratedShapes: torch.jit.trace bakes this net's attention reshapes at
    the traced batch, and CoreML's shape propagation rejects every other
    enumerated batch (E5RT "cannot reshape"). The engine reads the package's
    supported shapes and the shared planner feeds batch-1 rungs accordingly.
    Verified: CoreML argmax must match ORT argmax on a fixed input (>= 99.9%)
    before a package is accepted."""
    try:
        import coremltools as ct  # noqa: F401
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        import torch.nn as nn
        from onnx2torch import convert as o2t
    except ImportError as exc:
        print(f"[ane] SKIPPED — missing dependency ({exc}). Install torch, "
              "onnx2torch and coremltools (see README.md) and re-run with --ane.")
        return False

    sys.path.insert(0, HERE)
    from fix_autopad import fix  # SAME_UPPER -> explicit pads; onnx2torch need

    class RecArgmax(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.b = base

        def forward(self, x):
            y = self.b(x)          # [B, T, C] logits
            v, i = y.max(-1)
            return i.to(torch.int32), v

    rec_src = os.path.join(models, TIER_FILES[tier]["rec"])
    dest = os.path.join(out, "coreml", tier)
    os.makedirs(dest, exist_ok=True)
    for w in ANE_WIDTHS:
        pkg = os.path.join(dest, f"rec_ane_{w}.mlpackage")
        if os.path.exists(pkg):
            shutil.rmtree(pkg)  # regenerate: a half-written package is worse
        print(f"[ane] {os.path.basename(rec_src)} @ 48x{w} -> {pkg}")
        fixed = fix(rec_src, w)
        tmp = os.path.join(out, f"_ane_{tier}_{w}.onnx")
        onnx.save(fixed, tmp)
        base = o2t(tmp).eval()
        model = RecArgmax(base).eval()
        example = torch.randn(1, 3, 48, w)
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, 3, 48, w))],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS15,
        )
        # Golden: CoreML argmax vs ORT argmax on a fixed input. mlprogram runs
        # fp16 internally, so demand >=99.9% class-id agreement rather than
        # bit-equality (the measured packages were accepted the same way).
        x = np.random.default_rng(0).standard_normal((1, 3, 48, w)).astype(np.float32)
        sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        ref = sess.run(None, {sess.get_inputs()[0].name: x})[0].argmax(-1)
        out_d = ml.predict({"x": x})
        idx_key = [k for k, v in out_d.items()
                   if np.asarray(v).dtype.kind in "iu"][0]
        got = np.asarray(out_d[idx_key]).reshape(ref.shape)
        match = float((got == ref).mean()) * 100.0
        print(f"[ane]   argmax match vs ORT: {match:.2f}%")
        if match < 99.9:
            raise RuntimeError(
                f"ANE package argmax match {match:.2f}% < 99.9% at width {w}")
        ml.save(pkg)
        os.remove(tmp)
    return True


def pack(out: str, tier: str) -> str:
    """One tar.gz release asset holding the bundle, paths relative to the
    models dir so it extracts in place."""
    asset = os.path.join(out, f"apple_native_{tier}.tar.gz")
    files = TIER_FILES[tier]
    members = [os.path.splitext(files["det"])[0],
               os.path.splitext(files["rec"])[0]]
    if os.path.isdir(os.path.join(out, "cls")):
        members.append("cls")
    if os.path.isdir(os.path.join(out, "coreml", tier)):
        members.append(os.path.join("coreml", tier))
    with tarfile.open(asset, "w:gz") as tf:
        for m in members:
            tf.add(os.path.join(out, m), arcname=m)
    print(f"[pack] {asset}")
    return asset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", required=True, choices=sorted(TIER_FILES))
    ap.add_argument("--models", default="models",
                    help="directory holding the tier's ONNX files")
    ap.add_argument("--out", required=True,
                    help="output directory (models-dir-shaped)")
    ap.add_argument("--ane", action="store_true",
                    help="also build the CoreML ANE packages (heavy deps)")
    ap.add_argument("--pack", action="store_true",
                    help="pack the bundle into apple_native_<tier>.tar.gz")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    export_mpsgraph(args.models, args.tier, args.out)
    if args.ane:
        export_ane(args.models, args.tier, args.out)
    if args.pack:
        pack(args.out, args.tier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
