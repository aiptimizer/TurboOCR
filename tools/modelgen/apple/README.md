# Apple native-mode artefacts

The apple backend has two modes. `onnx` runs the ONNX models on the CoreML
execution provider — what a bare model download gets. `native` is the measured
~5× configuration: Metal kernels + MPSGraph detection/recognition on the GPU,
with the narrow recognition buckets on the **Neural Engine** in parallel
(`TURBO_APPLE_ANE_MAXW` sets the split, default 800). `mode="auto"` selects
native whenever its artefacts are present — this directory generates them.

## One command per tier

```bash
python tools/modelgen/apple/export_apple_native.py \
    --tier small --models models --out out/ [--ane] [--pack]
```

Produces, in the engine's own discovery layout (drop it into a models
directory and native mode lights up — no configuration):

| Artefact | What |
|---|---|
| `det_<tier>/graph.json` + `weights.bin` | MPSGraph detector, static **992×768** canvas (the conformance-validated one; the earlier 1280×1280 export was the conformance-red artifact) |
| `rec_<tier>/rec_b<W>/…` | MPSGraph recognizer, one static graph per width of the shared 9-bucket ladder (`recognition::kRecWidthBuckets`) |
| `cls/…` | MPSGraph 0°/180° line classifier (cls.onnx is statically 80×160) |
| `coreml/<tier>/rec_ane_<W>.mlpackage` | ANE-lane packages for W ∈ {320, 480, 800} — `--ane` only |

`--pack` additionally writes `apple_native_<tier>.tar.gz`, the release-asset
form: the Python wheel's `ModelStore.ensure_apple_native()` downloads and
extracts it into the model cache automatically on Apple silicon, and the
engine finds everything by path.

## Dependencies

MPSGraph exports need only `onnx`, `numpy`, `onnxruntime`. The `--ane` step
converts through PyTorch and needs `torch`, `onnx2torch`, `coremltools` —
heavy, so it is optional and skipped with a notice when the imports fail.

## Verification built in

Every MPSGraph export ships a fixed input plus an ONNX Runtime golden output
(`tools/modelgen/mps_export_rec.py` writes them; the engine's bring-up tests
consume them). Every ANE package is accepted only when its argmax agrees with
ONNX Runtime on a fixed input (≥ 99.9%; measured 100.00% on all shipped
widths — `mlprogram` runs fp16 internally, hence not bit-equality).

## Sharp edges, learned the hard way

- **ANE packages are fixed batch-1.** `torch.jit.trace` bakes this net's
  attention reshapes at the traced batch; CoreML's shape propagation rejects
  every other batch of an `EnumeratedShapes` package (E5RT "cannot reshape").
  The engine reads the package's supported shapes and the shared planner
  feeds batch-1 rungs — correct by construction, just don't "improve" this
  back to enumerated shapes without a runtime proof.
- **The ANE package encodes the tier's dictionary.** tiny has 6906 classes,
  small/medium 18710 — a wrong-tier package decodes garbage silently. The
  per-tier `coreml/<tier>/` layout exists so that cannot happen; keep it.
- **`fix_autopad.py` is load-bearing** for the ANE path: onnx2torch does not
  implement `auto_pad=SAME_UPPER`, so pads are rewritten explicitly at the
  frozen static shape first.
- **Do not export rec widths outside the shared ladder** — the recognizer
  restricts discovery to `kRecWidthBuckets`; anything else is dead weight
  (an earlier directory-scan behaviour picked up 42 stray exports and ran at
  half throughput).
