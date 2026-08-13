# Running the legacy PP-OCRv5 models

The previous-generation **PP-OCRv5** Latin models (`det_v5.onnx`,
`rec_v5.onnx`, `keys_v5.txt`) are **not** part of the release bundle —
`scripts/models/fetch/fetch_release_models.sh` ships only the v6 tiers and
the retained per-script v5 recognizers. If you have the files, the working
full-v5 configuration (validated 2026-08-06 on the RTX 5090) runs on the
**ONNX Runtime CUDA engine mode**, because TensorRT can no longer compile the
v5 detector (below):

```bash
TURBO_ENGINE_MODE=onnx \
DET_ONNX=models/det_v5.onnx \
REC_ONNX=models/rec_v5.onnx \
REC_DICT=models/keys_v5.txt \
./build/turboocr-server
```

The overrides win over `OCR_MODEL` per stage. To keep native-TRT speed for
everything except the legacy recognizer, swap only the recognizer instead —
the v5 rec engine still builds fine on TRT, and this is the same shape the
shipped per-script v5 bundles (`OCR_MODEL=arabic/…`) use:

```bash
REC_ONNX=models/rec_v5.onnx \
REC_DICT=models/keys_v5.txt \
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/turboocr-server
```

!!! warning "The v5 *detector* cannot be built by current TensorRT"
    Re-verified 2026-08-06 (RTX 5090, TensorRT 10.15.1): `DET_ONNX=det_v5.onnx`
    fails `buildSerializedNetwork` with `Error Code 10 … Could not find any
    implementation for node {ForeignNode[…]}` at every combination of build
    workspace (4/8 GiB), `TRT_OPT_LEVEL` (5/3), detection profile (1280/960),
    **and in both precisions** (default fp16 and the `TRT_FP16=0` fp32
    escape hatch — the fused-graph compiler rejects the whole network either
    way, even on an otherwise empty GPU). Hence the `TURBO_ENGINE_MODE=onnx`
    recipe above: ORT-CUDA runs the old graph without complaint.

**Coverage caveat — Latin only.** The retained PP-OCRv5 recognizer dictionary
(`keys_v5.txt`) is 836 characters with **no CJK**. On English forms/receipts v5 is
within ~1.5–2 points of PP-OCRv6-medium (measured this release: FUNSD 90.3% vs
91.9%, CORD 91.7% vs 93.4% word-F1), but on the EN+ZH **OmniDocBench-125** set it
scores far lower (**52.7% vs 91.0%** text accuracy) because it cannot read the
Chinese pages. Use the v6 default for mixed-script documents; the v5 path is for
Latin-only workloads or direct A/B comparison.
