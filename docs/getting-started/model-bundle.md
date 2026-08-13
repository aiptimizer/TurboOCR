# Models

!!! abstract "TL;DR"
    `scripts/models/fetch/fetch_release_models.sh` pulls every supported PP-OCRv6
    bundle plus the language-agnostic detection / classification /
    layout heads from a pinned GitHub Release, verifies SHA-256 against
    `SHA256SUMS.txt`, and lays them out on disk in the layout the
    server expects at startup.

## Release URL

```bash
MODELS_RELEASE_URL="${MODELS_RELEASE_URL:-https://github.com/aiptimizer/TurboOCR/releases/download/models-v3.0.0-ppocrv6}"
```

Override `MODELS_RELEASE_URL` to pin a different release tag (e.g.
inside an air-gapped mirror). The fetcher first downloads
`SHA256SUMS.txt` and then refuses any asset whose hash doesn't match
(`fetch_release_models.sh:34-46`).

## Per-language download flow

The PP-OCRv6 tiers cover Latin + Chinese + Japanese, so Chinese is **not** a
separate download. Only the non-Latin scripts have per-language recognizers —
for every script in `LANGS=(greek eslav arabic korean thai)`:

```text
${MODELS_RELEASE_URL}/rec-${lang}.onnx   →  models/rec/${lang}/rec.onnx
${MODELS_RELEASE_URL}/dict-${lang}.txt   →  models/rec/${lang}/dict.txt
```

The default PP-OCRv6 det/rec/cls tiers + shared stages land at:

```text
${MODELS_RELEASE_URL}/det.onnx        →  models/det.onnx        (PP-OCRv6 medium det)
${MODELS_RELEASE_URL}/det_small.onnx  →  models/det_small.onnx
${MODELS_RELEASE_URL}/det_tiny.onnx   →  models/det_tiny.onnx   (default tier)
${MODELS_RELEASE_URL}/rec.onnx        →  models/rec.onnx        (PP-OCRv6 medium, Latin+CJK)
${MODELS_RELEASE_URL}/rec_small.onnx  →  models/rec_small.onnx
${MODELS_RELEASE_URL}/rec_tiny.onnx   →  models/rec_tiny.onnx   (default tier)
${MODELS_RELEASE_URL}/keys.txt        →  models/keys.txt
${MODELS_RELEASE_URL}/keys_tiny.txt   →  models/keys_tiny.txt
${MODELS_RELEASE_URL}/cls.onnx        →  models/cls.onnx
${MODELS_RELEASE_URL}/layout.onnx     →  models/layout/layout.onnx
${MODELS_RELEASE_URL}/doc_ori.onnx    →  models/doc_ori.onnx
```

## On-disk inventory

Verified tree from this checkout (`find models -maxdepth 3 -type f`):

```text
cls.onnx
det.onnx              det_small.onnx   det_tiny.onnx     # PP-OCRv6 det tiers
rec.onnx             rec_small.onnx   rec_tiny.onnx      # PP-OCRv6 rec tiers (Latin+CJK)
keys.txt             keys_tiny.txt
doc_ori.onnx
MANIFEST.txt
layout/layout.onnx
rec/arabic/{rec.onnx,dict.txt}
rec/eslav/{rec.onnx,dict.txt}
rec/greek/{rec.onnx,dict.txt}
rec/korean/{rec.onnx,dict.txt}
rec/thai/{rec.onnx,dict.txt}
script_id/{script_id.onnx, script_id_v2.pt, meta.json, training_log.json}
table/slanext_encoder/{SLANeXt_wired_encoder.onnx, SLANeXt_wired_decoder.bin, SLANeXt_dict_infer.txt}   # SLANet-Plus — the only local table backend
formula/ppformulanet_s/{fast/{encoder,prep,step_batched}.onnx, inference_trt.onnx, tokenizer.json}        # PP-FormulaNet_plus-S weights, historical dir name (GPU fast graphs + CPU fused)
formula/ppformulanet_plus_m/{encoder,prep,decoder_step,decoder_step_384}.onnx, tokenizer.json             # plus-M (Chinese, opt-in)
```

!!! note "Superseded / archival files on disk"
    Older table and formula artifacts (`table/{table_cls,slanet_plus,table_struct_tatr,table_struct_nemotron}.onnx`,
    the 3-engine `formula/{encoder,decoder,image_resizer}.onnx` set, and `*.orig.onnx`/`*.bak`/intermediate
    `ppformulanet_s/inference*.onnx` exports) remain on disk for archival but are **not loaded and not shipped** —
    the live backends are SLANet-Plus (table) and PP-FormulaNet_plus-S / plus-M (formula) above.

!!! tip "MANIFEST.txt is the canonical ledger"
    `models/MANIFEST.txt` records the exact SHA-256 the server expects
    for each asset, plus inline surgery notes explaining why
    pre-surgery copies are retained.

## Persisted cache in Docker

Both images mount `/home/ocr/.cache/turbo-ocr` as the persistent cache
target. The Dockerfile symlinks `/app/models/rec → /home/ocr/.cache/
turbo-ocr/models/rec` **before** running `fetch_release_models.sh`, so
every per-language bundle lands directly in the cache volume
(`docker/Dockerfile`, the `nvidia` target). A single
`-v trt-cache:/home/ocr/.cache/turbo-ocr` thus persists:

- TensorRT engine plans (built from ONNX on first start, ~90 s).
- All per-language recognition bundles.

The `det`, `cls`, `rec.onnx`, `keys.txt`, and `layout/` heads stay
inside the image because they're language-agnostic.

## Formula & table backends

How the formula split graphs and the SLANet-Plus encoder/decoder actually
execute is documented once, on their model pages:
[Formula](../models/formula.md) (PP-FormulaNet_plus-S fast graphs + the
opt-in `ppformulanet_plus_m`) and [Table](../models/table.md) (SLANet-Plus
via `TABLE_BACKEND=slanext`). This page only defines what lands on disk.

## `FETCH_MODELS` CMake option

Both Dockerfiles pass `-DFETCH_MODELS=OFF` because the fetcher already
ran in an earlier image layer. For native builds, leave the default
behaviour and run `fetch_release_models.sh` once by hand:

```bash
bash scripts/models/fetch/fetch_release_models.sh
```

Re-running is idempotent — the SHA-256 verification will reject any
asset that drifts from the pinned hashes.

!!! info "See also"
    - [Build → Native](native.md) — when to invoke the fetcher by
      hand.
    - [Build → Docker](docker.md) — how the fetcher is staged into the
      image.
    - [Models → Formula](../models/formula.md) — PP-FormulaNet_plus-S, the live
      backend (the archival three-engine export is documented there too).
