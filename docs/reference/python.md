# Python library

The Python package wraps the **same C++ pipeline the server runs** — a
nanobind extension over `UnifiedOcrPipeline`, not a reimplementation. The GIL
is released during inference, output is identical to the server's, and
concurrency comes from a built-in replica pool.

> **v4.0.0-alpha.1 — the engine wheels are not on PyPI yet.** `pip install
> turboocr` today gets the published 0.3.0 *client* (talks to a server, no
> in-process engine). Until the wheels are published, build one from the
> checkout — the exact commands per backend are in
> [Install](../getting-started/install.md). Once published:
> `pip install --pre "turboocr[cpu]"` (or `[cuda12]` / `[cuda13]` /
> `[openvino]` / `[rocm]` — one per environment, mutually exclusive), and
> `turboocr doctor` names the right one for your machine.

```python
import turboocr_engine as turboocr   # `import turboocr` once the umbrella is installed

ocr = turboocr.OCR()                          # tiny tier, backend="auto"
page = ocr.read("invoice.png")
print(page.text)                              # lines joined in reading order

ocr = turboocr.OCR("medium", backend="cuda", replicas=3)
doc = ocr.read_batch(images)                  # fans out across the replica pool
doc = ocr.read_pdf("report.pdf", dpi=150)     # needs the `pdf` extra
print(doc.to_markdown())
```

## `OCR(...)` — the constructor

```python
OCR(model=None, backend="auto", *, lang=None, tier=None, models_dir=None,
    device=None, device_id=0, use_cls=False, mode="auto", replicas=1,
    fp16=True, allow_download=True, layout=False, tables=False,
    formulas=False, autorotate=False, verbose=False, keep_image=True)
```

**Model selection** — explicit `model=` wins; else `lang` (+ `tier`); else
`tier`; else the default `tiny`. `lang` picks a script recognizer (`korean`,
`arabic`, `thai`, `greek`, `eslav`) or, for Latin/CJK, selects the PP-OCRv6
tier. Models auto-download per tier with SHA256 verification (`~6` MB for
tiny) into the user cache (`~/.cache/turboocr` on Linux,
`~/Library/Caches/turboocr` on macOS); `models_dir=` points at your own
directory, `allow_download=False` forbids fetching. `tiny` omits Japanese
kana — use `small`/`medium` for Japanese
([model selection](../models/selection.md)).

| Parameter | Meaning |
|---|---|
| `backend` | Which silicon runs inference — see the table below |
| `device` | Vendor device *name*: for OpenVINO `CPU` \| `GPU` \| `NPU` (lands in `OV_DEVICE`) |
| `device_id` | Device *ordinal* for CUDA / ROCm / DirectML |
| `use_cls` | Run the 0°/180° line classifier on every line |
| `mode` | `"native"`/`"ultra"` = the vendor graph engine (TensorRT / MPSGraph / OpenVINO blob — fastest, needs a one-time build); `"onnx"`/`"fast"` = the ONNX model on that vendor's ORT provider, no graph build; `"auto"` takes native when its artefact exists. Resolved value: `info()["mode"]` |
| `replicas` | Independent native pipelines behind a checkout queue — one pipeline is single-flight, so this is where concurrency comes from. Each replica holds its own model copy in memory. `replicas=3` measured ~2.4× one replica (94% of the server's multi-replica throughput) on Apple silicon |
| `layout` / `tables` / `formulas` | Load the optional stage models at construction (tables/formulas imply layout). Per-call opt-in still applies at `read()` |
| `autorotate` | Load the page-orientation model (0/90/180/270) |
| `keep_image` | Keep each page's raster on the result (needed for `save_overlay` / `save_searchable_pdf`); pass `False` for large PDFs — a raster is ~6 MB per page at 150 DPI |

## Backends

`backend=` uses the same seam as the server's `--backend`. What each value
does depends on which engine wheel is installed — `turboocr doctor` reports
what the installed wheel carries.

| `backend=` | Runs | Notes |
|---|---|---|
| `"auto"` *(default)* | The installed wheel's best default | On the NVIDIA wheels this resolves to **`"turbo"`** — the first run builds and caches a TensorRT engine (one-time; `TRT_ENGINE_CACHE`, default `~/.cache/turbo-ocr`). Elsewhere: the CPU path |
| `"cpu"` | ONNX Runtime, MLAS | Works on every wheel |
| `"cuda"` | ORT CUDA execution provider | NVIDIA wheels; instant start, no engine build |
| `"turbo"` (`"tensorrt"`, `"trt"`) | Native TensorRT engine | NVIDIA wheels; peak throughput, one-time cached engine build |
| `"openvino"` (`"ov"`, `"intel"`) | **Native OpenVINO engine** on the openvino wheel; the ORT OpenVINO EP on builds that carry it | `device=` picks `CPU`/`GPU`/`NPU`; the OpenVINO runtime arrives as the wheel's own pip dependency, found automatically |
| `"apple"` (`"coreml"`) | CoreML EP opt-in on macOS | The macOS default is CPU — measured faster than the CoreML EP on these models |
| `"rocm"` / `"migraphx"` | ORT ROCm / MIGraphX EPs | rocm wheel (not yet hardware-verified) |

Asking for a backend the installed wheel cannot run raises
`BackendUnavailable` naming the wheel that can.

## Reading

```python
page = ocr.read(image, *, drop_score=0.5, rotate=0, layout=None,
                reading_order=False, tables=None, formulas=None,
                autorotate=None, text=True, keep_image=None)
```

`image` is a path, bytes, NumPy array, or PIL image. `layout=True` adds
layout regions; `tables=True` / `formulas=True` return HTML / LaTeX regions
(the matching `OCR(...)` flag must have loaded the model); `text=False` is a
layout-only run — validated by the same shared option gate as the HTTP
`?text=0`, so unsupported combinations raise the exact server error message.
`drop_score` below the engine's 0.5 floor is rejected rather than silently
ignored.

```python
doc = ocr.read_batch(images, *, batch_size=8, progress=None, ...)
```

Images go through the native whole-batch submission in groups of
`batch_size` (the server's `/ocr/batch` chunking), so the detector sees a
real batch. Requests needing per-image stages (layout, tables, formulas,
autorotate) fall back to one-at-a-time. `progress=True` logs to stderr, or
pass a callable `progress(done, total)`.

```python
doc = ocr.read_pdf(pdf, *, dpi=150, pages=None, max_pages=None, ...)
```

Renders with PDFium (the `pdf` extra: `pip install "turboocr[cpu,pdf]"`) and
OCRs each page. Pass `keep_image=False` for long documents.

## Results

- **`PageResult`** — `lines` (iterable; the page object itself iterates,
  indexes and `len()`s as its lines), `text` (lines joined in reading
  order), `width`/`height`, `page` (1-based for PDF pages), `orientation`
  (applied rotation when autorotate ran), `layout` (`LayoutBox`: label,
  confidence, quad), `tables` (`TableRegion`: `html`, score, quad),
  `formulas` (`FormulaRegion`: `latex`, score, quad), `warnings`,
  `filter(min_confidence=..., contains=...)`, `to_dict()`,
  `save_overlay(path)`, `save_searchable_pdf(path)`.
- **`TextLine`** — `text`, `confidence`, `box` (four corner points in
  original-image pixel coordinates).
- **`DocumentResult`** — `pages` (iterable), `text`,
  `to_markdown(structured=...)`, `to_dict()`, `source`.

## CLI

The engine wheel installs a `turboocr` command:

```bash
turboocr doctor           # detect hardware, name the right wheel + install line
turboocr models           # list available models/tiers
turboocr ocr page.png     # OCR images
turboocr pdf report.pdf   # OCR a PDF
turboocr version
```

## Errors

All exceptions derive from `TurboOCRError`: `BackendUnavailable` (this build
cannot run the requested backend — the message names the wheel that can),
`ModelLoadError`, `NativeExtensionMissing` (the native extension failed to
import; on the NVIDIA/OpenVINO wheels the message includes the exact
`pip install` line for the missing vendor runtime).

## Threading and processes

One `OCR` object is safe to share across threads: the replica pool serializes
access to each native pipeline while the GIL is released during inference, so
`replicas=N` is the way to scale, not user-side threading. Construction of
*different* backends is serialized internally (the engine reads process-global
environment at construction).

→ Repo-level docs: [`python/README.md`](https://github.com/aiptimizer/TurboOCR/blob/main/python/README.md) ·
[`python/DESIGN.md`](https://github.com/aiptimizer/TurboOCR/blob/main/python/DESIGN.md)
