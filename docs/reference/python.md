# Python library

The Python package wraps the **same C++ pipeline the server runs** — a
nanobind extension over `UnifiedOcrPipeline`, not a reimplementation. The GIL
is released during inference, output is identical to the server's, and
concurrency comes from a built-in replica pool.

> **v4.0.0-alpha — `[cpu]` and `[openvino]` install from PyPI today:**
> `pip install --pre "turboocr[cpu]"` (the `--pre` is required — it is a
> pre-release; the macOS cpu wheel carries the Apple backend). The NVIDIA
> `[cuda12]` / `[cuda13]` extras resolve once PyPI approves those wheels'
> file-size requests — until then build them from the checkout
> ([Install](../getting-started/install.md)). One engine wheel per
> environment, mutually exclusive; `turboocr doctor` names the right one.

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
`arabic`, `thai`, `greek`, `eslav` — ISO aliases like `ko`, `ar`, `th`, `el`,
`ru` work too) or, for Latin/CJK, selects the PP-OCRv6 tier. `turboocr
models` (or `turboocr_engine.list_models()`) prints every name. Models
auto-download per tier with SHA256 verification (`~6` MB for tiny) into the
user cache (`~/.cache/turboocr` on Linux, `~/Library/Caches/turboocr` on
macOS); `models_dir=` points at your own directory, `allow_download=False`
forbids fetching. `tiny` omits Japanese kana — use `small`/`medium` for
Japanese ([model selection](../models/selection.md)).

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
| `"apple"` | The Apple backend, in one of two modes — see below | macOS builds of the cpu wheel only |
| `"rocm"` / `"migraphx"` | ORT ROCm / MIGraphX EPs | rocm wheel (not yet hardware-verified) |

Asking for a backend the installed wheel cannot run raises
`BackendUnavailable` naming the wheel that can.

### Apple: one backend, two modes

Apple silicon has three engines — CPU, GPU, and the Neural Engine (ANE, which
is reachable only through CoreML, so there is no `backend="ane"`; it is a
lane, not an engine you select). `backend="apple"` picks the execution mode
by which artefacts are present (`info()["mode"]` reports the result):

| Mode | Runs on | When |
|---|---|---|
| `native` | Metal + MPSGraph on the **GPU**, narrow recognition buckets on the **ANE** in parallel (`TURBO_APPLE_ANE_MAXW`, default 800) — the measured ~5× configuration. Detection input is **dynamic**: the runtime specializes the compiled detector to each page's shape (the shared resize policy, snapped to a 128-px grid so near-equal sizes share one engine), paying a one-time ~50–350 ms compile per new shape and then running at full static-engine speed — the same cost model as a TensorRT engine cache. The live engine set is LRU-bounded (`TURBO_APPLE_DET_CANVAS_CACHE`, default 6), so memory stays fixed no matter how varied the corpus | The `apple_native_<tier>` export bundle is present. The wheel provisions it into the model cache automatically when the release asset is available; it can always be generated locally with `tools/modelgen/apple/export_apple_native.py` |
| `onnx` | The ONNX models on the CoreML execution provider (Apple's scheduler places ops on ANE/GPU/CPU) | No native bundle — the fallback |

The default `auto` on macOS stays on the CPU path (measured faster than the
CoreML-EP mode for these models); `backend="apple"` opts into the Apple
backend, and with the native bundle in place it is the fast path.

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

## Async

Every read method has an `async` twin — `aread`, `aread_batch`, `aread_pdf` —
with identical parameters and results:

```python
ocr = turboocr.OCR("small", backend="apple", replicas=3)

async def main():
    pages = await asyncio.gather(*[ocr.aread(img) for img in images])
```

The mechanics are deliberately transparent: each coroutine runs its sync twin
in a worker thread (`asyncio.to_thread`), and the parallelism is real because
the GIL is released during native inference and one `OCR` object is
thread-safe against its replica pool. Concurrency scales with `replicas` —
measured on Apple silicon: six gathered `aread` calls at `replicas=3` ran
**2.65×** faster than the same six serial reads, byte-identical output.
With `replicas=1` awaiting serializes exactly like the sync API. When the
image list is known up front, prefer `aread_batch` — the batch path feeds the
detector real batches.

**One-shots and introspection.** For a single call, the module-level
`turboocr_engine.read(image, model=..., backend=...)` and
`read_pdf(...)` reuse one cached default engine — `layout=True` /
`autorotate=True` there build (and cache) an engine with those capabilities
rather than being silently ignored. `ocr.info()` reports what a constructed
engine actually resolved to — backend, engine, `mode` (`native` vs `onnx`),
model paths, capabilities. `ocr.close()` releases the native pipelines. The
Python twins of the CLI's doctor are `turboocr_engine.doctor()` (prints the
panel) and `available_backends()`.

## Results

`read()` returns a **`PageResult`**; `read_batch()` and `read_pdf()` return a
**`DocumentResult`** holding one `PageResult` per page.

```python
page = ocr.read("invoice.png", layout=True)

for line in page.filter(min_confidence=0.9):     # a page iterates as its lines
    print(f"{line.confidence:.2f}  {line.text}")
for region in page.layout:
    print(region.label, region.box)

doc = ocr.read_pdf("report.pdf")
doc.to_markdown()                                # whole document as Markdown
doc.to_pandas()                                  # one DataFrame, `page` column
```

### `TextLine` — one recognized line

| Attribute | Meaning |
|---|---|
| `text` | The transcript |
| `confidence` | Recognition confidence, 0–1 |
| `box` | The four corner points it was read from, in original-image pixel coordinates |

### `PageResult` — one image or PDF page

The page behaves as a sequence of its lines: `for line in page`, `page[0]`,
`len(page)`.

| Attribute / method | Meaning |
|---|---|
| `lines` | The `TextLine`s, in reading order |
| `text` | All lines joined with newlines |
| `width`, `height` | Source image size in pixels |
| `page` | 1-based page number for PDF pages; `None` for standalone images |
| `orientation` | Rotation applied before OCR (0/90/180/270) when autorotate ran |
| `layout` | `LayoutBox` regions (`label`, `confidence`, `box`) — with `layout=True` |
| `tables` | `TableRegion`s (`html`, `score`, `box`) — with `tables=True` |
| `formulas` | `FormulaRegion`s (`latex`, `score`, `box`) — with `formulas=True` |
| `warnings` | Degradation notes (e.g. recognition produced boxes but no text) |
| `filter(min_confidence=…, contains=…, predicate=…)` | A new `PageResult` keeping only matching lines (page context carried over) |
| `to_dict()` | JSON-shaped dict, same keys as the server's response |
| `to_pandas()` | DataFrame of the lines (`[pandas]` extra) |
| `to_hocr()` | hOCR markup for this page |
| `save_overlay(path)` | The image with boxes drawn on it (needs `keep_image=True`, the default) |
| `save_searchable_pdf(path)` | Image page + invisible text layer as a PDF (needs `keep_image=True` and reportlab, from the `[pdf]` extra) |

### `DocumentResult` — a PDF or an image batch

The document iterates as its pages.

| Attribute / method | Meaning |
|---|---|
| `pages` | One `PageResult` per page |
| `text` | Whole-document text |
| `source` | The path it was read from |
| `to_markdown(structured=…)` | Markdown export |
| `to_dict()` | JSON-shaped dict |
| `to_pandas()` | One DataFrame across all pages, with a `page` column |
| `to_hocr()` | A single multi-page hOCR document |

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
