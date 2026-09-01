# Python library

The Python package wraps the **same C++ pipeline the server runs** — a
nanobind extension over `UnifiedOcrPipeline`, not a reimplementation. The GIL
is released during inference, output is identical to the server's, and
concurrency comes from a built-in replica pool.

> **v4.0.0-alpha — `[cpu]`, `[apple]` and `[openvino]` install from PyPI
> today:** `pip install --pre "turboocr[apple]"` on Apple silicon (Metal GPU +
> Neural Engine, native mode out of the box), `"turboocr[cpu]"` elsewhere (the
> `--pre` is required — it is a pre-release). The NVIDIA
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
doc = ocr.read_pdf("report.pdf", dpi=150)     # PDF support is built in
print(doc.to_markdown())
```

## `OCR(...)` — the constructor

```python
OCR(model=None, backend="auto", *, lang=None, tier=None, models_dir=None,
    device=None, device_id=0, use_cls=False, mode="auto", replicas=1,
    fp16=True, allow_download=True, layout=False, tables=False,
    formulas=False, autorotate=False, verbose=False, keep_image=None)
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
| `layout` / `tables` / `formulas` | **Load** the optional stage models and make those requests legal. They do **not** run unless a call asks — see *Load is not Run* below. tables/formulas imply layout |
| `autorotate` | Load the page-orientation model (0/90/180/270). Applies to `read()` **and** the PDF paths: each rendered page is detected, rotated upright before OCR, and the angle lands in `PageResult.orientation`. Deliberately **opt-in**: measured on the fixture corpus, the classifier false-fires (180°) on 2 of 19 upright images (a menu photo and a multi-language page) even through its confidence gate — rotating a correct page is worse than leaving a rotated one — and costs roughly +3 ms/page (cpu tiny) to +11 ms/page (apple tiny). Scanned-document corpora measure clean (0 false positives on rendered PDF pages) |
| `keep_image` | Keep each page's raster on the result (needed for `save_overlay` / `save_searchable_pdf` / `draw`). Default is per path: `read()` keeps it, `read_pdf`/`read_batch` **drop** it (a raster is ~6 MB per page at 150 DPI — long documents retained GBs). Set it here to override both, or per call |

### Load is not Run

**The constructor decides which models are in memory and which requests are
legal. The call decides which output stages run.**

```python
plain  = turboocr.OCR()
loaded = turboocr.OCR(layout=True, tables=True, formulas=True)

assert plain.read(img).to_dict() == loaded.read(img).to_dict()   # both ~61 ms
loaded.read(img, tables=True)                                    # ~434 ms, because you asked
plain.read(img, tables=True)                                     # ValueError: model not loaded
```

Constructing an engine with more models never changes what a call returns — only
what a call is *allowed* to ask for. This matches the HTTP and gRPC surfaces,
which have always worked this way (a request without `?layout=1` does not get
layout, however many models the server has resident).

Two consequences worth knowing:

* `read(img, layout=False, tables=True)` raises `ValueError`. Tables and formulas
  are recognized *inside* layout regions, so asking for them with layout off is a
  contradiction rather than a preference.
* **`autorotate` is the one exception, deliberately.** It is input preparation —
  it rotates the pixels every other stage then sees — not an output stage, and it
  never reaches the shared request gate. It stays inherited from the constructor,
  so `OCR(autorotate=True)` keeps straightening pages without a per-call flag.
  `page.stages` records whether it ran.

Stage costs, so the trade is visible at the call site (M3 Max, apple backend,
tiny tier, a text-dense letter, medians of 5):

| request | time |
|---|---|
| `read(img)` — text only | **61 ms** |
| `read(img, layout=True)` | 329 ms |
| `read(img, layout=True, tables=True, formulas=True)` | 418 ms |

Layout is ~82% of a full parse: a fixed 800×800 pass whose cost does not shrink
with page size, and on Apple it is not overlapped with recognition. Skipping it is
a **5.4× speedup**, which is why it must be asked for rather than inherited.

## Backends

`backend=` uses the same seam as the server's `--backend`. What each value
does depends on which engine wheel is installed — `turboocr doctor` reports
what the installed wheel carries.

| `backend=` | Runs | Notes |
|---|---|---|
| `"auto"` *(default)* | The installed wheel's best default | On the NVIDIA wheels this resolves to **`"tensorrt"`** — the first run builds and caches a TensorRT engine (one-time; `TRT_ENGINE_CACHE`, default `~/.cache/turbo-ocr`). Elsewhere: the CPU path |
| `"cpu"` | ONNX Runtime, MLAS | Works on every wheel |
| `"cuda"` | ORT CUDA execution provider | NVIDIA wheels; instant start, no engine build |
| `"tensorrt"` (`"trt"`, `"nvidia"`; legacy `"turbo"`) | Native TensorRT engine | NVIDIA wheels; peak throughput, one-time cached engine build. On wheels without the nvidia engine it raises `BackendUnavailable` (it used to silently run on CPU) |
| `"openvino"` (`"ov"`) | **Native OpenVINO engine** on the openvino wheel; the ORT OpenVINO EP on builds that carry it | `device=` picks `CPU`/`GPU`/`NPU`; the OpenVINO runtime arrives as the wheel's own pip dependency, found automatically |
| `"intel"` | The native intel engine ONLY | Unlike `"openvino"` it has no EP fallback: on wheels without the engine it raises `BackendUnavailable` naming turboocr-engine-openvino |
| `"apple"` | The Apple backend, in one of two modes — see below | macOS builds of the cpu wheel only |
| `"amd"` | Native MIGraphX engine | rocm wheel. Hardware-verified on an MI300X (all goldens, cross-backend conformance, the FUNSD accuracy gate). Layout runs the shared host stage (MIGraphX cannot parse that model — see `src/backends/amd/BRINGUP.md`) |
| `"rocm"` / `"migraphx"` | ORT ROCm / MIGraphX EPs | Needs an onnxruntime built with the MIGraphX EP (none is published officially; a working source-build recipe is in `src/backends/amd/BRINGUP.md`). Not yet hardware-verified |

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

`image` is a path, bytes, NumPy array, or PIL image. **A NumPy array is read as BGR** (OpenCV's order), everything else as RGB — so `read(np.array(Image.open(p)))` silently OCRs channel-swapped colour and gives different text from `read(p)`. Pass the PIL image itself, or `cv2.imread(p)`, or swap with `arr[..., ::-1]`. `layout=True` adds
layout regions; `tables=True` / `formulas=True` return HTML / LaTeX regions
(the matching `OCR(...)` flag must have loaded the model);
`reading_order=True` needs the layout model too (`OCR(layout=True)` — the
shared gate rejects it otherwise); `text=False` is a
layout-only run — validated by the same shared option gate as the HTTP
`?text=0`, so unsupported combinations raise the exact server error message.
`drop_score` below the engine's 0.5 floor is rejected rather than silently
ignored.

```python
doc = ocr.read_batch(images, *, layout=None, tables=None, formulas=None,
                     reading_order=False, batch_size=None, progress=None,
                     on_error="raise", keep_image=None, ...)
```

Stage flags work exactly as on `read()` — they are per-call requests, not inherited from the constructor (*Load is not Run*), so `read_batch(images, tables=True)` and `read_pdf(pdf, layout=True)` are how those paths ask for structure. Images go through the native whole-batch submission in groups of
`batch_size` (the server's `/ocr/batch` chunking), so the detector sees a
real batch. Requests needing per-image stages (layout, tables, formulas,
reading_order, autorotate) fall back to one-at-a-time. `progress=True` logs to stderr, or
pass a callable `progress(done, total)`. `on_error="skip"` contains a
failing image to its own page — an empty `PageResult` with a
`page_failed: ...` warning — instead of aborting the batch; the default
`"raise"` propagates the first failure and cancels the images not yet
started.

```python
doc = ocr.read_pdf(pdf, *, dpi=150, pages=None, max_pages=None,
                   mode="ocr", layout=None, tables=None, formulas=None,
                   reading_order=False, on_error="raise", autorotate=None,
                   keep_image=None, password=None, ...)
```

`mode` picks how each page's text is obtained, and is typed as a
`Literal["ocr", "auto", "text"]` so editors offer the three values inline.

**`"ocr"` (default)** — render every page and OCR it, ignoring any embedded
text layer. This is an OCR engine: unless you say otherwise, every character
in the result came from the recognizer, on every page.

**`"auto"`** — per page, the embedded text layer where the page has one AND a
quality gate trusts it, OCR for everything else. On born-digital PDFs this is
roughly **10x faster and more accurate** (no recognizer, so no misreads —
measured on a 2-page letter: 4.8 ms vs 58.5 ms, with OCR misreading the
letterhead `UNIA` as `UN1A`). Two things to weigh before opting in: a scan
whose text layer came from earlier, possibly worse, OCR software passes the
gate and is served as-is; and text a PDF holds only as an **image** (a logo,
a pasted screenshot) is invisible to the layer, so those lines are missing
from the result. `line.source` is `"pdf"` for layer-sourced lines and `""`
for OCR'd ones, so the two are always distinguishable.
The gate shares the
server's structure (garbled/`U+FFFD`-heavy, control-character-ridden, and
**thin** layers are refused — a Bates stamp or fax header on a scan must
not hijack the page) with a stricter trust threshold: pages
under ~50 visible layer chars simply OCR, where the server accepts 10.
`/Rotate`d pages serve their layers like any other (the rotation transform
is ink-verified; boxes land where the render puts the glyphs). Throughput
is density-dependent: ~2700 pages/s on sparse digital pages, ~260 on
typical body text, ~100 on very dense 7k-char pages. Engines built with
layout/tables/formulas still run those **structure stages** on text-layer
pages (the page renders for the structure pass). `reading_order` is never
computed on PDF pages — neither text-layer nor OCR'd (the PDF entry points
expose no reading_order parameter today).

**`"text"`** — the layer only, no OCR and NO gate (close to the server's
`geometric` mode, which does apply its gate; the server's `auto_verified`
mode is currently aliased to `auto` there and has no Python spelling).
Because PDFium is globally single-threaded, `replicas` and async buy nothing
in this mode — the speed comes from skipping rasterization and OCR. Since it
never renders and never runs a model, combining it with `layout`/`tables`/
`formulas`/`reading_order` raises `ValueError` instead of returning empty
lists — `mode="auto"` is the one that serves the text layer *and* runs
structure stages on the rendered raster.

`pdf_to_searchable()` always renders regardless of `mode` (its output embeds
the page rasters). The HTTP server's `/ocr/pdf` also defaults to `ocr`, so
library and service agree.
Renders with PDFium — built in, no extra needed (pypdfium2 and reportlab
ship with the engine wheel since 4.0.0a6) — and
OCRs each page. Pages fan out across the replica pool: with
`OCR(replicas=3)`, a 24-page scan measured **2.41×** faster than the
sequential read (24.5 → 59 pages/s on Apple silicon, tiny tier),
byte-identical output — results are assembled strictly in page order, and at
most `replicas + 1` page rasters are in flight, so memory stays bounded on
large documents. `replicas=1` is exactly the sequential read.
`pdf_to_searchable(...)` uses the same fan-out while still writing pages in
order. Page rasters are **dropped by default** on the PDF paths (since
4.0.0a6) — pass `keep_image=True` when you need `doc.save_searchable_pdf()`
or `draw()` afterwards (`pdf_to_searchable()` keeps them automatically).
`on_error="skip"` turns a failing page — OCR failures AND page
render/extract failures — into an empty result carrying a
`page_failed: ...` warning instead of aborting the document;
`autorotate=True` (or engine-level `OCR(autorotate=True)`) straightens
rotated scans per page. Honest scaling note: the
fan-out pays on **accelerator** backends (Apple measured 2.41×, CUDA
similar), where one replica underuses the device; on the plain CPU backend a
single replica already spreads across the cores, so extra replicas buy only
~1.2× there (measured on a 20-core Linux box).

`password=` opens an encrypted PDF (user or owner password) — accepted by
`read_pdf`, `read_pdf_stream`, and `pdf_to_searchable`. Input guards: a PDF
handed to `read()` (or an image handed to `read_pdf()`) is refused with a
pointer to the right method; a **multi-page TIFF** is refused with its page
count instead of silently decoding only page 1; and a PNG/JPEG header
claiming absurd dimensions fails fast under the `TURBO_MAX_IMAGE_MP` ceiling
(default 96 MP) instead of OOMing.

```python
for page in ocr.read_pdf_stream(pdf, ordered=False):   # generator of PageResult
    handle(page)          # each page as soon as it is ready

async for page in ocr.aread_pdf_stream(pdf):           # async twin
    await handle(page)
```

`read_pdf_stream` is the streaming form — `read_pdf` is exactly this drained
into a `DocumentResult`. It yields each page as soon as it is ready (measured:
first page after 57 ms where the full 24-page document takes ~400 ms), with
the same replica fan-out and bounded memory. `ordered=True` (default) yields
strictly in page order; `ordered=False` yields in completion order — no
finished page waits on a slower earlier one; reassemble by `PageResult.page`.
Breaking out of either loop cancels the queued pages. Concurrency: all of
an engine's streams share ONE pool of `replicas` page-worker threads, so
any number of concurrent documents — gathered, interleaved
(`zip(stream_a, stream_b)`), or nested inside each other's loops, sync or
async — make progress without stacking threads (workers never wait on
consumers, so there is no deadlock class). Each OPEN stream still holds its
own bounded look-ahead window of at most `replicas + 1` rendered pages, so
raster memory scales with the streams you hold open concurrently — close
or exhaust what you are done with. `mode="text"` streams use no workers,
and each async stream pumps on its own dedicated thread (never asyncio's
shared executor).

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
detector real batches. A single `aread_pdf` already uses every replica (the
pages fan out inside the call), so there is no need to split a document
yourself to get concurrency.

**One-shots and introspection.** For a single call, the module-level
`turboocr_engine.read(image, model=..., backend=...)` and
`read_pdf(...)` reuse one cached default engine — `layout=True` /
`autorotate=True` there build (and cache) an engine with those capabilities
rather than being silently ignored. `ocr.info()` reports what a constructed
engine actually resolved to — backend, engine, `mode` (`native` vs `onnx`),
model paths, capabilities. `ocr.close()` releases the native pipelines. The
Python twins of the CLI's doctor are `turboocr_engine.doctor()` (prints the
panel and returns a dict — `doctor()["native_backends"]` lists the seam
engines such as `apple`) and `available_backends()` (ORT execution
providers only; it does NOT list seam backends).

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
| `confidence` | Recognition confidence, 0–1 (1.0 for text-layer lines) |
| `box` | The four corner points it was read from, in original-image pixel coordinates |
| `bbox` | Axis-aligned `(x0, y0, x1, y1)` over the corners |
| `source` | `""` for OCR, `"pdf"` for a PDF's embedded text layer |
| `id` / `layout_id` | Reading-order index / owning layout region (−1 when not requested) |
| `crop(image)` | The rectified pixel strip of this line from the source image |

### `PageResult` — one image or PDF page

The page behaves as a sequence of its lines: `for line in page`, `page[0]`,
`len(page)`.

| Attribute / method | Meaning |
|---|---|
| `lines` | The `TextLine`s, in reading order |
| `text` | All lines joined with newlines |
| `width`, `height` | Source image size in pixels |
| `page` | 1-based page number for PDF pages; `None` for standalone images |
| `dpi` | Render DPI for PDF pages (serialized, so a restored page still sizes its searchable PDF correctly) |
| `orientation` | Rotation applied before OCR (0/90/180/270) when autorotate ran |
| `layout` | `LayoutBox` regions (`label`, `confidence`, `box`, `id`, `parent_id` for nesting) — with `layout=True`. NOTE: under the umbrella package, top-level `turboocr.LayoutBox` is the HTTP CLIENT's model; the engine's dataclass lives at `turboocr.engine.LayoutBox` |
| `tables` | `TableRegion`s (`html`, `score`, `box`; `confidence` aliases `score`) — with `tables=True` |
| `formulas` | `FormulaRegion`s (`latex`, `score`, `box`; `confidence` aliases `score`) — with `formulas=True` |
| `reading_order` | Engine reading-order indices (with `read(reading_order=True)`; empty on PDF pages) |
| `image` | The source raster when kept (see `keep_image`); not serialized |
| `stages` | Which pipeline stages actually **ran** for this page (`text`, `layout`, `reading_order`, `tables`, `formulas`, `autorotate`) — recorded, not inferred. An empty `layout` list is ambiguous between "never ran" and "ran and found nothing" (a blank scan legitimately yields zero regions), so the record is kept explicitly and serialized. `filter()` drops `reading_order` from both the list and the record |
| `warnings` | Degradation notes (e.g. recognition produced boxes but no text; `page_failed: ...` marks a page contained by `on_error="skip"`; `no_text_layer:` marks an empty `mode="text"` page) |
| `filter(min_confidence=…, contains=…, predicate=…)` | A new `PageResult` keeping only matching lines (page context carried over) |
| `to_dict()` | JSON-shaped dict, same keys as the server's response |
| `to_pandas()` | The text LINES as one DataFrame (`[pandas]` extra) |
| `tables_to_pandas()` | The RECOGNIZED TABLES, one DataFrame each (provenance in `df.attrs`); per region: `TableRegion.to_pandas()` |
| `draw(layout=True)` / `save_overlay(path, layout=True)` | Overlay the layout regions (stable per-label colors + captions); `lines=False` for layout-only |
| `to_hocr()` | hOCR markup for this page — **line-granular** (`ocr_page`/`ocr_line`; the engine recognizes whole lines, so no `ocrx_word` spans are fabricated) |
| `to_markdown(structured=…)` / `to_json()` / `to_tsv()` | Markdown (layout-aware when regions exist) / JSON string / per-page TSV (no page column — the document form adds it) |
| `save_overlay(path)` | The image with boxes drawn on it (needs the raster: `read()` keeps it by default, the PDF/batch paths need `keep_image=True`) |
| `save_searchable_pdf(path)` | Searchable PDF. A page WITH its raster becomes image + invisible text (the facsimile deliverable); a page WITHOUT one (text-layer pages under `mode="auto"`/`"text"`, or `keep_image=False` reads) becomes a VISIBLE re-typeset text-only page. For guaranteed facsimile output use `pdf_to_searchable()` or read with `keep_image=True, mode="ocr"` |
| `to_pdf_bytes()` | The same searchable PDF as bytes (e.g. for a web response) |
| `to_html()` | The page as structured HTML (layout-aware when layout ran) |
| `from_dict(d)` / `from_json(s)` | Rebuild a `PageResult` from its serialized form (classmethods). Accepts BOTH key spellings: this library's `box`/`label`/`score` and the HTTP server's `bounding_box`/`class`/`confidence`, so a server response parses directly. Text, boxes, layout, tables, formulas, page numbers and `dpi` round-trip exactly; **confidences are serialized to 4 decimal places** (matching the server's wire format), so a restored score can differ from the in-memory one by <1e-4 |

### `DocumentResult` — a PDF or an image batch

The document iterates as its pages.

| Attribute / method | Meaning |
|---|---|
| `pages` | One `PageResult` per page |
| `text` | Whole-document text |
| `source` | The path it was read from (`read_pdf`; empty for `read_batch` and byte inputs) |
| `to_markdown(structured=…)` | Markdown export |
| `to_dict()` | JSON-shaped dict |
| `to_pandas()` | The lines of all pages as one DataFrame, with a `page` column |
| `tables_to_pandas()` | Every table in the document as its own DataFrame (`attrs["page"]` = provenance) |
| `to_hocr()` | A single multi-page hOCR document |
| `to_html()` | The whole document as HTML (`full=True` wraps a complete page) |
| `save_searchable_pdf(path)` / `to_pdf_bytes()` | Searchable PDF from the page rasters (`keep_image=True` on PDF/batch reads) |
| `from_dict(d)` / `from_json(s)` | Rebuild a `DocumentResult` from its serialized form (classmethods) |

## CLI

The engine wheel installs a `turboocr` command:

```bash
turboocr doctor           # detect hardware, name the right wheel + install line
turboocr models           # list available models/tiers
turboocr warmup           # pay model download + engine compilation once, now
turboocr ocr page.png     # OCR images
turboocr pdf report.pdf   # OCR a PDF
turboocr info             # build the engine, print its resolved config (JSON)
turboocr version
```

`warmup` exists because the GPU engines compile models per machine on first
use (TensorRT engines, MIGraphX programs, CoreML specialization) — minutes on
a cold box. Run it once at install or image-build time (with the same
`--backend`/`--model`/stage flags you will serve with) and the first real
document is fast; once the caches are warm it costs seconds and is safe to
re-run.

Shared engine flags on `ocr`/`pdf`/`info`: `--backend`, `--model`/`--lang`/
`--tier`, `--replicas N` (parallel pages/images), `--layout`, `--tables`,
`--formulas`, `--autorotate`, `--cls`. The `pdf` subcommand adds
`--mode ocr|auto|text` (default `ocr` — every page is OCR'd; `auto` uses a trusted embedded text layer where present, `text` = layer only, never OCR),
`--searchable -o out.pdf`, `--dpi`, `--pages`, `--max-pages`;
`ocr` adds `--overlay boxes.png` (with `--layout`, regions are drawn too)
and `--on-error skip` (note unreadable images on stderr, keep going, exit
1 if any were skipped);
`pdf` also takes `--password` for encrypted PDFs. `-f hocr` emits one
complete hOCR document. Output shapes are STABLE regardless of how many
files a glob matched: `-f json` is always `{"pages": [...]}` (per-image
`source` on the `ocr` subcommand), `-f tsv` always carries a leading
`page` column, and `-f hocr` is always one parseable document — same
envelopes as the `pdf` subcommand.

## Errors

The `TurboOCRError` family covers ENGINE failures: `BackendUnavailable`
(this build cannot run the requested backend — the message names the wheel
that can, including seam-only names like `intel`/`amd`/`turbo` on wheels
without that engine), `ModelLoadError` (model or stage assets failed to
load or download; also raised at construction when a REQUESTED stage —
`OCR(tables=True)` — cannot come up, instead of silently returning zero
tables forever), and `NativeExtensionMissing` (the extension failed to
import; on the NVIDIA/OpenVINO wheels the message includes the exact
`pip install` line for the missing vendor runtime).

INPUT problems deliberately raise the standard types instead: bad
arguments and unreadable/hostile inputs are `ValueError` (wrong mode
strings, the drop_score floor, PDFs handed to `read()`, multi-page TIFFs,
the pixel ceiling, encrypted PDFs with a wrong/missing password), a
missing file is `FileNotFoundError`, and using a closed engine is
`RuntimeError`. Catch `(TurboOCRError, ValueError)` for "anything this
library refuses".

## Logging

The library is **silent by default** on both stdout and stderr: every
configuration measured — `cpu` and `apple`, with `layout`, `tables`,
`formulas`, `autorotate`, `use_cls` individually and all together, reading
images and PDFs in every mode — emits **zero** unrequested lines.
`OCR(verbose=True)` or `LOG_LEVEL=info` brings the diagnostics back (they are
silenced, not deleted); `LOG_LEVEL` accepts `debug|info|warn|error` and
defaults to `warn`.

One exception is outside the library's control: with **`replicas>1` on the
apple backend**, macOS emits a burst of `Context leak detected, CoreAnalytics
returned false` on stderr when CoreML sessions are created. It comes from
Apple's CoreAnalytics, not from TurboOCR; it is asynchronous, so a scoped
redirect cannot reliably catch it, and `OS_ACTIVITY_MODE=disable` has no
effect. It is bounded — construction-time only, it does not scale with pages
or documents — and `replicas=1` never triggers it.

### Accelerator degradation

The layout stage treats CoreML as an **accelerator, not a precondition**. If the
accelerated session fails to build — which happens transiently under GPU/ANE
contention, since the model compiles as many independent CoreML partitions —
the stage rebuilds once on the CPU provider instead of disappearing. Layout is
then slower but *available*, its output is equivalent (the CPU provider is what
every non-Apple backend runs by design), and the fallback is announced at WARN
with the original error. It is also queryable: `info()["layout_coreml_dropped"]`
is `True` once any layout session in the process has fallen back (`doctor`
cannot show this — it runs in a fresh process, where the latch is always
clear).

Two deliberate properties:

* **Out-of-memory is never retried.** A failed allocation is not contention, and
  the CPU build needs more host memory than the accelerated one, so retrying
  would turn an honest load failure into a success that dies later.
* **The drop is process-wide and one-way.** Replicas each build their own layout
  session, so once any of them falls back, every later load in that process
  skips CoreML too — otherwise a pool could answer the same page differently
  depending on which replica served it. Re-acquiring the accelerator means a new
  process.

## Threading and processes

One `OCR` object is safe to share across threads: the replica pool serializes
access to each native pipeline while the GIL is released during inference, so
`replicas=N` is the way to scale, not user-side threading. Construction of
*different* backends is serialized internally (the engine reads process-global
environment at construction).

→ Repo-level docs: [`python/README.md`](https://github.com/aiptimizer/TurboOCR/blob/main/python/README.md) ·
[`python/DESIGN.md`](https://github.com/aiptimizer/TurboOCR/blob/main/python/DESIGN.md)
