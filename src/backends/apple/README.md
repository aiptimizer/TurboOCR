# TurboOCR — Apple Backend (Metal + MPSGraph)

The device-resident Apple backend for the multi-backend rebuild. Unlike the
NVIDIA/AMD/Intel backends (structural on this Mac), this one is **real, proven,
and testable on the M3 Max**: it wraps the working, measured code from
`tools/probes/apple/mps_ocr.mm`, `tools/probes/apple/warp.metal`, and `mps_rec_build.h` behind the
shared `Backend` interfaces in `include/turbo_ocr/backend/`. The translator now
lives HERE (`src/backends/apple/engine/mps_rec_build.h`) because it is library code;
`tools/probes/apple/mps_rec_build.h` is a one-line forwarding header so the standalone
probes keep building with their `-Itools/probes/apple` recipe.

The design goal is the plan's core thesis: each vendor is a **device-resident
backend**, not a CPU pipeline with a swapped execution provider. Data stays on
the GPU end-to-end (`MTLBuffer` across warp → rec → argmax); only tiny token
indices cross to the host, at stage boundaries, as host POD types.

## File layout

One directory per concern — the same layout every vendor under `src/backends/`
uses. The rule each directory answers to is in
[`../README.md`](../README.md), and `python3 src/backends/layout_check.py`
enforces it. Sibling headers are included through the
vendor-rooted path off `-Isrc/backends` — `#import "apple/support/metal_common.h"`,
never a bare `"metal_common.h"`.

| File | Role | Interface implemented |
|------|------|-----------------------|
| `kernels_metal/shaders.metal` | Metal compute kernels: `warp_crops` (verbatim from `tools/probes/apple/warp.metal`), `resize_normalize`, `pack_bgr8_to_rgba`, `argmax`, `threshold_u8` | (device ops behind `IKernels`) |
| `kernels_metal/metal_kernels.{h,mm}` | `MetalKernels`: warp/resize/argmax/threshold native; `db_postprocess` host-fallback; `preprocess_region` TODO | `IKernels` |
| `support/metal_common.{h,mm}` | `MTLDevice`/library/`MPSGraphDevice` singletons; the `void*`↔`MTLBuffer` registry (unified-memory `.contents` pointers) | — |
| `support/apple_contention.{h,mm}` | contention counters + exit dump (`TURBO_APPLE_CONTENTION`) | — |
| `support/apple_profile.h` | env-gated per-stage wall-clock profiler (`TURBO_APPLE_PROFILE`) | — |
| `queue/metal_device_queue.{h,mm}` | `MetalDeviceQueue` / `MetalDeviceEvent`; the **one-command-buffer batch** (`begin/end_batch`) that is the residency lever | `DeviceQueue`, `DeviceEvent` |
| `memory/metal_allocator.{h,mm}` | `MetalAllocator` over `MTLResourceStorageModeShared` (unified memory ⇒ coherent memcpy H2D/D2H) | `IDeviceAllocator` |
| `memory/metal_image.{h,mm}` | `MetalImage` (owns an RGBA8 `MTLTexture` + BGR8 `MTLBuffer`); the texture registry that recovers a page's sampler source from an `ImageView` | `ImageView` (Metal) |
| `engine/mps_engine.{h,mm}` | `MpsEngine`: builds/compiles an `MPSGraphExecutable` per batch via `mps_rec_build.h`; `MTLBuffer`-backed `MPSGraphTensorData`; optional GPU argmax head | `IEngine` |
| `engine/mps_rec_build.h` | the MPSGraph translator for the rec export (shared with the `tools/probes/apple/mps_*.mm` probes) | — |
| `engine/ane_rec_engine.{h,mm}` | `AneRecEngine` + `AneBatchService`: the CoreML mlprogram rec head on the Neural Engine | `IEngine` |
| `stages/mps_stages.{h,mm}` | `MpsDetector` / `MpsRecognizer` / `MpsClassifier` / `MpsLayout` | `IDetector`/`IRecognizer`/`IClassifier`/`ILayout` |
| `backend/apple_backend.{h,mm}` | `AppleBackend`: factories + `load_stages()` + `make_image_decoder/make_orient_func` (NO make_infer_func — the ONE InferFunc is `pipeline::make_infer_func`) | `Backend` |
| `backend/apple_backend_registry.cpp` | one `BackendRegistrar`; registers "apple" (needs WHOLE_ARCHIVE) | — |

## Build (this Mac, today)

Requires Xcode Command Line Tools **plus the Metal toolchain** (`xcrun --find
metal`) and OpenCV 4 (`pkg-config opencv4`). Then:

The target is `turbo_ocr_backend_apple` in the root `CMakeLists.txt`, configured
with `-DTURBO_BACKENDS="cpu;apple"`. (There is no `src/backends/apple/build.sh`;
the standalone script this section used to name is gone.) What CMake runs, in
essence:

```sh
# shaders -> metallib (the Metal toolchain, exactly as tools/probes/apple/mps_ocr.mm does)
xcrun -sdk macosx metal   -O2 -c kernels_metal/shaders.metal -o build/shaders.air
xcrun -sdk macosx metallib     build/shaders.air -o build/turbo_apple.metallib

# each Obj-C++ TU (ARC), against both include trees + tools + OpenCV
clang++ -std=c++20 -fobjc-arc -O2 -Wall \
  -Iinclude -Isrc/backends $(pkg-config --cflags opencv4) \
  -c <file>.mm -o build/<file>.o

libtool -static -o build/libturbo_ocr_apple.a build/*.o
```

**Verified on this machine:** `shaders.metal` compiles+links to a `.metallib`;
all Obj-C++ TUs compile to object code cleanly under `-Wall` against the real
`backend/*.h` interfaces and the real host headers, producing
`turbo_apple.metallib` and `libturbo_ocr_backend_apple.a`.

Full server/bench linking additionally needs **turbo_ocr_common** (only three
host symbols are referenced: `detection::extract_boxes_from_bitmap`,
`recognition::ctc_greedy_decode`, `recognition::load_label_dict` —
`compute_perspective_inv` is header-inline), OpenCV, and the frameworks:

```
-L build -lturbo_ocr_apple  <turbo_ocr_common>  $(pkg-config --libs opencv4) \
  -framework Foundation -framework Metal \
  -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
```

At runtime, point `$TURBO_APPLE_METALLIB` at `turbo_apple.metallib` (or install
it next to the executable).

### Model artefacts

`MpsEngine::load(dir)` consumes the **export directory** that
`tools/modelgen/mps_export_rec.py` produces (`graph.json` + `weights.bin`, plus an ORT
`golden.bin` for the bit-accuracy check). Export det/rec/cls the same way:

```sh
python tools/modelgen/mps_export_rec.py models/rec.onnx <rec_export_dir> 48 320
python tools/modelgen/mps_export_rec.py models/det.onnx <det_export_dir> 640 640
# angle cls — models/cls.onnx's REAL input is [B,3,80,160] (PP-LCNet_x0_25
# text-line orientation), the same shape CpuPaddleCls uses:
python tools/modelgen/mps_export_rec.py models/cls.onnx ~/.apple_ocr_ml/exports/cls_b160 80 160
```

## How each interface is satisfied by the working code

- **`ImageView` (Metal)** — `MetalImage` owns an RGBA8 `MTLTexture` (hardware
  bilinear source for warp/resize) + a BGR8 Shared `MTLBuffer` (so `ImageView::
  data` is a real, resolvable Metal pointer). The RGBA8 B/R-swap upload is
  identical to `tools/probes/apple/mps_ocr.mm:63-69`.
- **`DeviceQueue` / one-submit batch** — `MetalDeviceQueue` wraps an
  `MTLCommandQueue`; `begin_batch()` opens one `MPSCommandBuffer` that every
  encoder appends to and `end_batch()` commits it once. That single command
  buffer spanning **warp → MPSGraph rec → argmax** is the residency guarantee the
  POC proved (`tools/probes/apple/mps_ocr.mm:151-154`, "FUSED one cmd buffer"). Events are
  `MTLSharedEvent`-based for true device-side cross-queue waits.
- **`IEngine`** — `MpsEngine` builds the graph through `mps_rec_build.h`'s
  `buildRecGraph` (~25 ops, bit-accurate to ORT), compiles an
  `MPSGraphExecutable` per batch (cached), and `encodeToCommandBuffer`s onto the
  queue with `MTLBuffer`-backed `MPSGraphTensorData` (zero-copy, device I/O).
  `caps()` = {io_space Metal, async, caller-owns-outputs, static shapes, graph
  transparent}. `enable_argmax_head()` appends `reductionArgMaximum` +
  `reductionMaximum` (`tools/probes/apple/mps_ocr.mm:119-120`) so only `[B,T]` indices/scores
  come back. Each executable is static-shape, but `load_shared()` re-specializes
  one parsed export to a new input H×W (shared graph dict + weights, ARC refs) —
  `MpsDetector` uses it to serve ANY page shape from one det export: per page it
  runs the shared `compute_det_resize` → `snap_det_canvas_grid` policy,
  letterboxes the content, and keeps an LRU-bounded cache of specialized engines
  (`TURBO_APPLE_DET_JIT`, `TURBO_APPLE_DET_CANVAS_CACHE`).
- **`IKernels`** — `MetalKernels`: `warp_crops` (the `tools/probes/apple/warp.metal` kernel),
  `resize_normalize`, `argmax`, `threshold` are **native**; `db_postprocess`
  (CCL + unclip — no MPS primitive) is a **host fallback** via
  `extract_boxes_from_bitmap`, reading the prob map through unified memory (a
  coherent read, not a PCIe D2H). `caps()` reports this per op.
- **Stages** — `MpsRecognizer` is the measured path: host homographies →
  `warp_crops` → rec + GPU argmax in **one command buffer** → host
  `ctc_greedy_decode`, mirroring `tools/probes/apple/mps_ocr.mm:140-161` exactly.
  `MpsDetector` runs the DB forward pass resident, then host DB post (bit-
  identical to `mps_ocr.mm:99-107`), returning original-coordinate boxes.
- **`Backend`** — `AppleBackend` wires the factories, `load_stages()` →
  `StageAvailability`, and the `server::InferFunc/ImageDecoder/OrientFunc` a
  unified `server_main` selects when the device is Apple.

## What runs on this Mac today vs TODO

**Runs + measured today** (`tools/probes/apple/mps_ocr.mm`, German receipt, 37 crops):

| stage | cost |
|-------|------|
| det GPU-exec | 0.42 ms |
| fused warp + rec + argmax GPU-exec (one cmd buffer) | 0.38 ms |
| **GPU-exec total** | **0.81 ms/img** |
| wall (median) | ~10 ms ⇒ **~100 img/s**, GPU utilization ~8 % |

The bottleneck is **CPU submit/sync overhead**, not GPU work — the GPU is idle
92 % of the wall. det/rec/cls all rebuild **bit-accurate vs ORT** through the
`mps_rec_build.h` translator.

**`MpsClassifier` — VALIDATED (was structural).** The reproducible check of
record is the generalized per-stage golden diff, `tests/cpp/backends/turbo_golden.cpp`
(also registered in ctest as `golden_apple_cls`):

```sh
DISABLE_COREML=1 ./build/turbo_golden --backend apple --ref cpu --stage cls \
    --images ~/compare-ocrs/funsd_cache --count 10
```

`DISABLE_COREML=1` keeps the CPU reference on plain ORT/MLAS. `--stage cls`
feeds the candidate the REFERENCE backend's boxes, so a disagreement is the
classifier's own, not detection leaking downstream.

| comparison | result | ctest tripwire |
|---|---|---|
| end-to-end **flip decision** agreement vs `CpuPaddleCls` | **98.81 %** (FUNSD pages 0-9, tiny) | ≥ 98 % |

Historical, from the now-deleted Apple-only harness `cls_golden_apple.mm`
(superseded by `turbo_golden`; the numbers below cannot be re-run and are kept
only as the provenance of the two bug fixes named next). It ran 2712 real FUNSD
line crops and reported: MPSGraph vs ORT on the **same** input tensor, max prob
delta **1.1e-5** (bit-accurate); Metal `warp_crops` vs `cv::warpPerspective`
crops, mean abs delta 0.17 (resampling); flip agreement **99.52 %** (59 vs 60
flips). An earlier 585-crop run of the same harness reported 4e-6 / 99.5 %
(13 vs 14 flips) — first pass, superseded by the 2712-crop figures. Only the
A-vs-C leg (end-to-end flip agreement) generalized; the Metal-internal
warp-vs-graph leg was dropped as a per-backend fork — see
`tests/cpp/backends/README.md`, "The one thing that was dropped, and why".

Two fixes were needed to get there: the export must use the model's real
**80x160** input (the header defaulted to the v4 48x192), and the decision rule
must be CpuPaddleCls's `s180 > s0 && s180 > 0.9` — the old code flipped on a
bare argmax, which disagrees with the CPU classifier on ~2.3 % of crops.

FUNSD-50 @ tiny through `UnifiedOcrPipeline`: **85.33 % → 85.85 % F1** with cls
enabled (7.6 → 7.2 img/s, ~5 % throughput cost). CPU parity reference: 85.31 %
without cls / 85.79 % with.

`MpsDetector`'s resident
`resize_normalize` preprocess should be golden-diffed against the host
`cv::resize` path (Metal bilinear vs `INTER_LINEAR`) before det is treated as
bit-locked; the recognizer is bit-exact regardless.

**TODO (MPSGraph-NATIVE versions not implemented — the capabilities
themselves ARE available through host fallbacks; see apple_backend.mm):**
1. `MpsLayout` — PP-DocLayoutV3 is **multi-IO** (image + im_shape +
   scale_factor); `mps_rec_build.h` handles one placeholder, so the MPSGraph
   build declines. **Layout still works**: apple_backend.mm falls back to the
   host ONNX stage (`HostLayoutOnDevice`) and sets the Layout capability bit.
2. `db_postprocess` on the GPU — a Metal union-find CCL + JFA-unclip to remove
   the one host round-trip in detection (`caps().db_postprocess` flips to true).
3. Local table/formula on MPSGraph — SLANeXt / PP-FormulaNet device builds.
   **Both modalities work today** through the host ORT recognizers wired in
   `make_table_recognizer`/`make_formula_recognizer` (VLM specs too).
4. `preprocess_region` (fused table/layout preproc) — stubbed.
5. Native image decode (vImage/VideoToolbox) — currently host `cv::imdecode` +
   resident upload.

## Top 3 things to do next for full residency / throughput

1. **Kill the submit/sync overhead → ~450+ img/s.** The GPU already finishes an
   image in 0.81 ms; the wall is dominated by per-op command-buffer create/
   commit/wait. Move to **one command buffer per image with full residency**
   (det + threshold + warp + rec + argmax in a single `MPSCommandBuffer`) and
   **pipeline throughput** across pool entries (N in-flight images, each its own
   `MetalDeviceQueue`, so submit/decode of image *i+1* overlaps GPU of image
   *i*). Unified memory means no staging copies to hide.
2. **Metal DB post-process** (union-find CCL + JFA-unclip) so detection needs no
   `queue.synchronize()` mid-image — the last host round-trip on the hot path —
   enabling the single-command-buffer full-image residency in (1).
3. **Multi-IO `MpsEngine` + `MpsLayout`**, unlocking layout → tables/formulas on
   device and closing the stage set (then port the fused table/layout
   `preprocess_region` kernels).

---

## In-process throughput ceiling — where the concurrency actually goes

Measured on an M3 Max (14 cores, 36 GB), FUNSD-50 tiny, replica-pool driver,
`TURBO_APPLE_REC_BUCKETS=320,480,800,1600`. Every figure below is from an
**interleaved paired A/B in one session** with cooldowns between runs — absolute
numbers drift up to ~40% under sustained load on this machine, so only paired
comparisons are meaningful. **F1 was 85.44% in every configuration listed**, so
these are purely throughput results.

### The ceiling is the Neural Engine's PER-PROCESS submission rate

| K (replicas, 1 process) | img/s | ANE rows/s | ANE predicts/s |
|---|---|---|---|
| 8  | ~98  | 4610 | ~500 |
| 16 | ~102 | 4790 | ~540 |
| 24 | ~106 | 4935 | ~550 |

Tripling in-flight work buys +8% because the ANE saturates at **~5000 rec rows/s
/ ~550 predicts/s per process**. At ~46 ANE rows per FUNSD page that is exactly
~107 img/s. Nothing else in the process is close to being a limiter:

* `resolve_buffer` / `mtl_pipeline` / `texture_for` global mutexes: **38 679
  `resolve_buffer` calls in a 7 s run blocked for 0.8 ms TOTAL**. Not a factor —
  carrying `id<MTLBuffer>` in `DeviceBuffer` would be a no-op for throughput.
* Per-page `newBufferWithLength` + `free`: 57 µs + 24 µs per page (~0.8%).
* The page texture pack on the process-global queue: real (11.9 ms of blocked
  time per page at K=24 vs 1.0 ms at K=8 — it degraded with concurrency), now
  fixed, but throughput-neutral because the ANE dominates.

### Levers that were tried and LOST (do not re-run these)

| change | K=24 img/s | why |
|---|---|---|
| baseline | **105** | W320 pinned batch 16, W480/W800 batch 8, 2 workers |
| `ANE_SHAPE_IDX=1` (all buckets one rung up) | 64 | a predict runs the FULL shape; cost scales with it |
| W480/W800 raised to batch 24 | 66 | same |
| 2 ms batch-fill linger | 40 | requests are ~9 rows vs a 16/8 shape — two never fit |
| `ANE_WORKERS=6` | 105 | latency 7.5 → 17.4 ms, rows/s flat |
| `ANE_WORKERS=6` + 3 distinct `.mlmodelc` | 105 | the limit is not per compiled model |
| `ANE_MAXW=480` (W800 rec → GPU) | 88 | the GPU cannot absorb it; it is a co-limiter |
| `ANE_MAXW=0` (all rec → GPU) | 61 | GPU-only reference point |
| pack on the caller's cb (kept) | 105 | correct, but neutral while the ANE binds |

### Why 3 processes beat 1

3 processes × 8 replicas reach **~7000 ANE rows/s summed** (2362 + 2569 + 2048)
against 4935 in one process — ~40% more, from the same silicon. Since worker
count, model-instance count and distinct compiled models all leave the
single-process number flat, the ~5000 rows/s ceiling lives in Apple's
**per-process CoreML/ANE client**, not in this code and not in the hardware.
Closing that gap in-process is not reachable from here; running a small number of
worker PROCESSES is.

**CAUTION — the multi-process configuration is not yet proven accuracy-clean.**
In 4 of 6 three-process trials one of the three processes returned **F1 76–80%**
instead of 85.44%, with whole pages carrying another page's transcript. It
reproduced identically with and without the texture-pack change (i.e. pre-existing
and independent of it), and never appeared in ~25 single-process runs at
K=8/16/24.

**The concurrency framing was wrong, and that matters.** 3 processes × 8 replicas
= 24 replicas — *exactly* single-process K=24. Same in-flight pages, same resident
engines, same per-page buffer churn. So the failing configuration is not "more
concurrency"; it is the same concurrency split across three address spaces. That
rules out any bug that scales with K — which is precisely why 25 single-process
runs found nothing — and points instead at something arbitrated **between Metal
clients at the driver level**, which raising K in one process cannot provoke.

**A sufficient mechanism was found and fixed.** A failed `MTLCommandBuffer`
executes *none* of its encoded work, so `MpsDetector::out_buf_`,
`MpsRecognizer::Bucket::{idx_buf,max_buf}` and the classifier's scratch — all
allocated once at `load()` and reused for every page — silently keep the
**previous page's** bytes. Every guard on those reads checked *encode* success
(`submit_forward_`, `engine->run()`), which says nothing about whether the GPU
ran the work. The result is the previous page's **complete, correct** transcript
attributed to this page: whole-page granularity, not garbled text. Exactly this
signature.

The detector for it already existed — `attach_error_watch` logs and counts every
failed command buffer — but `command_buffer_error_count()` had exactly two
references in the tree: its definition and its declaration. **Nothing consumed
it.** It now does: `MetalDeviceQueue::sync_ok(mark)` fails a page when the
timeline wait times out *or* this queue's error counter moved, and it is wired
into all four host reads of reused scratch (det sync, det async, the rec round,
cls). Losses go out through `IRecognizer::last_dropped_crops()`, because
`MpsRecognizer` pre-sizes its output so the pipeline's under-return check
structurally cannot see them.

**What is still unproven.** That aborts *are* what produced the 76–80% numbers.
The fix guarantees a GPU fault can no longer masquerade as a successful page, but
nobody has observed an abort in those trials. **One grep settles it:** capture
per-process stderr in a 3-process run and look for `COMMAND BUFFER FAILED`,
`queue synchronize TIMED OUT`, or `rec round INVALIDATED`. Do that before
declaring the multi-process number real.

**Two measurement gaps worth knowing.** Only the first pass (`j < N`) is scored,
so the F1 figure says nothing about corruption later in the window — `--words-all`
is what audits that. And a regression gate should key on **wrong-page count == 0**
from that audit, not on F1: F1 cannot distinguish "3 pages came back blank" from
"3 pages carried someone else's text", and only the second is a correctness bug.
(Note `UnifiedOcrPipeline::run()` returns only `std::vector<OCRResultItem>` and
discards `text_degraded`, so the bench cannot currently see degradation through
the API it uses — closing that is the next step.)

### Recommended next lever

Both engines are near saturation at ~107 img/s, so the only remaining in-process
gain is a *fractional* GPU/ANE split: keep the MPSGraph twin alive for the W800
bucket and divert a chunk to the GPU only when the ANE queue is deep, instead of
the static `ANE_MAXW` width cut (all-or-nothing, and full diversion costs 18%).
The upside is bounded by the gap between 107 and the co-saturation point — likely
single-digit percent, and below this machine's measurement noise without many
paired runs.

### Instrumentation

`TURBO_APPLE_CONTENTION=1` dumps lock-blocked time, queue depth, per-width ANE
predict counts/latency/rows and per-stage timings at exit
(`src/backends/apple/support/apple_contention.h`). Unlike `TURBO_APPLE_PROFILE` it takes no
global mutex per scope, so it does not itself serialize the thing being measured.
