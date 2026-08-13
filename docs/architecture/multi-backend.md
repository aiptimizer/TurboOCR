# TurboOCR multi-backend architecture (dedup-first)

*Originally the rebuild implementation plan; the rebuild has landed, so this is now both the
architecture record and the standing rules. Paths and claims last verified against the tree
2026-07-24.*

**Rule:** Work lands DIRECTLY in `src/backends/*`, `src/pipeline/*` and `include/turbo_ocr/backend/*`
— the old staging tree `rebuild/` is gone (merged into the main tree 2026-07-23). What is still frozen
is *hardware we cannot run*: the NVIDIA / AMD / Intel arms must stay bit-identical, so edits there are
mechanical only and verified by `tools/syntax_shims/check.sh` (every source in
`tools/syntax_shims/sources.txt` clean), never by a
real compile. Only the CPU and Apple arms are buildable and testable on this Mac. NEVER run git.

## DEDUPLICATION RULES (non-negotiable — read before writing any code)
The user's overriding requirement: **as few duplications as possible; never fix the same bug
separately per backend.** Speed/accuracy work must happen WITHIN this constraint, not around it.

1. **Generic policy is SHARED. Only device mechanics are per-backend.** Before writing anything in
   `src/backends/<vendor>/`, ask: *"would NVIDIA/CPU need this identical logic or this identical fix?"*
   If yes → it belongs in the shared layer (`src/pipeline/*` or `turbo_ocr_common` helpers), so
   every backend inherits it. Per-backend code is limited to: the inference engine, the device
   pre/post kernels, and thin glue (ImageView/DeviceQueue).
2. **Reuse existing shared helpers — never re-derive them.** e.g. width bucketing/crop sizing lives in
   `include/turbo_ocr/analysis/recognition/rec_geometry.h` (`rec_input_width`, `snap_width_bucket`,
   `kRecWidthBuckets`, `kMaxRecWidth=4000`); DB box extraction in `detection::extract_boxes_from_bitmap`;
   CTC in `recognition::ctc_greedy_decode`.
   **Cautionary precedent:** the rec-ladder clamping bug (lines >1600px squashed, −0.10pt) existed ONLY
   on Apple because `MpsRecognizer` hardcoded its own ladder instead of using the shared helpers. Shared
   policy makes that class of bug structurally impossible. Do not create more of these.
3. **Use the seam's existing mechanisms instead of inventing local ones.** Cross-stage residency /
   command-buffer fusion is what `DeviceQueue::begin_batch()/end_batch()` (BatchScope, device_queue.h)
   is for — Metal opens one MPSCommandBuffer, CUDA/Host no-op. Drive it from the shared pipeline.
4. **Device-specific hardware is allowed but must stay behind the seam.** The Apple GPU+ANE hybrid is
   legitimately Apple-only, but it lives *inside* an `IRecognizer` impl; routing decisions must never
   leak into the orchestration.
5. **If the seam blocks you, propose a seam change** (signature + rationale) rather than working around
   it locally. A local workaround is a future per-backend bug.
6. Every agent's final report must list what it put in the SHARED layer vs the vendor dir, and justify
   each vendor-local item as genuinely device-specific.

## PERFORMANCE GATE (binding, alongside the dedup rules)
Proven hand-tuned Apple hybrid = **~140 img/s aggregate** (3 concurrent streams, GPU+ANE) and
**~46 img/s single-stream**. User's acceptance criterion: **up to ~10% below is fine, a large
regression is not.**
- **FLOOR: >= ~126 img/s aggregate at F1 >= 85.2%** (tiny). Landing at 30-50 img/s = FAILURE.
- **A correct seam costs ~nothing.** If going through the shared layer costs >10%, that is EVIDENCE
  THE ABSTRACTION IS WRONG — fix the SHARED design so it's fast for every backend. It is NOT a licence
  to fork a vendor-private fast path, and NOT a slowdown to quietly accept.
- Near-zero-cost seam means: interfaces are per-batch/per-image (never per-crop virtual dispatch in a
  hot loop); `BatchScope` really fuses a whole image's warp+rec+argmax into ONE command buffer with one
  commit; no compilation/allocation in the hot path (executables cached by (width,batch) at warmup,
  buffers pre-sized and reused); shared bucketing policy computed once per image, handing the backend
  whole groups.
- **Always report throughput WITH its F1.** A speed number without accuracy is meaningless. If below
  the floor, profile and report exactly where the abstraction costs, and propose the shared-layer fix.
- **STATUS: the ANE port IS DONE; what binds is Apple's per-process CoreML/ANE submission rate, and
  it is NOT the seam.** `MpsRecognizer` (`src/backends/apple/stages/mps_stages.mm`) runs the GPU+ANE hybrid
  *inside* the `IRecognizer` seam, exactly as dedup rule 4 requires: narrow width buckets go to
  `AneRecEngine` (`src/backends/apple/ane_rec_engine.{h,mm}` — a CoreML mlprogram on
  CPU+NeuralEngine), wide buckets stay on MPSGraph, and the cut is `TURBO_APPLE_ANE_MAXW`
  (0 disables the ANE). Bucket→batch routing stays in the SHARED planner
  (`include/turbo_ocr/analysis/recognition/rec_batching.h`); the ANE engine only reports the batch shapes its
  package physically supports.
  Measured through the unified pipeline on an M3 Max, FUNSD-50 tiny: **~98 img/s at 8 replicas, ~102
  at 16, ~106 at 24, with F1 85.44% in every configuration** (table and full profile in
  `src/backends/apple/README.md`, "In-process throughput ceiling"). So the floor is still not met in
  ONE process — and the profile says the abstraction is not the reason: the ANE saturates at ~5000
  rec rows/s / ~550 predicts/s **per process** (~46 ANE rows per FUNSD page ⇒ ~107 img/s), while the
  shared-layer suspects total ~1% (the global `resolve_buffer` mutex blocked 0.8 ms in TOTAL across
  38 679 calls in a 7 s run; per-page buffer alloc+free is 81 µs). In the same paired session
  `TURBO_APPLE_ANE_MAXW=0` — all rec back on MPSGraph, i.e. the GPU-only reference point — 61 img/s
  against a 105 baseline, so the ANE is worth ~1.7x here.
  **The ~140 img/s target was never one process.** The [Apple backend log](../notes/apple-backend-log.md)
  R22/R23 measured "K concurrent
  hybrid instances, sum of per-stream rates", and the unified pipeline reproduces that same
  multi-process effect: 3 processes × 8 replicas sum to ~7000 ANE rows/s against 4935 in one, ~40%
  more from the same silicon. Open work, in order: (1) make the multi-process path accuracy-clean —
  README "CAUTION": 4 of 6 three-process trials had ONE process return F1 76-80%, with whole pages
  carrying another page's transcript; (2) a *fractional* GPU/ANE split instead of the static width
  cut. Do NOT re-run the levers the README's "tried and LOST" table has already measured.

## The seam (already exists — the contract, do not change signatures)
`include/turbo_ocr/backend/*.h`: `Backend`, `IEngine`, `IKernels`,
`IDetector/IRecognizer/IClassifier/ILayout` (stages.h), `ImageView` (image_view.h),
`DeviceQueue`/`DeviceEvent`/`BatchScope` (device_queue.h), de-CUDA'd `ITableRecognizer`/
`IFormulaRecognizer`, `BackendRegistrar` (backend_registry.h).
READ THESE FIRST — they define every method to implement.

**Every seam type lives in `namespace turbo_ocr::backend`.** `ITableRecognizer` / `IFormulaRecognizer`
were moved OUT of `turbo_ocr::table` / `turbo_ocr::formula` precisely to end an ODR collision with the
old CUDA-typed classes of the same name. New-world code must never name them through the old
namespaces.

## Current state (STATUS — updated after the dedup + unified-server pass)

### DONE
- `src/backends/nvidia/` — NVIDIA as pure wrapper of existing PaddleDet/Rec/Cls/Layout/TrtEngine (reference pattern).
- `src/backends/apple/` — Apple MPSGraph impl; CMake target `turbo_ocr_backend_apple`
  (`add_library(turbo_ocr_backend_apple ...)` in `CMakeLists.txt`) plus the `turbo_apple_metallib`
  custom target that compiles
  `shaders.metal` → `turbo_apple.metallib` next to the executables.
- `src/backends/amd/`, `src/backends/intel/` — no hardware here; shim-verified only
  (`tools/syntax_shims/check.sh`). Intel has been ported and verified on real Intel HW elsewhere.
- **GAP 1 CLOSED** — `src/backends/cpu/` (CpuBackend), CMake target `turbo_ocr_backend_cpu`
  (`add_library(turbo_ocr_backend_cpu ...)` in `CMakeLists.txt`). FUNSD-50 proof gate:
  tiny 85.79 / small 90.97 / medium 92.56.
- **GAP 2 CLOSED** — ONE orchestration: `src/pipeline/unified/unified_ocr_pipeline.{h,cpp}`;
  ONE `server::InferFunc` builder: `src/pipeline/unified/make_infer_func.{h,cpp}`.
  `Backend::make_infer_func()` **has been removed from the seam** (`backend/backend.h`) and every
  per-backend override deleted (apple / nvidia / cpu / amd / intel). Backends now expose only
  `load_stages()` + `make_image_decoder()` + `make_orient_func()`.
  AppleBackend through the same unified pipeline: FUNSD-50 tiny 85.72.
- **Shared remote/VLM factory** — `src/pipeline/unified/vlm_factory.{h,cpp}` is the ONE definition of
  `backend::make_table_recognizer(BackendSpec)` / `backend::make_formula_recognizer(BackendSpec)` for
  `kind:openai` specs (device-agnostic port of `src/backends/nvidia/stages/openai_endpoint.cpp`: GpuImage→ImageView,
  cudaStream_t→DeviceQueue&). Device→host page readback is registered per device — see the
  "Per-device VLM readback" bullet below. `tests/cpp/backends/vlm_factory_link_support.cpp`
  remains ONLY as the nullptr stub for offline text-only proof binaries.
- **ONE server** — `src/service/server/unified/server_main.cpp` + `src/service/server/unified/backend_stages.cpp`
  (`include/turbo_ocr/service/server/unified/backend_stages.h`) replace `src/server/cpu_server_main.cpp` +
  `bootstrap/stages_cpu` — both now deleted. (`src/cuda/server/gpu_server_main.cpp` + `stages_gpu.cpp`
  still exist and remain the GPU configure's `turboocr-server` until the nvidia backend's on-hardware
  bring-up — the GPU configure's own `add_executable(turboocr-server ...)` in `CMakeLists.txt`.)
  Vendor-neutral: it only
  calls `backend::make_backend(name)` (`--backend` / `TURBO_BACKEND` / `OCR_BACKEND`), builds
  `caps().recommended_pool_size` UnifiedOcrPipeline entries, and registers the EXISTING
  device-neutral HTTP routes + gRPC + `bootstrap::run_http_server`.
  `src/backends/apple/backend/apple_backend_registry.cpp` added (mirrors cpu/nv registries).
  Build: `src/service/server/unified/unified_server.cmake` (included from the root `CMakeLists.txt` in the CPU
  configure). Which vendors the binary can select among is the LINK-time list
  `-DTURBO_BACKENDS=cpu;apple;...`, force-linked per backend by `turbo_link_backends()`.
  **TURBO_BACKENDS=cpu, =apple and =cpu;apple all compile the SAME server_main clean.**
- **GAP 3 CLOSED — ONE shared, link-time-collecting backend registry.**
  `backend::make_backend` / `available_backends` are now defined ONCE in
  `src/backend/backend_registry.cpp`; the seam gains one additive header,
  `include/turbo_ocr/backend/backend_registry.h` (`BackendRegistrar`, `register_backend`,
  `kBackendPriority*`). Every `*_backend_registry.cpp` (cpu / apple / nvidia / amd / intel) is now
  PURE registration — one namespace-scope `BackendRegistrar` that self-registers at static init —
  so several can be linked into ONE binary. Nothing references their symbols, so the registration TUs
  must reach the linker whole: `turbo_link_backends()` links each backend archive under
  WHOLE_ARCHIVE / `-force_load`.
  PROVEN: the `turbo_backend_probe` target (`tests/cpp/backends/backend_probe.cpp`, registered with ctest
  as `backend_probe`) links every selected backend into one binary and prints
  `available_backends() [2]: apple cpu`, with `--backend cpu`→CpuBackend(host),
  `apple`/`metal`→AppleBackend(metal), `host`→CpuBackend, `""`→apple (priority), `nope`→nullptr.
- **ONE `/ocr/batch`** — `src/service/http/unified_routes.cpp`
  (`routes::register_ocr_batch_route_unified`), typed on `pipeline::UnifiedPipelinePool` and driving
  `UnifiedOcrPipeline::run_batch_with_layout` in chunks of 8 with a per-image `run_with_layout`
  retry for slot isolation. Contract is a line-for-line port of the since-deleted
  `register_ocr_batch_route_cpu` (`src/http/image/batch/batch_route_cpu.cpp`; provenance also recorded
  in `include/turbo_ocr/service/http/unified_routes.h:7`): same gate, same error codes/strings, same shared
  per-slot stages from `src/service/http/image/batch/batch_common.cpp`, same bounded jthread fan-out, same
  `{batch_results, errors}` body. Registered by server_main.
- **`backend_name` folded into `/capabilities` — DONE.** `common_routes.h` is no longer frozen (that
  ended with the 2026-07-23 merge), so `routes::CapabilitiesInfo` now carries `backend_name` and
  `device_name` directly (`include/turbo_ocr/service/http/common_routes.h:46`), emitted by
  `src/service/http/admin/capabilities_route.cpp::build_capabilities_json` and populated in
  `server_main.cpp`. `GET /capabilities/backend`
  (`routes::register_backend_capabilities_route`, `src/service/http/unified_routes.cpp`) is deliberately
  KEPT rather than deleted: it is the only endpoint reporting `available_backends[]` — everything
  compiled into the binary, not just the active one — and removing a live endpoint would break
  existing clients.
- **Per-device (not process-global) VLM readback** — `pipeline::set_device_readback()` is replaced by
  `pipeline::register_device_readback(DeviceKind, DeviceReadback)`, keyed on the device the pages
  belong to, so a multi-backend binary cannot have two owners of one global slot. The Metal
  unified-memory fast path is no longer a `kind == DeviceKind::Metal` test in the shared layer: it
  is the seam capability `IDeviceAllocator::host_coherent()` (additive, defaulted from
  `backend::device_is_host_coherent(DeviceKind)`, overridable by an APU/iGPU/managed-memory
  allocator).

### REMAINING
- ~~Full link on macOS blocked by the PDF subsystem.~~ **RESOLVED.** The old single `if(NOT APPLE)`
  conflated three independent facts and so dropped ALL of PDF whenever any one was missing. They are
  now decomposed (`src/service/server/unified/unified_server.cmake`, plus the `turbo_ocr_cpu` PDF block in
  `CMakeLists.txt` that sets `TURBO_HAVE_PDF_RENDER`):
  *(1)* pdfium IS vendored for mac-arm64 (`third_party/pdfium/lib/libpdfium.dylib`, via
  `scripts/setup/install_pdfium.sh`) — the old check looked only for `libpdfium.so`;
  *(2)* turbojpeg gates page-image export only;
  *(3)* `TURBO_HAVE_INOTIFY` gates the daemon-based renderer ONLY; every platform without it
  compiles `src/pdf/render/pdf_renderer_inprocess.cpp` (in-process PDFium, serial by design).
  macOS AND Windows therefore get text layer, `mode=auto_verified`, searchable-PDF output
  AND rasterization. That branch was keyed on `APPLE` until 2026-08-10, which silently cost
  Windows the renderer. When rendering is genuinely unavailable, `src/service/server/unified/pdf_unavailable.cpp`
  supplies the `PdfRenderer` definitions so the link still closes.
  (The `turboocr-cpu-server` target named here is gone — the CPU configure's `turboocr-server` IS
  the unified server.)
- **VLM readback is still injected, not constructed-in.** The properly-scoped fix is
  `RemoteOpenAIEndpoint(spec, IDeviceAllocator&)` — but the seam's free factories
  (`backend::make_table_recognizer(const BackendSpec&)` / `backend::make_formula_recognizer`) would
  need the allocator in their signature, which means editing every vendor's
  `Backend::make_table/formula_recognizer` (incl. `src/backends/apple/*`, owned elsewhere). The
  DeviceKind-keyed registration above removes the last-writer-wins hazard in the meantime.
- `src/pipeline`'s pool is `UnifiedPipelinePool`
  (`include/turbo_ocr/pipeline/unified/make_infer_func.h:48`), and it is now the only one. The
  name clash this entry used to describe is gone with the clash: the old
  `turbo_ocr::pipeline::PipelinePool<Pipeline>` template in
  `include/turbo_ocr/pipeline/pipeline_pool.h` was never instantiated and the
  header has been deleted. The plain name `PipelinePool` is therefore free — but
  renaming is not free, so it stays `UnifiedPipelinePool` until something else
  makes that file worth touching.

## Deliverable 1 — `src/backends/cpu/` (CpuBackend), fully buildable+testable on this Mac
**DELIVERED** — kept below as the design record. Where it disagrees with "Current state" above, the
STATUS section wins (notably: `cpu_backend_registry.cpp` is now PURE `BackendRegistrar` registration;
`make_backend` itself is defined once in `src/backend/backend_registry.cpp`).

Mirror `src/backends/nvidia/` wrapper style, but wrap the existing CPU classes (in main-tree src/):
`CpuPaddleDet`, `CpuPaddleRec`, `CpuPaddleCls`, `CpuDocOrientation`, `CpuPaddleLayout`, `OrtEngine`.
Files:
- `cpu/host_device_queue.{h,cpp}` — `DeviceQueue` where device==Host: is_async()=false, record/wait/
  synchronize/begin_batch/end_batch are no-ops. `DeviceEvent` trivial.
- `cpu/host_allocator.{h,cpp}` — allocations are plain host malloc/`std::vector`; `ImageView{kind=Host}`
  wraps a `cv::Mat`'s data/step (zero-copy, no upload).
- `cpu/host_kernels.{h,cpp}` — `IKernels`: warp_crops → `cv::warpPerspective` (see cpu_paddle_rec.cpp),
  resize_normalize → `cv::resize`+convertTo, threshold → `cv::threshold`, db_postprocess →
  call shared `detection::extract_boxes_from_bitmap` (turbo_ocr_common), argmax → the loop already in ctc.
  `KernelCaps`: all host-native (no fallback needed since host IS the fallback).
- `cpu/cpu_stages.{h,cpp}` — `CpuDetector`/`CpuRecognizer`/`CpuClassifier`/`CpuLayout` implementing the
  stages.h interfaces by delegating to the wrapped Cpu* classes. Thin, like nv_stages.cpp.
- `cpu/cpu_backend.{h,cpp}` — `CpuBackend : Backend`: make_queue/allocator/make_kernels/make_engine
  (returns a OrtEngine wrapper), load_stages(BackendConfig)->StageSet, make_table/formula_recognizer
  (VLM/Openai specs work now; local table=CpuSlanextTableRecognizer, formula=CpuFormulaRecognizer wrapped),
  caps(){device=Host,name="cpu",native_image_decode=false,async=false,recommended_pool_size=hw_concurrency}.
- `cpu/backend/cpu_backend_registry.cpp` — registers "cpu" in make_backend().
BUILD IT: `add_library(turbo_ocr_backend_cpu STATIC ...)` at `CMakeLists.txt:1178`, linking
`turbo_ocr_pipeline` (the seam + `turbo_ocr_common`) + `turbo_ocr_cpu` (ONNX Runtime + the main-tree
host stages) + OpenCV. Must compile clean on this Mac.

## Deliverable 2 — `src/pipeline/` UnifiedOcrPipeline + shared infer-func (the dedup core)
**DELIVERED** — kept below as the design record.
- `pipeline/unified_ocr_pipeline.{h,cpp}` — ONE orchestration written against the seam interfaces
  (IDetector/IRecognizer/IClassifier/ILayout + DeviceQueue + Backend for table/formula registries).
  Port the control flow from the unified pipeline (`src/pipeline/unified/unified_ocr_pipeline.cpp`,
  `include/turbo_ocr/cuda/pipeline/ocr/ocr_pipeline.h`) — det→sort_boxes→(cls)→rec→combine,
  + run_with_layout adding
  layout→CUA router→table/formula dispatch. Retype GpuImage→ImageView, cudaStream_t→DeviceQueue&.
  Device-specific speed tricks (double-buffer/events) stay INSIDE the backend's IDetector/IRecognizer,
  invisible here. This compiles device-agnostically (no CUDA/Metal), links against any backend.
- `pipeline/make_infer_func.{h,cpp}` — the ONE `server::InferFunc` builder over a pipeline pool;
  replaces every per-backend make_infer_func. Backends must DROP their make_infer_func override
  (keep only load_stages + make_image_decoder).
Compile the pipeline .cpp to an object against the seam (host-only) on this Mac.

## Verification gates (on this Mac)
The "tiny driver" is now three real cross-backend binaries — one set for EVERY backend, selected at
runtime by `--backend`, built by `turbo_add_backend_test_exe()` (defined in `CMakeLists.txt`):
`turbo_bench` (accuracy + throughput), `turbo_conformance` (same images through every linked backend,
diffed), `turbo_golden` (per-stage golden diff vs the CPU reference), plus `turbo_backend_probe`.
They are registered with ctest; the FUNSD gates activate when `-DTURBO_FUNSD_CACHE=<dir>` points at
an image cache.
1. CpuBackend: `turbo_bench --backend cpu` over FUNSD 50 → score with tools/bench/score_funsd.py.
   Currently tiny 0.8579 / medium 0.9256, matching the pre-seam CpuOcrPipeline. This proves the seam
   is correct + dedup didn't regress.
2. AppleBackend: same binary, `--backend apple` → FUNSD det+rec F1 0.8572 (tiny), served through the
   unified pipeline instead of the standalone .mm harness.
NVIDIA/AMD/Intel: no hardware here — `tools/syntax_shims/check.sh` only (every source in `tools/syntax_shims/sources.txt` clean);
real regression needs their hardware.
Current whole-tree state: C++ 3283 assertions / 447 cases pass; ctest 12/12.

## Notes
- Metal 4 warp+rec fusion (MTL4MachineLearningCommandEncoder): VERIFIED works (metal-package-builder
  converts our models) but saves <1% for compute-bound small/medium rec — DE-PRIORITIZED, not a lever.
- Apple small/medium tiers use CoreML mlprogram (transformer arch), not MPSGraph; tiny uses MPSGraph.
