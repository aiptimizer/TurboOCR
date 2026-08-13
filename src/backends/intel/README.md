# TurboOCR — Intel backend (OpenVINO + SYCL)

> **STATUS: UNVALIDATED ON INTEL HARDWARE.** This backend was written on an
> Apple M3 Max. There is no Intel GPU, no Level Zero driver and no oneAPI DPC++
> compiler on that machine, so **not one SYCL kernel here has ever been compiled,
> let alone executed, and no accuracy or throughput number has been measured.**
> What *has* been verified is stated precisely in [What is actually
> verified](#what-is-actually-verified) — read that section before trusting
> anything else in this file.

The Intel vendor backend for the multi-backend rebuild. Device data lives in
**SYCL USM** memory, pre/post runs as **SYCL kernels**, inference runs on
**OpenVINO Runtime** (CPU / GPU / NPU). Everything above the `Backend` seam — the
one `UnifiedOcrPipeline`, box routing, CTC, DB post, reading order, doc assembly,
table/formula dispatch — is shared and is **not** in this directory.

| NVIDIA (reference) | Intel (this backend) |
|---|---|
| `cudaStream_t` / `cudaEvent_t` | `sycl::queue` / `sycl::event` (`L0DeviceQueue`) |
| `cudaMalloc` | `sycl::malloc_device` USM (`L0Allocator`) |
| `src/cuda/kernels/*.cu` | SYCL kernels (`SyclKernels`) |
| `TrtEngine` (profiles) | `OpenVINOEngine` (per-shape `CompiledModel` cache) |
| `nvJPEG` decode | host OpenCV decode + H2D (VAAPI/oneVPL is a TODO) |

## Files

```
src/backends/intel/
├── queue/l0_device_queue.{h,cpp}        DeviceQueue/DeviceEvent over an in-order sycl::queue
├── memory/l0_allocator.{h,cpp}          IDeviceAllocator over USM + native L0 handles
├── kernels_sycl/sycl_kernels.{h,cpp}    IKernels: 5 native SYCL ops + 2 declared host fallbacks
├── engine/openvino_engine.{h,cpp}       IEngine: per-(width,batch) CompiledModel/InferRequest cache
├── stages/intel_stages.{h,cpp}          IDetector/IRecognizer/IClassifier/ILayout (thin)
├── backend/intel_backend.{h,cpp}        Backend: factories + load_stages + service fns
├── backend/intel_backend_registry.cpp   registers "intel" in make_backend()
└── probes/ov_engine_probe.cpp           off-hardware functional test of the engine
```

This per-concern layout is the one **every** vendor under `src/backends/` uses —
see [`../README.md`](../README.md). Headers are included through the
vendor-rooted path off `-Isrc/backends` (`#include "intel/memory/l0_allocator.h"`).

## What lives here vs. what is shared

Per the plan's **deduplication rules**, this directory contains only device
mechanics. Every policy question is answered above the seam:

| Concern | Where it lives | Called from |
|---|---|---|
| crop geometry, vertical-text rotation, width clamp | `compute_crop_transform` (`common/geometry/perspective.h`) | rec + cls |
| rec width buckets, `kMaxRecWidth` | `recognition::kRecWidthBuckets`, `rec_input_width` | via the planner |
| batch ladder, element budget, routing, chunking | `recognition::plan_rec_batches` / `batch_ladder_for_width` / `snap_batch` / `rec_shape_matrix` | rec + cls |
| det resize policy + DB thresholds | `detection::read_det_resize` / `compute_det_resize` / `read_db_params` / `effective_det_max_side` | det |
| DB box extraction (CCL, unclip, corner order) | `detection::extract_boxes_from_bitmap` | `SyclKernels::db_postprocess` |
| CTC greedy decode + dictionary | `recognition::ctc_greedy_decode` / `load_label_dict` | rec |
| box reading order | `turbo_ocr::sorted_boxes` | det |
| det→cls→rec→layout→router orchestration | `src/pipeline/` (`UnifiedOcrPipeline`, `make_infer_func`) | — |
| table / formula backend selection | shared `table::` / `formula::make_*_recognizer` | `IntelBackend` |

There is deliberately **no** `make_infer_func()` override and **no** Intel-private
table/formula dispatch.

> **A defect that was found and removed.** The pre-existing Intel scaffold in this
> directory had hardcoded `kMaxW = 320` with its own crop-width formula, an
> inline CTC decode loop, a hand-rolled `findContours` + unclip DB post, its own
> det canvas rule, and the wrong classifier normalization and corner-flip. That
> is the *exact* failure mode the plan warns about with the Apple rec-ladder bug
> (a private ladder squashed every line > 1600 px and cost 0.10 pt of F1 on one
> backend only). All six now route through the shared helpers, so a fix anywhere
> lands everywhere and an Intel-vs-CPU golden diff on those stages is exact by
> construction rather than "close".

## Design decisions worth knowing

### The engine caches artefacts per `(width, batch)` — nothing compiles in the hot path
`ov::Core::compile_model` costs 10²–10³ ms and a *static-shape* CompiledModel is
materially faster on the GPU plugin than a dynamic one. So `OpenVINOEngine` keeps
a `Variant{CompiledModel, InferRequest, staging}` per primary-input shape, built
by `prebuild()` at `load()` from `recognition::rec_shape_matrix(...)` — the shared
ladder. `run()` is a hash lookup + bind + infer. A shape that was never prebuilt
falls back to one dynamic variant and increments `shape_misses()`, which is
exposed precisely so a wrong warmup matrix is *observable* instead of silently
costing reshape time. Set `OV_CACHE_DIR` so the warmup compile is paid once ever.

`output_shape()` reports the compiled model's real output dims, so stages size
their logits/argmax buffers **from the model** rather than assuming a `/8` stride
or a dict-sized class head.

### `caps().async == false` is deliberate
OpenVINO's GPU plugin runs on its own stream. Until that stream is provably the
same Level Zero queue as our `sycl::queue`, returning "async" would be a data
race: the caller's contract is "sync the DeviceQueue, then read", and syncing a
SYCL queue says nothing about an OpenVINO request. `run()` therefore barriers the
queue, infers synchronously, and returns with outputs valid. **This is the single
biggest known performance cost of the current design** — one host sync per
forward pass — and removing it is bring-up item 2, not a code change here.

### `db_postprocess` is a *declared* host fallback, and that is the right call
`caps().db_postprocess == false`. Connected-component labelling + Clipper unclip
has no portable SYCL primitive, and the shared host function is the same one the
CPU and NVIDIA-contour paths use. Falling back costs **zero accuracy**, inherits
every future fix, and touches only two small maps once per image. Hand-writing a
SYCL union-find would be a *second implementation of shared post-processing
policy* — what the dedup rules forbid — and would need its own validation. This
mirrors the Apple backend exactly. Same reasoning for `decode_image`.

### `begin_batch()`/`end_batch()` are near-no-ops, on purpose
Metal needs an explicit `MTLCommandBuffer` to get one submission per image. SYCL
does not: an **in-order** `sycl::queue` already guarantees program order and lets
the Level Zero backend coalesce consecutive appends into one command-list
submission. So `begin_batch()` only marks the region and — critically —
`end_batch()` does **not** call `q.wait()`; a flush there would destroy the
coalescing the seam is asking for, turning every `BatchScope` into a host round
trip. A true single-submit region via `sycl_ext_oneapi_graph` is available under
`TURBO_OCR_HAS_SYCL_GRAPH` and is unvalidated.

## What is actually verified

The `build.sh` this section once referred to was retired when the backend
moved into the CMake tree; the current equivalents are:

```bash
tools/syntax_shims/check.sh                 # host type-check of every vendor TU
cmake -S . -B build-intel -DTURBO_BACKENDS=intel && ninja -C build-intel \
    turbo_backend_probe ov_engine_probe     # real-OpenVINO compile + link + probe
# No -DUSE_CPU_ONLY needed: naming a non-nvidia backend selects the host build
# path automatically (the configure no longer enters CUDA mode). See SETUP.md.
```

**Verified originally with the retired script (all passing; the shim check and
the CMake targets reproduce each item today):**

1. **Host syntax check** (`check`) — all 7 TUs compile with `-Wall -Wextra`, zero
   warnings, with SYCL and OpenVINO both off. Since the seam interfaces are
   abstract and every class is `final` and instantiated, this proves **every
   `override` matches the seam's pure virtuals**, and that all shared-helper call
   sites type-check.
2. **Real OpenVINO compile** (`ov`) — all 7 TUs compile with
   `-DTURBO_OCR_HAS_OPENVINO` against **OpenVINO 2026.0.0** headers (the Docker `intel` stage now pins the `openvino/ubuntu24_dev:2026.2.1` image — re-run the shim check against 2026.2 headers on first bring-up; the OV C++ API is stable across 2026.x minors, so no source change is expected). Every
   `ov::Core` / `CompiledModel` / `InferRequest` / `RemoteTensor` / reshape call
   is confirmed to exist with the signature used.
3. **Link closure** (`ov`) — the whole backend links into an executable against
   `libopenvino` + `build-cpu/libturbo_ocr_common.a`, so every shared helper it
   calls really resolves.
4. **Engine functional test** (`probe`, `probes/ov_engine_probe.cpp`) — drives the
   real `OpenVINOEngine` on the OpenVINO **CPU plugin** with the real
   `models/rec_tiny.onnx` and `models/det_tiny.onnx`. Passing assertions:
   `load()` + IO discovery; `prebuild()` compiles every shape from the shared
   ladder; `output_shape()` returns the model's true logits geometry
   (`rec_tiny` @ w=320 → `[B, 40, 6906]`); a prebuilt shape runs and does **not**
   increment `shape_misses()`; a non-ladder shape still runs and **does**
   increment it; the engine-owned-output `OutputLease` path returns data; the det
   model writes a valid `[0,1]` probability map into the caller's buffer.

**NOT verified — nothing on an Apple machine can verify it:**

- **No SYCL code has been compiled.** Every kernel body is inside
  `#if defined(TURBO_OCR_HAS_SYCL)`. Expect ordinary compile errors on first
  `icpx` run.
- **No USM / RemoteTensor interop.** The `ClContext::create_tensor(type, shape,
  usm_ptr)` signature was read from the installed headers, but the path needs
  `CL/cl2.hpp` (absent here) and is gated behind `TURBO_OCR_HAS_OV_USM`. Whether
  a SYCL-allocated pointer is accepted by the plugin's context is **the
  make-or-break unknown of this backend** (item 1 below).
- **No SYCL device build** — the `icpx` flags remain reviewed, not executed;
  Level Zero / USM interop is untested (see above).
- Accuracy and throughput ARE now measured — on real Intel silicon, twice
  (Core Ultra 7 265T and i5-13600K + UHD 770; the OpenVINO **CPU and GPU
  plugins** both ran the full pipeline through Docker). The numbers, the
  container recipes and the per-stage profile live in **SETUP.md** in this
  directory; this file's earlier "any figure quoted would be fabricated"
  claim predates those runs.

## Bring-up checklist (in order)

1. **USM ↔ RemoteTensor context sharing** — *the make-or-break item.* Confirm the
   DPC++ runtime's `sycl::context` and the OpenVINO GPU plugin's context are the
   same `ze_context_handle_t`. If not, either (a) build the plugin context *from*
   our handle via `ov::intel_gpu::ocl::ClContext`, or (b) invert ownership and
   allocate through `ClContext::create_usm_device_tensor()`, handing SYCL those
   pointers. Until it works, `has_remote` stays false and every tensor stages
   through a host mirror — **correct, but with a copy per tensor**.
2. **One lane** — put OpenVINO's inference stream on the same L0 command queue as
   `L0DeviceQueue`, then flip `caps().async` to true and drop the barrier in
   `run()`. This is the largest single expected win.
3. **Compile the SYCL kernels** and golden-diff each one against
   `src/backends/cpu/kernels_host/host_kernels.cpp` (see below).
4. **Pool sizing** — `recommended_pool_size` is currently a reasoned guess
   (iGPU 1 / Arc 2 / CPU `hw/4`). Size it from `ov::device` memory properties.
5. **Hardware decode** — VAAPI / oneVPL JPEG straight into USM, to flip
   `caps().decode_image`.
6. **Page orientation** — `make_orient_func()` returns empty (autorotate off),
   deliberately, rather than a stub that always answers 0°. Implementing it is
   mechanical (224² ImageNet preprocess + 4-class argmax, structurally identical
   to `IntelClassifier`); validate the class→angle mapping against
   `CpuDocOrientation`.
7. **Det canvas shapes** — det runs on the dynamic variant because there is no
   shared canvas-bucketing policy. If dynamic reshape measures expensive, add
   `detection::snap_det_canvas()` to the **shared** layer (NVIDIA's TRT profiles
   and Apple would both use it) — do not add a private ladder here.
8. **NPU** — MEASURED 2026-08-12 (Core Ultra 9 285K, native Windows, OpenVINO
   2026.3): the NPU plugin **does** require fully static shapes — every dynamic
   compile of our det/rec ONNX fails with `[NPU_VCL] Upper bounds were not
   specified`, while the same models compile fine once reshaped. So the
   prebuild machinery is the right mechanism, and `prebuild()` failures
   degrading to the dynamic variant means **silent NPU failure** — check
   `shape_misses()`. Two hard limits before investing: **`layout.onnx` cannot
   run on the NPU at all** — it is rejected on op support (`String attribute
   reduction is not supported`), static shapes or not, so layout must fall back
   to CPU/GPU in any NPU-routed pipeline; and the throughput verdict is
   negative — the NPU was 3.2x slower than the CPU on `det_tiny`, lost on both
   recognizers, and won only marginally on the medium detector. See SETUP.md
   §0b for the full matrix. NPU is a power/CPU-offload play, not a speed one.
9. **Native DB post** — only after everything above, and only if measurement says
   the D2H matters. It buys no accuracy.
10. **Device-resident table/formula** — run the SLANeXt / PP-FormulaNet encoders
    on this backend's engine instead of the portable path.

## Validation on real hardware (do these in order — do not skip to step 3)

**Step 1 — per-stage golden diff vs. the CPU backend.** The CPU backend already
passed its proof gate, so it is the reference. On a fixed set (say 50 FUNSD
images), run both backends and compare *per stage*, not end-to-end:

| Stage | Compare | Expected |
|---|---|---|
| `resize_normalize` | the CHW tensor | max abs diff ≤ ~1e-4 (float order) |
| `warp_crops` | per-crop CHW tensor | same tolerance; check padded columns are exactly 0 |
| `threshold` | the u8 bitmap | **bit-identical** |
| `db_postprocess` | box list | **identical** (same shared function on both sides) |
| `argmax` | indices + scores | indices identical (verify the lowest-index tie-break) |
| det `run()` | box list | identical, or the resize/normalize step is at fault |
| rec `run()` | strings | identical; any diff is a bucket/geometry bug, not a model bug |
| cls `run()` | flip count + flipped indices | identical |
| layout `run()` | class/score/box rows | identical (watch `im_shape` vs `scale_factor`) |

A diff at any row localises the bug to one op. Do not proceed while any row
mismatches — an F1 number computed on top of a broken kernel tells you nothing.

**Step 2 — FUNSD F1 through the shared pipeline.** Only once step 1 is clean,
run `UnifiedOcrPipeline` over `IntelBackend` on FUNSD-50 and score with
`tools/bench/score_funsd.py`. The target is the CPU/NVIDIA number for the same tier
(tiny ≈ 85.5%). Materially below that means a stage is subtly wrong, not that
"Intel is less accurate".

**Step 3 — throughput, and only then.** Report **throughput WITH its F1**; a
speed number without accuracy is meaningless (plan, performance gate). Measure
single-stream first, then a pool at `recommended_pool_size`. Before quoting any
number, assert `OpenVINOEngine::shape_misses() == 0` after warmup — otherwise you
are timing graph reshapes. Then profile in this order, because the design already
names the two suspects: (a) the per-forward-pass `queue.synchronize()` in
`run()`, (b) staging copies from a failed USM interop (check
`caps().io_space == L0`). If going through the shared seam costs > 10 %, the
plan is explicit: **fix the shared design so every backend gets it**, do not fork
a vendor-private fast path.

## Environment

| Var | Meaning |
|---|---|
| `OV_DEVICE` | `CPU` \| `GPU` \| `NPU` (default GPU) |
| `OV_CACHE_DIR` | on-disk compiled-blob cache; makes the warmup prebuild nearly free after the first boot |
| `OV_REC_MAX_PREBUILD_WIDTH` | how much of the rec width ladder to compile at boot (default 1600). Wider buckets still work via the dynamic variant |
| `REC_IMAGE_H` | recognizer input height (default 48) |
| `TURBO_POOL_SIZE` | override `recommended_pool_size` |
| `DET_*` | shared detection knobs (`det_config.h`) — same names and meanings as every other backend |

Build macros: `TURBO_OCR_HAS_SYCL` (SYCL kernels/queue/allocator),
`TURBO_OCR_HAS_OPENVINO` (the engine), `TURBO_OCR_HAS_OV_USM` (zero-copy USM
binding; needs the OpenCL C++ headers), `TURBO_OCR_HAS_SYCL_GRAPH` (single-submit
batch region). With all of them undefined the tree still compiles, and `caps()`
reports the reduced capability honestly rather than pretending.
