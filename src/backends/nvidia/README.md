# TurboOCR — NVIDIA backend (`turbo_ocr_nvidia`)

This directory is the **NVIDIA implementation of the device-agnostic `Backend`
seam** (`include/turbo_ocr/backend/*.h`). Per the approved plan, the
NVIDIA backend is the **non-regression reference**: it does **not** re-implement
anything — it **wraps the existing, proven CUDA/TensorRT/ORT-CUDA classes** and
forwards to them, converting only the interface vocabulary at the seam. The goal
is **bit-identical output and zero perf regression** vs today's `OcrPipeline`.

Nothing here modifies `src/` or `include/`. Every file `#include`s the real
headers and delegates.

---

## What each interface wraps

| New interface (`turbo_ocr::backend`) | Wrapped existing class | Files |
|---|---|---|
| `ImageView` ↔ `decode::GpuImage` | `decode::GpuImage` (add/drop `kind=Cuda`) | `support/cuda_common.h` |
| `DeviceQueue` / `DeviceEvent` | raw `cudaStream_t` / `cudaEvent_t` | `queue/cuda_device_queue.{h,cpp}` |
| `IDeviceAllocator` / `DeviceBuffer` | `cudaMalloc`/`cudaMallocHost` (as in `CudaPtr`) | `memory/cuda_allocator.{h,cpp}` |
| `IEngine` (+`IProfiles`,`IGraphCapture` ExplicitSlot) | `engine::TrtEngine` | `engine/trt_engine_adapter.{h,cpp}` |
| `IEngine` (+`IGraphCapture` Transparent) | `formula::OrtSession` (ORT-CUDA) | `engine/ort_cuda_engine.{h,cpp}` |
| `IKernels` (9 CUDA pre/post ops) | free functions in `kernels/kernels.h` + `NvJpegDecoder` | `kernels_cuda/cuda_kernels.{h,cpp}` |
| `IDetector` | `detection::PaddleDet` | `stages/nv_stages.{h,cpp}` |
| `IRecognizer` | `recognition::PaddleRec` | `stages/nv_stages.{h,cpp}` |
| `IClassifier` | `classification::PaddleCls` | `stages/nv_stages.{h,cpp}` |
| `ILayout` (two-phase enqueue/collect) | `layout::PaddleLayout` | `stages/nv_stages.{h,cpp}` |
| `OrientFunc` source | `classification::DocOrientation` | `stages/nv_doc_orientation.h` |
| `backend::ITableRecognizer (was table::ITableRecognizer before the ODR-fix rename)` (de-CUDA'd) | `table::SlanextTableRecognizer` | `stages/nv_table_recognizer*.{h,cpp}` + `stages/nv_table_bridge.h` |
| `backend::IFormulaRecognizer` (de-CUDA'd) | `formula::PPFormulaNetOrt` | `stages/nv_formula_recognizer*.{h,cpp}` + `stages/nv_formula_bridge.h` |
| `Backend` + `make_backend`/`available_backends` | `stages_gpu.cpp` helpers (`load_gpu_stages`, `make_gpu_*`, `probe_nvjpeg`) | `backend/cuda_backend.{h,cpp}`, `backend/nv_backend_registry.cpp` |

### The old/new interface collision — and the pimpl bridge

`backend::ITableRecognizer (was table::ITableRecognizer before the ODR-fix rename)` and `backend::IFormulaRecognizer` (plus
`formula::FormulaEngineResult`) are declared **in the same namespaces** by both
the **old** headers (`include/turbo_ocr/{table,formula}/…`) and the **new**
headers (`include/turbo_ocr/backend/…`). A single translation unit
**cannot see both generations** (ODR clash). So table and formula each use a
**pimpl-across-a-generation-gap** split:

- `nv_*_recognizer.cpp` — **new-headers** TU: implements the new interface,
  forwards to an opaque `Nv*Impl` (declared in `nv_*_bridge.h`).
- `nv_*_recognizer_impl.cpp` — **old-headers** TU: defines `Nv*Impl` by wrapping
  the existing `SlanextTableRecognizer` / `PPFormulaNetOrt`.
- `nv_*_bridge.h` + `nv_image_pod.h` — the **neutral boundary**: only shared,
  non-colliding types (`Box`, `OCRResultItem`, `router::TableResult`, a
  `GpuImagePod`, a `FormulaResultPod`, `void* stream`). Includes **neither**
  interface header.

det/rec/cls/layout need no such split — `turbo_ocr::backend` does not collide
with `turbo_ocr::{detection,recognition,…}`, so one TU wraps them directly.

---

## Toolchain — what compiles where

**None of this compiles on the dev Mac** (no CUDA / TensorRT / ORT-CUDA / nvJPEG
SDKs). It is written to compile in the existing `turbo_ocr_gpu` CUDA build
(`nvcc` + host `clang++`/`g++`, TRT 10, CUDA 13, ORT-CUDA), against **both**
include roots (`-Iinclude`).

| File | Toolchain | Pulls |
|---|---|---|
| `support/cuda_common.h`, `queue/cuda_device_queue.*`, `memory/cuda_allocator.*`, `kernels_cuda/cuda_kernels.*`, `engine/trt_engine_adapter.*`, `engine/ort_cuda_engine.*`, `stages/nv_stages.*`, `stages/nv_doc_orientation.h`, `stages/nv_*_impl.cpp`, `backend/cuda_backend.*`, `backend/nv_backend_registry.cpp` | CUDA host compiler (TRT/CUDA/ORT/nvJPEG) | `NvInfer.h`, `cuda_runtime.h`, `nvjpeg.h`, ORT via pImpl |
| `support/nv_image_pod.h`, `stages/nv_*_bridge.h`, `stages/nv_table_recognizer.h`, `stages/nv_formula_recognizer.h` | plain C++20 (no CUDA) | shared common headers only |

**Verified on the Mac** (`clang++ -std=c++20 -fsyntax-only` against both include
roots): the CUDA-free headers — `nv_image_pod.h`, `nv_table_bridge.h`,
`nv_formula_bridge.h`, `nv_table_recognizer.h`, `nv_formula_recognizer.h` — all
parse clean. The CUDA-dependent TUs are **compile-verified on hardware only**.

### CMake — `turbo_ocr_backend_nvidia`

The target lives in the root `CMakeLists.txt`, guarded by
`nvidia IN_LIST TURBO_BACKENDS`. It went a long time never configured; the
breakages that had accumulated behind that are fixed and it now compiles. Its
shape:

* every `.cpp` in this directory, including **both** `nv_*_recognizer.cpp` and
  their `nv_*_recognizer_impl.cpp` siblings — same library, separate TUs (that is
  the whole point of the pimpl-across-a-generation-gap split above);
* `nv_backend_registry.cpp`, force-linked with `$<LINK_LIBRARY:WHOLE_ARCHIVE,…>`
  via `turbo_link_backends()` — its `BackendRegistrar` is referenced by
  nothing, so an ordinary archive link drops it and the backend silently vanishes;
* the `.cu` files this directory adds beyond `TURBO_GPU_CU_SRCS`, which the glob
  subtracts (see `kernels_cuda/README.md`);
* links `turbo_ocr_pipeline`, `turbo_ocr_backend_onnx` (the shared CUDA-EP fast
  path) and `turbo_ocr_gpu`, which PUBLICly carries the CUDA/TRT/ORT include
  dirs, `${TENSORRT_DIR}/lib`, and
  `nvinfer`/`nvinfer_plugin`/`nvonnxparser`/`nvjpeg`/`cudart`. Nothing here is
  recompiled that `turbo_ocr_gpu` already owns.

### `kernels_cuda/` holds both the kernels and the seam adapter

`kernels_cuda/` contains the four `.cu` kernel files (CCL, JFA, fused
preprocess, reductions) **and** `cuda_kernels.{h,cpp}`, the `IKernels` adapter
that translates seam vocabulary into calls on them.

This used to be an asymmetry with AMD — the hipified twins of the same kernels
lived inside the backend at `src/backends/amd/kernels_hip/*.hip` while NVIDIA's
sat in a top-level `src/cuda/`, because NVIDIA's kernel header had consumers in
the analysis layer and moving it in would have inverted the layering. Routing
those consumers through `backend::IKernels` removed them, and the kernels moved
here. The asymmetry is gone; NVIDIA now follows the same rule as every other
vendor.

The remaining `.cu` files outside this directory are
`src/backends/nvidia/stages/table_kernels.cu` (table/layout region preprocess)
and `src/analysis/formula/ppformulanet/*.cu` (PP-FormulaNet decode / gating).
See `kernels_cuda/README.md` for how the four kernel TUs are named once and
subtracted from this target's glob.

Two factual notes from the superseded argument are worth keeping, because they
were themselves corrections of earlier mistakes:

> The signature header is **NVIDIA-private**, not cross-vendor. It includes
> `<cuda_runtime.h>`; AMD's `kernels_hip.h` deliberately does *not* include it
> and mirrors the POD types instead, precisely to avoid the CUDA dependency. An
> earlier claim to the contrary came from grepping the *filename* and matching a
> comment that said the opposite.
>
> Being NVIDIA-private is what made the header *movable*; it was never what
> decided *where* it goes. The consumer fan-out and the link graph decided that,
> and the fan-out is now zero.

**→ [`kernels_cuda/README.md`](kernels_cuda/README.md)** is the authoritative
record for how the kernel TUs are compiled and why this target's
`file(GLOB_RECURSE ... nvidia/*.cu)` must subtract them.

Bring-up checklist: `tests/cpp/backends/README.md` § "UNVERIFIED — first-configure
checklist".

---

## On-hardware bring-up TODOs

1. **InferFunc — DONE, nothing NVIDIA-specific left.** `CudaBackend::make_infer_func()`
   and `attach_dispatcher()` have been DELETED: they were the NVIDIA copy of the
   orchestration/pooling. `src/service/server/unified/server_main.cpp` builds a pool of
   `UnifiedOcrPipeline` entries from `load_stages()` and calls the ONE
   `pipeline::make_infer_func(pool)`. On-hardware work here is validating that the
   unified pipeline reaches the same throughput as `make_gpu_infer_func` did — if it
   does not, the fix goes in the SHARED layer, never back into `src/backends/nvidia/`.
2. **`IClassifier::run` flip-count.** `PaddleCls::run` returns `void`;
   `NvClassifier::run` returns `0`. If the merged pipeline needs the flipped
   count, thread it out of `PaddleCls` (it knows its `kClsThresh` hits).
3. **`CudaKernels::db_postprocess` vs `PaddleDet`.** The authoritative,
   regression-gated DB post-process stays inside `PaddleDet`
   (`run_gpu_ccl`/`run_gpu_ccl_fast`, owned by `NvDetector`). The standalone
   `IKernels::db_postprocess` reproduces the **mode-2 axis-aligned JFA** path for
   generic callers and **must be byte-diffed against `PaddleDet` on hardware**
   before it replaces any detector call site (mode-1 per-ROI `findContours`
   rotated quads are intentionally NOT reproduced here).
4. **`CudaKernels::resize_normalize` normalization variants.** Only the two baked
   full-frame variants (det, layout `pixel/255`) are exposed; arbitrary mean/std
   goes through `warp_crops`. Extend the `.cu` if a third full-frame norm is
   needed. `preprocess_region(LayoutSubRect)` hard-codes the 800×800 cell-det
   size — parameterize if a model differs.
5. **`OrtCudaEngine` construction knobs.** `device_id` / `cuda_stream` /
   `do_copy_default_stream` / `enable_cuda_graph` are ctor args (no common
   constructor). The production formula path uses `PPFormulaNetOrt` (which drives
   `OrtSession` directly); this engine adapter is for generic single-shot
   ORT-CUDA inference and needs its knobs wired from config on the formula host.
6. **`TrtEngineAdapter::reset()` is a no-op.** `engine::TrtEngine` exposes no
   public graph re-bake; NVIDIA never re-binds buffers post-warmup. Expose
   `destroy_graphs()` only if a caller needs to re-bake against new addresses.
7. **`make_backend` auto-detect.** `nv_backend_registry.cpp` returns the CUDA
   backend unconditionally (this lib only exists in a CUDA build). In the final
   multi-vendor tree, replace it with the **shared** common registry that
   link-collects every compiled backend and probes `cudaGetDeviceCount()` for
   empty-name auto-selection (fall back to `CpuBackend`).
8. **Regression gate.** Build `turbo_ocr_nvidia`, run the existing suite + a
   fixed image set through the server, and **byte-diff JSON output** against
   pre-rebuild `main`. Any diff blocks the phase.
