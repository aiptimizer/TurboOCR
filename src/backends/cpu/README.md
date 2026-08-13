# TurboOCR — CPU backend (`turbo_ocr_backend_cpu`)

The portable arm of the device seam (`include/turbo_ocr/backend/*.h`), and the
only one that is built and gated in CI on every commit. Three roles, and it is
worth keeping them apart because only the first is "the cpu backend":

1. **`CpuBackend`** — the `Backend` a CPU build constructs, and the auto-detect
   fallback when no accelerator is present.
2. **The SHARED host kernel set** (`kernels_host/`) — the host fallback ops that
   *every* vendor links for the ops it has no native kernel for. Target
   `turbo_ocr_host_kernels`.
3. **The SHARED ONNX ("fast") stage set** (`stages/cpu_stages.h`) — the .onnx
   through an ONNX Runtime execution provider with host pre/post. NVIDIA-on-CUDA-EP,
   Intel-on-OpenVINO, Apple-on-CoreML and AMD-on-MIGraphX are all *this* code with
   a different `EpConfig`. Target `turbo_ocr_backend_onnx`.

Roles 2 and 3 are why `src/backends/cpu/` is a dependency of every other vendor
directory and not a peer of them. **A vendor-local copy of anything in
`kernels_host/` or `stages/cpu_stages.*` is the duplication this architecture
exists to prevent** — see the shared-policy rule in
[`docs/contributing/adding-a-backend.md`](../../../docs/contributing/adding-a-backend.md) §2.

## Files

The per-concern layout every vendor uses — see [`../README.md`](../README.md)
for the rule each directory answers to. Headers are included through the
vendor-rooted path off `-Isrc/backends`, e.g. `#include "cpu/memory/host_allocator.h"`.

```
src/backends/cpu/
├── README.md                             this file
├── support/host_common.h                 seam <-> host vocabulary: ImageView (Host) <-> cv::Mat
├── queue/host_device_queue.{h,cpp}       DeviceQueue: the degenerate synchronous lane
├── memory/host_allocator.{h,cpp}         IDeviceAllocator over plain host RAM
├── kernels_host/host_kernels.{h,cpp}     IKernels — the SHARED host fallback op set (role 2)
├── engine/cpu_engine_adapter.{h,cpp}     IEngine over ORT (any execution provider)
├── stages/cpu_stages.{h,cpp}             IDetector/IRecognizer/IClassifier/ILayout
│                                           + make_vendor_onnx_stages / resolve_engine_mode (role 3)
├── stages/cpu_table_recognizer.{h,cpp}   ITableRecognizer over the SLANeXt path
├── stages/cpu_formula_recognizer.{h,cpp} IFormulaRecognizer over the PP-FormulaNet path
└── backend/cpu_backend.{h,cpp}           Backend: factories + load_stages + service fns
    backend/cpu_backend_registry.cpp      registers "cpu" (needs WHOLE_ARCHIVE)
```

## What lives here vs. what is shared

The Host address space **is** host RAM, so this backend has the least device
mechanics of any of them: the queue is a synchronous no-op, the allocator is
plain malloc, and an `ImageView` and a `cv::Mat` alias the same bytes — there is
no transfer and no vendor pointer type (`support/host_common.h`).

What is left is thin wrapping. `stages/cpu_stages.h` forwards to the proven
main-tree classes and re-implements none of them:

| Seam interface | Wrapped class |
|---|---|
| `backend::IDetector` | `detection::CpuPaddleDet` |
| `backend::IRecognizer` | `recognition::CpuPaddleRec` |
| `backend::IClassifier` | `classification::CpuPaddleCls` |
| `backend::ILayout` | `layout::CpuPaddleLayout` (synchronous only) |
| `OrientFunc` source | `classification::CpuDocOrientation` |

Every policy question — det resize and DB thresholds, rec width buckets and the
batch ladder, CTC decode and the dictionary, cls geometry and threshold,
normalization constants, crop geometry, the PicoDet row decode, the layout
post-filter, SLANeXt/formula path resolution, and the whole
det→cls→rec→layout→router flow — is answered **above** the seam by a shared
header this directory calls. There is deliberately **no** `make_infer_func()`
override here.

## What is actually verified

Unlike every other vendor directory in this tree, this one is not a scaffold:

- Built on every commit by CI (`Dockerfile.cpu`, unit suite, smoke test).
- It is the `--ref` side of `turbo_golden`: the other backends are diffed
  **against this code**, per stage, so a divergence anywhere else is measured
  relative to what is here.
- It is the reference for `turbo_conformance` and carries a FUNSD accuracy gate
  in ctest.

That also sets the standard of proof for changing anything in `kernels_host/` or
`stages/cpu_stages.*`: a change here moves every backend at once, including the
baseline the others are judged against.

## Reading order for a new vendor

This is the directory `docs/contributing/adding-a-backend.md` tells you to read first,
and the order that pays off is:

1. `backend/cpu_backend.cpp` — the whole `Backend` surface, small enough to hold
   in your head.
2. `stages/cpu_stages.h` — what a stage class is required to do, and
   `resolve_engine_mode()` / `make_vendor_onnx_stages()`, which your backend
   calls rather than re-derives.
3. `support/host_common.h` — the seam↔vendor translation pattern in its simplest
   possible form; `nvidia/support/cuda_common.h` is the same idea with a real
   device under it.
