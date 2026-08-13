# `src/backends/` — the vendor arms of the device seam

One directory per vendor. Each implements
`include/turbo_ocr/backend/backend.h`'s `Backend` and registers itself into the
ONE link-time registry (`src/backend/backend_registry.cpp` — *backend*, singular:
the vendor-neutral contract side, see src/README.md), so several
backends co-link into a single binary and `--backend` selects at runtime.

| Directory | Device path | Inference | Status |
|---|---|---|---|
| `cpu/` | host RAM | ORT (MLAS or any EP) | reference; also hosts the SHARED host kernels and the SHARED ONNX stage set |
| `apple/` | Metal / unified memory | MPSGraph + ANE | measured on M3 Max |
| `nvidia/` | CUDA | TensorRT / ORT-CUDA | wraps the proven main-tree classes; fully configured in CMake (`turbo_ocr_backend_nvidia`) and hardware-validated 2026-08-02 |
| `amd/` | HIP | MIGraphX | written, never compiled — needs ROCm hardware |
| `intel/` | SYCL USM / Level Zero | OpenVINO | ported and verified on real Intel hardware |

Read each directory's `README.md` for what is and is not verified there.

## The layout, which is the same for every vendor

A vendor arm is not an arbitrary bag of files. It is **an implementation of a
known interface set**, so the directory names are the *interface* names: the
mapping from "which interface" to "which directory" is mechanical rather than a
matter of taste, and a reader who learns one vendor knows all five.

| Directory | The one thing it holds | Declared in |
|---|---|---|
| `backend/` | the `Backend` implementation + the one `BackendRegistrar` — the vendor's entry point; everything else here is something `Backend` hands out | `backend.h`, `backend_registry.h` |
| `engine/` | `IEngine` — how this vendor runs a **model** | `engine.h`, `engine_mode.h` |
| `kernels_<toolchain>/` | `IKernels` — how this vendor runs a **hand-written op** — plus that toolchain's kernel sources | `kernels.h` |
| `memory/` | `IDeviceAllocator` and the device image/buffer types — where this vendor's bytes live | `backend.h`, `image_view.h` |
| `queue/` | `DeviceQueue` / `DeviceEvent` — how this vendor orders work | `device_queue.h` |
| `stages/` | `IDetector` / `IRecognizer` / `IClassifier` / `ILayout`, plus table/formula | `stages.h`, `{table,formula}_recognizer.h` |
| `support/` | *(no interface)* everything used by ≥2 of the above that implements no seam interface itself | — |
| `probes/` | *(no interface)* standalone executables that exercise ONE interface off the full pipeline | — |

**A directory exists only if that column-two sentence can be written for it.**
The two that map to no interface carry an explicit membership rule instead, so
neither becomes a junk drawer:

* `support/` — seam↔vendor type translation, the error-check macro, device
  singletons, instrumentation. Admission test: *used by two or more concern
  directories, and implements nothing from `include/turbo_ocr/backend/`.*
  `intel/` has none, which is information: it does its translation inline.
* `probes/` — one `add_executable` per file, each answering *"is this one
  interface correct?"* separately from *"is the wiring correct?"*, which an
  end-to-end F1 cannot distinguish. `intel/probes/ov_engine_probe.cpp` is the
  model.

`kernels_<toolchain>/` is suffixed because the toolchain is the thing that
actually varies — `.hip`→hipcc, `.cu`→nvcc, `.metal`→metal, `.cpp`→host C++,
each with its own CMake `LANGUAGE`. Today: `kernels_host`, `kernels_cuda`,
`kernels_metal`, `kernels_hip`, `kernels_sycl`.

A directory is simply absent when the vendor has nothing for it. Headers are
included through the **vendor-rooted** path off `-Isrc/backends`, which every
backend target and `tools/syntax_shims/check.sh` put on the include path:

```cpp
#include "nvidia/memory/cuda_allocator.h"   // yes
#include "cuda_allocator.h"                 // no — breaks the moment a file moves
```

### This is enforced, not just documented

```bash
python3 src/backends/layout_check.py     # exit 0 = clean
```

It checks the rules above plus the two failure modes that had **already**
happened here: a bare sibling include that breaks on any move, and a source file
compiled by nothing (the Apple target used a non-recursive `file(GLOB .../apple/*.mm)`,
which subdirectories would have silently emptied — a broken binary, not a build
error). It also catches a `CMakeLists.txt` or `tools/syntax_shims/sources.txt`
entry that no longer points at a real file. No build required.

## Adding a new vendor

**→ [`docs/contributing/adding-a-backend.md`](../../docs/contributing/adding-a-backend.md)** — the
guide: what a backend is and is not, the shared-policy rule, the minimum viable
backend, the build order, host-fallback-first bring-up, mode handling, the
correctness gates with exact commands, the traps, and a reviewer checklist.

Start from the generator, which emits a compiling, correct, host-delegated
backend and prints the exact CMake lines to add:

```bash
python3 tools/new_backend.py --name foo          # --dry-run to preview
```

## The one rule that matters

**Generic policy is SHARED; only device mechanics live here.** Detection resize
and DB thresholds, rec width buckets and the batch ladder, CTC decode, the
character dictionary, cls geometry and its threshold, normalization constants,
crop geometry, the PicoDet row decode, the layout post-filter, SLANeXt/formula
path resolution, engine-mode policy, and the whole
det→cls→rec→layout→router flow are shared headers a backend **calls**. Every one
of them is a shared header *because* a backend once forked it and drifted — the
list, the owning header, and the bug each is a scar from are in §2 of the guide.

---

## Where a piece goes — the rule

1. **Device mechanics** — anything whose correctness or speed depends on which
   silicon runs it — lives in `src/backends/<vendor>/<concern>/` and is reached
   only through an interface in `include/turbo_ocr/backend/`. **No vendor header
   ever appears under `include/`.**
2. **Policy** — anything where the same answer is correct on every device (box
   sorting, drop score, reading order, degradation flags, result assembly,
   request scheduling) — lives above the seam, exactly once, and is deliberately
   **not** swappable. Fixing it in one backend is a bug in all the others.
3. **If the generic pipeline performs a step, that step must be an interface on
   the seam** — even when only one implementation exists today. A step the
   orchestration does inline is a step no vendor can replace.
4. **A vendor that lacks a piece borrows another vendor's by including its
   header** (`#include "cpu/…"`), never by copying it.
   `src/backends/apple/backend/apple_backend.mm` is the worked example: it
   borrows `cpu::HostDeviceQueue`, `cpu::HostAllocator`, `cpu::HostKernels`,
   `cpu::CpuEngineAdapter`, `cpu::CpuTableRecognizer`, `cpu::CpuLayout` and
   replaces only det/rec/cls + Metal kernels + the MPSGraph engine.
5. **A vendor changes behaviour only by returning a different implementation
   from its `Backend`.** If the shared layer must branch on `caps().name` or on
   `DeviceKind` to make a vendor fast, the seam is missing a piece: add the
   piece, not the branch.

**The test that settles arguments.** Name the piece, then ask: *"If NVIDIA had a
CUDA version of exactly this, where would it plug in?"* If the honest answer is
"it would have to fork the pipeline", the piece is missing from the seam — and
the speed gap you are about to measure is that fork, not that device.

### Why there is no `src/cuda/`

There was, until 2026-08-01: a complete second implementation for NVIDIA — its
own pipeline, server main, HTTP routes and model classes — while amd, apple,
intel and cpu each got by with a plugin. It is deleted. NVIDIA's device pieces
live in `src/backends/nvidia/` and it runs the same generic pipeline as
everyone else. `PipelineDispatcher` went with it: it was typed on the CUDA
pipeline, and that is exactly what forced a second `*_gpu.cpp` route family to
exist alongside every CPU twin.

**Known seam gaps** (a vendor cannot replace these yet, so they are where a
speed gap will appear): the page H2D upload is hardcoded in
`unified_ocr_pipeline.cpp::upload_image_`, and the pipeline owns a single
`DeviceQueue` where the old CUDA path used five streams with event handoffs, so
layout serialises behind detection instead of overlapping recognition. Both are
seam defects to close, not device costs to accept.
