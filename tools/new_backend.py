#!/usr/bin/env python3
"""Generate a COMPILING skeleton for a new TurboOCR GPU-vendor backend.

    python3 tools/new_backend.py --name foo
    python3 tools/new_backend.py --name foo --device-kind Hip --dry-run
    python3 tools/new_backend.py --name foo --out src/backends/foo --force

What you get is a backend that is *correct on day one* and *slow on day one*:
every seam factory delegates to the shared host implementation (HostAllocator /
HostDeviceQueue / HostKernels / CpuEngineAdapter) and load_stages() goes through
the ONE shared ONNX stage factory, so the new vendor serves real OCR the moment
it links.  You then replace the delegations one at a time, native path first.

Read docs/contributing/adding-a-backend.md before editing what this emits — in
particular the SHARED-POLICY RULE, which is the thing that has broken every
backend added to this tree so far.

Dependency-free: standard library only, Python 3.8+.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# The DeviceKind enum is CLOSED (include/turbo_ocr/backend/image_view.h).  A
# genuinely new memory space needs a new enumerator there PLUS entries in
# device_is_host_coherent() and device_kind_name() — a SHARED edit, reviewed as
# such.  Until that lands, a new backend runs in Host space, which is exactly
# what the host-delegating skeleton below does anyway.
KNOWN_DEVICE_KINDS = ("Host", "Cuda", "Metal", "Hip", "L0")

HEADER_TMPL = r'''#pragma once

// @Name@Backend — the @name@ implementation of the ONE device seam
// (include/turbo_ocr/backend/backend.h).
//
// A Backend is a FACTORY, not a pipeline.  It hands the shared layer five
// things — a queue, an allocator, a kernel set, an engine, and a constructed
// StageSet — plus two genuinely device-shaped service functions (image decode
// and page orientation).  It must contain NO orchestration: there is
// deliberately no make_infer_func() on this interface, because the ONE
// det->cls->rec->layout->router flow lives above the seam in
// pipeline::UnifiedOcrPipeline + pipeline::make_infer_func.  Every backend that
// once shipped its own copy of that flow is why the seam looks like this.
//
// AS GENERATED this backend is the SHARED ONNX ("fast") path: every factory
// below returns the host implementation and load_stages() calls
// cpu::make_vendor_onnx_stages("@name@", cfg).  That is a complete, correct,
// portable backend — the same code Intel and AMD started from — and it is the
// baseline the golden diff compares your device path against.  Bring the device
// up underneath it in this order:
//
//   1. IDeviceAllocator + DeviceQueue  (device memory and one ordered lane)
//   2. IEngine                         (the vendor's model runner)
//   3. IKernels, one op at a time      (caps() reports which are native)
//   4. the stage classes               (resident det/rec/cls/layout)
//
// At every step caps() must describe REALITY — the mode you actually came up
// on, the device space you actually allocate in, the ops you actually run
// natively.  The shared layer branches on caps(); an over-claiming caps() is
// not an optimism bug, it is a wrong-output bug.

#include <memory>
#include <string>

#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/analysis/classification/ort_doc_orientation.h" // OrtDocOrientation

namespace turbo_ocr::@name@ {

class @Name@Backend final : public backend::Backend {
public:
  @Name@Backend();
  ~@Name@Backend() override;

  [[nodiscard]] backend::BackendCaps caps() const override;

  // --- Low-level device factories -------------------------------------------
  [[nodiscard]] std::unique_ptr<backend::DeviceQueue> make_queue() override;
  [[nodiscard]] std::shared_ptr<backend::IDeviceAllocator> allocator() override;
  [[nodiscard]] std::unique_ptr<backend::IKernels> make_kernels() override;
  [[nodiscard]] std::unique_ptr<backend::IEngine> make_engine() override;

  [[nodiscard]] std::unique_ptr<backend::ITableRecognizer>
  make_table_recognizer(const backend_routing::BackendSpec &spec) override;
  [[nodiscard]] std::unique_ptr<backend::IFormulaRecognizer>
  make_formula_recognizer(const backend_routing::BackendSpec &spec) override;

  // --- Stage bootstrap (the stages_* altitude) -------------------------------
  [[nodiscard]] backend::StageSet
  load_stages(const backend::BackendConfig &cfg) override;

  // --- Service-boundary functions -------------------------------------------
  // NOTE (dedup): no make_infer_func(). See the header comment.
  [[nodiscard]] server::ImageDecoder make_image_decoder() override;
  [[nodiscard]] server::OrientFunc make_orient_func() override;

private:
  // THE DEVICE SIDE FOLLOWS THE MODE.  In Onnx mode the stages are the shared
  // HOST ones: they take an ImageView in host memory and dereference it on the
  // CPU.  Handing them a device queue + a device allocator makes the shared
  // pipeline upload every page into device memory and pass a device pointer to
  // host code — not a slow path, a wrong one (it aborted on the first image on
  // Apple).  So every factory checks this before returning anything device.
  [[nodiscard]] bool native_device_() const noexcept {
    return mode_ == backend::EngineMode::Native;
  }

  // Resolved in the CONSTRUCTOR, not in load_stages(): the server snapshots
  // caps() once, BEFORE it calls load_stages() (src/service/server/unified/backend_stages.cpp:78
  // vs :107), and everything it reports afterwards — device, async, mode,
  // /capabilities, the HTTP thread count — comes from that snapshot.  A backend
  // that only learns its mode inside load_stages() reports the pre-resolution
  // default forever.
  backend::EngineMode mode_ = backend::EngineMode::Onnx;

  std::shared_ptr<backend::IDeviceAllocator> host_alloc_;
  std::unique_ptr<classification::OrtDocOrientation> doc_ori_;
  int pool_size_ = 0; // 0 => the default in caps()
};

// The vendor entry point the registrar hands to the shared registry.  Returns
// nullptr when this machine has no @name@ device: the registry treats that as
// "compiled in but unusable here" and walks down to the next priority.
[[nodiscard]] std::unique_ptr<backend::Backend> make_@name@_backend();

} // namespace turbo_ocr::@name@
'''

SOURCE_TMPL = r'''// @Name@Backend implementation — see @name@_backend.h.
//
// Everything here delegates to the SHARED host/ONNX layer.  That is not a
// placeholder to be embarrassed about: it is the correctness baseline the
// device path is diffed against (tools: turbo_golden --backend @name@ --ref cpu).
// Replace one delegation at a time and re-run the gates after each.

#include "@name@/backend/@name@_backend.h"

#include <memory>
#include <utility>

#include <opencv2/core.hpp>

// THE SHARED HOST LAYER.  These are NOT "the cpu backend" — they are the seam's
// portable implementations, linked by every vendor (CMake target
// turbo_ocr_backend_onnx / turbo_ocr_host_kernels).  Writing a @name@-local copy
// of any of them is the duplication this architecture exists to prevent.
#include "cpu/engine/cpu_engine_adapter.h"     // CpuEngineAdapter   : IEngine over ORT
#include "cpu/kernels_host/host_kernels.h"     // HostKernels        : IKernels
#include "cpu/memory/host_allocator.h"         // HostAllocator      : IDeviceAllocator
#include "cpu/queue/host_device_queue.h"       // HostDeviceQueue    : DeviceQueue
#include "cpu/stages/cpu_formula_recognizer.h" // CpuFormulaRecognizerAdapter
#include "cpu/stages/cpu_stages.h"             // make_vendor_onnx_stages, resolve_engine_mode
#include "cpu/stages/cpu_table_recognizer.h"   // CpuTableRecognizer

#include "turbo_ocr/backend/formula_recognizer.h"     // backend::make_formula_recognizer
#include "turbo_ocr/backend/table_recognizer.h"       // backend::make_table_recognizer
#include "turbo_ocr/backend/routing_config.h" // BackendSpec, Kind
#include "turbo_ocr/image/cpu_image_decode.h"        // decode_cpu_fallback

// TODO(@name@): the SHARED POLICY headers your device stages must call once they
// exist.  Every one of these is here because a backend forked it and drifted;
// none of them may be re-derived, re-typed or "adapted" inside this directory.
// See docs/contributing/adding-a-backend.md section 2.
//
//   turbo_ocr/analysis/detection/det_config.h        read_det_resize / compute_det_resize /
//                                           read_db_params  (resize + DB thresholds)
//   turbo_ocr/core/db_post_config.h    kDbDefaults and the geometry limits
//   turbo_ocr/analysis/recognition/rec_geometry.h    rec_input_width / kRecWidthBuckets
//   turbo_ocr/analysis/recognition/rec_batching.h    plan_rec_batches / batch_ladder_for_width /
//                                           snap_batch / rec_shape_matrix
//   turbo_ocr/analysis/recognition/ctc_decode.h      ctc_greedy_decode / load_label_dict
//   turbo_ocr/analysis/classification/cls_config.h   kClsImageH / kClsImageW / the threshold
//   turbo_ocr/core/norm_params.h          norm::rec_norm / cls_norm / imagenet_bgr
//   turbo_ocr/base/geometry/perspective.h compute_crop_transform
//   turbo_ocr/analysis/layout/picodet_decode.h       decode_picodet_rows
//   turbo_ocr/analysis/layout/layout_postfilter.h    postfilter_layout_boxes
//   turbo_ocr/onnx/host_ort_threads.h     set_host_ort_intra_op_threads (only if
//                                           your det/rec do NOT run on the CPU)

namespace turbo_ocr::@name@ {

@Name@Backend::@Name@Backend() {
  // TODO(@name@): resolve the engine mode HERE, once the native engine exists:
  //
  //   const bool native_available = @Name@Engine::device_available();
  //   mode_ = cpu::resolve_engine_mode("@name@", cfg, native_available);
  //
  // resolve_engine_mode() is the SHARED policy — Auto falls back to Onnx loudly,
  // an explicit "native" with no artefact is a hard error, and every vendor gets
  // the same answer on the corner that matters.  Do not re-invent it here.
  //
  // It takes a BackendConfig, which the constructor does not have; the two ways
  // out are (a) read TURBO_ENGINE_MODE in the factory below and construct with
  // the resolved mode, or (b) resolve in load_stages() and accept that caps()
  // was already snapshotted.  (a) is correct — see the note on mode_.
}

@Name@Backend::~@Name@Backend() = default;

backend::BackendCaps @Name@Backend::caps() const {
  backend::BackendCaps c;
  c.name = "@name@";

  // WHICH PATH WE ACTUALLY CAME UP ON.  An Auto run that fell back from native
  // to onnx must SAY onnx: /capabilities and the Python info() read this, and an
  // operator debugging "why is my ultra engine slow" must not be told it is
  // running when it is not.
  c.mode = mode_;
  c.has_native_engine = false; // TODO(@name@): true once @Name@Engine exists
  c.has_onnx_engine = true;    // the .onnx through this vendor's ORT provider

  // The ONNX path is host pre/post on cv::Mat — not device-resident, and with no
  // async device queue to overlap against.  Claiming otherwise makes the shared
  // pipeline schedule overlap that never happens.
  c.device = native_device_() ? backend::DeviceKind::@DeviceKind@
                              : backend::DeviceKind::Host;
  c.async = native_device_();
  c.native_image_decode = false; // TODO(@name@): true with an on-device decoder
  c.supports_batch = false;      // TODO(@name@): true once run_batch() is native

  // POLICY hint, not a capability: how many images this device WANTS coalesced
  // into one detection submission.  The hard ceiling is
  // IDetector::max_batch_size(); the shared batcher takes the smaller.  1 means
  // "do not coalesce" and keeps today's batch-1 detection exactly.
  c.preferred_batch_size = 1;

  // Memory-tier decision, not a core-count one: each entry is a full model set
  // plus its scratch.  Size it from real device memory once you have a device.
  c.recommended_pool_size = pool_size_ > 0 ? pool_size_ : 4;

  // IMPLEMENTED: what this backend+mode could EVER build given the right models
  // — the axis an operator CANNOT fix by configuration.  Default all() is right
  // for the shared ONNX path.  Narrow it (c.implemented.set(id, false)) only for
  // a stage your NATIVE path structurally cannot build, so /capabilities can say
  // "unsupported" instead of sending an operator hunting for a model path.
  return c;
}

// --- Low-level device factories ---------------------------------------------
// All four follow native_device_().  See the note in the header for why mixing
// a device queue with host stages is a wrong-output bug, not a slow one.

std::unique_ptr<backend::DeviceQueue> @Name@Backend::make_queue() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostDeviceQueue>();
  // TODO(@name@): return std::make_unique<@Name@DeviceQueue>();
  return std::make_unique<turbo_ocr::cpu::HostDeviceQueue>();
}

std::shared_ptr<backend::IDeviceAllocator> @Name@Backend::allocator() {
  if (!host_alloc_)
    host_alloc_ = std::make_shared<turbo_ocr::cpu::HostAllocator>();
  // TODO(@name@): return the shared @Name@Allocator when native_device_().
  // Override IDeviceAllocator::host_coherent() if this device's pointers ARE
  // host-dereferenceable (a UMA part) — the shared layer branches on that
  // method, never on the DeviceKind.
  return host_alloc_;
}

std::unique_ptr<backend::IKernels> @Name@Backend::make_kernels() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostKernels>();
  // TODO(@name@): return std::make_unique<@Name@Kernels>(...).  Move ops onto
  // the device ONE AT A TIME and report each honestly in KernelCaps; an op that
  // is still a host fallback is correct and visible, an op that silently no-ops
  // produces zero boxes at full inference cost (that is exactly what Intel's
  // no-SYCL build did before it delegated to HostKernels).
  return std::make_unique<turbo_ocr::cpu::HostKernels>();
}

std::unique_ptr<backend::IEngine> @Name@Backend::make_engine() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::CpuEngineAdapter>();
  // TODO(@name@): return std::make_unique<@Name@Engine>(...).  Answer
  // EngineCaps honestly up front — io_space, async, caller_owns_outputs,
  // multi_io, dynamic_shapes, thread_safe_concurrent — because callers branch on
  // it instead of assuming a memory/stream/ownership model.
  return std::make_unique<turbo_ocr::cpu::CpuEngineAdapter>();
}

// --- Table / formula: registry dispatch, never a private table ---------------

std::unique_ptr<backend::ITableRecognizer>
@Name@Backend::make_table_recognizer(const backend_routing::BackendSpec &spec) {
  // VLM/OpenAI specs are device-independent — straight to the shared factory.
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_table_recognizer(spec);
  // A LOCAL spec must resolve to the CUDA-free ORT structure backend EXPLICITLY,
  // exactly as CpuBackend/AppleBackend/IntelBackend do.  Handing a local spec to
  // the common registry asks for the CUDA-tied sibling, which a CPU-configured
  // build never compiles: the factory returns null and the server ABORTS AT BOOT
  // rather than serve without tables.
  if (spec.engine.empty() || spec.engine == "slanext")
    return std::make_unique<turbo_ocr::cpu::CpuTableRecognizer>();
  return backend::make_table_recognizer(spec);
}

std::unique_ptr<backend::IFormulaRecognizer>
@Name@Backend::make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_formula_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "ppformulanet_s")
    return std::make_unique<turbo_ocr::cpu::CpuFormulaRecognizerAdapter>();
  return backend::make_formula_recognizer(spec);
}

// --- Stage bootstrap ---------------------------------------------------------

backend::StageSet @Name@Backend::load_stages(const backend::BackendConfig &cfg) {
  // THE SHARED FAST PATH.  make_vendor_onnx_stages() builds det/cls/rec/layout +
  // doc-orientation on engine::CpuEngine over this vendor's execution provider
  // (backend::onnx_provider_for("@name@"), overridable with TURBO_EP_PROVIDER).
  // NVIDIA-on-CUDA-EP, Intel-on-OpenVINO, Apple-on-CoreML and AMD-on-MIGraphX are
  // all THIS call with a different provider string.  Do not fork it.
  //
  // TODO(@name@): when the native path exists, branch here —
  //   if (native_device_()) return load_native_stages_(cfg);
  // and keep this call as the Onnx arm.  Both arms must fill StageAvailability
  // truthfully: detector/recognizer are REQUIRED (the server refuses to start
  // without them, src/service/server/unified/backend_stages.cpp:108), everything else is opt-in
  // and reported through the optional CapabilityMask.
  auto built = turbo_ocr::cpu::make_vendor_onnx_stages("@name@", cfg);
  doc_ori_ = std::move(built.doc_ori);
  if (cfg.pool_size > 0)
    pool_size_ = cfg.pool_size;
  return std::move(built.stages);
}

// --- Service-boundary functions ----------------------------------------------

server::ImageDecoder @Name@Backend::make_image_decoder() {
  // Host JPEG/PNG decode.  TODO(@name@): swap in the vendor decoder (nvJPEG /
  // vImage / VAAPI / rocJPEG) and flip caps().native_image_decode.  For
  // device-RESIDENT decode into an ImageView the pipeline uses
  // IKernels::decode_image instead — this one deliberately returns a host
  // cv::Mat, which is what the routes consume.
  return [](const unsigned char *data, std::size_t len) -> cv::Mat {
    return decode::decode_cpu_fallback(data, len);
  };
}

server::OrientFunc @Name@Backend::make_orient_func() {
  // Capture THIS BACKEND, never the raw model pointer.  build_backend_runtime
  // calls load_stages() once PER POOL ENTRY and each call REPLACES doc_ori_,
  // destroying the object an earlier entry's closure captured — and
  // UnifiedOcrPipeline's constructor calls make_orient_func() for every entry.
  // A raw-pointer capture therefore leaves every entry but the last holding a
  // dangling pointer.  The backend outlives every pipeline and route.
  if (!doc_ori_)
    return {}; // empty == autorotate off; NEVER a closure that always answers 0
  return [this](const cv::Mat &page) -> int {
    return doc_ori_ ? doc_ori_->detect(page) : 0;
  };
}

std::unique_ptr<backend::Backend> make_@name@_backend() {
  // TODO(@name@): probe for the device and return nullptr when it is absent —
  // that is the registry's "compiled in but not usable on this machine" signal,
  // and it is what lets auto-detect walk down to the next backend instead of
  // booting onto one whose engine will fail to load.
  return std::make_unique<@Name@Backend>();
}

} // namespace turbo_ocr::@name@
'''

REGISTRY_TMPL = r'''// @name@_backend_registry.cpp — registers "@name@" into the ONE shared link-time
// registry (src/backend/backend_registry.cpp).
//
// PURE REGISTRATION.  This TU must NOT define backend::make_backend or
// backend::available_backends: those are defined once, in the shared layer, and
// a per-vendor definition is why only one backend could ever be linked into a
// binary.  All this file contributes is one namespace-scope BackendRegistrar
// whose constructor runs at static-init time.
//
// PULL-IN CONTRACT: this TU defines no symbol anybody references, so a plain
// static archive is entitled to drop the whole object and the backend silently
// vanishes from available_backends().  The CMake target must be force-linked —
// turbo_link_backends() applies $<LINK_LIBRARY:WHOLE_ARCHIVE,...>.  The
// regression test for exactly this is `turbo_backend_probe --list`.

#include <memory>

#include "@name@/backend/@name@_backend.h"
#include "turbo_ocr/backend/backend_registry.h"

namespace turbo_ocr::@name@ {
namespace {

std::unique_ptr<backend::Backend> make_@name@_backend_entry() {
  return make_@name@_backend(); // nullptr when there is no @name@ device here
}

// AUTO-DETECT PRIORITY.  A new, unmeasured backend must NOT win the auto slot:
// on any machine that compiles it in, auto-detect would silently prefer it over
// a backend that is known to be faster, and the only startup line afterwards
// names whoever won.  Intel sits below cpu for exactly this reason (measured
// 4.3 vs 8.8 img/s), and it says so in a comment with the number.
//
// `--backend @name@` still selects this explicitly at any priority.
//
// TODO(@name@): once turbo_bench shows this beats the backends above it ON THIS
// HARDWARE, raise this to the vendor's slot in backend_registry.h
// (kBackendPriorityNvidia/Amd/Apple/Intel) and record the measurement here.
constexpr int k@Name@AutoPriority = backend::kBackendPriorityCpu - 1;

// The aliases are extra spellings accepted by --backend / TURBO_BACKEND.
// TODO(@name@): add the ones an operator would reach for first (a vendor name,
// an SDK name), the way cpu accepts "host" and amd accepts "rocm"/"hip".
const backend::BackendRegistrar g_@name@_registrar{
    "@name@", {}, k@Name@AutoPriority, &make_@name@_backend_entry};

} // namespace
} // namespace turbo_ocr::@name@
'''

README_TMPL = r'''# TurboOCR — @name@ backend

> **STATUS: SCAFFOLD.** Generated by `tools/new_backend.py`. Nothing here has
> been run on @name@ hardware. Delete this block when that stops being true, and
> replace it with what IS verified and what is not — see
> `src/backends/intel/README.md` "What is actually verified" for the shape.

The @name@ vendor arm of the multi-backend seam
(`include/turbo_ocr/backend/*.h`). Everything above the seam — the one
`UnifiedOcrPipeline`, box routing, CTC decode, DB post-process, reading order,
document assembly, table/formula dispatch, capability policy — is **shared** and
is not in this directory.

## Files

Every vendor directory under `src/backends/` uses the SAME per-concern layout, so
"where is the allocator" has one answer across all of them. The scaffold starts
with only `backend/`; the other directories appear as you implement them, and
they go by concern, never as a flat dump at the vendor root.

```
src/backends/@name@/
├── backend/@name@_backend.{h,cpp}        Backend: factories + load_stages + service fns
├── backend/@name@_backend_registry.cpp   one BackendRegistrar; needs WHOLE_ARCHIVE
└── README.md                             this file

  ... and, as the device path is built out (see "Bring-up order"):

├── memory/     IDeviceAllocator
├── queue/      DeviceQueue / DeviceEvent
├── engine/     IEngine (the vendor's model runner)
├── kernels_@name@/  IKernels + the device kernel sources (model this on
│                   amd/kernels_hip/, NOT on nvidia/kernels_cuda/ — NVIDIA's
│                   .cu files sit at src/backends/nvidia/kernels_cuda/ because they
│                   have consumers outside src/backends/; yours will not)
├── stages/     IDetector / IRecognizer / IClassifier / ILayout
├── support/    used by >=2 of the above, implements no seam interface itself
└── probes/     standalone probes (see intel/probes/ov_engine_probe.cpp)
```

Sibling headers are included through the vendor-rooted path off
`-Isrc/backends`, e.g. `#include "@name@/memory/@name@_allocator.h"` — never a
bare `"@name@_allocator.h"`, which breaks the moment a file moves.

## What it does today

Every seam factory delegates to the SHARED host implementation and
`load_stages()` calls `cpu::make_vendor_onnx_stages("@name@", cfg)` — the .onnx
through this vendor's ONNX Runtime execution provider, with host pre/post. That
is a correct, complete backend. It is also the reference the device path is
diffed against.

## Bring-up order

1. `IDeviceAllocator` + `DeviceQueue` — device memory and one ordered lane.
2. `IEngine` — the vendor's model runner. Answer `EngineCaps` honestly.
3. `IKernels` — one op at a time; `KernelCaps` reports which are native.
4. The four stage classes — resident det/rec/cls/layout.

After each step: `tools/syntax_shims/check.sh`, then
`turbo_golden --backend @name@ --ref cpu --stage all`, then `turbo_conformance`,
then `turbo_bench`. Details and the exact commands:
**`docs/contributing/adding-a-backend.md`**.

## What must NOT live here

Detection resize policy and DB thresholds, rec width buckets and the batch
ladder, CTC decode, the character dictionary, cls geometry and its threshold,
normalization constants, crop geometry, the PicoDet row decode, the layout
postfilter, SLANeXt/formula path resolution, and the whole
det→cls→rec→layout→router flow. Every one of those is a shared header this
directory CALLS. See `docs/contributing/adding-a-backend.md` §2 for the list, the header
that owns each, and the bug each one is a scar from.
'''


def camel(name: str) -> str:
    """foo_bar -> FooBar; foo -> Foo."""
    return "".join(p[:1].upper() + p[1:] for p in name.split("_") if p)


def render(tmpl: str, name: str, device_kind: str) -> str:
    return (
        tmpl.replace("@Name@", camel(name))
        .replace("@NAME@", name.upper())
        .replace("@name@", name)
        .replace("@DeviceKind@", device_kind)
    )


def files_for(name: str, device_kind: str):
    """Ordered mapping of relative path -> rendered content.

    The keys carry the per-concern subdirectory (`backend/...`) that every vendor
    under src/backends/ uses. Emitting a flat vendor directory is how the tree got
    to 27 files at one level; the generator is the place that has to stop it.
    """
    return {
        f"backend/{name}_backend.h": render(HEADER_TMPL, name, device_kind),
        f"backend/{name}_backend.cpp": render(SOURCE_TMPL, name, device_kind),
        f"backend/{name}_backend_registry.cpp": render(REGISTRY_TMPL, name, device_kind),
        "README.md": render(README_TMPL, name, device_kind),
    }


CMAKE_BLOCK = r'''    # -----------------------------------------------------------------------
    # turbo_ocr_backend_@name@
    # -----------------------------------------------------------------------
    if(@name@ IN_LIST TURBO_BACKENDS)
        add_library(turbo_ocr_backend_@name@ STATIC
            src/backends/@name@/backend/@name@_backend.cpp
            src/backends/@name@/backend/@name@_backend_registry.cpp   # <- registrar; needs WHOLE_ARCHIVE
        )
        target_include_directories(turbo_ocr_backend_@name@ PUBLIC "${CMAKE_SOURCE_DIR}/src/backends")
        target_link_libraries(turbo_ocr_backend_@name@ PUBLIC
            turbo_ocr_pipeline
            turbo_ocr_backend_onnx      # THE shared fast path (host pre/post + ORT EP)
            turbo_ocr_host_kernels      # SHARED host fallback ops
            ${OpenCV_LIBS}
        )
        target_compile_options(turbo_ocr_backend_@name@ PRIVATE ${TURBO_BACKEND_CXX_FLAGS})
        add_library(turbo_ocr::backend_@name@ ALIAS turbo_ocr_backend_@name@)
    endif()
'''


def print_cmake_instructions(name: str, out_rel: str) -> None:
    n = name
    bar = "=" * 72
    print(f"\n{bar}\nCMakeLists.txt — THREE edits, in this order\n{bar}")

    print(
        f"""
[1/3]  Register the name.  Search CMakeLists.txt for _turbo_known_backends:

           set(_turbo_known_backends cpu apple nvidia amd intel)
       becomes
           set(_turbo_known_backends cpu apple nvidia amd intel {n})

       Without this, -DTURBO_BACKENDS="cpu;{n}" FATAL_ERRORs with
       "TURBO_BACKENDS contains unknown backend '{n}'".
"""
    )

    print(
        f"""[2/3]  Let this backend pull in the shared ONNX stage set.  Search
       CMakeLists.txt for TURBO_NEEDS_HOST_STAGES and add "{n}" to the vendor
       list in its foreach:

           foreach(_b cpu apple intel nvidia amd)
       becomes
           foreach(_b cpu apple intel nvidia amd {n})

       That guard builds turbo_ocr_backend_onnx, which the generated
       load_stages() links against.  Skip it and you get undefined references to
       turbo_ocr::cpu::make_vendor_onnx_stages.
"""
    )

    print(
        "[3/3]  Add the target.  Paste this AFTER the turbo_ocr_backend_cpu block\n"
        "       (search CMakeLists.txt for `add_library(turbo_ocr_backend_cpu`)\n"
        "       so it is inside the `if(TURBO_BACKENDS)` scope where\n"
        "       TURBO_BACKEND_CXX_FLAGS and turbo_ocr_host_kernels are already\n"
        "       defined:\n"
    )
    print(render(CMAKE_BLOCK, name, "Host"))

    print(
        f"""Nothing else is needed:
  * turbo_link_backends() already loops over TURBO_BACKENDS and force-links
    turbo_ocr_backend_${{_b}} with $<LINK_LIBRARY:WHOLE_ARCHIVE,...>, so
    turbo_bench / turbo_conformance / turbo_golden / turbo_backend_probe /
    _turboocr pick "{n}" up automatically;
  * the ctest golden diffs golden_{n}_{{det,cls,rec}} register themselves for
    every non-cpu backend in TURBO_BACKENDS once -DTURBO_FUNSD_CACHE is set;
  * a FUNSD accuracy GATE is NOT automatic — add a row to _turbo_gates
    (search for _turbo_gates) once you have a measured floor:
        "{n}_tiny|{n}|tiny|<measured F1 minus a hair>"

{bar}
CONFIGURE + FIRST CHECK
{bar}

    cmake -B build-{n} -S . -G Ninja \\
          -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON -DFETCH_MODELS=OFF \\
          -DTURBO_BACKENDS="cpu;{n}" \\
          -DTURBO_FUNSD_CACHE=$HOME/compare-ocrs/funsd_cache
    ninja -C build-{n} turbo_backend_probe turbo_bench turbo_conformance turbo_golden

    ./build-{n}/turbo_backend_probe --list        # MUST list cpu AND {n}

If "{n}" is missing from --list, the target linked but the registrar was dropped:
check that the registry TU is in the add_library() source list and that
turbo_link_backends() saw the name.

Then, in order (docs/contributing/adding-a-backend.md §7):

    tools/syntax_shims/check.sh src/backends/{n}/backend/{n}_backend.cpp
    ./build-{n}/turbo_conformance --images $HOME/compare-ocrs/funsd_cache --count 20
    ./build-{n}/turbo_golden --backend {n} --ref cpu --stage all \\
        --images $HOME/compare-ocrs/funsd_cache --count 10
    ./build-{n}/turbo_bench --backend {n} --tier tiny \\
        --images $HOME/compare-ocrs/funsd_cache --count 50 --threads 8 --repeat 40 \\
        --words /tmp/{n}.words.json --out /tmp/{n}.metrics.json

Sources generated in: {out_rel}
Guide: docs/contributing/adding-a-backend.md
"""
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="new_backend.py",
        description="Generate a compiling skeleton for a new TurboOCR vendor backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Read docs/contributing/adding-a-backend.md first — especially the shared-policy rule.",
    )
    ap.add_argument(
        "--name",
        required=True,
        help="canonical backend id, lowercase (the --backend value, the "
             "TURBO_BACKENDS entry, the directory name)",
    )
    ap.add_argument(
        "--device-kind",
        default="Host",
        help="backend::DeviceKind enumerator this backend allocates in when "
             "NATIVE (%s). Default Host — the DeviceKind enum is closed, and "
             "adding one is a shared edit to include/turbo_ocr/backend/image_view.h."
             % "|".join(KNOWN_DEVICE_KINDS),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output directory (default: src/backends/<name>). The generated "
             "#include uses \"<name>/<name>_backend.h\", so the PARENT of this "
             "directory must be on the include path — src/backends is, via "
             "target_include_directories.",
    )
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that exist and differ")
    ap.add_argument("--dry-run", action="store_true",
                    help="write nothing; print what would change and the CMake lines")
    args = ap.parse_args(argv)

    name = args.name.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        ap.error(
            f"--name {name!r} must be lowercase [a-z][a-z0-9_]* — it becomes a C++ "
            "namespace, a directory name, a CMake target suffix and the --backend value"
        )
    reserved = ("cpu", "apple", "nvidia", "amd", "intel", "backend", "host")
    if name in reserved:
        ap.error(f"--name {name!r} is already taken (existing backend or reserved word)")

    device_kind = args.device_kind
    if device_kind not in KNOWN_DEVICE_KINDS:
        ap.error(
            f"--device-kind {device_kind!r} is not a backend::DeviceKind enumerator. "
            f"Known: {', '.join(KNOWN_DEVICE_KINDS)}. A genuinely new device space "
            "needs an enumerator added to include/turbo_ocr/backend/image_view.h "
            "plus device_is_host_coherent() and device_kind_name() — a SHARED edit. "
            "Until then generate with Host: the skeleton runs host-delegated anyway."
        )

    out = args.out or os.path.join("src", "backends", name)
    out = os.path.abspath(out)
    out_rel = os.path.relpath(out, os.getcwd())
    if out_rel.startswith(".."):
        out_rel = out # outside the repo: a ../../.. chain reads as noise

    files = files_for(name, device_kind)

    # Idempotence: a re-run that would produce byte-identical files is a no-op and
    # succeeds. Only a DIFFERING existing file needs --force, so the script can be
    # re-run safely (in a Makefile, in CI, after a --dry-run) without a flag.
    existing = os.path.isdir(out)
    same, differing, missing = [], [], []
    for fn, content in files.items():
        path = os.path.join(out, fn)
        if not os.path.exists(path):
            missing.append(fn)
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                same.append(fn) if fh.read() == content else differing.append(fn)
        except OSError as e:
            print(f"error: cannot read {path}: {e}", file=sys.stderr)
            return 1

    if differing and not args.force and not args.dry_run:
        print(
            f"error: {out_rel} already has {len(differing)} file(s) that differ from "
            f"the template: {', '.join(differing)}\n"
            "       Refusing to overwrite hand-written code. Re-run with --force to "
            "replace them, or --dry-run to see the generated content first.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(f"[dry-run] would write to {out_rel}/")
        for fn in files:
            state = ("unchanged" if fn in same
                     else "OVERWRITE" if fn in differing
                     else "create")
            print(f"[dry-run]   {state:>9}  {fn}  ({len(files[fn])} bytes)")
        if differing and not args.force:
            print("[dry-run] NOTE: the OVERWRITE files need --force.")
        print_cmake_instructions(name, out_rel)
        return 0

    if not existing:
        os.makedirs(out, exist_ok=True)

    wrote = 0
    for fn, content in files.items():
        if fn in same:
            continue
        path = os.path.join(out, fn)
        os.makedirs(os.path.dirname(path), exist_ok=True)  # fn carries a subdir
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        wrote += 1
        print(f"  {'overwrote' if fn in differing else 'wrote'}  {out_rel}/{fn}")

    if wrote == 0:
        print(f"{out_rel}/ is already up to date ({len(same)} files unchanged).")

    print_cmake_instructions(name, out_rel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
