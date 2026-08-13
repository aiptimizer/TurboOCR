// IntelBackend — Intel vendor Backend adapter.
//
// The factory plumbing here is toolchain-agnostic; every device-specific piece
// lives behind L0Allocator / SyclKernels / OpenVINOEngine, so this file is
// syntax-checkable without oneAPI or OpenVINO.

#include "intel/backend/intel_backend.h"
#include "intel/kernels_sycl/sycl_kernels.h"

#if !defined(TURBO_OCR_HAS_SYCL)
// Built without DPC++: SyclKernels' native ops (resize_normalize, warp_crops,
// threshold, argmax, preprocess_region) compile to NO-OPS. Running the pipeline
// on those silently feeds the engine an untouched buffer — detection then emits
// ZERO boxes at full inference cost, which is exactly what an end-to-end F1 of
// 0.00% at 31 img/s looked like before this was tracked down.
//
// The fix is NOT an Intel-local host implementation of those five ops. The CPU
// backend's HostKernels already implements every one of them against the same
// seam, with OpenCV, and already delegates db_postprocess to the same shared
// detection::extract_boxes_from_bitmap. Per the deduplication rule (generic
// policy is SHARED; a fix must not be written once per backend), the no-SYCL
// Intel build uses THAT implementation. The win is practical, not just tidy:
// the OpenVINO engine — where the speed lives — becomes usable with no oneAPI
// toolchain at all.
#include "cpu/kernels_host/host_kernels.h"
#endif
#include "intel/queue/l0_device_queue.h"
#include "intel/stages/intel_stages.h"

#include "turbo_ocr/base/env_utils.h"

#include <algorithm>
#include <memory>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cpu/stages/cpu_stages.h"
#include "cpu/queue/host_device_queue.h"
#include "cpu/memory/host_allocator.h"
#include "cpu/engine/cpu_engine_adapter.h"
#include "cpu/stages/cpu_table_recognizer.h"
#include "cpu/stages/cpu_formula_recognizer.h"
#include "turbo_ocr/backend/formula_recognizer.h" // backend::make_formula_recognizer
#include "turbo_ocr/backend/table_recognizer.h"   // backend::make_table_recognizer
#include "turbo_ocr/backend/routing_config.h" // BackendSpec, Kind
#include "turbo_ocr/backend/stage_device.h"   // StageKind, <STAGE>_DEVICE
#include "turbo_ocr/base/log/logger.h"        // TOCR_LOG_*

namespace turbo_ocr::intel {

struct IntelBackend::Impl {
  OpenVINOEngine::DeviceType device;
  std::shared_ptr<L0Allocator> alloc;
  int pool_size = 0;
  // Which path load_stages() settled on (backend/engine_mode.h). Native =
  // OpenVINO compiled models through OpenVINOEngine; Onnx = the .onnx through
  // the OpenVINO EXECUTION PROVIDER (precision=FP16, no compiled blob), which
  // is the shared host stage set every vendor uses.
  backend::EngineMode mode = backend::EngineMode::Native;
  std::shared_ptr<backend::IDeviceAllocator> host_alloc;
  std::unique_ptr<classification::OrtDocOrientation> doc_ori_onnx;
  [[nodiscard]] bool native_device() const noexcept {
    return mode == backend::EngineMode::Native;
  }

  // THE device predicate — one definition, used by caps() AND by everything
  // that hands out a device-shaped object (make_queue / allocator /
  // make_kernels). Splitting it is what produced the bug this exists to stop:
  // caps() required all three conditions while make_queue() checked only
  // native_device(), so on a build with the OpenVINO plugin but no SYCL/L0 USM
  // context (Windows, SYCL kernels OFF) caps() reported device=Host while
  // make_queue() returned an ASYNC L0DeviceQueue. Callers that key off
  // is_async() then took the copy-then-sync staging branch over memory that was
  // already host-readable. Caught by tests/cpp/backends/test_backend_caps.cpp
  // "a Host-device backend's queue is never async" on real Intel hardware.
  //
  // All three are required: the mode must have resolved to native (the OpenVINO
  // plugin came up), the device must not be plain CPU, and the allocator must
  // actually have a USM context — has_device() answers "is zero-copy
  // available", which is a different question from "did the plugin load"
  // (intel_backend_registry.cpp: "Losing L0 costs zero-copy, not the device").
  [[nodiscard]] bool device_path() const noexcept {
    return native_device() && device != OpenVINOEngine::DeviceType::CPU &&
           alloc && alloc->has_device();
  }

  explicit Impl(OpenVINOEngine::DeviceType d)
      : device(d), alloc(std::make_shared<L0Allocator>(/*device_id=*/-1)) {}

  // Kernels for this build. With DPC++ the native SYCL ops run on the Intel
  // device; without it the SHARED host implementation runs, so the backend is
  // correct either way rather than silently no-op.
  [[nodiscard]] std::unique_ptr<backend::IKernels> make_kernels_impl() {
#if defined(TURBO_OCR_HAS_SYCL)
    return std::make_unique<SyclKernels>(alloc);
#else
    return std::make_unique<turbo_ocr::cpu::HostKernels>();
#endif
  }

  // One StageDeps bundle: this stage's own kernels + engine over the SHARED
  // allocator. Each stage gets its own OpenVINOEngine because an
  // ov::InferRequest is not thread-safe (EngineCaps::thread_safe_concurrent).
  // Resolve THIS stage's device: <STAGE>_DEVICE if the operator set one, else
  // the backend-wide device already resolved from OV_DEVICE. An unparseable or
  // unavailable value falls back rather than failing — a typo in DET_DEVICE
  // should cost the placement, not the server.
  [[nodiscard]] OpenVINOEngine::DeviceType
  device_for(backend::StageKind k) const {
    const std::string want = backend::stage_device_override(k);
    if (want.empty()) return device;

    OpenVINOEngine::DeviceType d = device;
    if (want == "CPU")      d = OpenVINOEngine::DeviceType::CPU;
    else if (want == "GPU") d = OpenVINOEngine::DeviceType::GPU;
    else if (want == "NPU") d = OpenVINOEngine::DeviceType::NPU;
    else {
      TOCR_LOG_WARN("stage device override not understood; keeping the backend "
                    "default", "stage", backend::stage_kind_name(k).c_str(),
                    "value", want.c_str());
      return device;
    }
    // Availability is checked here, not at first inference: a stage pinned to a
    // device the runtime does not enumerate would otherwise fail deep inside
    // load() with an OpenVINO message that never mentions the override.
    if (!OpenVINOEngine::device_available(d)) {
      TOCR_LOG_WARN("stage device override unavailable; keeping the backend "
                    "default", "stage", backend::stage_kind_name(k).c_str(),
                    "value", want.c_str());
      return device;
    }
    TOCR_LOG_INFO("stage device override",
                  "stage", backend::stage_kind_name(k).c_str(),
                  "device", want.c_str());
    return d;
  }

  [[nodiscard]] StageDeps make_deps(backend::StageKind k) {
    StageDeps d;
    d.alloc = alloc;
    d.kernels = make_kernels_impl();
    d.engine = std::make_unique<OpenVINOEngine>(device_for(k), alloc);
    return d;
  }
};

IntelBackend::IntelBackend(OpenVINOEngine::DeviceType device)
    : impl_(std::make_unique<Impl>(device)) {}
IntelBackend::~IntelBackend() = default;

backend::BackendCaps IntelBackend::caps() const {
  const auto &I = *impl_;
  backend::BackendCaps c;
  // native_device() is the SAME signal load_stages() resolved the mode from —
  // NOT a bare alloc->has_device(). The two can disagree (has_device answers
  // "is there a SYCL/L0 USM context for zero-copy", not "did the OpenVINO
  // plugin come up"; intel_backend_registry.cpp documents the split as
  // "Losing L0 costs zero-copy, not the device"). Testing has_device alone
  // reported device=L0/async=true while make_queue/allocator/make_kernels
  // were all handing out Host implementations — the pipeline then staged
  // through the device ring over host memory, the exact AppleBackend bug the
  // downgrade block below exists to prevent.
  const bool device_path = I.device_path();
  c.device = device_path ? backend::DeviceKind::L0 : backend::DeviceKind::Host;
  c.name = "intel";
  c.native_image_decode = false; // host OpenCV decode (no VAAPI/oneVPL yet)
  // The engine is synchronous by construction (see OpenVINOEngine::run) but the
  // QUEUE genuinely is async — SYCL pre/post kernels return before completing —
  // so async follows the device, not the engine.
  c.async = device_path;
  c.supports_batch = true;
  // ONNX MODE MUST DOWNGRADE — the honesty contract every other arm carries
  // (rocm_backend.cpp / cuda_backend.cpp: "an Auto run that fell back to onnx
  // must SAY onnx; the rule is shared, so it belongs on every arm"). This arm
  // never set c.mode at all, so /capabilities reported the default whatever
  // load_stages() had actually resolved.
  c.mode = I.mode;
  if (!I.native_device()) {
    c.device = backend::DeviceKind::Host;
    c.async = false;
    c.supports_batch = false;
  }

  // Pool sizing. Each entry costs one OpenVINOEngine per stage plus its USM
  // scratch, so this is a memory-tier decision, not a core-count one:
  //   * iGPU (UMA): the GPU shares system RAM and one media/compute slice —
  //     more entries mostly add contention. Default 1.
  //   * Arc dGPU:   dedicated VRAM, so 2-4 entries overlap host and device work.
  //   * CPU plugin: OpenVINO already threads internally; oversubscribing pools
  //     is counterproductive, so cap well below hardware_concurrency.
  // TURBO_POOL_SIZE (or BackendConfig::pool_size) overrides. The default below
  // is a REASONED GUESS, not a measurement — there is no Intel GPU here. Sizing
  // it properly from ov::device::capabilities / global memory is a bring-up item.
  int pool = 1;
  if (I.device == OpenVINOEngine::DeviceType::CPU) {
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    pool = std::clamp(hw > 0 ? hw / 4 : 1, 1, 4);
  } else if (I.device == OpenVINOEngine::DeviceType::GPU) {
    pool = 2;
  }
  pool = env::env_int("TURBO_POOL_SIZE", pool, 1, 1024);
  c.recommended_pool_size = I.pool_size > 0 ? I.pool_size : pool;
  // OpenVINO IS this arm's native graph engine; the ORT path (with or without
  // the OpenVINO EP) is the fallback. has_native_engine reports whether the
  // native engine is the one ACTUALLY RUNNING (mode), matching the other arms
  // — a build where OpenVINO failed to come up must not advertise it.
  c.has_native_engine = I.native_device();
  c.has_onnx_engine = true;
  return c;
}

// In onnx mode the stages are the shared HOST ones, so the device side must be
// the host side too — a device queue/allocator would hand host code a device
// pointer (see the Apple backend for the abort this caused).
std::unique_ptr<backend::DeviceQueue> IntelBackend::make_queue() {
  // device_path(), not native_device(): must match caps().device exactly.
  if (!impl_->device_path())
    return std::make_unique<turbo_ocr::cpu::HostDeviceQueue>();
  return std::make_unique<L0DeviceQueue>(/*device_id=*/-1);
}
std::shared_ptr<backend::IDeviceAllocator> IntelBackend::allocator() {
  // device_path(), not native_device(): a caller that got DeviceKind::Host from
  // caps() must not be handed a device allocator.
  if (!impl_->device_path()) {
    if (!impl_->host_alloc)
      impl_->host_alloc = std::make_shared<turbo_ocr::cpu::HostAllocator>();
    return impl_->host_alloc;
  }
  return impl_->alloc;
}
std::unique_ptr<backend::IKernels> IntelBackend::make_kernels() {
  // device_path(): SyclKernels operate on USM buffers from `alloc`, so they are
  // only usable when that allocator actually has a device — the same condition
  // caps() reports as DeviceKind::L0.
  if (!impl_->device_path())
    return std::make_unique<turbo_ocr::cpu::HostKernels>();
  return impl_->make_kernels_impl();
}
std::unique_ptr<backend::IEngine> IntelBackend::make_engine() {
  // native_device(), NOT device_path() — and the difference is deliberate.
  // Inference and device MEMORY are separate capabilities: OpenVINO runs models
  // perfectly well on a device with no L0/USM context for zero-copy (that is
  // exactly the Windows-without-SYCL case). Gating the engine on device_path()
  // would drop the whole OpenVINO path — including the NPU — for the sake of an
  // allocator the engine does not use.
  if (!impl_->native_device())
    return std::make_unique<turbo_ocr::cpu::CpuEngineAdapter>();
  return std::make_unique<OpenVINOEngine>(impl_->device, impl_->alloc);
}

std::unique_ptr<backend::ITableRecognizer>
IntelBackend::make_table_recognizer(const backend_routing::BackendSpec &spec) {
  // DEDUP: no Intel-private dispatch. VLM/OpenAI specs are device-independent
  // and local specs currently resolve to the shared portable structure backend.
  // A genuinely device-resident SLANeXt (encoder on this OpenVINOEngine, fused
  // preproc on SyclKernels) would be a legitimate vendor class — it is a
  // follow-up.
  //
  // Until then the LOCAL case must resolve to the CUDA-free ORT structure
  // backend explicitly, exactly as CpuBackend and AppleBackend do. Handing a
  // local spec to the common registry asks for the CUDA-tied sibling, which a
  // CPU-configured build never compiles: the factory returns null and the
  // server ABORTS AT BOOT rather than serve without tables.
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_table_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "slanext")
    return std::make_unique<turbo_ocr::cpu::CpuTableRecognizer>();
  return backend::make_table_recognizer(spec);
}

std::unique_ptr<backend::IFormulaRecognizer>
IntelBackend::make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  // Same reasoning as tables: a LOCAL spec must resolve to the CUDA-free ORT
  // PP-FormulaNet class explicitly, or a CPU-configured build gets a null
  // recognizer from the CUDA-tied registry entry and refuses to boot.
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_formula_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "ppformulanet_s")
    return std::make_unique<turbo_ocr::cpu::CpuFormulaRecognizerAdapter>();
  return backend::make_formula_recognizer(spec);
}

backend::StageSet IntelBackend::load_stages(const backend::BackendConfig &cfg) {
  auto &I = *impl_;

  // NATIVE here means "OpenVINOEngine can compile these models for the chosen
  // device". That needs the runtime to actually enumerate the device; without
  // it there is nothing to compile against, so Auto drops to the ONNX path
  // (which reaches the same silicon through the OpenVINO EP and needs no
  // compiled blob). UNVERIFIED: not compiled here (no OpenVINO on the dev Mac);
  // the shared policy it calls is exercised by the Apple ctest gates.
  const bool native_available = OpenVINOEngine::device_available(I.device);
  I.mode = turbo_ocr::cpu::resolve_engine_mode("intel", cfg, native_available);
  if (!I.native_device()) {
    auto built = turbo_ocr::cpu::make_vendor_onnx_stages("intel", cfg);
    I.doc_ori_onnx = std::move(built.doc_ori);
    if (cfg.pool_size > 0) I.pool_size = cfg.pool_size;
    return std::move(built.stages);
  }

  backend::StageSet set;

  // Detection + recognition are required; the rest are opt-in per model path.
  // Every load() below does its (width,batch) prebuild internally, so all
  // compilation happens HERE, at startup, and never during traffic.
  if (!cfg.det_model.empty()) {
    auto det = std::make_unique<IntelDetector>(I.make_deps(backend::StageKind::Detection));
    if (det->load(cfg.det_model)) {
      set.available.detector = true;
      set.detector = std::move(det);
    }
  }

  if (!cfg.rec_model.empty()) {
    auto rec = std::make_unique<IntelRecognizer>(I.make_deps(backend::StageKind::Recognition));
    bool ok = rec->load(cfg.rec_model);
    if (ok && !cfg.rec_dict.empty())
      ok = rec->load_dict(cfg.rec_dict);
    if (ok) {
      set.available.recognizer = true;
      set.recognizer = std::move(rec);
    }
  }

  if (!cfg.cls_model.empty()) {
    auto cls = std::make_unique<IntelClassifier>(I.make_deps(backend::StageKind::Classification));
    if (cls->load(cfg.cls_model)) {
      set.available.classifier = true;
      set.classifier = std::move(cls);
    }
  }

  if (cfg.want_layout && !cfg.layout_model.empty()) {
    auto lay = std::make_unique<IntelLayout>(I.make_deps(backend::StageKind::Layout));
    if (lay->load(cfg.layout_model)) {
      set.available.optional.set(capability::CapabilityId::Layout, true);
      set.layout = std::move(lay);
    }
  }

  // Page orientation is a service-boundary function, not a StageSet member; the
  // flag reports whether make_orient_func() will return a live callable.
  set.available.optional.set(capability::CapabilityId::DocOrientation, false);
  set.available.optional.set(capability::CapabilityId::Table, cfg.want_tables);
  set.available.optional.set(capability::CapabilityId::Formula, cfg.want_formulas);
  if (cfg.pool_size > 0)
    I.pool_size = cfg.pool_size;
  return set;
}

server::ImageDecoder IntelBackend::make_image_decoder() {
  // Host cv::Mat decode; there is no on-device decoder yet (caps()
  // .native_image_decode == false). For device-resident decode into an
  // ImageView the pipeline uses IKernels::decode_image instead.
  return [](const unsigned char *data, std::size_t len) -> cv::Mat {
    const cv::Mat enc(1, static_cast<int>(len), CV_8UC1,
                      const_cast<unsigned char *>(data));
    return cv::imdecode(enc, cv::IMREAD_COLOR);
  };
}

server::OrientFunc IntelBackend::make_orient_func() {
  // ONNX ("fast") mode loads a real doc-orientation model through the shared
  // stage factory — returning {} regardless would load it and then THROW IT
  // AWAY, leaving autorotate silently dead on a backend that can do it.
  // Capture THIS BACKEND, not the raw model pointer — load_stages() runs once
  // per pool entry and REPLACES the model, so a raw pointer captured by an
  // earlier pipeline's constructor dangles (see CpuBackend::make_orient_func).
  if (impl_->doc_ori_onnx) {
    return [this](const cv::Mat &page) -> int {
      return impl_->doc_ori_onnx ? impl_->doc_ori_onnx->detect(page) : 0;
    };
  }
  // NATIVE mode: page-orientation is NOT implemented on the OpenVINO engine.
  // Empty is the seam's documented way to say "autorotate off" — strictly
  // better than a closure that always answers 0 degrees, which would look like
  // a working detector that thinks every page is upright. Implementing it is
  // mechanical (224x224 ImageNet preprocess + 4-class argmax, structurally
  // identical to IntelClassifier); see README bring-up item 6.
  return {};
}

std::unique_ptr<backend::Backend>
make_intel_backend(OpenVINOEngine::DeviceType device) {
  return std::make_unique<IntelBackend>(device);
}

} // namespace turbo_ocr::intel
