#include "amd/backend/rocm_backend.h"
#include "cpu/engine/cpu_engine_adapter.h"
#include "cpu/kernels_host/host_kernels.h"
#include "cpu/memory/host_allocator.h"
#include "cpu/queue/host_device_queue.h"
#include "cpu/stages/cpu_stages.h"
#include "cpu/stages/cpu_table_recognizer.h"     // Local-spec table fallback
#include "cpu/stages/cpu_formula_recognizer.h"   // Local-spec formula fallback
#include "turbo_ocr/pipeline/pool_sizing.h"      // compute_pipeline_pool_size

#include "amd/support/hip_check.h"
#include "amd/engine/migraphx_engine.h"
#include "amd/kernels_hip/hip_kernels.h"
#include "amd/memory/hip_allocator.h"
#include "amd/queue/hip_queue.h"
#include "amd/stages/rocm_stages.h"

#include "turbo_ocr/backend/formula_recognizer.h"
#include "turbo_ocr/backend/routing_config.h" // BackendSpec (complete type for Kind/engine)
#include "turbo_ocr/backend/table_recognizer.h"
#include "turbo_ocr/base/log/logger.h"

#include <hip/hip_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace turbo_ocr::amd {

struct RocmBackend::Impl {
  int device_id = 0;
  std::shared_ptr<HipAllocator> alloc;
  bool have_device = false;

  explicit Impl(int dev) : device_id(dev) {
    int count = 0;
    if (hipGetDeviceCount(&count) == hipSuccess && count > 0) {
      have_device = true;
      hipSetDevice(device_id);
      alloc = std::make_shared<HipAllocator>(device_id);
    }
  }
};

RocmBackend::RocmBackend(int device_id)
    : p_(std::make_unique<Impl>(device_id)) {}
RocmBackend::~RocmBackend() = default;

backend::BackendCaps RocmBackend::caps() const {
  backend::BackendCaps c;
  c.device = backend::DeviceKind::Hip;
  c.name = "amd";
  c.native_image_decode = false; // no rocJPEG wired yet (host decode)
  c.async = true;
  c.supports_batch = true;
  // Pool sizing through the SHARED helper — the same call cuda_backend.cpp
  // makes. This used to be a hand-rolled 48/24/12 GB ladder that (a) disagreed
  // with the shared 14/12/8 tiers and (b) read hipMemGetInfo's FREE value and
  // then never used it — dropping exactly the free-VRAM safety floor that
  // exists because the flat tier once "duly died with 'CUDA Error … out of
  // memory' at startup". Arithmetic over two memory numbers is not
  // vendor-specific; forking it per backend is how the two arms diverged.
  c.recommended_pool_size = 1;
  if (p_->have_device) {
    // Same formula-engine surcharge as the NVIDIA arm (pool_sizing.h): a
    // routed plus-M/auto engine adds GiB-scale per-replica decode buffers the
    // base footprint constant does not include.
    size_t formula_extra = 0;
    try {
      const auto routing = backend_routing::load_routing_config();
      if (const auto *f = backend_routing::resolve(routing, "formula"))
        formula_extra = pipeline::formula_engine_scratch_bytes(f->engine);
    } catch (...) {
    }
    size_t freeb = 0, totalb = 0;
    if (hipMemGetInfo(&freeb, &totalb) == hipSuccess)
      c.recommended_pool_size = pipeline::compute_pipeline_pool_size(
          freeb, totalb, pipeline::kPerPipelineFootprintBytes + formula_extra);
  }

  // ONNX MODE MUST DOWNGRADE. This is not cosmetic: UnifiedOcrPipeline picks its
  // staging path from caps().device (stages_through_ring_(), unified_ocr_pipeline
  // .cpp) — a non-Host device means "stage through the device ring". In onnx mode
  // this backend hands out cpu::HostDeviceQueue + cpu::HostAllocator, so claiming
  // a device here made the pipeline run the device path over host memory and tag
  // the ImageView with a device kind it does not have.
  //
  // Also the honesty contract at backend.h:158-160: an Auto run that fell back to
  // onnx must SAY onnx. AppleBackend was the only arm doing this; the rule is
  // shared, so it belongs on every arm.
  c.mode = mode_;
  c.has_native_engine = native_device_();
  if (!native_device_()) {
    c.device = backend::DeviceKind::Host;
    c.async = false;
    c.supports_batch = false;
  }

  return c;
}

std::unique_ptr<backend::DeviceQueue> RocmBackend::make_queue() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostDeviceQueue>();
  return std::make_unique<HipStreamQueue>(p_->device_id);
}

std::shared_ptr<backend::IDeviceAllocator> RocmBackend::allocator() {
  if (!native_device_()) {
    if (!host_allocator_)
      host_allocator_ = std::make_shared<turbo_ocr::cpu::HostAllocator>();
    return host_allocator_;
  }
  return p_->alloc;
}

std::unique_ptr<backend::IKernels> RocmBackend::make_kernels() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostKernels>();
  return std::make_unique<HipKernels>(p_->alloc);
}

std::unique_ptr<backend::IEngine> RocmBackend::make_engine() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::CpuEngineAdapter>();
  return std::make_unique<MIGraphXEngine>(p_->device_id);
}

std::unique_ptr<backend::ITableRecognizer>
RocmBackend::make_table_recognizer(const backend_routing::BackendSpec &spec) {
  // A LOCAL spec must resolve to the CUDA-free ORT structure backend HERE,
  // exactly as the cpu/intel/apple arms do: the shared factory answers only
  // OpenAI specs and returns nullptr for Local ones by design
  // (vlm_factory.cpp), which DISABLES the modality. This arm used to delegate
  // everything to it under a comment claiming "tables still work" — the exact
  // opposite of what that factory does — so `--backend amd` silently served
  // no tables and no formulas, a bring-up landmine pointing at MIGraphX.
  // TODO(on-hardware): register a HIP/MIGraphX SLANeXt structure backend and
  // its hipified fused preproc behind the "slanext" key; the host path below
  // stays the fallback.
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_table_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "slanext")
    return std::make_unique<turbo_ocr::cpu::CpuTableRecognizer>();
  return backend::make_table_recognizer(spec);
}

std::unique_ptr<backend::IFormulaRecognizer>
RocmBackend::make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  // Same rule as tables: Local resolves to the CUDA-free ORT PP-FormulaNet
  // class here; the shared factory serves only OpenAI specs.
  // TODO(on-hardware): a MIGraphX PP-FormulaNet-S host-AR-loop backend (the
  // AMD peer of PPFormulaNetOrt) plugs in behind "ppformulanet_s".
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_formula_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "ppformulanet_s")
    return std::make_unique<turbo_ocr::cpu::CpuFormulaRecognizerAdapter>();
  return backend::make_formula_recognizer(spec);
}

backend::StageSet RocmBackend::load_stages(const backend::BackendConfig &cfg) {
  // NATIVE = MIGraphX, which compiles a program per gfx target on first use.
  // Availability is "is there a HIP device"; the ONNX/MIGraphX-EP path skips
  // the compile entirely. UNVERIFIED (no ROCm hardware / no build target).
  int hip_devices = 0;
  if (hipGetDeviceCount(&hip_devices) != hipSuccess) hip_devices = 0;
  mode_ = turbo_ocr::cpu::resolve_engine_mode("amd", cfg, hip_devices > 0);
  if (!native_device_()) {
    auto built = turbo_ocr::cpu::make_vendor_onnx_stages("amd", cfg);
    onnx_doc_ori_ = std::move(built.doc_ori);
    return std::move(built.stages);
  }

  backend::StageSet set;
  if (!p_->have_device) {
    TOCR_LOG_ERROR("AMD native path selected but no HIP device is open; "
                   "stages unavailable",
                   "backend", "amd");
    return set;
  }
  // A FRESH HipKernels per StageSet. load_stages() is called once per pipeline
  // pool entry, and each entry runs concurrently on its own queue; HipKernels
  // owns mutable device scratch (decode pool, CCL label map, JFA seeds, pinned
  // box staging), so a shared instance would let two entries corrupt each
  // other's connected-component state. The stages hold it alive via shared_ptr.
  // The ALLOCATOR is still shared — it is stateless and hipMalloc is
  // thread-safe.
  StageDeps deps{std::make_shared<HipKernels>(p_->alloc), p_->alloc.get(),
                 p_->device_id};

  // Detection (required).
  {
    auto det = std::make_unique<RocmDetector>(deps);
    if (!cfg.det_model.empty() && det->load(cfg.det_model)) {
      set.detector = std::move(det);
      set.available.detector = true;
    }
  }
  // Recognition (required).
  {
    auto rec = std::make_unique<RocmRecognizer>(deps);
    bool ok = !cfg.rec_model.empty() && rec->load(cfg.rec_model);
    if (ok && !cfg.rec_dict.empty())
      ok = rec->load_dict(cfg.rec_dict);
    if (ok) {
      set.recognizer = std::move(rec);
      set.available.recognizer = true;
    }
  }
  // Classification (optional).
  if (!cfg.cls_model.empty()) {
    auto cls = std::make_unique<RocmClassifier>(deps);
    if (cls->load(cfg.cls_model)) {
      set.classifier = std::move(cls);
      set.available.classifier = true;
    }
  }
  // Layout (optional).
  if (cfg.want_layout && !cfg.layout_model.empty()) {
    auto lay = std::make_unique<RocmLayout>(deps);
    if (lay->load(cfg.layout_model)) {
      set.layout = std::move(lay);
      set.available.optional.set(capability::CapabilityId::Layout, true);
    }
  }
  set.available.optional.set(capability::CapabilityId::Table, cfg.want_tables);
  set.available.optional.set(capability::CapabilityId::Formula, cfg.want_formulas);
  return set;
}

// NOTE (dedup): make_infer_func() removed from the seam — server_main builds
// pipeline::make_infer_func() over a UnifiedOcrPipeline pool from load_stages().

server::ImageDecoder RocmBackend::make_image_decoder() {
  // Host decode to cv::Mat (nvJPEG has no ROCm analog). For device-resident
  // decode into an ImageView, the pipeline uses IKernels::decode_image instead.
  return [](const unsigned char *data, size_t len) -> cv::Mat {
    cv::Mat enc(1, static_cast<int>(len), CV_8UC1,
                const_cast<unsigned char *>(data));
    return cv::imdecode(enc, cv::IMREAD_COLOR);
  };
}

server::OrientFunc RocmBackend::make_orient_func() {
  // The ONNX doc-orientation model IS loaded in onnx mode (load_stages populates
  // onnx_doc_ori_), and this used to return {} unconditionally — so autorotate
  // was dead even when the operator supplied the model and the capability
  // reported it. NVIDIA and Intel both have this guard; AMD was the one arm
  // missing it. Generic policy is shared, never fixed per backend.
  //
  // Captured by `this`, not by raw pointer: load_stages() runs once per pool
  // entry and REPLACES the model, so a pointer captured by an earlier pipeline's
  // constructor would dangle (the same UAF fixed in AppleBackend).
  if (onnx_doc_ori_) {
    return [this](const cv::Mat &page) -> int {
      return onnx_doc_ori_ ? onnx_doc_ori_->detect(page) : 0;
    };
  }
  // TODO(on-hardware): a RocmDocOrientation (MIGraphX 224x224 classifier)
  // mirroring NvDocOrientation for the NATIVE arm. Until then native mode has no
  // orientation model, so autorotate stays off there.
  return {};
}

std::unique_ptr<backend::Backend> make_rocm_backend(int device_id) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
    // No line of our own here: returning nullptr is the registrar contract's
    // "compiled in, not usable on this machine", and backend_registry.cpp logs
    // exactly that (structured) on both the auto-detect and the --backend amd
    // path. A raw fprintf would double the message AND emit non-JSON into a
    // log stream every other line of which is JSON.
    return nullptr;
  }
  return std::make_unique<RocmBackend>(device_id);
}

} // namespace turbo_ocr::amd
