// CudaBackend implementation. Each factory returns the NVIDIA adapter object;
// load_stages builds the wrapped stage classes; the service-boundary functions
// forward to the existing stages_gpu.cpp helpers (make_gpu_image_decoder /
// make_gpu_infer_func / probe_nvjpeg) so the NVIDIA server stays byte-for-byte
// the proven path.

#include "nvidia/backend/cuda_backend.h"
#include "cpu/engine/cpu_engine_adapter.h"
#include "cpu/kernels_host/host_kernels.h"
#include "cpu/memory/host_allocator.h"
#include "cpu/queue/host_device_queue.h"
#include "cpu/stages/cpu_stages.h"

#include "nvidia/engine/onnx_to_trt.h"
#include "nvidia/engine/trt_engine_adapter.h"
#include "nvidia/kernels_cuda/cuda_kernels.h"
#include "nvidia/memory/cuda_allocator.h"
#include "nvidia/queue/cuda_device_queue.h"
#include "nvidia/stages/nv_formula_recognizer.h"
#include "nvidia/stages/nv_stages.h"
#include "nvidia/stages/nv_table_recognizer.h"

#include "turbo_ocr/backend/routing_config.h" // BackendSpec, Kind
#include "turbo_ocr/pipeline/pool_sizing.h"          // compute_pipeline_pool_size
#include "nvidia/support/nv_image_decode.h"   // probe_nvjpeg / make_nv_image_decoder
#include "nvidia/support/nvjpeg_encoder.h"   // NvJpegEncoder (installed below)
#include "turbo_ocr/image/page_image_encoder.h" // set_jpeg_encode_hook
#include "nvidia/support/nvjpeg_decoder.h"    // NvJpegDecoder::is_jpeg (header sniff)

#include <opencv2/core.hpp>

namespace turbo_ocr::nvidia {

// Number of visible CUDA devices, 0 when the driver/runtime is unusable. Used
// only to decide native-vs-onnx; every real CUDA error still surfaces later.
static int cuda_device_count() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}

// nvJPEG page-image encode, installed on the device-neutral encoder.
//
// The encoder used to reach INTO this vendor arm with a
// `#include "nvidia/support/nvjpeg_encoder.h"` under `#ifndef USE_CPU_ONLY` —
// the one file outside src/backends/ that named a vendor, against the rule in
// src/README.md. The dependency now points the correct way: the arm hands its
// encoder to src/image/, which knows nothing about who supplied it.
//
// thread_local because nvJPEG encoder state is single-threaded, and empty on any
// failure so the caller falls back to libjpeg-turbo.
static std::vector<uint8_t> encode_jpeg_nvjpeg(const cv::Mat &bgr, int quality) {
  static thread_local turbo_ocr::encode::NvJpegEncoder enc;
  if (!enc.available()) return {};
  return enc.encode(bgr, quality);
}

CudaBackend::CudaBackend() : allocator_(std::make_shared<CudaAllocator>()) {
  nvjpeg_available_ = nvidia::probe_nvjpeg();
  pdf::set_jpeg_encode_hook(&encode_jpeg_nvjpeg);
}
CudaBackend::~CudaBackend() = default;

backend::BackendCaps CudaBackend::caps() const {
  backend::BackendCaps c;
  c.device = backend::DeviceKind::Cuda;
  c.name = "nvidia";
  c.native_image_decode = nvjpeg_available_;
  c.async = true;
  c.supports_batch = true;
  // VRAM-tier sizing — the SAME compute_pipeline_pool_size the v3.5.0 GPU main
  // used (pool_sizing.h), not a hard-coded 4: a 4-entry pool left most of a
  // large card idle and over-committed a small one. Falls back to the previous
  // default when no device answers (mode resolution then avoids native anyway).
  if (pool_size_ > 0) {
    c.recommended_pool_size = pool_size_;
  } else {
    // The routed formula engine changes the per-replica footprint: plus-M and
    // the auto ladder allocate GiB-scale decode buffers per replica that the
    // measured base constant does not include (pool_sizing.h). A malformed
    // routing config must not turn a caps() query into a throw — sizing falls
    // back to the base footprint; the config error itself fatals at load.
    size_t formula_extra = 0;
    try {
      const auto routing = backend_routing::load_routing_config();
      if (const auto *f = backend_routing::resolve(routing, "formula"))
        formula_extra = pipeline::formula_engine_scratch_bytes(f->engine);
    } catch (...) {
    }
    size_t free_mem = 0, total_mem = 0;
    c.recommended_pool_size =
        (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess)
            ? pipeline::compute_pipeline_pool_size(
                  free_mem, total_mem,
                  pipeline::kPerPipelineFootprintBytes + formula_extra)
            : 4;
  }
  // Report the path this backend actually came up on, so /capabilities and
  // Python info() can never disagree with reality (an Auto run that fell back
  // from TRT to the CUDA-EP fast path must SAY onnx).
  c.mode = mode_;
  c.has_native_engine = true; // TRT builds its engine from the .onnx on demand
  c.has_onnx_engine = true;

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
  if (!native_device_()) {
    c.device = backend::DeviceKind::Host;
    c.async = false;
    c.supports_batch = false;
  }

  return c;
}

// In onnx mode the stages are the shared HOST ones, so the device side must be
// the host side too (a CUDA pointer handed to host code is a crash, not a
// slowdown — see the Apple backend, where exactly that aborted).
std::unique_ptr<backend::DeviceQueue> CudaBackend::make_queue() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostDeviceQueue>();
  return std::make_unique<CudaDeviceQueue>(/*owns=*/true);
}

std::shared_ptr<backend::IDeviceAllocator> CudaBackend::allocator() {
  if (!native_device_()) {
    if (!host_allocator_)
      host_allocator_ = std::make_shared<turbo_ocr::cpu::HostAllocator>();
    return host_allocator_;
  }
  return allocator_;
}

std::unique_ptr<backend::IKernels> CudaBackend::make_kernels() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostKernels>();
  return std::make_unique<CudaKernels>();
}

std::unique_ptr<backend::IEngine> CudaBackend::make_engine() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::CpuEngineAdapter>();
  // The generic engine slot is TRT (det/rec/cls/layout). The formula stage
  // needs an ORT-CUDA engine constructed with a device_id + stream + graph
  // flags, so it is built inside NvFormulaRecognizer rather than here (there is
  // no common constructor — wf_engine.txt).
  return std::make_unique<TrtEngineAdapter>();
}

std::unique_ptr<turbo_ocr::backend::ITableRecognizer>
CudaBackend::make_table_recognizer(const backend_routing::BackendSpec &spec) {
  // Openai/VLM table specs are device-independent and served by the shared
  // common factory (table::make_table_recognizer(spec)); this backend only
  // mints the DEVICE-resident local structure recognizer.
  if (spec.kind == backend_routing::Kind::Openai)
    return turbo_ocr::backend::make_table_recognizer(spec); // shared endpoint
  // Local: SLANeXt encoder-split (default). Unknown local engine keys fall
  // through to the shared factory too.
  if (spec.engine.empty() || spec.engine == "slanext")
    return std::make_unique<NvTableRecognizer>();
  return turbo_ocr::backend::make_table_recognizer(spec);
}

std::unique_ptr<turbo_ocr::backend::IFormulaRecognizer>
CudaBackend::make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind == backend_routing::Kind::Openai)
    return turbo_ocr::backend::make_formula_recognizer(spec); // shared endpoint
  // Every LOCAL device engine — incl. the auto CJK ladder — goes through the
  // bridge, which forwards the key to the one old-side factory. Passing the
  // engine is load-bearing: the bridge used to hardcode plus-S, so plus-M got
  // the wrong graph contract (fatal at load) and "auto" never resolved here.
  if (spec.engine.empty() || spec.engine == "ppformulanet_s" ||
      spec.engine == "ppformulanet_plus_m" || spec.engine == "auto")
    return std::make_unique<NvFormulaRecognizer>(spec.engine);
  return turbo_ocr::backend::make_formula_recognizer(spec);
}

backend::StageSet CudaBackend::load_stages(const backend::BackendConfig &cfg) {
  // NATIVE = TensorRT. It is "available" whenever a CUDA device is present:
  // unlike Apple's exports, TRT BUILDS its engine from the .onnx on first use
  // (cached thereafter), so there is no artefact to probe for — the cost is
  // minutes of build time, which is exactly what the ONNX/CUDA-EP fast path
  // exists to skip.
  const bool native_available = (cuda_device_count() > 0);
  mode_ = turbo_ocr::cpu::resolve_engine_mode("nvidia", cfg, native_available);
  if (!native_device_()) {
    auto built = turbo_ocr::cpu::make_vendor_onnx_stages("nvidia", cfg);
    onnx_doc_ori_ = std::move(built.doc_ori);
    return std::move(built.stages);
  }

  backend::StageSet set;

  // BackendConfig carries PORTABLE model paths (.onnx) because every vendor gets
  // the same config; materializing them into this vendor's artefact is this
  // backend's job. TRT deserializes a serialized plan, never an .onnx, so the
  // engine must be built-or-cache-hit here — the stage loaders below take the
  // resulting .trt path. An empty return means the ONNX was missing or the build
  // failed (ensure_trt_engine logs which); it propagates as a load failure,
  // which the caller turns into a refusal to start.
  engine::sweep_orphan_engine_temps();
  const auto plan = [](const std::string &onnx, const char *type) {
    return onnx.empty() ? std::string{} : engine::ensure_trt_engine(onnx, type);
  };

  // Detection + recognition are required.
  auto det = std::make_unique<NvDetector>();
  set.available.detector = det->load(plan(cfg.det_model, "det"));
  set.detector = std::move(det);

  auto rec = std::make_unique<NvRecognizer>();
  bool rec_ok = rec->load(plan(cfg.rec_model, "rec"));
  if (rec_ok && !cfg.rec_dict.empty())
    rec_ok = rec->load_dict(cfg.rec_dict);
  set.available.recognizer = rec_ok;
  set.recognizer = std::move(rec);

  // Optional stages: null member + false flag when the model path is empty or
  // load fails (mirrors CpuStageAvailability / the GPU model probes).
  if (!cfg.cls_model.empty()) {
    auto cls = std::make_unique<NvClassifier>();
    if (cls->load(plan(cfg.cls_model, "cls"))) {
      set.available.classifier = true;
      set.classifier = std::move(cls);
    }
  }
  if (cfg.want_layout && !cfg.layout_model.empty()) {
    auto lay = std::make_unique<NvLayout>();
    if (lay->load(plan(cfg.layout_model, "layout"))) {
      set.available.optional.set(capability::CapabilityId::Layout, true);
      set.layout = std::move(lay);
    }
  }
  if (!cfg.doc_orient_model.empty()) {
    doc_ori_ = std::make_unique<NvDocOrientation>();
    set.available.optional.set(capability::CapabilityId::DocOrientation, doc_ori_->load(plan(cfg.doc_orient_model, "doc_ori")));
    if (!set.available.optional.get(capability::CapabilityId::DocOrientation))
      doc_ori_.reset();
  }

  set.available.optional.set(capability::CapabilityId::Table, cfg.want_tables);
  set.available.optional.set(capability::CapabilityId::Formula, cfg.want_formulas);
  if (cfg.pool_size > 0)
    pool_size_ = cfg.pool_size;
  return set;
}

// NOTE (dedup): CudaBackend::make_infer_func() (and attach_dispatcher) were
// DELETED. They forwarded to make_gpu_infer_func(dispatcher) — the NVIDIA-only
// copy of the orchestration/pooling. The shared layer now owns that: server_main
// builds a pool of UnifiedOcrPipeline entries over load_stages() and calls
// pipeline::make_infer_func(pool). NVIDIA keeps only the device mechanics.

bool CudaBackend::can_device_decode(const std::uint8_t *data,
                                    std::size_t len) const {
  // nvJPEG decodes JPEG only; PNG/WebP/BMP/TIFF all fall to the host tail.
  return nvjpeg_available_ &&
         decode::NvJpegDecoder::is_jpeg(
             reinterpret_cast<const unsigned char *>(data), len);
}

bool CudaBackend::device_memory(std::size_t &used, std::size_t &total) const {
  std::size_t free_mem = 0, total_mem = 0;
  if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess)
    return false;
  used = total_mem - free_mem;
  total = total_mem;
  return true;
}

server::ImageDecoder CudaBackend::make_image_decoder() {
  // nvJPEG (GPU) for JPEG when available, Wuffs/OpenCV host tail otherwise —
  // exactly the existing helper.
  return nvidia::make_nv_image_decoder(nvjpeg_available_);
}

server::OrientFunc CudaBackend::make_orient_func() {
  // ONNX ("fast") mode never builds NvDocOrientation — it loads the portable
  // doc-orientation model through the shared stage factory instead. Checking
  // only doc_ori_ would silently disable autorotate for the whole fast path
  // even though the model loaded fine.
  // Capture THIS BACKEND, not the raw model pointer — load_stages() runs once
  // per pool entry and REPLACES the model, so a raw pointer captured by an
  // earlier pipeline's constructor dangles (see CpuBackend::make_orient_func).
  if (onnx_doc_ori_) {
    return [this](const cv::Mat &page) -> int {
      return onnx_doc_ori_ ? onnx_doc_ori_->detect(page) : 0;
    };
  }
  if (!doc_ori_ || !doc_ori_->is_ready())
    return {}; // autorotate off
  return [this](const cv::Mat &page) -> int {
    return (doc_ori_ && doc_ori_->is_ready()) ? doc_ori_->detect(page) : 0;
  };
}

} // namespace turbo_ocr::nvidia
