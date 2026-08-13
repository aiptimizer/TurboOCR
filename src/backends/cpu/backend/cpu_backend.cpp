// CpuBackend implementation. Each factory returns the host adapter object;
// load_stages builds the wrapped CpuPaddle* stage classes; the table/formula
// factories mint the CUDA-free local recognizers (or pass VLM/OpenAI specs
// through to the shared factories). make_infer_func is intentionally empty here
// — Deliverable 2's UnifiedOcrPipeline + shared make_infer_func replaces every
// per-backend InferFunc — so this backend supplies only load_stages +
// make_image_decoder + make_orient_func at Deliverable-1 altitude.

#include "cpu/backend/cpu_backend.h"

#include <thread>

#include <opencv2/core.hpp>

#include "cpu/engine/cpu_engine_adapter.h"
#include "cpu/kernels_host/host_kernels.h"
#include "cpu/memory/host_allocator.h"
#include "cpu/queue/host_device_queue.h"
#include "cpu/stages/cpu_formula_recognizer.h"
#include "cpu/stages/cpu_stages.h"
#include "cpu/stages/cpu_table_recognizer.h"

#include "turbo_ocr/backend/routing_config.h" // BackendSpec, Kind
#include "turbo_ocr/image/cpu_image_decode.h"         // decode_cpu_fallback

namespace turbo_ocr::cpu {

CpuBackend::CpuBackend() : allocator_(std::make_shared<HostAllocator>()) {}
CpuBackend::~CpuBackend() = default;

backend::BackendCaps CpuBackend::caps() const {
  backend::BackendCaps c;
  c.device = backend::DeviceKind::Host;
  c.name = "cpu";
  c.native_image_decode = false; // host stb/OpenCV decode, not an on-device path
  c.async = false;               // synchronous host queues
  c.supports_batch = false;      // rec batches internally but no batch route
  // 4, matching the v3.5.0 CPU server (pipeline_pool_size.value_or(4)) — NOT
  // hardware_concurrency. Each entry is a full model set; sizing by core count
  // multiplied boot time and memory on many-core hosts with no config change,
  // and the ORT sessions already parallelise internally. PIPELINE_POOL_SIZE /
  // --pool-size still override.
  c.recommended_pool_size = pool_size_ > 0 ? pool_size_ : 4;
  // The host arm is ONNX Runtime end to end: there is no vendor graph engine
  // to fall back from. Stated rather than left to the struct default, because
  // /capabilities now serves these and a default is not an answer.
  c.has_native_engine = false;
  c.has_onnx_engine = true;
  return c;
}

std::unique_ptr<backend::DeviceQueue> CpuBackend::make_queue() {
  return std::make_unique<HostDeviceQueue>();
}

std::shared_ptr<backend::IDeviceAllocator> CpuBackend::allocator() {
  return allocator_;
}

std::unique_ptr<backend::IKernels> CpuBackend::make_kernels() {
  return std::make_unique<HostKernels>();
}

std::unique_ptr<backend::IEngine> CpuBackend::make_engine() {
  return std::make_unique<CpuEngineAdapter>();
}

std::unique_ptr<turbo_ocr::backend::ITableRecognizer>
CpuBackend::make_table_recognizer(const backend_routing::BackendSpec &spec) {
  // Openai/VLM table specs are device-independent — served by the shared common
  // factory; this backend only mints the host-local SLANeXt structure backend.
  if (spec.kind == backend_routing::Kind::Openai)
    return turbo_ocr::backend::make_table_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "slanext")
    return std::make_unique<CpuTableRecognizer>();
  return turbo_ocr::backend::make_table_recognizer(spec);
}

std::unique_ptr<turbo_ocr::backend::IFormulaRecognizer>
CpuBackend::make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind == backend_routing::Kind::Openai)
    return turbo_ocr::backend::make_formula_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "ppformulanet_s")
    return std::make_unique<CpuFormulaRecognizerAdapter>();
  return turbo_ocr::backend::make_formula_recognizer(spec);
}

backend::StageSet CpuBackend::load_stages(const backend::BackendConfig &cfg) {
  // The CPU backend IS the shared ONNX path on the default provider — one
  // caller of make_onnx_stages() among several (Apple/Intel/NVIDIA/AMD reach
  // the same code with their own EpConfig). cfg.ep lets an operator point even
  // the "cpu" backend at another provider (ORT_EP-style) without a new backend.
  // Route through make_vendor_onnx_stages like every OTHER backend, rather than
  // calling make_onnx_stages directly. This was the one arm that skipped it, so
  // TURBO_EP_PROVIDER had no effect on `--backend cpu` while working on all four
  // others — an operator setting it saw four backends honour it and one silently
  // ignore it. onnx_provider_for("cpu") is "" (the default provider), so the
  // resolved EpConfig is IDENTICAL to cfg.ep when the env var is unset: this
  // adds the override path without changing any existing behaviour.
  auto built = make_vendor_onnx_stages("cpu", cfg);
  doc_ori_ = std::move(built.doc_ori);
  doc_ori_ready_ = static_cast<bool>(doc_ori_);
  if (cfg.pool_size > 0)
    pool_size_ = cfg.pool_size;
  return std::move(built.stages);
}

// NOTE (dedup): CpuBackend::make_infer_func() was DELETED (it was already an
// empty placeholder). The ONE InferFunc is pipeline::make_infer_func() over a
// pool of UnifiedOcrPipeline entries built from load_stages().

server::ImageDecoder CpuBackend::make_image_decoder() {
  // Host JPEG/PNG decode (Wuffs/OpenCV fallback) — no on-device decoder.
  return [](const unsigned char *data, std::size_t len) -> cv::Mat {
    return decode::decode_cpu_fallback(data, len);
  };
}

server::OrientFunc CpuBackend::make_orient_func() {
  // Capture THIS BACKEND, not the raw model pointer. UnifiedOcrPipeline's
  // constructor also calls make_orient_func(), and build_backend_runtime calls
  // load_stages() once per pool entry — each call REPLACES the doc-ori model,
  // destroying the object the previous entry's closure captured. A raw-pointer
  // capture therefore leaves every pool entry but the last holding a dangling
  // pointer (latent UAF for any future detect_orientation caller). The backend
  // itself outlives every pipeline and route (documented destruction order in
  // BackendRuntime), and reading the CURRENT model at call time is exactly the
  // old semantics: all consumers always shared the last-loaded model.
  if (!doc_ori_ || !doc_ori_ready_)
    return {}; // autorotate off
  return [this](const cv::Mat &page) -> int {
    return doc_ori_ ? doc_ori_->detect(page) : 0;
  };
}

} // namespace turbo_ocr::cpu
