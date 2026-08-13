// vlm_factory — the ONE definition of the seam's remote (kind:openai) table and
// formula factories. See vlm_factory.h for the design rationale.
//
// This is the device-agnostic face of the OpenAI endpoint. The main-tree class
// turbo_ocr::vlm::OpenAIEndpoint cannot be reused directly: it implements the
// OLD CUDA-typed table::ITableRecognizer / formula::IFormulaRecognizer
// (GpuImage + cudaStream_t) and its header includes <cuda_runtime.h>, so it does
// not exist in a non-CUDA build at all. The NEW interfaces implemented here are
// turbo_ocr::backend::ITableRecognizer / backend::IFormulaRecognizer, over:
//
//   GpuImage      -> backend::ImageView
//   cudaStream_t  -> backend::DeviceQueue&   (+ the readback hook, see header)
//
// ONLY the seam types differ. The parse/health-check/crop/submit logic, the
// CropOutcome ok-flag semantics, the self-contained async_result_parser()
// snapshot, and the shared global VLMCropPool all live in ONE place —
// turbo_ocr/analysis/vlm/openai_policy.h — which both classes call, so the remote path
// behaves identically on every backend and a fix there fixes it everywhere.
// This TU therefore contains exactly one thing the policy cannot express:
// host_page(), which reads an ImageView back through the registered device
// readback (the CUDA class supplies a cudaMemcpyAsync closure instead).
//
// This TU is deliberately SEPARATE from unified_ocr_pipeline.cpp /
// make_infer_func.cpp because it drags in libcurl + the VLM crop pool. Binaries
// that never route a remote modality (e.g. the offline FUNSD proof-gates) link
// tests/cpp/backends/vlm_factory_link_support.cpp instead, which returns nullptr.

#include "turbo_ocr/pipeline/unified/vlm_factory.h"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/backend/device_queue.h"
#include "turbo_ocr/backend/formula_recognizer.h"
#include "turbo_ocr/backend/image_view.h"
#include "turbo_ocr/backend/table_recognizer.h"
#include "turbo_ocr/backend/backend.h" // IDeviceAllocator
#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/analysis/vlm/openai_policy.h" // the shared device-free endpoint policy

namespace turbo_ocr::pipeline {
namespace {

std::mutex &readback_mu() {
  static std::mutex m;
  return m;
}
// One slot per DeviceKind — NOT one slot per process. See vlm_factory.h: with a
// multi-backend binary a single slot is a global with two owners.
std::array<DeviceReadback, 8> &readback_table() {
  static std::array<DeviceReadback, 8> t;
  return t;
}
std::size_t kind_index(backend::DeviceKind k) {
  const auto i = static_cast<std::size_t>(k);
  return i < readback_table().size() ? i : 0;
}

// Looked up by host_page() below (same TU; reachable from turbo_ocr::vlm as
// turbo_ocr::pipeline::readback_for thanks to the unnamed namespace's implicit
// using-directive).
DeviceReadback readback_for(backend::DeviceKind kind) {
  std::lock_guard<std::mutex> lk(readback_mu());
  return readback_table()[kind_index(kind)];
}

} // namespace

void register_device_readback(backend::DeviceKind kind, DeviceReadback rb) {
  std::lock_guard<std::mutex> lk(readback_mu());
  readback_table()[kind_index(kind)] = std::move(rb);
}

DeviceReadback
make_allocator_readback(std::shared_ptr<backend::IDeviceAllocator> alloc) {
  if (!alloc) return {};
  DeviceReadback rb;
  // The allocator answers the unified-memory question itself (seam capability,
  // default derived from the device class) — the shared layer must never test
  // `kind == DeviceKind::Metal`.
  rb.host_coherent = alloc->host_coherent();
  // CO-OWN the allocator. This closure is stored in a function-local static
  // table that lives until static destruction, i.e. it OUTLIVES the
  // BackendRuntime that owns the Backend. Backend::allocator() already returns
  // a shared_ptr, so making the lifetime a fact costs one refcount at boot —
  // whereas capturing the raw pointer made it a convention documented in a
  // comment, and `.get()` at the one call site was taken off a temporary.
  rb.copy = [alloc = std::move(alloc)](void *dst, const void *src,
                                       std::size_t bytes,
                                       backend::DeviceQueue &queue) -> bool {
    alloc->copy_d2h(dst, src, bytes, queue);
    queue.synchronize(); // the crops are built host-side right after this
    return true;
  };
  return rb;
}

} // namespace turbo_ocr::pipeline

namespace turbo_ocr::vlm {
namespace {

// ---------------------------------------------------------------------------
// Page readback: ImageView (any device) -> host-addressable rows.
// ---------------------------------------------------------------------------
//
// This is the ONE device-specific step of the remote endpoint — everything else
// is openai_policy. Returns a HostPage covering `page.rows * page.step`
// contiguous host bytes, or an empty HostPage (the policy's decline signal) when
// the page cannot be reached from the host. `scratch` owns the bytes when a copy
// was required; it stays empty on the zero-copy paths, and must outlive the
// policy call (the crops are encoded and copied into the pool before it
// returns).
openai_policy::HostPage host_page(const backend::ImageView &page,
                                  backend::DeviceQueue &queue,
                                  std::vector<std::uint8_t> &scratch) {
  if (page.empty()) return {};
  const std::size_t need =
      static_cast<std::size_t>(page.rows) * page.step;

  // Wrap a host-addressable base pointer in the descriptor the policy wants.
  const auto as_host_page = [&](const void *base) {
    return openai_policy::HostPage{static_cast<const std::uint8_t *>(base),
                                   page.cols, page.rows, page.step};
  };

  // Host backend: the view already wraps host RAM (zero copy).
  if (page.kind == backend::DeviceKind::Host) return as_host_page(page.data);

  // Everything else goes through the entry registered FOR THIS PAGE'S DEVICE, so
  // a binary holding several backends always reads a page back through its own
  // backend's allocator (see vlm_factory.h).
  const auto rb = turbo_ocr::pipeline::readback_for(page.kind);

  // Unified memory: the device pointer IS host-addressable once the queue has
  // drained, so skip the copy entirely. The allocator answers this (seam
  // capability) — the shared layer does not know which vendor is unified.
  if (rb.host_coherent) {
    queue.synchronize();
    return as_host_page(page.data);
  }

  // Discrete VRAM: the allocator's copy_d2h is the ONLY correct route.
  if (rb.copy) {
    scratch.resize(need);
    if (!rb.copy(scratch.data(), page.data, need, queue)) return {};
    return as_host_page(scratch.data());
  }

  // Nothing registered for this device. Fall back to the device CLASS default
  // (seam helper, not a vendor test) so binaries that never call
  // register_device_readback — the offline proof gates — still work on UMA.
  if (backend::device_is_host_coherent(page.kind)) {
    queue.synchronize();
    return as_host_page(page.data);
  }

  TOCR_LOG_ERROR_RL("OpenAI VLM endpoint: no host readback for device page "
                    "(call pipeline::register_device_readback at startup)",
                    "device", backend::device_kind_name(page.kind));
  return {};
}

// ---------------------------------------------------------------------------
// The endpoint. Implements BOTH de-CUDA'd recognizer interfaces, exactly like
// the main-tree OpenAIEndpoint implements both CUDA-typed ones.
// ---------------------------------------------------------------------------
class BackendOpenAIEndpoint final : public backend::IFormulaRecognizer,
                                   public backend::ITableRecognizer {
public:
  explicit BackendOpenAIEndpoint(backend_routing::BackendSpec spec)
      : spec_(std::move(spec)) {}

  // --- shared -------------------------------------------------------------
  [[nodiscard]] bool is_ready() const noexcept override { return ready_; }
  [[nodiscard]] std::string_view backend_name() const noexcept override {
    return "openai";
  }
  [[nodiscard]] bool supports_async() const noexcept override { return ready_; }
  [[nodiscard]] std::string
  parse_async_result(const std::string &raw) const override {
    return openai_policy::parse_with(spec_.parser, raw);
  }
  [[nodiscard]] std::function<std::string(const std::string &)>
  async_result_parser() const override {
    // Capture the parser enum BY VALUE — the callable holds no pointer back
    // into this object, so it stays valid after a pipeline recycle frees it.
    const backend_routing::Parser parser = spec_.parser;
    return [parser](const std::string &raw) {
      return openai_policy::parse_with(parser, raw);
    };
  }
  // Read the page back once on the device worker; PNG bytes are COPIED into the
  // pool at submit, so the futures outlive `scratch`.
  [[nodiscard]] std::vector<std::future<std::string>>
  submit_async(const backend::ImageView &page, const std::vector<Box> &boxes,
               backend::DeviceQueue &queue) override {
    std::vector<std::uint8_t> scratch;
    return openai_policy::submit_crops_async(
        spec_, ready_, boxes,
        [&] { return host_page(page, queue, scratch); });
  }

  // GET <base_url>/v1/models; resolves the served-model-name when unset.
  bool health_check() { return openai_policy::health_check(spec_, ready_); }

  // --- IFormulaRecognizer -------------------------------------------------
  [[nodiscard]] bool load_model_dir(const std::string &) override {
    return health_check();
  }
  [[nodiscard]] bool load_tokenizer(const std::string &) override { return true; }
  [[nodiscard]] std::vector<backend::FormulaEngineResult>
  run(const backend::ImageView &page, const std::vector<Box> &boxes,
      backend::DeviceQueue &queue) override {
    auto crops = infer_crops(page, boxes, queue);
    return openai_policy::to_formula_results<backend::FormulaEngineResult>(
        std::move(crops));
  }

  // --- ITableRecognizer ---------------------------------------------------
  [[nodiscard]] bool load() override { return health_check(); }
  [[nodiscard]] std::vector<router::TableResult>
  run(const backend::ImageView &page, const std::vector<Box> &regions,
      const std::vector<OCRResultItem> & /*page_ocr*/,
      backend::DeviceQueue &queue) override {
    auto crops = infer_crops(page, regions, queue);
    return openai_policy::to_table_results(std::move(crops), regions);
  }

private:
  // The sync path, with this backend's readback closure plugged into the shared
  // policy. `scratch` lives for the whole call: the policy has finished encoding
  // (and copying) every crop by the time it returns.
  std::vector<openai_policy::CropOut>
  infer_crops(const backend::ImageView &page, const std::vector<Box> &boxes,
              backend::DeviceQueue &queue) {
    std::vector<std::uint8_t> scratch;
    return openai_policy::infer_crops(
        spec_, ready_, boxes,
        [&] { return host_page(page, queue, scratch); });
  }

  backend_routing::BackendSpec spec_;
  bool ready_ = false;
};

} // namespace
} // namespace turbo_ocr::vlm

// ---------------------------------------------------------------------------
// The two seam factories.
// ---------------------------------------------------------------------------
//
// Only the REMOTE (kind==Openai) branch is served here. kind==Local specs are
// device-specific by definition (SLANeXt / PP-FormulaNet on this backend's
// engine) and are answered by the Backend itself — CpuBackend returns its
// CpuTableRecognizer, CudaBackend its Nv* wrappers — BEFORE reaching this
// factory. Returning nullptr for a Local spec here is therefore correct: it
// means "no backend claimed it", and the pipeline disables the modality.

namespace turbo_ocr::backend {

std::unique_ptr<ITableRecognizer>
make_table_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind != backend_routing::Kind::Openai) return nullptr;
  return std::make_unique<vlm::BackendOpenAIEndpoint>(spec);
}

std::unique_ptr<IFormulaRecognizer>
make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  if (spec.kind != backend_routing::Kind::Openai) return nullptr;
  return std::make_unique<vlm::BackendOpenAIEndpoint>(spec);
}

} // namespace turbo_ocr::backend
