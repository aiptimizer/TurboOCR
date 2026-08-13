#pragma once

// Backend — the ONE device seam, at the stages_* adapter altitude.
//
// Each vendor ships exactly one Backend implementation (NvidiaBackend,
// AppleBackend, AmdBackend, IntelBackend, CpuBackend), selected at startup
// (env/config) among whichever backends were compiled in. It is the single
// object the merged server_main constructs; from it flow (a) the low-level
// device factories the ONE OcrPipeline uses to assemble stages and keep data
// resident — queue, allocator, kernels, engine, and the constructed stage
// interfaces — and (b) the high-level service-boundary functions the HTTP/gRPC
// routes already consume (server::InferFunc / ImageDecoder / OrientFunc from
// service_fns.h). This collapses stages_gpu.h + stages_cpu.h and
// gpu_server_main.cpp + cpu_server_main.cpp into one backend-neutral bootstrap.
//
// The Backend owns NO device SDK types in its interface — everything device is
// behind ImageView / DeviceQueue / IEngine / IKernels.

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/backend/device_queue.h"       // DeviceQueue, DeviceKind
#include "turbo_ocr/backend/engine.h"             // IEngine
#include "turbo_ocr/backend/engine_mode.h"        // EngineMode, EpConfig
#include "turbo_ocr/backend/formula_recognizer.h" // backend::IFormulaRecognizer
#include "turbo_ocr/backend/image_view.h"         // ImageView, DeviceKind
#include "turbo_ocr/backend/kernels.h"            // IKernels
#include "turbo_ocr/backend/stages.h"             // IDetector/IRecognizer/...
#include "turbo_ocr/backend/table_recognizer.h"   // backend::ITableRecognizer
#include "turbo_ocr/core/capability.h"      // CapabilityId, CapabilityMask
#include "turbo_ocr/core/service_fns.h"         // server::InferFunc/ImageDecoder/OrientFunc

namespace turbo_ocr::backend_routing { struct BackendSpec; }

namespace turbo_ocr::backend {

// Owning RAII handle for a device allocation, freed through its allocator on
// destruction. Non-copyable, movable. `data()` lives in `device()`'s space.
class IDeviceAllocator;
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(IDeviceAllocator *alloc, void *data, std::size_t bytes,
               DeviceKind kind) noexcept
      : alloc_(alloc), data_(data), bytes_(bytes), kind_(kind) {}
  ~DeviceBuffer();
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&o) noexcept
      : alloc_(o.alloc_), data_(o.data_), bytes_(o.bytes_), kind_(o.kind_) {
    o.alloc_ = nullptr;
    o.data_ = nullptr;
    o.bytes_ = 0;
  }
  DeviceBuffer &operator=(DeviceBuffer &&o) noexcept;

  [[nodiscard]] void *data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] DeviceKind device() const noexcept { return kind_; }
  [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

private:
  IDeviceAllocator *alloc_ = nullptr;
  void *data_ = nullptr;
  std::size_t bytes_ = 0;
  DeviceKind kind_ = DeviceKind::Host;
};

// Device memory factory. Allocates buffers in the backend's device space and
// stages ordered copies on a DeviceQueue. Pinned-host helpers are optional (used
// for fast H2D/D2H staging; default to plain host allocation).
class IDeviceAllocator {
public:
  virtual ~IDeviceAllocator() = default;
  [[nodiscard]] virtual DeviceKind device() const noexcept = 0;

  // Can the host dereference a pointer from allocate() directly, once the queue
  // has drained? Additive capability with a device-class default, so no vendor
  // has to implement it: unified-memory devices (Metal today) and the host say
  // yes, discrete VRAM says no. Override where the device class is not the whole
  // story — an AMD APU, an Intel iGPU, a CUDA managed-memory allocator.
  //
  // The shared layer branches on THIS, never on `kind == DeviceKind::Metal`
  // (dedup rule 1); see src/pipeline/unified/vlm_factory.cpp::host_pixels.
  [[nodiscard]] virtual bool host_coherent() const noexcept {
    return device_is_host_coherent(device());
  }

  [[nodiscard]] virtual void *allocate(std::size_t bytes) = 0;
  virtual void free(void *p) noexcept = 0;

  // Convenience owning wrapper.
  [[nodiscard]] DeviceBuffer allocate_buffer(std::size_t bytes) {
    return DeviceBuffer{this, allocate(bytes), bytes, device()};
  }

  // Pinned / page-locked host staging memory (optional; default plain host).
  [[nodiscard]] virtual void *allocate_host(std::size_t bytes) = 0;
  virtual void free_host(void *p) noexcept = 0;

  // Ordered copies on `queue` (async on device backends; immediate on Host).
  virtual void copy_h2d(void *dst, const void *src, std::size_t bytes,
                        DeviceQueue &queue) = 0;
  virtual void copy_d2h(void *dst, const void *src, std::size_t bytes,
                        DeviceQueue &queue) = 0;
  virtual void copy_d2d(void *dst, const void *src, std::size_t bytes,
                        DeviceQueue &queue) = 0;
};

inline DeviceBuffer::~DeviceBuffer() {
  if (alloc_ && data_)
    alloc_->free(data_);
}
inline DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&o) noexcept {
  if (this != &o) {
    if (alloc_ && data_)
      alloc_->free(data_);
    alloc_ = o.alloc_;
    data_ = o.data_;
    bytes_ = o.bytes_;
    kind_ = o.kind_;
    o.alloc_ = nullptr;
    o.data_ = nullptr;
    o.bytes_ = 0;
  }
  return *this;
}

// What the Backend reports about itself. Drives routing (the device axis) and
// server capability responses.
struct BackendCaps {
  DeviceKind device = DeviceKind::Host;
  std::string name;               // "nvidia" | "apple" | "amd" | "intel" | "cpu"
  bool native_image_decode = false; // has an on-device decoder (nvJPEG / vImage)
  bool async = false;             // device queues are asynchronous
  bool supports_batch = false;    // native batched inference / batch route
  int recommended_pool_size = 1;  // pipeline entries (VRAM/UMA-tier sized)

  // How many images this backend WANTS coalesced into one detection submission
  // — Triton's `preferred_batch_size`, and the vendor's own advice about its
  // device.
  //
  // This is a policy hint, not a capability: the hard ceiling is
  // IDetector::max_batch_size() (stages.h), and the shared cross-request
  // batcher (include/turbo_ocr/pipeline/unified/stage_batcher.h) takes the smaller of the two.
  // 1 (the default) means "do not coalesce", so a vendor that has not opted in
  // keeps today's batch-1 detection exactly.
  //
  // Sizing it is a per-device judgement the shared layer must not make: it is
  // bounded by device memory for N detection canvases, by how quickly the
  // forward pass saturates the ALUs (past that point a bigger batch only adds
  // latency), and — on a fixed-shape runtime like MPSGraph or a TRT engine with
  // a static profile — by which batch shapes were actually compiled.
  int preferred_batch_size = 1;

  // Which path this backend actually came up on, and what it could offer.
  // Reported so /capabilities and the Python `info()` can never disagree with
  // reality — an Auto run that fell back from native to onnx must SAY onnx.
  EngineMode mode = EngineMode::Onnx;
  bool has_native_engine = false; // a vendor graph engine exists in this build
  bool has_onnx_engine = true;    // the .onnx/EP path is available

  // IMPLEMENTED (see capability/capability.h): which optional capabilities this
  // backend+mode could EVER build, given the right models. This is the axis an
  // operator CANNOT fix by configuration — distinct from StageAvailability,
  // which says what actually loaded THIS boot and usually can be fixed.
  //
  // Defaults to all(): a backend that has not spoken keeps exactly today's
  // behaviour ("nothing is structurally impossible; if it did not load, it is a
  // config problem"). A vendor that genuinely cannot build a stage in some mode
  // should narrow this so /capabilities can say "unsupported" instead of
  // sending an operator hunting for a model path that would never be used.
  capability::CapabilityMask implemented = capability::CapabilityMask::all();
};

// LOADED (see capability/capability.h): which stages this backend actually
// brought up this boot. Detection/recognition are always required; the rest are
// opt-in per model availability.
//
// The optional stages are a CapabilityMask rather than one bool each: a mask
// cannot be transposed with a neighbouring argument when it is passed along,
// and it can be iterated, so consumers loop over the capability table instead
// of hand-listing stages and drifting from each other.
struct StageAvailability {
  bool detector = false;
  bool recognizer = false;
  bool classifier = false; // text-line angle cls (not an optional capability:
                           // it has no request flag — it is always applied when
                           // loaded, so it is not in the capability table)
  capability::CapabilityMask optional; // layout / tables / formulas / autorotate
};

// The constructed device stages plus availability, handed to the ONE OcrPipeline
// so it assembles its orchestration against interfaces (never a vendor class).
// Detector/recognizer are always present when the backend loaded; the optional
// members are null when the corresponding StageAvailability bit is clear.
struct StageSet {
  StageAvailability available;
  std::unique_ptr<IDetector> detector;
  std::unique_ptr<IRecognizer> recognizer;
  std::unique_ptr<IClassifier> classifier; // may be null
  std::unique_ptr<ILayout> layout;         // may be null
};

// Model paths + pool sizing for load_stages(). Backends read finer per-model
// knobs from env (matching today's stages_gpu/stages_cpu behaviour); this struct
// carries only what the merged server_main resolves generically.
struct BackendConfig {
  std::string det_model;
  std::string rec_model;
  std::string cls_model;      // empty => angle cls disabled
  std::string layout_model;   // empty => layout disabled
  std::string doc_orient_model; // empty => autorotate disabled
  std::string rec_dict;       // recognition character dictionary
  int pool_size = 0;          // 0 => backend picks (VRAM/UMA tier)
  bool want_layout = false;
  bool want_tables = false;
  bool want_formulas = false;

  // WHICH PATH to the silicon (backend/engine_mode.h): the vendor's native
  // graph engine ("ultra") or the .onnx through its ORT provider ("fast").
  // Auto prefers native and falls back — loudly — when the native artefact is
  // absent, which is what lets `--backend apple` work on a plain models/ tree.
  EngineMode mode = EngineMode::Auto;
  // Provider settings for the fast path. Backends normally leave this at the
  // default and let onnx_provider_for(<vendor>) fill it in; an operator can
  // override the device/precision through it.
  EpConfig ep{};
};

// The one device seam. One implementation per vendor.
class Backend {
public:
  virtual ~Backend() = default;

  [[nodiscard]] virtual BackendCaps caps() const = 0;

  // --- Low-level device factories (used by the ONE OcrPipeline) -------------
  // A queue is one ordered lane of device work (a pipeline entry owns one, as a
  // GpuPipelineEntry owns a cudaStream_t today).
  [[nodiscard]] virtual std::unique_ptr<DeviceQueue> make_queue() = 0;

  // The device memory allocator (may be a shared singleton per device).
  [[nodiscard]] virtual std::shared_ptr<IDeviceAllocator> allocator() = 0;

  // The device pre/post kernel op set.
  [[nodiscard]] virtual std::unique_ptr<IKernels> make_kernels() = 0;

  // NOTE (removed): make_engine(). It was a PURE VIRTUAL that all five vendors
  // implemented and NOTHING ever called — IEngine does not appear above the
  // seam at all; every arm builds its own engines inside load_stages() and
  // hands the shared layer stage interfaces, which is the right shape. Five
  // overrides is interface tax paid for a dispatch that never happens, and a
  // pure virtual is worse than a dead function: it forces the next vendor to
  // implement it too. Each arm keeps its factory as its own member.

  // Per-request table/formula recognizer construction (registry dispatch). The
  // Backend supplies device-appropriate local backends; VLM specs route to the
  // shared OpenAI endpoint regardless of device.
  [[nodiscard]] virtual std::unique_ptr<backend::ITableRecognizer>
  make_table_recognizer(const backend_routing::BackendSpec &spec) = 0;
  [[nodiscard]] virtual std::unique_ptr<backend::IFormulaRecognizer>
  make_formula_recognizer(const backend_routing::BackendSpec &spec) = 0;

  // --- Stage bootstrap (stages_* altitude) ----------------------------------
  // Build/resolve the models and return the constructed device stages plus their
  // availability. Called once at startup (per pool entry, or shared where the
  // engine is concurrency-safe).
  [[nodiscard]] virtual StageSet load_stages(const BackendConfig &cfg) = 0;

  // --- High-level service-boundary functions (consumed by routes/gRPC) ------
  //
  // NOTE (dedup): there is deliberately NO make_infer_func() here. The fully-
  // assembled OCR entry point is built ONCE, above the seam, by
  //   turbo_ocr::pipeline::make_infer_func(pool)   [include/turbo_ocr/pipeline/unified/make_infer_func.h]
  // over a pool of UnifiedOcrPipeline entries constructed from THIS backend's
  // load_stages() + make_queue(). Every backend that used to override
  // make_infer_func() carried a private copy of the det->cls->rec->layout->
  // router orchestration — exactly the duplication this rebuild removes. The
  // only legitimate per-vendor service-boundary functions are the two below
  // (image decode and page orientation), because both are genuinely device
  // mechanics (nvJPEG / vImage, per-vendor doc-orientation model).
  //
  // Encoded-bytes -> host cv::Mat decoder (nvJPEG-accelerated where available,
  // else host). For device-resident decode into an ImageView, use
  // IKernels::decode_image instead.
  [[nodiscard]] virtual server::ImageDecoder make_image_decoder() = 0;

  // Would IKernels::decode_image() handle these bytes on-device, or decline and
  // leave them to make_image_decoder()? A header sniff: no device work, no
  // state. Asked BEFORE a pipeline replica is leased, so a container this
  // vendor cannot device-decode is host-decoded OUTSIDE the lease — decoding
  // inside one pins a replica for the whole CPU decode, which costs more
  // throughput than the device decode saves. Default false = always host.
  [[nodiscard]] virtual bool can_device_decode(const std::uint8_t * /*data*/,
                                               std::size_t /*len*/) const {
    return false;
  }

  // WHOLE-DEVICE memory in bytes, false when this vendor cannot report it (the
  // default, and correct for a host backend that has no separate device pool).
  //
  // Exists so /metrics can export VRAM without naming a vendor: it used to call
  // cudaMemGetInfo straight out of service/server/metrics.h under
  // `#ifndef USE_CPU_ONLY` — the one place CUDA had leaked outside
  // src/backends/nvidia/, and a build break waiting for the first non-NVIDIA
  // GPU configure, since that flag is off for those too.
  //
  // The numbers are DEVICE-WIDE, not this process's: on a shared GPU they count
  // every tenant. The exported gauges' HELP strings repeat that caveat so a leak
  // hunt is not misled by them.
  [[nodiscard]] virtual bool device_memory(std::size_t & /*used*/,
                                           std::size_t & /*total*/) const {
    return false;
  }

  // Page-orientation detector (0/90/180/270), empty when the doc-orientation
  // model isn't loaded (autorotate off).
  [[nodiscard]] virtual server::OrientFunc make_orient_func() = 0;
};

// Runtime backend selection among the compiled-in vendors. `name` from config/env
// ("nvidia"/"apple"/"amd"/"intel"/"cpu"); empty => auto-detect the best available
// device, falling back to CpuBackend. Returns nullptr if the named backend was
// not compiled into this build.
std::unique_ptr<Backend> make_backend(std::string_view name);

// The vendors compiled into this binary (for /capabilities and auto-selection).
[[nodiscard]] std::vector<std::string_view> available_backends();

} // namespace turbo_ocr::backend
