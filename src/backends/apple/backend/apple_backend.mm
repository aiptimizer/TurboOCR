// AppleBackend implementation (see apple_backend.h).

#import "apple/backend/apple_backend.h"

#import <Foundation/Foundation.h>

#include <filesystem>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

// The SHARED "fast" stage set — Apple's ONNX path is the same host pre/post +
// ORT code every other vendor uses, on the CoreML provider.
#include "cpu/queue/host_device_queue.h"
#include "cpu/memory/host_allocator.h"
#include "cpu/kernels_host/host_kernels.h"
#include "cpu/engine/cpu_engine_adapter.h"
#include "cpu/stages/cpu_stages.h"
#include "turbo_ocr/backend/routing_config.h" // BackendSpec (complete type)
#include "cpu/stages/cpu_formula_recognizer.h"
#include "turbo_ocr/onnx/host_ort_threads.h" // set_host_ort_intra_op_threads
#include "cpu/stages/cpu_table_recognizer.h"
#include "turbo_ocr/base/log/logger.h"

#include "apple/engine/mps_engine.h"
#include "apple/kernels_metal/metal_kernels.h"
#include "apple/memory/metal_allocator.h"
#include "apple/memory/metal_image.h"
#include "apple/queue/metal_device_queue.h"
#include "apple/support/metal_common.h" // mtl_library — eager metallib probe
#include "apple/stages/mps_stages.h"

#include "turbo_ocr/backend/formula_recognizer.h" // backend::make_formula_recognizer
#include "turbo_ocr/backend/table_recognizer.h"   // backend::make_table_recognizer
#include "turbo_ocr/base/geometry/box.h"         // sorted_boxes
#include "turbo_ocr/core/types.h"                 // OCRResultItem
#include "turbo_ocr/core/infer_result.h"          // InferResult (complete type)

namespace turbo_ocr::apple {

AppleBackend::AppleBackend() = default;
AppleBackend::~AppleBackend() {
  // TURBO_APPLE_PAGE_AUDIT=1 only: totals for the wrong-page (stale sampler
  // texture) probe. Silent when the audit is off.
  page_audit_report("backend teardown");
}

backend::BackendCaps AppleBackend::caps() const {
  backend::BackendCaps c;
  c.device = backend::DeviceKind::Metal;
  c.name = "apple";
  c.native_image_decode = true;  // ImageIO (hardware JPEG on Apple silicon) ->
                                 // MetalImage::from_host_bgr; falls back to
                                 // cv::imdecode for anything ImageIO declines.
  c.async = true;
  c.supports_batch = true;
  c.recommended_pool_size = 4; // unified memory; a few resident pipeline entries
  c.mode = mode_;
  c.has_native_engine = true;  // MPSGraph, given an export
  c.has_onnx_engine = true;    // CoreML EP over the .onnx
  // The ONNX path runs host pre/post on cv::Mat, so it is NOT device-resident
  // and has no async device queue to overlap against; saying otherwise would
  // make the pipeline schedule work that never overlaps.
  if (mode_ != backend::EngineMode::Native) {
    c.device = backend::DeviceKind::Host;
    c.async = false;
    c.supports_batch = false;
  }
  return c;
}

// THE DEVICE SIDE FOLLOWS THE MODE, and it has to.
//
// The ONNX ("fast") path is the shared HOST stage set: CpuDetector & co. take
// an ImageView in HOST memory and a queue they can treat as a no-op. Handing
// them a MetalDeviceQueue + a Metal allocator (which is what this backend
// returns for its native path) makes UnifiedOcrPipeline upload every page into
// an MTLBuffer and pass a device pointer to code that will dereference it on
// the CPU. That is not a slow path, it is a wrong one — it aborted on the
// first image. So in onnx mode every device factory below returns the host
// implementation, exactly as CpuBackend would.
bool AppleBackend::native_device_() const noexcept {
  return mode_ == backend::EngineMode::Native;
}

std::unique_ptr<backend::DeviceQueue> AppleBackend::make_queue() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostDeviceQueue>();
  return std::make_unique<MetalDeviceQueue>();
}

std::shared_ptr<backend::IDeviceAllocator> AppleBackend::allocator() {
  if (!native_device_()) {
    if (!host_allocator_)
      host_allocator_ = std::make_shared<turbo_ocr::cpu::HostAllocator>();
    return host_allocator_;
  }
  return shared_allocator();
}

std::unique_ptr<backend::IKernels> AppleBackend::make_kernels() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::HostKernels>();
  return std::make_unique<MetalKernels>();
}

std::unique_ptr<backend::IEngine> AppleBackend::make_engine() {
  if (!native_device_())
    return std::make_unique<turbo_ocr::cpu::CpuEngineAdapter>();
  return std::make_unique<MpsEngine>();
}

std::unique_ptr<backend::ITableRecognizer>
AppleBackend::make_table_recognizer(const backend_routing::BackendSpec &spec) {
  // VLM specs route to the shared OpenAI endpoint regardless of device. A local
  // SLANeXt-on-MPSGraph structure backend is still a TODO(apple-slanext-mpsgraph),
  // so the host-local
  // engine is the CUDA-free ORT one — the SAME choice CpuBackend makes
  // (cpu_backend.cpp). Delegating the local case to the common registry instead
  // asks for the CUDA-tied sibling, which a CPU-configured build never
  // compiles: the factory returns null and the server aborts at boot rather
  // than running without tables.
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_table_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "slanext")
    return std::make_unique<turbo_ocr::cpu::CpuTableRecognizer>();
  return backend::make_table_recognizer(spec);
}

std::unique_ptr<backend::IFormulaRecognizer>
AppleBackend::make_formula_recognizer(const backend_routing::BackendSpec &spec) {
  // Same posture as tables, and the same reason for not using the registry for
  // the local case: PP-FormulaNet-on-MPSGraph is a TODO, so run the CUDA-free
  // ORT recognizer CpuBackend uses.
  if (spec.kind == backend_routing::Kind::Openai)
    return backend::make_formula_recognizer(spec);
  if (spec.engine.empty() || spec.engine == "ppformulanet_s")
    return std::make_unique<turbo_ocr::cpu::CpuFormulaRecognizerAdapter>();
  return backend::make_formula_recognizer(spec);
}

// Defined together below — same rule, so they cannot drift.
static std::string mps_export_dir(const std::string &model_path);
static bool has_mps_export(const std::string &model_path);

namespace {

// LAYOUT IS THE ONE STAGE THAT STAYS ON ONNX IN NATIVE MODE.
//
// PP-DocLayoutV3 is not a det/rec-shaped graph: three inputs (im_shape, image,
// scale_factor), three outputs, ~1100 nodes including GridSample, ScatterND,
// GatherND, Einsum, TopK and Mod. The single-input MPSGraph builder cannot
// express it and several of those ops have no MPSGraph equivalent at all, so
// MpsLayout::load() has always returned false and native mode reported
// layout:0. backend::StageSet has independent slots, so the fix is not to port
// the graph — it is to run THIS slot on the shared ONNX stage while det/rec/cls
// stay resident on MPSGraph.
//
// The hazard that makes this a wrapper rather than a one-line substitution is
// documented at native_device_() above: in native mode the pipeline hands every
// stage a MetalAllocator-backed ImageView (kind=Metal), and the shared host
// stage will dereference that pointer on the CPU. That is safe here ONLY
// because MetalAllocator hands out MTLBuffer.contents from a
// StorageModeShared (unified-memory) buffer — the same address is valid on both
// sides. We do not assume it: the allocator is asked (host_coherent()), the
// queue is drained so the H2D memcpy is ordered before the read, and a
// non-coherent allocator falls back to an explicit D2H into host staging. Only
// the layout stage pays this; det/rec/cls never leave the device.
class HostLayoutOnDevice final : public backend::ILayout {
public:
  explicit HostLayoutOnDevice(std::shared_ptr<MetalAllocator> alloc)
      : alloc_(std::move(alloc)) {}

  [[nodiscard]] bool load(const std::string &model_path) override {
    // Thread count is NOT set here — AppleBackend::load_native_stages_ sets the
    // host-idle hint once for every host ORT stage (layout, formula, table's
    // cell recognizer), so this stage no longer has an opinion of its own. See
    // common/host_ort_threads.h.
    if (auto *native = inner_.native()) {
      // CoreML runs the trunk on GPU/ANE. Correct again as of the ORT 1.27.1
      // bump — on 1.24.4 it returned NaN for every score and box (see
      // OrtPaddleLayout::set_use_coreml), and the finite-check in
      // OrtPaddleLayout::run is what makes trusting it reasonable at all.
      //
      // CoreML WINS ON BOTH SINGLE-PAGE AND MULTI-PAGE. Same binary, quiet
      // machine, ORT 1.27.1:
      //
      //   single page   CoreML 327 ms (319-342)  vs  CPU/4thr 580 ms (572-585)
      //   10-page PDF   CoreML 3.52-3.60 s       vs  CPU/4thr 6.33-6.44 s
      //                 (0.35 s/page)                (0.63 s/page)
      //                 160 regions both, identical
      //
      // A SYNTHETIC CONCURRENCY NUMBER SAYS THE OPPOSITE — DO NOT ACT ON IT.
      // Firing 8 independent requests at once measures CoreML at 1.74 req/s vs
      // CPU's 3.17, because CoreML puts layout on the GPU/ANE where MPSGraph
      // det/rec already are and they contend, whereas CPU layout overlaps with
      // GPU det/rec on idle cores. That contention is real but it costs LESS
      // than the per-page compute CoreML saves, so on the shape of request this
      // service actually gets — Stirling posting a whole multi-page PDF — the
      // pipelined wall clock is still 1.8x better with CoreML. The 8-way figure
      // measures a load pattern nobody sends.
      //
      // Hence DISABLE_COREML=1 is a DEBUG switch (rule out a provider
      // regression without a rebuild), NOT a throughput tuning knob. Reaching
      // for it on the strength of the req/s number alone makes real requests
      // slower.
      //
      // Accuracy cost of CoreML is small but NOT zero (fp16 on ANE): same
      // class ids, same region count, boxes within 1 px, scores within 0.006
      // of the CPU path on the verification page.
      native->set_use_coreml(true);
    }
    return inner_.load(model_path);
  }

  [[nodiscard]] std::vector<turbo_ocr::layout::LayoutBox>
  run(const backend::ImageView &img, int orig_h, int orig_w,
      float score_threshold, backend::DeviceQueue &queue) override {
    if (img.empty()) return {};
    if (img.is_host())
      return inner_.run(img, orig_h, orig_w, score_threshold, queue);

    // Device-resident page: make the pixels both VISIBLE (the upload was
    // staged on this queue) and HOST-ADDRESSABLE before the ONNX stage reads
    // them.
    queue.synchronize();
    backend::ImageView host = img;
    host.kind = backend::DeviceKind::Host;
    if (!alloc_ || !alloc_->host_coherent()) {
      const std::size_t bytes =
          (img.step ? img.step : static_cast<std::size_t>(img.cols) * 3) *
          static_cast<std::size_t>(img.rows);
      staging_.resize(bytes);
      alloc_->copy_d2h(staging_.data(), img.data, bytes, queue);
      host.data = staging_.data();
    }
    return inner_.run(host, orig_h, orig_w, score_threshold, queue);
  }

  [[nodiscard]] bool is_ready() const noexcept override {
    return inner_.is_ready();
  }

private:
  turbo_ocr::cpu::CpuLayout inner_;
  std::shared_ptr<MetalAllocator> alloc_;
  std::vector<unsigned char> staging_; // only on a non-coherent allocator
};

} // namespace

backend::StageSet AppleBackend::load_native_stages_(const backend::BackendConfig &cfg) {
  auto alloc = shared_allocator();
  backend::StageSet ss;

  // THE HOST IS IDLE ON THIS PATH — say so once, here, for every host ORT
  // stage rather than per stage.
  //
  // det/rec/cls below are MPSGraph, so the CPU is not doing the OCR trunk. The
  // host ORT stages that remain (layout, formula, table's cell recognizer) each
  // cap their intra-op pool on the assumption that they are competing with
  // det/rec for cores, which is simply false here. Measured on this box (10
  // performance / 14 logical cores):
  //
  //     layout ... 1010 ms -> 583 ms      formula .. 1.08 s -> 0.53 s
  //     table .... 1.30 s  -> 0.68 s
  //
  // 4, not "as many as ORT wants": pool_size replicas x a machine-sized pool
  // oversubscribes, and on layout that cost 40% of throughput to buy 130 ms of
  // latency. 4 measured better than the old cap on BOTH latency and throughput
  // (0.583 s / 3.26 req/s vs 1.140 s / 2.72 req/s at 8 concurrent), which is
  // why it is safe as a default rather than a tuning knob.
  //
  // ORT_NUM_THREADS still overrides everything, unchanged.
  turbo_ocr::set_host_ort_intra_op_threads(4);

  auto det = std::make_unique<MpsDetector>(alloc);
  ss.available.detector = det->load(mps_export_dir(cfg.det_model));
  ss.detector = std::move(det);

  auto rec = std::make_unique<MpsRecognizer>(alloc, cfg.rec_dict);
  ss.available.recognizer = rec->load(mps_export_dir(cfg.rec_model));
  ss.recognizer = std::move(rec);

  if (!cfg.cls_model.empty()) {
    auto cls = std::make_unique<MpsClassifier>(alloc);
    if (cls->load(mps_export_dir(cfg.cls_model))) {
      ss.available.classifier = true;
      ss.classifier = std::move(cls);
    }
  }

  // Layout: MPSGraph if an export somehow exists, otherwise the ONNX stage on
  // host memory (see HostLayoutOnDevice). NOT a silent fallback for the whole
  // backend — det/rec/cls above are still resident MPSGraph.
  if (cfg.want_layout && !cfg.layout_model.empty()) {
    // Probe FIRST rather than letting MpsLayout::load() fail: with no export
    // present that failure is the expected case, and its "not implemented"
    // message printed once per pool entry reads like a fault when it is in fact
    // the designed route.
    std::unique_ptr<MpsLayout> mps_lay;
    if (has_mps_export(cfg.layout_model)) {
      mps_lay = std::make_unique<MpsLayout>(alloc);
      if (!mps_lay->load(mps_export_dir(cfg.layout_model))) mps_lay.reset();
    }
    if (mps_lay) {
      ss.available.optional.set(capability::CapabilityId::Layout, true);
      ss.layout = std::move(mps_lay);
    } else {
      auto lay = std::make_unique<HostLayoutOnDevice>(alloc);
      if (lay->load(cfg.layout_model)) {
        ss.available.optional.set(capability::CapabilityId::Layout, true);
        ss.layout = std::move(lay);
        TOCR_LOG_INFO("apple native: layout runs on the ONNX stage (host memory)",
                      "model", std::string_view(cfg.layout_model));
      } else {
        TOCR_LOG_WARN("apple native: layout model failed to load — layout off",
                      "model", std::string_view(cfg.layout_model));
      }
    }
  }

  // Table/formula are constructed per-request via make_*_recognizer(spec);
  // advertise availability from the operator's opt-in flags.
  ss.available.optional.set(capability::CapabilityId::Table, cfg.want_tables);
  ss.available.optional.set(capability::CapabilityId::Formula, cfg.want_formulas);

  // Document orientation, same posture as layout: no MPSGraph export is wired
  // for it, so run the ONNX stage rather than leave ?autorotate=1 dead.
  //
  // This one needs no host/device dance at all. Orientation is not a StageSet
  // slot — it leaves the backend as a server::OrientFunc, which is
  // `int(const cv::Mat&)`, so it is handed a HOST cv::Mat by construction and
  // never sees the Metal-resident ImageView that forced the HostLayoutOnDevice
  // wrapper above. Load it and hand make_orient_func() the pointer.
  if (!cfg.doc_orient_model.empty()) {
    auto ori = std::make_unique<classification::OrtDocOrientation>();
    if (ori->load_model(cfg.doc_orient_model)) {
      onnx_doc_ori_ = std::move(ori);
      ss.available.optional.set(capability::CapabilityId::DocOrientation, true);
      TOCR_LOG_INFO("apple native: doc-orientation runs on the ONNX stage",
                    "model", std::string_view(cfg.doc_orient_model));
    } else {
      ss.available.optional.set(capability::CapabilityId::DocOrientation, false);
      TOCR_LOG_WARN("apple native: doc-orientation model failed to load — "
                    "autorotate off",
                    "model", std::string_view(cfg.doc_orient_model));
    }
  } else {
    ss.available.optional.set(capability::CapabilityId::DocOrientation, false); // no model configured
  }
  return ss;
}

// The FAST path: the plain .onnx through the CoreML execution provider (ANE +
// GPU, fp16 natively), assembled by the SHARED factory. Nothing Apple-specific
// happens here beyond the provider string — that is the point.
backend::StageSet AppleBackend::load_onnx_stages_(const backend::BackendConfig &cfg) {
  auto built = turbo_ocr::cpu::make_vendor_onnx_stages("apple", cfg);
  onnx_doc_ori_ = std::move(built.doc_ori);
  return std::move(built.stages);
}

// Does a native MPSGraph export actually exist for these models? MpsEngine
// wants a directory holding graph.json + weights.bin; a normal models/ tree
// holds .onnx files, and asking MPSGraph to open one produces exactly the
// "missing graph.json/weights.bin" failure that made `--backend apple`
// unusable before the two paths existed.
static bool has_mps_export(const std::string &model_path) {
  if (model_path.empty()) return false;
  std::error_code ec;
  // Resolve through the SAME rule the loaders use, then ask whether what we
  // land on is really an export. Checking a different thing here than
  // mps_export_dir() resolves is how "native available" and "native loadable"
  // drifted apart twice already: first the .onnx path was probed for a
  // graph.json inside a file, then a rec LADDER (rec_b<W>/ children, no
  // top-level graph.json) was declared "not exported" and refused to start.
  std::filesystem::path p(mps_export_dir(model_path));
  if (!std::filesystem::is_directory(p, ec)) return false;
  if (std::filesystem::exists(p / "graph.json", ec)) return true;
  // A ladder: every rec_b<W>/ child is its own export.
  for (const auto &e : std::filesystem::directory_iterator(p, ec)) {
    if (e.is_directory(ec) && std::filesystem::exists(e.path() / "graph.json", ec))
      return true;
  }
  return false;
}

// The path the MPSGraph stages must actually be given: an export DIRECTORY.
// has_mps_export() resolves `…/det_tiny.onnx` to its sibling `…/det_tiny/`,
// but the stages were handed the raw .onnx — so the probe said "native is
// available", the loaders then looked for graph.json inside a file, and
// startup died with "did not produce the required detector + recognizer".
// One rule, applied in both places.
static std::string mps_export_dir(const std::string &model_path) {
  if (model_path.empty()) return model_path;
  std::error_code ec;
  std::filesystem::path p(model_path);
  if (std::filesystem::is_directory(p, ec)) return model_path;
  std::filesystem::path stem = p.parent_path() / p.stem();
  if (std::filesystem::exists(stem / "graph.json", ec)) return stem.string();
  // A rec ladder: the stem dir holds rec_b<W>/ children, each its own export.
  if (std::filesystem::is_directory(stem, ec)) {
    for (const auto &e : std::filesystem::directory_iterator(stem, ec)) {
      if (e.is_directory(ec) && std::filesystem::exists(e.path() / "graph.json", ec))
        return stem.string();
    }
  }
  return model_path;
}

backend::StageSet AppleBackend::load_stages(const backend::BackendConfig &cfg) {
  cfg_ = cfg;
  configured_ = true;

  // Apple's own probe; the POLICY (auto-fallback vs hard error) is shared with
  // every other vendor in cpu::resolve_engine_mode.
  const bool native_available =
      has_mps_export(cfg.det_model) && has_mps_export(cfg.rec_model);
  mode_ = turbo_ocr::cpu::resolve_engine_mode("apple", cfg, native_available);

  if (mode_ == backend::EngineMode::Native) {
    // Force the shader-library load NOW, while no command encoder exists.
    // mtl_library() throws a catchable error on a missing/broken metallib;
    // deferring it to the first kernel dispatch would throw mid-encoding and
    // trip Metal's uncommitted-encoder assertion (an abort) during unwind.
    (void)mtl_library();
    return load_native_stages_(cfg);
  }
  return load_onnx_stages_(cfg);
}

server::ImageDecoder AppleBackend::make_image_decoder() {
  // Encoded bytes -> host cv::Mat (BGR). For device-resident decode into an
  // ImageView, callers use IKernels::decode_image instead.
  return [](const unsigned char *data, std::size_t len) -> cv::Mat {
    cv::Mat enc(1, (int)len, CV_8U, const_cast<unsigned char *>(data));
    return cv::imdecode(enc, cv::IMREAD_COLOR);
  };
}

server::OrientFunc AppleBackend::make_orient_func() {
  // Empty means "no rotation applied" (service_fns.h contract) — returned only
  // when no doc-orientation model actually loaded, in EITHER mode. Both
  // load_native_stages_ and load_onnx_stages_ populate onnx_doc_ori_, so this
  // one accessor serves both.
  if (!onnx_doc_ori_) return {};
  // Capture THIS BACKEND, not the raw model pointer. UnifiedOcrPipeline's
  // constructor calls make_orient_func(), and build_backend_runtime calls
  // load_stages() once per pool entry — each call REPLACES onnx_doc_ori_,
  // destroying the object the previous entry's closure captured. A raw-pointer
  // capture therefore leaves every pool entry but the last holding a dangling
  // pointer (Apple defaults to 4 entries, so 3 of 4 dangle). The backend
  // outlives every pipeline and route (BackendRuntime's documented destruction
  // order), and reading the CURRENT model at call time is exactly the old
  // semantics: all consumers always shared the last-loaded model.
  return [this](const cv::Mat &page) -> int {
    return onnx_doc_ori_ ? onnx_doc_ori_->detect(page) : 0;
  };
}

// NOTE (dedup): AppleBackend::make_infer_func() was DELETED. It was a private
// copy of the det -> sort -> (cls) -> rec orchestration (and had no layout /
// router / table / formula path at all). The ONE orchestration now lives in
// src/pipeline/unified/unified_ocr_pipeline.{h,cpp} and the ONE server::InferFunc
// builder in src/pipeline/unified/make_infer_func.{h,cpp}, both driven over the
// StageSet that AppleBackend::load_stages() returns. tests/cpp/backends/
// funsd_unified_apple.mm already drives that path (FUNSD-50 tiny 85.70% F1).

std::unique_ptr<backend::Backend> make_apple_backend() {
  return std::make_unique<AppleBackend>();
}

} // namespace turbo_ocr::apple
