#pragma once

// UnifiedOcrPipeline — the ONE device-agnostic OCR orchestration.
//
// This collapses the two forked pipelines (turbo_ocr::pipeline::OcrPipeline for
// the GPU, turbo_ocr::pipeline::CpuOcrPipeline for the host) into a single class
// written entirely against the Backend seam
// (include/turbo_ocr/backend/*.h). Every device concept is behind an
// interface:
//
//   GpuImage         -> backend::ImageView
//   cudaStream_t     -> backend::DeviceQueue&
//   PaddleDet/Rec/…  -> backend::IDetector/IRecognizer/IClassifier/ILayout
//   Paddle/CpuTable… -> backend::ITableRecognizer / backend::IFormulaRecognizer
//
// The control flow (det -> sort -> (cls) -> rec -> combine, plus layout -> CUA
// router -> table/formula dispatch -> reading order) is ported verbatim from the
// main-tree GPU pipeline. Device-specific speed tricks (double-buffered uploads,
// per-modality streams/events, MTLCommandBuffer batching) live INSIDE each
// backend's stage impls and are invisible here — this file only calls
// det_->run / rec_->run / layout_->run against one DeviceQueue.
//
// THREADING: one instance is driven by one thread at a time (same contract as
// the pipelines it replaces). Concurrency is a POOL of instances, one queue
// each — see make_infer_func.h.

#include <cstdint>   // std::uint8_t (host_staging)
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/backend/backend.h" // Backend, StageSet, DeviceQueue
#include "turbo_ocr/pipeline/unified/staging_ring.h"
#include "turbo_ocr/backend/formula_recognizer.h" // backend::IFormulaRecognizer
#include "turbo_ocr/backend/image_view.h"         // ImageView, DeviceKind
#include "turbo_ocr/backend/stages.h"             // IDetector/IRecognizer/...
#include "turbo_ocr/backend/table_recognizer.h"   // backend::ITableRecognizer
#include "turbo_ocr/backend/routing_config.h" // RequestRouting/BackendSpec
#include "turbo_ocr/core/types.h"               // OCRResultItem
#include "turbo_ocr/pipeline/pipeline_result.h"   // OcrPipelineResult
#include "turbo_ocr/pipeline/router/cua_router.h"          // router::CuaRouter
#include "turbo_ocr/pipeline/router/routing_plan.h"        // router::RoutingPlan
#include "turbo_ocr/core/service_fns.h"         // server::OrientFunc
#include "turbo_ocr/pipeline/unified/stage_batcher.h"                        // DetectionBatcher

namespace cv { class Mat; }

namespace turbo_ocr::pipeline {

// The optional-stage switches, as ONE POD instead of a positional bool run.
//
// WHY: the three entry points below used to take the same four booleans in
// three DIFFERENT orders (run_with_layout put routing/defer_external between
// reading_order and tables; the batch pair put tables/formulas before routing).
// Every one of them is `bool`, so moving an argument list from one call to
// another compiled cleanly while silently swapping "wants tables" for "wants
// reading order" — the same transposition class capability::CapabilityMask
// already eliminated one layer up, at the HTTP/gRPC seam. With a struct the
// call site names each flag (RunFlags{.layout = true, .tables = want_tables}),
// so a transposition takes a deliberate act rather than a slip.
//
// Precisely: RunFlags is an AGGREGATE, so `RunFlags{true, false, true, false}`
// and `run_with_layout(img, {true, true})` are still well-formed and still
// positional. C++ cannot both allow designated initialisers and forbid
// positional aggregate init, so the guarantee is a RULE, not a type property:
// DESIGNATED INITIALISERS ARE MANDATORY at every call site. Every current caller
// complies (make_infer_func.cpp, unified_routes.cpp, bindings.cpp,
// unified_ocr_pipeline.cpp) and a reviewer can enforce it by inspection.
//
// RunFlags is PIPELINE-owned on purpose. The pipeline must not depend on
// validation/ or server/ request types (that dependency edge is what forked the
// old per-device pipelines), so each caller translates ITS request options into
// a RunFlags at the call site.
//
// Defaults are all-false, matching the defaults the individual bool parameters
// carried, so `run_with_layout(img)` still means "text only".
struct RunFlags {
  bool layout = false;
  bool reading_order = false;
  bool tables = false;
  bool formulas = false;
  // text=false is the LAYOUT-ONLY run (?text=0&layout=1): det/cls/rec are
  // skipped entirely and the result carries layout regions with no items.
  // Validation (options_core.h) guarantees tables/formulas/blocks/
  // reading_order are all off when text is — each of those consumes
  // recognized text.
  bool text = true;
};

class UnifiedOcrPipeline {
public:
  // Takes the constructed device stages (moved out of `stages`) plus the one
  // ordered device lane this instance schedules against. `backend` outlives the
  // pipeline (owned by the server bootstrap) and supplies the device allocator,
  // the per-request table/formula recognizer factories, orientation, and the
  // image decoder.
  UnifiedOcrPipeline(backend::Backend &backend, backend::StageSet stages,
                     std::unique_ptr<backend::DeviceQueue> queue);
  ~UnifiedOcrPipeline();

  UnifiedOcrPipeline(const UnifiedOcrPipeline &) = delete;
  UnifiedOcrPipeline &operator=(const UnifiedOcrPipeline &) = delete;

  // --- Optional stage bootstrap (ported from load_router_models / load_table_
  //     backend / load_formula_model) --------------------------------------
  // Build the CUA router and the table/formula recognizer registries from the
  // routing config (TURBO_ROUTING_CONFIG or env-synth). Each backend supplies a
  // device-appropriate local recognizer via Backend::make_table/formula_
  // recognizer; VLM specs route to the shared OpenAI endpoint. Returns false
  // only when an explicitly-configured LOCAL backend fails to load (caller
  // aborts boot rather than serve a silently structure-less pipeline).
  [[nodiscard]] bool load_router_models();
  [[nodiscard]] bool load_table_backend();
  [[nodiscard]] bool load_formula_model();

  // Run a dummy image through the full pipeline to trigger lazy allocations /
  // graph JIT in the backend stages.
  void warmup();

  // --- Core entry points (device-agnostic) --------------------------------
  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img);

  // ENCODED-BYTES entry point — the one that lets a backend decode on-device.
  //
  // The cv::Mat overload below forces a HOST decode before the pipeline ever
  // sees the page, so on NVIDIA it replaces a ~200 KB H2D of the JPEG with a
  // ~25 MB H2D of the decoded A4@300dpi and throws away nvJPEG entirely. That
  // is why `Backend::make_kernels()` and `IKernels::decode_image()` existed with
  // no caller: the seam had the capability and the service boundary
  // (`InferFunc` takes a cv::Mat) made it unreachable.
  //
  // Falls back to the host decoder when the backend has no on-device decode
  // (`caps().native_image_decode == false`), so every vendor can call this and
  // only the ones that benefit change behaviour.
  [[nodiscard]] OcrPipelineResult
  run_encoded(const std::uint8_t *data, std::size_t len,
              const RunFlags &flags = {},
              const backend_routing::RequestRouting &routing = {},
              bool defer_external = false);

  [[nodiscard]] OcrPipelineResult
  run_with_layout(const cv::Mat &img, const RunFlags &flags = {},
                  const backend_routing::RequestRouting &routing = {},
                  bool defer_external = false);

  [[nodiscard]] std::vector<std::vector<OCRResultItem>>
  run_batch(const std::vector<cv::Mat> &imgs);

  [[nodiscard]] std::vector<OcrPipelineResult>
  run_batch_with_layout(const std::vector<cv::Mat> &imgs,
                        const RunFlags &flags = {},
                        const backend_routing::RequestRouting &routing = {});

  // --- Cross-image stage pipelining (SHARED — every backend inherits it) ----
  //
  // Runs `imgs` in order while keeping TWO images in flight: image N+1's
  // detection is submitted to the device the instant image N's boxes come back,
  // so it executes while image N's classification/recognition run. Results are
  // identical to calling run_with_layout() per image — only the submission
  // schedule differs.
  //
  // It is written ONCE against IDetector::enqueue()/StageFuture, so:
  //   * a backend with supports_async()==true (Apple today, CUDA by
  //     implementing enqueue as "record the det event") gets the overlap;
  //   * every other backend inherits the DEFAULT enqueue(), which runs
  //     detection eagerly — so this degrades to exactly today's sequential
  //     control flow with no branch in the caller and no behavioural change.
  //
  // Lifetime (this is what makes overlap safe): ImageView is non-owning, so the
  // pipeline holds a 2-deep ring of uploaded pages. Image N's device buffer is
  // released only after its recognition has completed, and image N+1's is
  // allocated before its detection is submitted.
  [[nodiscard]] std::vector<OcrPipelineResult>
  run_pipelined(const std::vector<cv::Mat> &imgs, const RunFlags &flags = {},
                const backend_routing::RequestRouting &routing = {});

  // True when the detector can overlap across images (drives run_pipelined's
  // choice inside run_batch_with_layout; also useful to a server pool).
  [[nodiscard]] bool supports_stage_pipelining() const noexcept;

  // NOTE (removed): set_detection_batcher(). Zero callers — the constructor
  // installs pipeline::shared_detection_batcher() already, which is what makes
  // two concurrent requests on two replicas meet in one run_batch submission.
  // The setter described a bootstrap-driven alternative that was never built.
  [[nodiscard]] const std::shared_ptr<DetectionBatcher> &
  detection_batcher() const noexcept {
    return det_batcher_;
  }

  // Page's detected clockwise rotation (0/90/180/270); 0 when unavailable.
  [[nodiscard]] int detect_orientation(const cv::Mat &bgr);

  // Tier-B ad-hoc: run ONE crop (whole image is the region) through a
  // table/formula backend, returning the raw recognized string.
  [[nodiscard]] std::string
  infer_one(const cv::Mat &img, const std::string &modality,
            const std::string &backend_name,
            const backend_routing::BackendSpec *inline_spec = nullptr);

  // --- Availability introspection -----------------------------------------
  [[nodiscard]] bool has_layout() const noexcept {
    return layout_ != nullptr;
  }
  [[nodiscard]] bool has_default_table_backend() const noexcept {
    return table_recognizer_ != nullptr;
  }
  [[nodiscard]] bool has_default_formula_backend() const noexcept {
    return formula_ != nullptr;
  }
  [[nodiscard]] bool has_table_backend(const std::string &name) const {
    return table_registry_.find(name) != table_registry_.end();
  }
  [[nodiscard]] bool has_formula_backend(const std::string &name) const {
    return formula_registry_.find(name) != formula_registry_.end();
  }
  [[nodiscard]] bool has_doc_ori() const noexcept {
    return static_cast<bool>(orient_);
  }

private:
  // A page uploaded to the backend's device space. Non-host backends own the
  // device buffer here; `host_staging` keeps the H2D source alive for the whole
  // run so a truly-async copy never reads freed memory. For the Host backend the
  // view is a zero-copy wrap of the caller's cv::Mat (buf/host_staging empty).
  struct Uploaded {
    backend::ImageView view{};
  };
  [[nodiscard]] Uploaded upload_image_(const cv::Mat &img);


  // The shared core of both entry points: everything after the page is device-
  // resident. Expressed against an ImageView so it is indifferent to whether the
  // pixels came from a host decode + H2D or from IKernels::decode_image().
  [[nodiscard]] OcrPipelineResult
  run_from_view_(const backend::ImageView &view, int rows, int cols,
                 const RunFlags &flags,
                 const backend_routing::RequestRouting &routing,
                 bool defer_external);

  // False when upload_image_ zero-copy-wraps the caller's cv::Mat (Host backend,
  // or any backend with no device allocator) — the staging ring is untouched on
  // that path, so its depth calls must be skipped too.
  [[nodiscard]] bool stages_through_ring_() const noexcept;

  StagingRing staging_;

  // The ONE detection call site. Routes through the shared cross-request
  // batcher when one is installed, and otherwise is literally det_->run().
  [[nodiscard]] std::vector<Box> detect_(const backend::ImageView &view,
                                         int orig_h, int orig_w);

  // Submit layout on its own lane when the backend can overlap it, inserting the
  // cross-lane barrier that makes the staged page visible there. Returns the
  // outstanding future; an INVALID future means this backend cannot overlap and
  // the caller must run layout synchronously — at the point of its own choosing,
  // which is not the same point on both paths (see run_pipelined).
  //
  // ONE helper because the single-image and the pipelined paths must schedule
  // layout the same way. They did not: run_with_layout overlapped it and
  // run_pipelined — the path every multi-image request takes — ran it blocking,
  // in front of cls+rec, so a batch lost exactly the overlap a single image got.
  [[nodiscard]] backend::LayoutFuture
  enqueue_layout_(const backend::ImageView &view, int orig_h, int orig_w);

  // Angle classification with the CLS_ALL_BOXES / vertical-only gate applied
  // identically on every path. Flips 180° boxes in place.
  void classify_angles_(const backend::ImageView &view, std::vector<Box> &boxes);

  // CUA router + table/formula dispatch on one page. No-op unless the router is
  // loaded and out.layout is non-empty (text-only pages pay nothing). Takes the
  // whole RunFlags (it reads .tables/.formulas) for the same reason the public
  // entry points do: three of its call sites are internal, and they used to pass
  // two adjacent bools that nothing but argument order distinguished.
  void dispatch_router_(OcrPipelineResult &out, const backend::ImageView &view,
                        const std::vector<Box> &boxes, const RunFlags &flags,
                        const backend_routing::RequestRouting &routing,
                        bool defer_external);
  void dispatch_tables_(OcrPipelineResult &out, const backend::ImageView &view,
                        backend::ITableRecognizer *table_rec, bool defer_external);
  void dispatch_formulas_(OcrPipelineResult &out, const backend::ImageView &view,
                          backend::IFormulaRecognizer *formula_rec,
                          bool defer_external);

  // Named-override-or-default recognizer pick (Tier-A). nullptr only when
  // neither the named entry nor a route default exists.
  [[nodiscard]] backend::ITableRecognizer *
  pick_table_recognizer_(const std::string &name) const;
  [[nodiscard]] backend::IFormulaRecognizer *
  pick_formula_recognizer_(const std::string &name) const;

  // Phase 4 of run_batch_with_layout: per-image layout + router + reading order.
  void run_batch_layout_stage_(const std::vector<backend::ImageCrops> &crops,
                               std::vector<OcrPipelineResult> &outs,
                               const RunFlags &flags,
                               const backend_routing::RequestRouting &routing);

  backend::Backend &backend_;
  backend::BackendCaps caps_;
  std::shared_ptr<backend::IDeviceAllocator> allocator_;
  std::unique_ptr<backend::DeviceQueue> queue_;
  // LAYOUT LANE. Layout depends only on the page, not on detection's boxes, so
  // it overlaps cls+rec instead of serialising in front of them — but only if it
  // is submitted on a DIFFERENT queue. Everything on `queue_` is one ordered
  // stream, so enqueueing layout there would overlap nothing.
  //
  // Created lazily on first async-capable layout use: a backend whose ILayout
  // has supports_async()==false never pays for a second queue, and on a host
  // backend HostDeviceQueue is a synchronous no-op anyway.
  std::unique_ptr<backend::DeviceQueue> layout_queue_;
  // On-device image decode. Created lazily on the first run_encoded() call and
  // only when caps_.native_image_decode — a backend without one never pays for
  // the object. NOTE the seam contract: the ImageView decode_image() returns is
  // valid only until the NEXT decode_image() on this object, which is why the
  // pipeline (one request at a time per replica) may hold exactly one.
  std::unique_ptr<backend::IKernels> kernels_;
  // Records the point on `queue_` at which the page's H2D has landed, so the
  // layout lane can wait on it DEVICE-SIDE. Two lanes of one device are
  // unordered with respect to each other: submitting layout on `layout_queue_`
  // without this barrier let the layout kernels read the page buffer before the
  // upload copy completed. It happened to be masked because detection blocks
  // (it returns host boxes), which is precisely the kind of accident that stops
  // holding the moment a detector goes async. Minted lazily; never used on a
  // backend whose queues are synchronous.
  std::unique_ptr<backend::DeviceEvent> upload_event_;

  std::unique_ptr<backend::IDetector> det_;
  std::unique_ptr<backend::IRecognizer> rec_;
  std::unique_ptr<backend::IClassifier> cls_;
  std::unique_ptr<backend::ILayout> layout_;
  server::OrientFunc orient_;

  // Shared by every replica of this backend; nullptr => detection is called
  // directly, exactly as before this existed.
  std::shared_ptr<DetectionBatcher> det_batcher_;

  bool use_cls_ = false;

  // CUA router + per-modality recognizer registries. The registry OWNS every
  // per-request-routable backend; table_recognizer_/formula_ are NON-OWNING
  // pointers to the route-default entry (the hot-path pick).
  std::unique_ptr<router::CuaRouter> router_;
  router::RoutingPlan plan_;
  std::map<std::string, std::unique_ptr<backend::ITableRecognizer>> table_registry_;
  std::map<std::string, std::unique_ptr<backend::IFormulaRecognizer>>
      formula_registry_;
  backend::ITableRecognizer *table_recognizer_ = nullptr;   // registry-owned default
  backend::IFormulaRecognizer *formula_ = nullptr;        // registry-owned default

  // Reusable scratch for the vertical-only cls gate.
  std::vector<int> vertical_box_indices_;
  std::vector<Box> vertical_boxes_buf_;
};

} // namespace turbo_ocr::pipeline
