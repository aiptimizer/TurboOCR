#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/classification/doc_orientation.h"
#include "turbo_ocr/classification/paddle_cls.h"
#include "turbo_ocr/common/log/timing.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/detection/paddle_det.h"
#include "turbo_ocr/layout/paddle_layout.h"
#include "turbo_ocr/pipeline/ocr/i_ocr_pipeline.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/recognition/paddle_rec.h"
#include "turbo_ocr/router/routing_plan.h"
#include "turbo_ocr/backend_routing/routing_config.h"

namespace turbo_ocr::router  { class CuaRouter; }
namespace turbo_ocr::table   { class ITableRecognizer; }
namespace turbo_ocr::formula { class IFormulaRecognizer; }

namespace turbo_ocr::pipeline {

// THREADING CONTRACT (was undocumented): ONE OcrPipeline instance is driven by
// exactly ONE thread at a time. The double-buffered upload (cur_img_buf_,
// h_pinned_buf_), the per-modality streams/events, and the grow-only device
// buffers are all unsynchronized instance state — concurrency is achieved by a
// POOL of instances (one per worker thread), never by sharing one instance
// across threads. Driving a single instance from two threads torns cur_img_buf_
// and corrupts device memory. `in_use_` below is a debug reentrancy tripwire that
// aborts loudly on a contended entry instead of corrupting silently.
class OcrPipeline : public IOcrPipeline {
public:
  OcrPipeline();
  ~OcrPipeline() noexcept override;

  [[nodiscard]] bool init(const std::string &det_model, const std::string &rec_model,
                          const std::string &rec_dict,
                          const std::string &cls_model = "",
                          const DetInferConfig &det_cfg =
                              {turbo_ocr::detection::kDetResizeDefault,
                               turbo_ocr::detection::kDbDefaults}) override;

  // Optionally load a PP-DocLayoutV3 model. Must be called after init().
  // Call once per pipeline. After a successful call, run_with_layout(...)
  // returns both text results and layout detections in one struct.
  // Plain run(...) continues to return text only (layout is computed and
  // discarded to keep a stable API for non-layout callers).
  [[nodiscard]] bool load_layout_model(const std::string &layout_trt_path);

  // Pluggable table backend. TABLE_BACKEND selects slanext|vlm;
  // each backend reads its own env (TABLE_SLANEXT_*,
  // VLLM_TABLE_*) in load() and self-skips when unconfigured. When loaded,
  // detected table regions are recognized by this one ITableRecognizer.
  [[nodiscard]] bool load_table_backend();

  // Load the CUA router + table-stage + formula engines in one call.
  // Pass an empty path for any optional component (e.g. wireless table
  // engines, or the formula engine entirely) to skip it — the loader
  // returns true as long as the *required* parts (router + wired table)
  // initialised. Streams and events for the new modalities are created
  // lazily inside this call; pipelines that never invoke it allocate
  // zero new CUDA resources (required by plan 04 §7).
  [[nodiscard]] bool
  load_router_models(const std::string &table_cls_trt,
                     const std::string &cell_wired_trt,
                     const std::string &cell_wireless_trt,
                     const std::string &slanext_wired_trt,
                     const std::string &slanext_wireless_trt,
                     const std::string &formula_onnx,
                     const std::string &formula_tokenizer_json);

  // Load the document-orientation model (PP-LCNet_x1_0_doc_ori). Optional;
  // when loaded, detect_orientation() powers /ocr/pdf?autorotate=1. Must be
  // called after init(). Returns false on load failure (caller logs + treats
  // autorotate as unavailable).
  [[nodiscard]] bool load_doc_ori_model(const std::string &doc_ori_trt_path);
  [[nodiscard]] bool has_doc_ori() const noexcept { return use_doc_ori_; }

  // True once (and clears) when a recoverable CUDA fault in the table/formula
  // stage flagged this pipeline for rebuild. The dispatcher worker consumes it
  // after each task and recycles the entry, so a wedge-free slot self-heals
  // instead of returning 500s on every subsequent request.
  [[nodiscard]] bool consume_recycle_request() noexcept {
    return recycle_requested_.exchange(false, std::memory_order_relaxed);
  }

  // --- Per-request routing (Tier-A/B) introspection + ad-hoc inference -----

  // True when `name` is a registered table/formula backend (i.e. a legal
  // per-request override target). Used by Tier-B /infer; Tier-A route-layer
  // validation uses backend_routing::routable_backend_names (same set, no pipeline
  // round-trip needed).
  [[nodiscard]] bool has_table_backend(const std::string &name) const {
    return table_registry_.find(name) != table_registry_.end();
  }
  [[nodiscard]] bool has_formula_backend(const std::string &name) const {
    return formula_registry_.find(name) != formula_registry_.end();
  }

  // Default (no-override) opt-in availability: these are the exact pointers
  // dispatch_router_ picks for routing.table=="default" / the default formula
  // route, so the server's `tables=1`/`formulas=1` fail-loud gate reads the
  // truth from what actually loaded rather than re-deriving it from config.
  [[nodiscard]] bool has_default_table_backend() const {
    return table_recognizer_ != nullptr;
  }
  [[nodiscard]] bool has_default_formula_backend() const {
    return formula_ != nullptr;
  }

  // Tier-B: run ONE crop (the whole image is the region) through a backend for
  // `modality` ("table"|"formula") and return the raw recognized string (HTML
  // for table, LaTeX/text for formula). Empty string for the legitimately-empty
  // result (recognizer ran, produced nothing) or an empty/undecodable crop.
  //   - inline_spec == nullptr -> use the NAMED registry backend `backend_name`
  //     (or the route default when empty).
  //   - inline_spec != nullptr -> build a TRANSIENT recognizer from that spec
  //     ON THIS WORKER THREAD (so any TRT build binds to the worker's CUDA
  //     context), run it once, and discard it. `backend_name` is ignored.
  // Throws RoutingConfigError if an inline spec names an engine invalid for the
  // modality (the route maps that to a 400).
  // Throws BackendUnavailableError when the EXPLICITLY requested backend (named
  // pick or inline spec) is null or not ready — so a failed-to-load backend
  // can't masquerade as an empty-but-successful result (route -> 503).
  [[nodiscard]] std::string
  infer_one(const cv::Mat &img, cudaStream_t stream, const std::string &modality,
            const std::string &backend_name,
            const backend_routing::BackendSpec *inline_spec = nullptr);

  // Detect a rendered page's clockwise rotation (0/90/180/270). Returns 0 when
  // the model isn't loaded or confidence is low. Used by the autorotate path
  // to de-rotate the page before OCR + image encode.
  [[nodiscard]] int detect_orientation(const cv::Mat &bgr, cudaStream_t stream);

  // IOcrPipeline interface — delegates to stream-aware overloads with stream=0
  void warmup() override { warmup_gpu(0); }
  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img) override { return run(img, 0); }

  // GPU-specific: Pre-allocate all buffers and run a dummy inference to warm up
  // TensorRT engines and lazy GPU allocations.
  void warmup_gpu(cudaStream_t stream);

  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img, cudaStream_t stream);

  // Run OCR on an image already in GPU memory (skips CPU→GPU upload).
  // The GpuImage must contain BGR8 data with the given pitch/step.
  // The caller retains ownership of the GPU buffer.
  [[nodiscard]] std::vector<OCRResultItem> run(const GpuImage &gpu_img, cudaStream_t stream = 0);

  // Text + layout in one struct. Returns an empty `layout` vector when the
  // pipeline was never loaded with a layout model, so this overload is
  // always safe to call. Preferred over `run()` for HTTP/gRPC paths that
  // surface layout results to clients — it removes the need for a
  // side-channel getter on the pipeline object.
  // Layout is OPT-IN per call: `want_layout=false` (default) runs only
  // det/cls/rec; `want_layout=true` additionally runs layout if the model
  // is loaded. When the pipeline has no layout model, the flag has no
  // effect (the output `.layout` is always empty). Callers (HTTP/gRPC
  // routes) parse `?layout=1` from the request and pass it through.
  // `routing` is an OPTIONAL per-request backend override (Tier-A). Defaulted
  // empty so existing callers are unchanged; when a modality's name is set, the
  // table/formula regions of THIS request are dispatched to that named registry
  // backend instead of the configured route default. An unknown name falls back
  // to the default (route-layer validation rejects unknown names with a 400
  // before they ever reach here).
  // `defer_external`: when true AND the resolved formula/table backend is a
  // remote async-capable one (supports_async()), this call does all GPU work +
  // a NON-BLOCKING crop submit and returns immediately with the pending futures
  // stashed in result.pending — freeing the GPU pipeline slot during the VL
  // network round-trip. The caller MUST then call finalize_deferred() (off the
  // GPU worker) to await + assemble. Defaulted false → fully synchronous, so
  // gRPC / PDF / batch / cpu_main callers are byte-identical. Only /ocr/raw
  // (single-image, the throughput path) opts in today.
  [[nodiscard]] OcrPipelineResult run_with_layout(const cv::Mat &img,
                                                   cudaStream_t stream,
                                                   bool want_layout = false,
                                                   bool want_reading_order = false,
                                                   const backend_routing::RequestRouting &routing = {},
                                                   bool defer_external = false,
                                                   bool want_tables = false,
                                                   bool want_formulas = false);
  [[nodiscard]] OcrPipelineResult run_with_layout(GpuImage gpu_img,
                                                   cudaStream_t stream = 0,
                                                   bool want_layout = false,
                                                   bool want_reading_order = false,
                                                   const backend_routing::RequestRouting &routing = {},
                                                   bool defer_external = false,
                                                   bool want_tables = false,
                                                   bool want_formulas = false);

  // Layout-only path: upload the image, run the PP-DocLayoutV3 inference,
  // collect the boxes, and return. Skips detection, angle classification,
  // and recognition entirely — no CTC decode, no rec_stream synchronisation,
  // no wasted inference.
  //
  // Used by /ocr/pdf mode=geometric and mode=auto (native-text pages) when
  // ENABLE_LAYOUT=1: the page's text comes from the PDFium text layer and
  // only the visual `layout` array still needs the rendered image. Returns
  // an OcrPipelineResult with empty `.results` (caller fills from the text
  // layer) and populated `.layout` (or empty when the pipeline has no
  // layout model, in which case this method is a no-op and still safe).
  [[nodiscard]] OcrPipelineResult run_layout_only(const cv::Mat &img,
                                                   cudaStream_t stream);
  // Same, for an image already on the device (the JPEG paths decode straight
  // into the pipeline's buffer): no upload, no host copy of the pixels.
  [[nodiscard]] OcrPipelineResult run_layout_only(GpuImage gpu_img,
                                                   cudaStream_t stream);

  // Layout + table/formula recognition WITHOUT detection/recognition. Used by
  // /ocr/pdf mode=geometric/auto when the client asked for structure: the page
  // text comes from the PDFium text layer (passed in as `text_results`, already
  // in this image's pixel space), so det/rec is skipped, but tables (→ HTML) and
  // formulas (→ LaTeX) still need the rendered image and the layout regions.
  // Uploads the image once, runs layout, then the CUA router over the supplied
  // text boxes — the same dispatch_router_ the OCR path uses. Returns the text
  // results verbatim plus populated `.layout` / `.tables` / `.formulas`. A no-op
  // (returns just the text) when layout isn't loaded.
  [[nodiscard]] OcrPipelineResult run_layout_and_structure(
      const cv::Mat &img, cudaStream_t stream,
      std::vector<OCRResultItem> text_results, bool want_tables,
      bool want_formulas, const backend_routing::RequestRouting &routing = {});

  // Ensure the GPU upload buffer can hold an image of the given size.
  // Returns {d_img_buf_, d_img_pitch_} after grow-only reallocation.
  // Useful for callers that want to decode directly into the pipeline's GPU buffer.
  [[nodiscard]] std::pair<void *, size_t> ensure_gpu_buf(int rows, int cols);

  // Batch: process multiple images, return text results per image.
  // Delegates to run_batch_with_layout with want_layout=false, so
  // text-only batches pay zero layout/router cost.
  [[nodiscard]] std::vector<std::vector<OCRResultItem>>
  run_batch(const std::vector<cv::Mat> &imgs, cudaStream_t stream = 0);

  // Batch counterpart of run_with_layout: batched det, cross-image batched
  // rec, and — when want_layout — per-page layout + CUA router per
  // page. Returns full per-page results: text, layout, tables, formulas,
  // and reading order when requested. Callers should chunk at most
  // kMaxBatchImages images per call (extra images are dropped with a
  // warning, same contract as run_batch).
  [[nodiscard]] std::vector<OcrPipelineResult>
  run_batch_with_layout(const std::vector<cv::Mat> &imgs,
                        cudaStream_t stream = 0,
                        bool want_layout = false,
                        bool want_reading_order = false,
                        bool want_tables = false,
                        bool want_formulas = false,
                        const backend_routing::RequestRouting &routing = {});

private:
  std::unique_ptr<detection::PaddleDet> det_;
  std::unique_ptr<classification::PaddleCls> cls_;
  std::unique_ptr<recognition::PaddleRec> rec_;
  std::unique_ptr<layout::PaddleLayout> layout_;
  std::unique_ptr<classification::DocOrientation> doc_ori_;

  bool use_cls_ = false;
  bool use_layout_ = false;
  bool use_doc_ori_ = false;

  // Shared GPU upload: wait for previous rec, toggle double-buffer, grow-only
  // realloc, pinned staging memcpy, async H2D. Used by run_with_layout and
  // run_layout_only.
  GpuImage upload_image(const cv::Mat &img, cudaStream_t stream,
                        PipelineTimer &timer);

  // Block until every consumer of the previous request's image buffers is
  // done: recognition (rec_event_) AND the table/formula streams
  // (table_done_event_ / formula_done_event_). Called before any buffer
  // reuse. Today the local structure backends host-sync before returning,
  // so the table/formula waits are no-ops — but that is their contract
  // enforcement, not an assumption: a future truly-async backend would
  // otherwise read a buffer the next request is overwriting.
  void wait_prior_readers_();

  // Angle classification with the CLS_ALL_BOXES / vertical-only gate applied
  // identically on every pipeline path (single cv::Mat, GpuImage, batch).
  // Flips 180-degree boxes in place. timer may be null (batch path).
  void classify_angles_(const GpuImage &img, std::vector<Box> &boxes,
                        cudaStream_t stream, PipelineTimer *timer);

  // Dedicated stream for recognition — allows det on the caller's stream to
  // overlap with rec on this stream across consecutive requests.
  cudaStream_t rec_stream_ = nullptr;

  // Event recorded on rec_stream_ after recognition GPU work is launched.
  // The next run() waits on this before reusing the image buffer that rec
  // might still be reading.  This enables det/rec overlap across calls:
  //   Call N:    [...det_N on stream...] → launch rec_N on rec_stream_ (async)
  //   Call N+1:  wait rec_event_ → [...det_{N+1} overlaps with tail of rec_N...]
  cudaEvent_t  rec_event_  = nullptr;

  // Event recorded on the caller's stream after det+cls finish.
  // rec_stream_ waits on this before launching recognition, ensuring proper
  // data dependency without a full stream synchronize.
  cudaEvent_t  det_event_  = nullptr;

  // Dedicated stream for optional layout detection. Layout reads gpu_img
  // (written by det's preprocess) and runs independently of cls/rec, so
  // running it on its own stream lets it overlap with rec on rec_stream_
  // and with cls on the caller's stream. Only allocated when layout is
  // enabled via load_layout_model().
  cudaStream_t layout_stream_ = nullptr;

  // Event recorded on the caller's stream right after detection, BEFORE cls.
  // layout_stream_ waits on this so layout can start as soon as det is done
  // without also waiting for cls/rec. Only used when layout is enabled.
  cudaEvent_t  det_only_event_ = nullptr;

  // Double-buffered GPU upload buffers (grow-only).
  // Alternating between two buffers lets recognition read the previous image
  // on rec_stream_ while the current image is uploaded + detected on the
  // caller's stream without a data race.
  int cur_img_buf_ = 0; // toggles 0/1 each run() call
  struct ImgBuf {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int cap_rows = 0;
    int cap_cols = 0;
  };
  ImgBuf img_bufs_[2];

  // Pinned host staging buffer for truly async CPU->GPU uploads
  void *h_pinned_buf_ = nullptr;
  size_t h_pinned_size_ = 0;

  // Grow-only realloc for a pitched device image buffer (ImgBuf /
  // BatchImgBuf fields passed by reference). Zeroes the caps BEFORE the
  // alloc so an OOM throw leaves {nullptr, cap 0} — the next request then
  // re-enters the realloc instead of writing through a stale-cap null buffer.
  static void grow_pitch_buf_(void *&d_buf, size_t &pitch, int &cap_rows,
                              int &cap_cols, int rows, int cols);
  // Same grow-only / zero-before-alloc contract for the shared pinned
  // staging buffer.
  void grow_pinned_(size_t needed);

  // Reusable buffers for selective angle classification
  std::vector<int> vertical_box_indices_;
  std::vector<Box> vertical_boxes_buf_;

  // Pre-allocated batch image GPU buffers (avoid cudaMalloc per batch image)
  static constexpr int kMaxBatchImages = 8;
  struct BatchImgBuf {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int cap_rows = 0, cap_cols = 0;
  };
  BatchImgBuf batch_img_bufs_[kMaxBatchImages];

  // ---- CUA router + table/formula stages (lazy-allocated) ----------------
  std::unique_ptr<router::CuaRouter>   router_;

  // Per-modality recognizer registry: backend NAME -> recognizer. The registry
  // OWNS every per-request-routable backend (the route default + all kind:openai
  // backends, per backend_routing::routable_backend_names). In zero-config / env-synth
  // mode there is exactly one entry per modality, so this is byte-identical to a
  // single recognizer and costs zero extra VRAM. table_recognizer_ / formula_
  // are NON-OWNING pointers to the route-default entry, keeping the hot path a
  // plain pointer pick.
  std::map<std::string, std::unique_ptr<table::ITableRecognizer>>   table_registry_;
  std::map<std::string, std::unique_ptr<formula::IFormulaRecognizer>> formula_registry_;
  table::ITableRecognizer       *table_recognizer_ = nullptr;     // default entry (registry-owned)
  formula::IFormulaRecognizer   *formula_          = nullptr;     // default entry (registry-owned)

  cudaStream_t table_stream_       = nullptr;
  cudaStream_t formula_stream_     = nullptr;
  cudaEvent_t  table_done_event_   = nullptr;
  cudaEvent_t  formula_done_event_ = nullptr;

  // Set by dispatch_router_ on a recoverable CUDA fault in the table/formula
  // stage; consumed by the dispatcher worker to trigger an entry rebuild.
  std::atomic<bool> recycle_requested_{false};
  // Debug reentrancy tripwire for the single-thread-per-instance contract above.
  std::atomic<int> in_use_{0};
  // RAII tripwire scope: aborts loudly on contended entry (ctor out-of-line to
  // keep <iostream> out of this header). Acquired once per EXTERNAL call at the
  // real entry bodies only — pure-delegation wrappers (run -> run_with_layout,
  // run_batch -> run_batch_with_layout -> n==1 run_with_layout) must NOT
  // acquire it or the nested self-call would trip its own guard.
  struct UseGuard {
    std::atomic<int> &c;
    UseGuard(std::atomic<int> &counter, const char *entry);
    ~UseGuard() { c.fetch_sub(1, std::memory_order_release); }
    UseGuard(const UseGuard &) = delete;
    UseGuard &operator=(const UseGuard &) = delete;
  };

  // Lazily create the table/formula stream + done-event on first use. The
  // pipeline owns and destroys these CUDA resources; the recognizer-registry
  // construction helpers call these (via a callback) when they register a
  // backend that will run on the corresponding stream.
  void ensure_table_stream_();
  void ensure_formula_stream_();

  // Reusable per-call routing plan — member, not local, to avoid the
  // per-page heap churn of the inner vectors.
  router::RoutingPlan plan_;

  // Apply the router + dispatch table/formula stages on a single page.
  // No-op (returns immediately) unless `router_` is loaded AND `out.layout`
  // is non-empty — text-only callers see zero new CUDA API calls.
  void dispatch_router_(OcrPipelineResult &out,
                        const GpuImage &gpu_img,
                        const std::vector<Box> &boxes,
                        PipelineTimer &timer,
                        const backend_routing::RequestRouting &routing = {},
                        bool defer_external = false,
                        bool want_tables = false,
                        bool want_formulas = false);

  // Resolve the recognizer for a per-request dispatch: the named override entry
  // when `name` is non-empty AND present in the registry, else the route
  // default. Returns nullptr only when neither exists.
  [[nodiscard]] table::ITableRecognizer *
  pick_table_recognizer_(const std::string &name) const;
  [[nodiscard]] formula::IFormulaRecognizer *
  pick_formula_recognizer_(const std::string &name) const;

  // Table / formula halves of dispatch_router_ (ocr_pipeline_dispatch.cpp).
  // Preconditions established by the dispatch_router_ shell: plan_ is
  // classified, the recognizer pointer is non-null, and the corresponding
  // plan_ region-id list is non-empty.
  void dispatch_tables_(OcrPipelineResult &out, const GpuImage &gpu_img,
                        PipelineTimer &timer,
                        table::ITableRecognizer *table_rec,
                        bool defer_external);
  void dispatch_formulas_(OcrPipelineResult &out, const GpuImage &gpu_img,
                          PipelineTimer &timer,
                          formula::IFormulaRecognizer *formula_rec,
                          bool defer_external);

  // Build + register every kind:openai backend (for `modality`) that isn't
  // already the route default, so per-request overrides can address them with
  // no first-request build latency. Openai clients are cheap (no GPU/VRAM);
  // local heavy engines are intentionally NOT prewarmed here.
  void prewarm_openai_registry_(const std::string &modality,
                                const backend_routing::RoutingTable &tbl);

  // Phase 4 of run_batch_with_layout: per-image layout, CUA router dispatch,
  // and reading-order assignment. `image_crops[i].img` is page i on GPU; `.boxes` are
  // the det boxes recognition ran on. Caller guarantees layout_ is loaded.
  void run_batch_layout_stage_(
      const std::vector<recognition::PaddleRec::ImageCrops> &image_crops,
      bool want_reading_order,
      cudaStream_t stream,
      std::vector<OcrPipelineResult> &outs,
      bool want_tables = false,
      bool want_formulas = false,
      const backend_routing::RequestRouting &routing = {});
};

} // namespace turbo_ocr::pipeline
