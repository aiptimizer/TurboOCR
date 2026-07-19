#include <iostream>
#include "turbo_ocr/pipeline/ocr_pipeline.h"
#include <unordered_map>
#include "infer_one.h"
#include "ocr_pipeline_detail.h"
#include "recognizer_registry.h"
#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/backend_error.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/timing.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/layout/reading_order.h"
#include "turbo_ocr/router/cua_router.h"
#include "turbo_ocr/routing/routing_config.h"
#include "turbo_ocr/table/table_recognizer.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"
#include "turbo_ocr/table/slanext_enc_split.h"
#include "turbo_ocr/table/table_types.h"
#include "turbo_ocr/engine/onnx_to_trt.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>

#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::pipeline;
using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::GpuImage;
using turbo_ocr::PipelineTimer;
// is_vertical_box / sorted_boxes are called unqualified below; ADL resolves them
// to turbo_ocr:: from their Box / vector<Box> arguments, so no using-decl needed.
using turbo_ocr::detection::PaddleDet;
using turbo_ocr::classification::PaddleCls;
using turbo_ocr::recognition::PaddleRec;
using turbo_ocr::layout::PaddleLayout;
using turbo_ocr::pipeline::OcrPipelineResult;

using turbo_ocr::pipeline::detail::adjust_table_region;

OcrPipeline::OcrPipeline() {
  det_ = std::make_unique<PaddleDet>();
  rec_ = std::make_unique<PaddleRec>();
}

OcrPipeline::~OcrPipeline() noexcept {
  if (rec_stream_)
    cudaStreamDestroy(rec_stream_);
  if (layout_stream_)
    cudaStreamDestroy(layout_stream_);
  if (table_stream_)
    cudaStreamDestroy(table_stream_);
  if (formula_stream_)
    cudaStreamDestroy(formula_stream_);
  if (rec_event_)
    cudaEventDestroy(rec_event_);
  if (det_event_)
    cudaEventDestroy(det_event_);
  if (det_only_event_)
    cudaEventDestroy(det_only_event_);
  if (table_done_event_)
    cudaEventDestroy(table_done_event_);
  if (formula_done_event_)
    cudaEventDestroy(formula_done_event_);
  for (auto &buf : img_bufs_) {
    if (buf.d_buf)
      cudaFree(buf.d_buf);
  }
  cudaFreeHost(h_pinned_buf_);
  for (auto &buf : batch_img_bufs_) {
    if (buf.d_buf)
      cudaFree(buf.d_buf);
  }
}

bool OcrPipeline::init(const std::string &det_model,
                       const std::string &rec_model,
                       const std::string &rec_dict,
                       const std::string &cls_model,
                       const DetInferConfig &det_cfg) {
  if (!det_->load_model(det_model, det_cfg.resize, det_cfg.db))
    return false;
  if (!rec_->load_model(rec_model))
    return false;
  if (!rec_->load_dict(rec_dict))
    return false;

  if (!cls_model.empty()) {
    cls_ = std::make_unique<PaddleCls>();
    if (!cls_->load_model(cls_model)) {
      std::cerr << std::format("[Pipeline] Failed to load GPU cls model: {}", cls_model) << '\n';
      return false;
    }
    use_cls_ = true;
    std::cout << "[Pipeline] Angle classifier enabled ("
              << (classification::cls_all_boxes_enabled() ? "all boxes"
                                                          : "vertical-only gate")
              << ")" << '\n';
  }

  // Pre-allocate double-buffered GPU upload buffers for a typical image
  // (1920x1080). Grow-only: reused for smaller images, reallocated only if a
  // larger image arrives. Two device buffers let recognition on rec_stream_
  // read the previous image while the next uploads on the caller's stream —
  // NOTE: today that overlap is mostly latent, because upload_image host-waits
  // on the LATEST rec_event_ and stages through ONE shared pinned buffer; a
  // pinned ring matched to img_bufs_ (waiting only on the OLDER buffer's rec)
  // is the known follow-up that would realize it.
  constexpr int kDefaultRows = 1080;
  constexpr int kDefaultCols = 1920;
  for (auto &buf : img_bufs_) {
    CUDA_CHECK(
        cudaMallocPitch(&buf.d_buf, &buf.pitch, kDefaultCols * 3, kDefaultRows));
    buf.cap_rows = kDefaultRows;
    buf.cap_cols = kDefaultCols;
  }

  // Eagerly allocate rec/cls buffers (avoids first-request latency)
  rec_->allocate_buffers();
  if (use_cls_)
    cls_->allocate_buffers();

  // Dedicated recognition stream — allows det on the caller's stream to
  // overlap with rec on rec_stream_ across consecutive pipeline invocations.
  CUDA_CHECK(cudaStreamCreateWithFlags(&rec_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&rec_event_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&det_event_, cudaEventDisableTiming));

  return true;
}

bool OcrPipeline::load_router_models(
    const std::string &table_cls_trt,
    const std::string &cell_wired_trt,
    const std::string &cell_wireless_trt,
    const std::string &slanext_wired_trt,
    const std::string &slanext_wireless_trt,
    const std::string &formula_onnx,
    const std::string &formula_tokenizer_json) {
  // Router itself is CPU-only and cheap; build it unconditionally so the
  // classify path is available even when no downstream engines load.
  router_ = std::make_unique<router::CuaRouter>();

  // The live table path is the SlanextTableRecognizer registry built in
  // load_table_backend(); the table-engine arguments above are retained for
  // ABI stability but no longer drive a separate cell-det stage here.
  (void)table_cls_trt;
  (void)cell_wired_trt;
  (void)cell_wireless_trt;
  (void)slanext_wired_trt;
  (void)slanext_wireless_trt;

  // Formula half — backend selected via FORMULA_BACKEND
  // (formulanet | ppformulanet_s | vlm). The file-based backends are guarded
  // against missing files so the server starts cleanly while upstream
  // re-exports the split-ONNX bundle (see internal engineering notes);
  // the "vlm" backend talks to a remote endpoint and needs no local files.
  // Formula backend from the routing table (TURBO_ROUTING_CONFIG, or
  // env-synthesized from FORMULA_BACKEND/VLLM_* = today's behavior). kind:openai
  // -> generic OpenAIEndpoint; kind:local -> engine factory. load_model_dir/
  // load_tokenizer are health-check/no-op for the remote backend, so the load
  // flow below stays uniform. Construction + registry insertion lives in
  // recognizer_registry; this pipeline only owns the CUDA stream lifetime,
  // passed in as the ensure-callback below.
  const routing::RoutingTable formula_routing = routing::load_routing_config();
  const bool formula_ok =
      load_formula_into_registry(formula_registry_, formula_, formula_routing,
                                 formula_onnx, formula_tokenizer_json,
                                 [this] { ensure_formula_stream_(); });

  // Prewarm every kind:openai formula-routable backend that ISN'T the default
  // (cheap HTTP clients) so a per-request override can target them with no
  // first-request build latency. Local heavy engines are NOT prewarmed (would
  // cost extra VRAM); they're only built when they're the route default above.
  prewarm_openai_registry_("formula", formula_routing);

  // false only when a CONFIGURED local formula backend failed to load; the
  // caller turns that into a fatal boot rather than a silently formula-less server.
  return formula_ok;
}

bool OcrPipeline::load_layout_model(const std::string &layout_trt_path) {
  if (layout_trt_path.empty()) return false;
  auto layout = std::make_unique<PaddleLayout>();
  if (!layout->load_model(layout_trt_path)) {
    std::cerr << std::format("[Pipeline] Failed to load layout model: {}",
                             layout_trt_path)
              << '\n';
    return false;
  }
  layout_ = std::move(layout);
  use_layout_ = true;
  // Dedicated layout stream — allows layout TRT execute to overlap with
  // cls/rec on their own streams. Only allocated here because non-layout
  // pipelines don't need this stream or its event.
  CUDA_CHECK(cudaStreamCreateWithFlags(&layout_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&det_only_event_, cudaEventDisableTiming));
  std::cout << "[Pipeline] Layout detection enabled" << '\n';
  return true;
}

bool OcrPipeline::load_table_backend() {
  // Pluggable table backend: TABLE_BACKEND selects slanext|vlm; each
  // backend reads its own env (TABLE_SLANEXT_*, VLLM_TABLE_*)
  // in load() and returns false when unconfigured (so this no-ops cleanly).
  // Table backend from the routing table (TURBO_ROUTING_CONFIG, or env-synth
  // from TABLE_BACKEND/VLLM_TABLE_* = today). kind:openai -> OpenAIEndpoint;
  // kind:local -> slanext|vlm. Construction + registry insertion lives in
  // recognizer_registry; the pipeline owns the CUDA stream (ensure-callback).
  const routing::RoutingTable table_routing = routing::load_routing_config();
  if (!load_table_into_registry(table_registry_, table_recognizer_, table_routing,
                                [this] { ensure_table_stream_(); }))
    return false;
  // Prewarm kind:openai table-routable overrides (cheap HTTP clients).
  prewarm_openai_registry_("table", table_routing);
  return true;
}

bool OcrPipeline::load_doc_ori_model(const std::string &doc_ori_trt_path) {
  if (doc_ori_trt_path.empty()) return false;
  auto doc_ori = std::make_unique<classification::DocOrientation>();
  if (!doc_ori->load_model(doc_ori_trt_path)) {
    std::cerr << std::format("[Pipeline] Failed to load doc-orientation model: {}",
                             doc_ori_trt_path)
              << '\n';
    return false;
  }
  doc_ori->allocate_buffers();
  doc_ori_ = std::move(doc_ori);
  use_doc_ori_ = true;
  std::cout << "[Pipeline] Document-orientation (autorotate) enabled" << '\n';
  return true;
}

int OcrPipeline::detect_orientation(const cv::Mat &bgr, cudaStream_t stream) {
  if (!use_doc_ori_) return 0;
  return doc_ori_->detect(bgr, stream);
}

void OcrPipeline::warmup_gpu(cudaStream_t stream) {
  // Run full pipeline with a dummy image to trigger TRT JIT and lazy GPU
  // allocations. If layout is enabled, the run(...) body hits layout too.
  {
    cv::Mat dummy(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::rectangle(dummy, cv::Point(10, 30), cv::Point(90, 70), cv::Scalar(0, 0, 0), 2);
    (void)run(dummy, stream);
    cudaStreamSynchronize(stream);
  }

  // Record the per-(batch,width)-profile CUDA graphs before the bucket warm
  // loop below, so the loop exercises the graph-replay path end to end.
  rec_->bake_graphs(stream);

  // Warm all 5 rec width buckets to eliminate TRT JIT latency on first use.
  // The initial run() above only hits one bucket; each unseen bucket pays ~5-50ms
  // on first inference. We call rec_->run() directly with fake boxes sized to
  // produce crops at each bucket width.
  static constexpr int warm_widths[] = {320, 480, 800, 1600, 4000};
  auto &buf = img_bufs_[0]; // use first buffer for warmup
  for (int w : warm_widths) {
    // Create a white dummy image wide enough for this bucket
    cv::Mat dummy_wide(48, w, CV_8UC3, cv::Scalar(255, 255, 255));

    // Upload to GPU (reuses the grow-only buffer). Grow BOTH dims to the max
    // of current and needed: sizing to the dummy alone (48 rows x 4000 cols)
    // would SHRINK the row cap and force a realloc on the first real page
    // after warmup — a grow-only violation.
    if (dummy_wide.rows > buf.cap_rows || dummy_wide.cols > buf.cap_cols) {
      const int want_rows = std::max(buf.cap_rows, dummy_wide.rows);
      const int want_cols = std::max(buf.cap_cols, dummy_wide.cols);
      cudaFree(buf.d_buf);
      buf.d_buf = nullptr;
      // Zero the cap BEFORE the alloc (matches upload_image/ensure_gpu_buf): an
      // OOM throw must leave {nullptr, cap 0}, never a null buffer with a stale
      // non-zero cap that a later request would skip re-allocating.
      buf.cap_rows = 0;
      buf.cap_cols = 0;
      CUDA_CHECK(cudaMallocPitch(&buf.d_buf, &buf.pitch,
                                  static_cast<size_t>(want_cols) * 3, want_rows));
      buf.cap_rows = want_rows;
      buf.cap_cols = want_cols;
    }
    auto needed = static_cast<size_t>(dummy_wide.rows) * dummy_wide.step;
    if (needed > h_pinned_size_) {
      cudaFreeHost(h_pinned_buf_);
      h_pinned_buf_ = nullptr;
      h_pinned_size_ = 0;  // zero before alloc — see cap reset above
      // Upload-only pinned buffer: CPU writes (memcpy) once and the GPU DMAs
      // it. Write-combined uncached memory is ~10-15% faster for this access
      // pattern (no read-back from CPU).
      CUDA_CHECK(cudaHostAlloc(&h_pinned_buf_, needed, cudaHostAllocWriteCombined));
      h_pinned_size_ = needed;
    }
    std::memcpy(h_pinned_buf_, dummy_wide.data, needed);
    CUDA_CHECK(cudaMemcpy2DAsync(buf.d_buf, buf.pitch, h_pinned_buf_,
                                  dummy_wide.step, dummy_wide.cols * 3,
                                  dummy_wide.rows, cudaMemcpyHostToDevice, stream));

    GpuImage gpu_img{buf.d_buf, buf.pitch, dummy_wide.rows, dummy_wide.cols};

    // A single box spanning the full image -> crop width = w
    Box box{};
    box[0] = {0, 0};                      // top-left
    box[1] = {w - 1, 0};                  // top-right
    box[2] = {w - 1, dummy_wide.rows - 1}; // bottom-right
    box[3] = {0, dummy_wide.rows - 1};     // bottom-left
    std::vector<Box> boxes = {box};

    auto rec_res = rec_->run(gpu_img, boxes, stream);
    cudaStreamSynchronize(stream);
    (void)rec_res;
  }
}

void OcrPipeline::wait_prior_readers_() {
  CUDA_CHECK(cudaEventSynchronize(rec_event_));
  // Table/formula backends read the image buffer on their own streams. The
  // local backends host-sync inside run(), so these waits complete instantly
  // today — they exist to make buffer reuse safe even for a backend that
  // returns with device work still in flight. Null until the lazy stream
  // setup runs (no table/formula backend configured).
  if (table_done_event_)
    CUDA_CHECK(cudaEventSynchronize(table_done_event_));
  if (formula_done_event_)
    CUDA_CHECK(cudaEventSynchronize(formula_done_event_));
}

void OcrPipeline::classify_angles_(const GpuImage &img, std::vector<Box> &boxes,
                                   cudaStream_t stream, PipelineTimer *timer) {
  if (!use_cls_ || boxes.empty()) return;
  // Default gate: only classify boxes that look vertical (h >= w*1.5) —
  // horizontal text (the majority) skips the classifier. CLS_ALL_BOXES=1
  // classifies every crop instead: geometry gives the axis but cannot detect
  // an upside-down horizontal line, so scans with mixed per-line orientations
  // need the flip check on all boxes.
  if (classification::cls_all_boxes_enabled()) {
    if (timer) timer->gpu_start("angle_classification");
    cls_->run(img, boxes, stream); // flips 180° boxes in place
    if (timer) timer->gpu_stop();
    return;
  }
  vertical_box_indices_.clear();
  for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
    if (is_vertical_box(boxes[i]))
      vertical_box_indices_.push_back(i);
  }
  if (vertical_box_indices_.empty()) return;
  vertical_boxes_buf_.clear();
  vertical_boxes_buf_.reserve(vertical_box_indices_.size());
  for (int idx : vertical_box_indices_)
    vertical_boxes_buf_.push_back(boxes[idx]);

  if (timer) timer->gpu_start("angle_classification");
  cls_->run(img, vertical_boxes_buf_, stream);
  if (timer) timer->gpu_stop();

  for (size_t j = 0; j < vertical_box_indices_.size(); ++j)
    boxes[vertical_box_indices_[j]] = vertical_boxes_buf_[j];
}

GpuImage OcrPipeline::upload_image(const cv::Mat &img, cudaStream_t stream,
                                   PipelineTimer &timer) {
  // LOAD-BEARING INVARIANT: reusing h_pinned_buf_ below is safe only because
  // rec_event_ is recorded AFTER rec_->run() returns (ocr_pipeline_run.cpp),
  // and rec_->run() itself synchronizes past the crop kernels that consume
  // this buffer's H2D DMA. If rec ever becomes fully async with the event
  // recorded at issue time (not consumption), the memcpy below could clobber
  // the pinned source mid-DMA — torn image, silent garbage recognition.
  wait_prior_readers_();
  cur_img_buf_ ^= 1;
  auto &buf = img_bufs_[cur_img_buf_];

  if (img.rows > buf.cap_rows || img.cols > buf.cap_cols) [[unlikely]] {
    cudaFree(buf.d_buf);
    buf.d_buf = nullptr;
    // Zero the cap BEFORE the alloc: if cudaMallocPitch throws (OOM), the
    // slot must stay {nullptr, cap 0} so the NEXT request re-enters this
    // realloc branch instead of skipping it and writing to a null buffer
    // with a stale (larger) cap.
    buf.cap_rows = 0;
    buf.cap_cols = 0;
    CUDA_CHECK(cudaMallocPitch(&buf.d_buf, &buf.pitch, img.cols * 3, img.rows));
    buf.cap_rows = img.rows;
    buf.cap_cols = img.cols;
  }

  timer.gpu_start("image_upload");
  auto needed = static_cast<size_t>(img.rows) * img.step;
  if (needed > h_pinned_size_) [[unlikely]] {
    cudaFreeHost(h_pinned_buf_);
    h_pinned_buf_ = nullptr;
    h_pinned_size_ = 0;  // zero before alloc — see d_buf cap reset above
    // Upload-only pinned buffer: CPU writes (memcpy) once and the GPU DMAs
    // it. Write-combined uncached memory is ~10-15% faster for this access
    // pattern (no read-back from CPU).
    CUDA_CHECK(cudaHostAlloc(&h_pinned_buf_, needed, cudaHostAllocWriteCombined));
    h_pinned_size_ = needed;
  }
  std::memcpy(h_pinned_buf_, img.data, needed);
  CUDA_CHECK(cudaMemcpy2DAsync(buf.d_buf, buf.pitch, h_pinned_buf_, img.step,
                                img.cols * 3, img.rows,
                                cudaMemcpyHostToDevice, stream));
  timer.gpu_stop();

  return GpuImage{buf.d_buf, buf.pitch, img.rows, img.cols};
}

void OcrPipeline::ensure_table_stream_() {
  if (!table_stream_) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&table_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&table_done_event_, cudaEventDisableTiming));
  }
}

void OcrPipeline::ensure_formula_stream_() {
  if (!formula_stream_) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&formula_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&formula_done_event_, cudaEventDisableTiming));
  }
}

void OcrPipeline::prewarm_openai_registry_(const std::string &modality,
                                           const routing::RoutingTable &tbl) {
  prewarm_openai_into_registry(modality, tbl, table_registry_, formula_registry_,
                               [this] { ensure_table_stream_(); },
                               [this] { ensure_formula_stream_(); });
}

turbo_ocr::table::ITableRecognizer *
OcrPipeline::pick_table_recognizer_(const std::string &name) const {
  if (!name.empty()) {
    auto it = table_registry_.find(name);
    if (it != table_registry_.end()) return it->second.get();
  }
  return table_recognizer_;  // route default (may be nullptr if tables disabled)
}

turbo_ocr::formula::IFormulaRecognizer *
OcrPipeline::pick_formula_recognizer_(const std::string &name) const {
  if (!name.empty()) {
    auto it = formula_registry_.find(name);
    if (it != formula_registry_.end()) return it->second.get();
  }
  return formula_;  // route default (may be nullptr if formulas disabled)
}

OcrPipeline::UseGuard::UseGuard(std::atomic<int> &counter, const char *entry)
    : c(counter) {
  // Two threads driving one instance = torn buffers / device corruption.
  // Abort loudly rather than corrupt silently (single-thread contract).
  if (c.fetch_add(1, std::memory_order_acq_rel) != 0) {
    std::cerr << "[OcrPipeline] FATAL: concurrent use of one pipeline instance ("
              << entry << ") — it is single-thread-per-instance; use a "
                          "pipeline pool\n";
    std::abort();
  }
}

std::string OcrPipeline::infer_one(const cv::Mat &img, cudaStream_t stream,
                                   const std::string &modality,
                                   const std::string &backend_name,
                                   const routing::BackendSpec *inline_spec) {
  UseGuard _ug{in_use_, "infer_one"};
  if (img.empty()) return "";
  // Upload the crop once; the whole image is the single region.
  PipelineTimer timer;
  timer.init(stream);
  timer.reset();
  GpuImage gpu_img;
  try {
    gpu_img = upload_image(img, stream, timer);
  } catch (const std::exception &e) {
    // Degenerate crop (e.g. zero-aligned pitch tripping cudaMemcpy2DAsync): no
    // pixels to recognize. Surface the cause instead of silently dropping it.
    std::cerr << "[Pipeline] infer_one upload failed for "
              << img.cols << "x" << img.rows << " — returning empty: "
              << e.what() << '\n';
    return "";
  }
  return infer_one_region(
      gpu_img, img.cols, img.rows, stream, modality, backend_name, inline_spec,
      [this](const std::string &n) { return pick_table_recognizer_(n); },
      [this](const std::string &n) { return pick_formula_recognizer_(n); });
}

std::pair<void *, size_t> OcrPipeline::ensure_gpu_buf(int rows, int cols) {
  auto &buf = img_bufs_[cur_img_buf_];
  if (rows > buf.cap_rows || cols > buf.cap_cols) {
    cudaFree(buf.d_buf);
    buf.d_buf = nullptr;
    // Zero the cap BEFORE the alloc so an OOM throw leaves {nullptr, cap 0}:
    // otherwise a later smaller request would skip the realloc and hand a
    // null device buffer to nvjpeg decode → sticky illegal-address → abort.
    buf.cap_rows = 0;
    buf.cap_cols = 0;
    CUDA_CHECK(cudaMallocPitch(&buf.d_buf, &buf.pitch, cols * 3, rows));
    buf.cap_rows = rows;
    buf.cap_cols = cols;
  }
  return {buf.d_buf, buf.pitch};
}
