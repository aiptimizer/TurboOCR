// Runtime helpers shared by every OcrPipeline entry point: stream/event
// bookkeeping, angle classification, the grow-only upload buffers, the
// recognizer registry pickers, and infer_one.

#include <iostream>
#include "turbo_ocr/pipeline/ocr/ocr_pipeline.h"
#include <unordered_map>
#include "infer_one.h"
#include "ocr_pipeline_detail.h"
#include "recognizer_registry.h"
#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/timing.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/router/cua_router.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/table/table_recognizer.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"
#include "turbo_ocr/table/slanext/slanext_enc_split.h"
#include "turbo_ocr/table/table_types.h"
#include "turbo_ocr/engine/trt/onnx_to_trt.h"

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
void OcrPipeline::grow_pitch_buf_(void *&d_buf, size_t &pitch, int &cap_rows,
                                  int &cap_cols, int rows, int cols) {
  if (d_buf) cudaFree(d_buf);
  d_buf = nullptr;
  // Zero the caps BEFORE the alloc: if cudaMallocPitch throws (OOM), the slot
  // must stay {nullptr, cap 0} so the NEXT request re-enters the realloc
  // instead of skipping it and writing to a null buffer with a stale cap.
  cap_rows = 0;
  cap_cols = 0;
  CUDA_CHECK(cudaMallocPitch(&d_buf, &pitch, static_cast<size_t>(cols) * 3, rows));
  cap_rows = rows;
  cap_cols = cols;
}

void OcrPipeline::grow_pinned_(size_t needed) {
  if (h_pinned_buf_) cudaFreeHost(h_pinned_buf_);
  h_pinned_buf_ = nullptr;
  h_pinned_size_ = 0;  // zero before alloc — same contract as grow_pitch_buf_
  // Upload-only pinned buffer: CPU writes (memcpy) once and the GPU DMAs it.
  // Write-combined uncached memory is ~10-15% faster for this access pattern
  // (no read-back from CPU).
  CUDA_CHECK(cudaHostAlloc(&h_pinned_buf_, needed, cudaHostAllocWriteCombined));
  h_pinned_size_ = needed;
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

  if (img.rows > buf.cap_rows || img.cols > buf.cap_cols) [[unlikely]]
    grow_pitch_buf_(buf.d_buf, buf.pitch, buf.cap_rows, buf.cap_cols,
                    img.rows, img.cols);

  timer.gpu_start("image_upload");
  auto needed = static_cast<size_t>(img.rows) * img.step;
  if (needed > h_pinned_size_) [[unlikely]]
    grow_pinned_(needed);
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
                                           const backend_routing::RoutingTable &tbl) {
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
                                   const backend_routing::BackendSpec *inline_spec) {
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
  // A failed grow here would otherwise hand a null device buffer to nvjpeg
  // decode -> sticky illegal-address -> abort; grow_pitch_buf_'s
  // zero-before-alloc contract prevents exactly that.
  if (rows > buf.cap_rows || cols > buf.cap_cols)
    grow_pitch_buf_(buf.d_buf, buf.pitch, buf.cap_rows, buf.cap_cols, rows,
                    cols);
  return {buf.d_buf, buf.pitch};
}
