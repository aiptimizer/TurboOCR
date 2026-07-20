// Batched entry points: whole-batch upload, batched detection/recognition,
// and the per-page layout + router stage (phase 4).

#include "turbo_ocr/pipeline/ocr/ocr_pipeline.h"
#include <unordered_map>
#include "infer_one.h"
#include "ocr_pipeline_detail.h"
#include "recognizer_registry.h"
#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/log/timing.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/formula/routing/auto_cjk_formula.h"
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
#include "turbo_ocr/pipeline/reading_order_util.h"

std::vector<std::vector<OCRResultItem>> OcrPipeline::run_batch(
    const std::vector<cv::Mat> &imgs, cudaStream_t stream) {
  auto outs = run_batch_with_layout(imgs, stream,
                                    /*want_layout=*/false,
                                    /*want_reading_order=*/false);
  std::vector<std::vector<OCRResultItem>> results;
  results.reserve(outs.size());
  for (auto &out : outs)
    results.push_back(std::move(out.results));
  return results;
}

std::vector<OcrPipelineResult> OcrPipeline::run_batch_with_layout(
    const std::vector<cv::Mat> &imgs, cudaStream_t stream,
    bool want_layout, bool want_reading_order,
    bool want_tables, bool want_formulas,
    const backend_routing::RequestRouting &routing) {
  if (imgs.empty())
    return {};

  // If only one image, just use single-image path
  if (imgs.size() == 1) {
    std::vector<OcrPipelineResult> single;
    single.push_back(run_with_layout(imgs[0], stream, want_layout,
                                     want_reading_order, routing,
                                     /*defer_external=*/false,
                                     want_tables, want_formulas));
    return single;
  }

  // Oversized batches are processed in full by chunking here (BEFORE the
  // UseGuard, like the n==1 delegation) — never silently truncated to
  // kMaxBatchImages, which would drop pages behind a clean 200 (the
  // no-silent-failure contract, and parity with PaddleDet::run_batch's
  // overflow handling). Each chunk is a fresh guarded call over the shared
  // batch buffers.
  if (imgs.size() > static_cast<size_t>(kMaxBatchImages)) [[unlikely]] {
    std::vector<OcrPipelineResult> all;
    all.reserve(imgs.size());
    for (size_t beg = 0; beg < imgs.size(); beg += kMaxBatchImages) {
      const size_t end = std::min(beg + kMaxBatchImages, imgs.size());
      std::vector<cv::Mat> chunk(imgs.begin() + beg, imgs.begin() + end);
      auto part = run_batch_with_layout(chunk, stream, want_layout,
                                        want_reading_order, want_tables,
                                        want_formulas, routing);
      for (auto &r : part) all.push_back(std::move(r));
    }
    return all;
  }

  // Guard AFTER the n==1 delegation and chunk fan-out above — run_with_layout /
  // the recursive call self-guard, and a guard here would trip on them.
  UseGuard _ug{in_use_, "run_batch_with_layout"};

  const int n = static_cast<int>(imgs.size());
  const int batch_n = n;  // <= kMaxBatchImages by the chunking above

  // Batch buffers and the shared pinned staging buffer are reused across
  // requests — same reuse contract as upload_image().
  wait_prior_readers_();

  // --- Phase 1: Upload all images to GPU, run batched detection + cls ---
  // We need all images alive on GPU simultaneously for batched recognition.
  struct PerImage {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int rows = 0, cols = 0;
    std::vector<Box> boxes;
  };
  std::vector<PerImage> per_img(batch_n);

  // Upload all images to GPU first
  for (int i = 0; i < batch_n; i++) {
    const auto &img = imgs[i];
    auto &pi = per_img[i];
    pi.rows = img.rows;
    pi.cols = img.cols;

    // Use pre-allocated GPU buffer (grow-only, avoids cudaMalloc per batch)
    auto &bbuf = batch_img_bufs_[i];
    if (img.rows > bbuf.cap_rows || img.cols > bbuf.cap_cols) [[unlikely]]
      grow_pitch_buf_(bbuf.d_buf, bbuf.pitch, bbuf.cap_rows, bbuf.cap_cols,
                      img.rows, img.cols);
    pi.d_buf = bbuf.d_buf;
    pi.pitch = bbuf.pitch;

    // Upload via the shared pinned staging buffer
    auto needed = static_cast<size_t>(img.rows) * img.step;
    if (needed > h_pinned_size_) [[unlikely]]
      grow_pinned_(needed);
    std::memcpy(h_pinned_buf_, img.data, needed);
    CUDA_CHECK(cudaMemcpy2DAsync(pi.d_buf, pi.pitch, h_pinned_buf_, img.step,
                                  img.cols * 3, img.rows,
                                  cudaMemcpyHostToDevice, stream));
    // Sync before next iteration: h_pinned_buf_ is shared and will be
    // overwritten, so the async copy must complete first.
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Batched detection across the whole batch (one TRT call per <=8-image
  // chunk via the batch-capable det profile) instead of batch-1 per image.
  // det run_batch resizes all images to the batch-max shape and maps boxes
  // back per image — ~40% cheaper det/image at batch 8 (measured). Only the
  // batch path uses this; single /ocr/raw still takes the batch-1 fast path.
  std::vector<std::vector<Box>> all_det_boxes(batch_n);
  {
    std::vector<GpuImage> gpu_imgs;
    std::vector<std::pair<int, int>> dims;
    gpu_imgs.reserve(batch_n);
    dims.reserve(batch_n);
    for (int i = 0; i < batch_n; i++) {
      gpu_imgs.push_back({per_img[i].d_buf, per_img[i].pitch,
                          per_img[i].rows, per_img[i].cols});
      dims.emplace_back(per_img[i].rows, per_img[i].cols);
    }
    try {
      all_det_boxes = det_->run_batch(gpu_imgs, dims, stream);
    } catch (const turbo_ocr::CudaError &e) {
      // A batched det fault cannot be attributed to one member image, so the
      // degenerate-input downgrade of the single paths does not apply: sticky
      // check, then loud for the whole batch (never a silent empty 200 x N).
      cudaStreamSynchronize(stream);
      turbo_ocr::abort_on_sticky_cuda_fault("run_batch_with_layout/det");
      cudaGetLastError();
      throw turbo_ocr::InferenceError(
          std::string("batched detection GPU fault: ") + e.what());
    } catch (const turbo_ocr::InferenceError &e) {
      // TRT enqueue failures surface as InferenceError (infer_dynamic already
      // ran the sticky check internally); rewrap so the batch context isn't
      // lost while keeping the same loud path as the CudaError branch.
      throw turbo_ocr::InferenceError(
          std::string("batched detection failed: ") + e.what());
    }
  }

  // Assign detection results and run angle classification per-image
  // (CLS_ALL_BOXES / vertical-only gate, same policy as the single paths —
  // the batch path previously ignored CLS_ALL_BOXES entirely).
  for (int i = 0; i < batch_n; i++) {
    per_img[i].boxes = std::move(all_det_boxes[i]);
    sorted_boxes(per_img[i].boxes);
    GpuImage gpu_img{per_img[i].d_buf, per_img[i].pitch,
                     per_img[i].rows, per_img[i].cols};
    classify_angles_(gpu_img, per_img[i].boxes, stream, nullptr);
  }

  // --- Phase 2: Batched recognition across ALL images ---
  std::vector<PaddleRec::ImageCrops> image_crops(batch_n);
  for (int i = 0; i < batch_n; i++) {
    image_crops[i].img = GpuImage{per_img[i].d_buf, per_img[i].pitch,
                                  per_img[i].rows, per_img[i].cols};
    image_crops[i].boxes = std::move(per_img[i].boxes);
  }

  // Launch batched recognition on rec_stream_ (pipeline parallelism)
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));
  auto all_rec_results = rec_->run_multi(image_crops, rec_stream_);
  // Note: rec_->run_multi() syncs rec_stream_ internally for D2H + CTC decode,
  // so no additional cudaStreamSynchronize needed here.

  // --- Phase 3: Combine results and filter by drop_score ---
  std::vector<OcrPipelineResult> all_results(batch_n);
  const auto &dropped = rec_->last_dropped_per_image();
  for (int i = 0; i < batch_n; i++) {
    detail::combine_recognition(all_results[i], image_crops[i].boxes,
                                all_rec_results[i]);
    if (static_cast<size_t>(i) < dropped.size())
      detail::flag_dropped_crops(all_results[i], dropped[i]);
  }

  // --- Phase 4 (opt-in): layout + CUA router (table/formula) per page ---
  // Engages only when the caller asked for layout AND the operator loaded
  // a layout model — text-only batches pay zero layout/router cost.
  if (want_layout && use_layout_ && layout_)
    run_batch_layout_stage_(image_crops, want_reading_order, stream,
                            all_results, want_tables, want_formulas, routing);

  // No cleanup needed — batch_img_bufs_ are pre-allocated and reused

  return all_results;
}

void OcrPipeline::run_batch_layout_stage_(
    const std::vector<PaddleRec::ImageCrops> &image_crops,
    bool want_reading_order, cudaStream_t stream,
    std::vector<OcrPipelineResult> &outs,
    bool want_tables, bool want_formulas,
    const backend_routing::RequestRouting &routing) {
  const int batch_n = static_cast<int>(image_crops.size());

  // All det/rec GPU work is host-synced by this point (run_multi syncs
  // internally), but table/formula streams gate on det_only_event_, so
  // record it on the caller's stream to give them a valid ordering point.
  CUDA_CHECK(cudaEventRecord(det_only_event_, stream));

  for (int i = 0; i < batch_n; i++) {
    const GpuImage &gpu_img = image_crops[i].img;
    // collect() only after a successful enqueue — on failure it would
    // sync the stale d2h_event_ of the last successful execute and hand
    // this page the PREVIOUS page's boxes (which the router would then
    // crop at wrong coordinates on the wrong image).
    if (!layout_->enqueue(gpu_img, gpu_img.rows, gpu_img.cols,
                          layout_stream_))
      continue;
    outs[i].layout = layout_->collect();
    PipelineTimer t;
    dispatch_router_(outs[i], gpu_img, image_crops[i].boxes, t,
                     routing, /*defer_external=*/false,
                     want_tables, want_formulas);
  }

  // Reading order — same contract as run_with_layout: helper handles
  // orphan results (missing layout match) via synthetic XY-cut entries.
  for (auto &out : outs)
    maybe_assign_reading_order(want_reading_order, out.results, out.layout,
                               out.reading_order);
}
