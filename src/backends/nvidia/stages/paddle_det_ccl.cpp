#include "nvidia/stages/paddle_det.h"
#include "turbo_ocr/analysis/detection/det_config.h"
#include "nvidia/kernels_cuda/kernels_cuda.h"
#include "turbo_ocr/analysis/detection/det_postprocess.h"

#include "turbo_ocr/base/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>

#include <opencv2/imgproc.hpp>

using turbo_ocr::engine::TrtEngine;

namespace turbo_ocr::detection {

// GPU CCL path: connected component labeling on GPU, then extract real contours
// from bitmap within each component's bbox for accurate unclip polygons.
std::vector<Box>
PaddleDet::run_gpu_ccl(const float *d_pred, const uint8_t *d_bitmap,
                        int resize_h, int resize_w,
                        int orig_h, int orig_w,
                        cudaStream_t stream,
                        int content_h, int content_w) {
  if (content_h <= 0) content_h = resize_h;
  if (content_w <= 0) content_w = resize_w;
  float ratio_h = static_cast<float>(content_h) / orig_h;
  float ratio_w = static_cast<float>(content_w) / orig_w;

  int h_num_boxes = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      d_bitmap, d_pred, resize_w, resize_h,
      box_thresh_,
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream);

  std::vector<Box> boxes;
  if (h_num_boxes == 0)
    return boxes;

  // Download ONLY the bitmap (not pred_map -- GPU CCL already computed scores).
  // We need the bitmap for per-ROI findContours to get accurate polygon contours.
  const size_t bitmap_pixels = static_cast<size_t>(resize_h) * resize_w;
  if (bitmap_pixels > h_bitmap_pixels_) [[unlikely]] {
    h_bitmap_ = CudaHostPtr<uint8_t>(bitmap_pixels);
    h_bitmap_pixels_ = bitmap_pixels;
  }
  cv::Mat bitmap(resize_h, resize_w, CV_8UC1, h_bitmap_.get());
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, d_bitmap, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // No pred_map download -- use gb.score from GPU CCL instead of box_score_fast

  boxes.reserve(h_num_boxes);

  // For each GPU-detected component, extract the real contour from the bitmap
  // within the bbox region, then feed to the same unclip pipeline as CPU path.
  for (int i = 0; i < h_num_boxes; i++) {
    const auto &gb = h_ccl_boxes_.get()[i];

    int bw = gb.xmax - gb.xmin + 1;
    int bh = gb.ymax - gb.ymin + 1;
    if (bw < 3 || bh < 3)
      continue;

    // Extract the small bitmap ROI for this component's bbox
    // Pad by 1 pixel to ensure findContours can find closed contours at edges
    int roi_x = std::max(0, gb.xmin - 1);
    int roi_y = std::max(0, gb.ymin - 1);
    int roi_x2 = std::min(resize_w - 1, gb.xmax + 1);
    int roi_y2 = std::min(resize_h - 1, gb.ymax + 1);
    int roi_w = roi_x2 - roi_x + 1;
    int roi_h = roi_y2 - roi_y + 1;

    // Clone the ROI since findContours may modify the source image
    cv::Mat roi = bitmap(cv::Rect(roi_x, roi_y, roi_w, roi_h)).clone();

    // Find contours within this small ROI (~50x20 pixels, negligible cost)
    ccl_roi_contours_buf_.clear();
    cv::findContours(roi, ccl_roi_contours_buf_, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    if (ccl_roi_contours_buf_.empty())
      continue;

    // Pick the largest contour in the ROI (should be the component itself)
    const auto &best_contour = (ccl_roi_contours_buf_.size() == 1)
      ? ccl_roi_contours_buf_[0]
      : *std::ranges::max_element(ccl_roi_contours_buf_, {}, [](const std::vector<cv::Point> &c) {
          return cv::contourArea(c);
        });

    if (best_contour.size() <= 2)
      continue;

    // Shift contour from ROI-local coords to global bitmap coords
    ccl_contour_buf_.clear();
    ccl_contour_buf_.reserve(best_contour.size());
    for (const auto &pt : best_contour)
      ccl_contour_buf_.push_back(cv::Point(pt.x + roi_x, pt.y + roi_y));

    // Use GPU CCL score (already filtered by box_thresh_ in the GPU kernel)
    // Skip box_score_fast — saves downloading pred_map (2.4MB) entirely

    float ssid = 0;
    (void)get_mini_boxes(ccl_contour_buf_, ssid);
    if (ssid < kMinBoxSide)
      continue;

    auto unclipped = unclip(ccl_contour_buf_, unclip_ratio_ * unclip_scale_);
    if (unclipped.size() < 3)
      continue;

    float ssid2 = 0;
    auto box = get_mini_boxes(unclipped, ssid2);
    if (ssid2 < kMinUnclippedSide)
      continue;

    // Scale back to original image
    for (int k = 0; k < 4; ++k) {
      box[k][0] = std::clamp(static_cast<int>(std::round(box[k][0] / ratio_w)), 0, orig_w - 1);
      box[k][1] = std::clamp(static_cast<int>(std::round(box[k][1] / ratio_h)), 0, orig_h - 1);
    }

    // Filter tiny boxes
    int rw = static_cast<int>(std::sqrt(((box[0][0] - box[1][0]) * (box[0][0] - box[1][0])) +
                                        ((box[0][1] - box[1][1]) * (box[0][1] - box[1][1]))));
    int rh = static_cast<int>(std::sqrt(((box[0][0] - box[3][0]) * (box[0][0] - box[3][0])) +
                                        ((box[0][1] - box[3][1]) * (box[0][1] - box[3][1]))));
    if (rw <= 3 || rh <= 3)
      continue;

    boxes.push_back(box);
  }

  return boxes;
}

// GPU CCL + JFA per-component Euclidean unclip (all-GPU).
// 1. CCL on original → compact_ids + bboxes + moments
// 2. JFA propagates nearest-foreground coords (unsigned SDF)
// 3. Expand: pixels within `expand` distance assigned to nearest component
//    via compact_ids lookup → no component merging (Voronoi boundary)
// 4. GPU bbox extraction: one block per component scans expanded labels
// 5. Copy expanded bboxes → scale → filter → output
std::vector<Box>
PaddleDet::run_gpu_ccl_fast(const float *d_pred, const uint8_t *d_bitmap,
                              int resize_h, int resize_w,
                              int orig_h, int orig_w,
                              cudaStream_t stream,
                              int content_h, int content_w) {
  if (content_h <= 0) content_h = resize_h;
  if (content_w <= 0) content_w = resize_w;
  float ratio_h = static_cast<float>(content_h) / orig_h;
  float ratio_w = static_cast<float>(content_w) / orig_w;

  // Step 1: CCL on original mask → compact IDs + original bboxes
  int h_num_boxes = 0;
  int h_num_total = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      d_bitmap, d_pred, resize_w, resize_h,
      box_thresh_,
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream, &h_num_total);

  std::vector<Box> boxes;
  if (h_num_boxes == 0) return boxes;

  // Process all PRE-filter compact_ids — that's what compact_ids[] stores.
  // Score+size filter is applied inside compute_expand_per_comp_kernel.
  // LOAD-BEARING BOUND: every id stored in compact_ids[] is < kMaxGpuComponents
  // or -1 — ccl_buf_compact_assign_kernel (kernels.cu) writes -1 for roots past
  // the cap, so the [compact_id]-indexed buffers below (perim/expand/exp_bboxes,
  // all sized kMaxGpuComponents) can never be written out of bounds even when
  // h_num_total exceeds the cap. num_slots clamps only the COUNT.
  using turbo_ocr::kernels::GpuDetBox;
  using turbo_ocr::kernels::kMaxGpuComponents;
  int num_slots = std::min(h_num_total, (int)kMaxGpuComponents);
  if (num_slots == 0) return boxes;

  // Step 2a: Per-component contour (crack) perimeter — the true divisor for the
  // area*ratio/perimeter unclip, matching cv2.arcLength. Must precede the expand
  // computation below. Fully device-resident.
  turbo_ocr::kernels::cuda_accumulate_crack_perimeter(
      d_ccl_compact_ids_.get(), d_bitmap, resize_w, resize_h, num_slots,
      d_perim_per_comp_.get(), stream);

  // Step 2b: Per-component expand distance = area*ratio/perimeter, indexed by
  // PRE-filter compact_id. kMaxExpand is the global cutoff; it also bounds the
  // JFA jump range below — keep them equal.
  constexpr float kMaxExpand = 24.0f;
  turbo_ocr::kernels::cuda_compute_expand_per_comp(
      d_ccl_bboxes_.get(), d_perim_per_comp_.get(), num_slots,
      unclip_ratio_ * unclip_scale_, /*min*/ 2.0f, kMaxExpand,
      box_thresh_, d_expand_per_comp_.get(), stream);

  // Step 3: JFA + per-component label expansion (variable cutoff per component).
  // Pass kMaxExpand so JFA bounds its jump range to it (no pixel beyond it survives).
  turbo_ocr::kernels::cuda_jfa_expand_labels(
      d_bitmap, d_ccl_compact_ids_.get(), d_expand_per_comp_.get(),
      d_jfa_labels_.get(), resize_w, resize_h, kMaxExpand,
      d_jfa_seeds_.get(), d_jfa_seeds_alt_.get(), stream);

  // Step 4: GPU extraction over expanded region — axis-aligned bbox (for the
  // score/size filters) AND the PCA-oriented min-area-rect corners, matching
  // the CPU minAreaRect geometry so skewed lines get a tight rotated quad.
  // Launcher inits sentinels for atomic scatter, so no pre-memset needed.
  GpuDetBox *exp_bboxes = d_ccl_bboxes_.get() + kMaxGpuComponents;
  turbo_ocr::kernels::cuda_jfa_extract_oriented(
      d_jfa_labels_.get(), resize_w, resize_h,
      exp_bboxes, num_slots,
      d_ccl_moments_.get(), d_ccl_orient_.get(), stream);

  // Step 5: Copy expanded bboxes to host via the pre-allocated pinned buffer.
  CUDA_CHECK(cudaMemcpyAsync(h_exp_boxes_.get(), exp_bboxes,
      num_slots * sizeof(GpuDetBox), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Filter, scale, output. pixel_count==0 means slot was empty or filtered out.
  boxes.reserve(h_num_boxes);
  for (int i = 0; i < num_slots; i++) {
    const auto &eb = h_exp_boxes_.get()[i];
    if (eb.pixel_count < 9) continue;
    int bw = eb.xmax - eb.xmin + 1, bh = eb.ymax - eb.ymin + 1;
    if (bw < kMinUnclippedSide || bh < kMinUnclippedSide) continue;

    // Order the GPU oriented-rect corners as [tl, tr, br, bl] via the SAME
    // shared helper get_mini_boxes uses (mode 0/1) — one convention, no drift.
    float px[4], py[4];
    for (int k = 0; k < 4; ++k) { px[k] = eb.ox[k]; py[k] = eb.oy[k]; }
    order_quad_tl_tr_br_bl(px, py);

    Box box;
    for (int k = 0; k < 4; ++k) {
      box[k] = {std::clamp(static_cast<int>(std::round(px[k] / ratio_w)), 0, orig_w - 1),
                std::clamp(static_cast<int>(std::round(py[k] / ratio_h)), 0, orig_h - 1)};
    }

    // Tiny-box reject on the oriented edge lengths (mirrors mode 1).
    int rw = static_cast<int>(std::sqrt(
        double((box[0][0]-box[1][0])*(box[0][0]-box[1][0]) +
               (box[0][1]-box[1][1])*(box[0][1]-box[1][1]))));
    int rh = static_cast<int>(std::sqrt(
        double((box[0][0]-box[3][0])*(box[0][0]-box[3][0]) +
               (box[0][1]-box[3][1])*(box[0][1]-box[3][1]))));
    if (rw <= 3 || rh <= 3) continue;
    boxes.push_back(box);
  }
  return boxes;
}

// CPU fallback path (original findContours)
std::vector<Box>
PaddleDet::run_cpu_contours(const float *d_pred, const uint8_t *d_bitmap,
                             int resize_h, int resize_w,
                             int orig_h, int orig_w,
                             cudaStream_t stream,
                             int content_h, int content_w) {
  if (content_h <= 0) content_h = resize_h;
  if (content_w <= 0) content_w = resize_w;
  // Download raw probability map for score filtering. Both D2H copies are queued
  // async and covered by a single stream sync — the minimal, and required, sync
  // for the host-side findContours below.
  cv::Mat pred_map(resize_h, resize_w, CV_32F);
  CUDA_CHECK(cudaMemcpyAsync(pred_map.data, d_pred,
                              resize_h * resize_w * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));

  cv::Mat bitmap(resize_h, resize_w, CV_8UC1);
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, d_bitmap, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // extract_boxes_from_bitmap takes the map extent from the Mats; the
  // resize params only feed the box->original ratios, so pass content dims.
  return extract_boxes_from_bitmap(pred_map, bitmap, orig_h, orig_w, content_h, content_w,
                                   box_thresh_, unclip_ratio_ * unclip_scale_, kMinBoxSide,
                                   kMinUnclippedSide, shifted_buf_, mask_buf_, contours_buf_,
                                   hierarchy_buf_);
}


} // namespace turbo_ocr::detection
