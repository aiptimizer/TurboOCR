#include "turbo_ocr/detection/paddle_det.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/detection/det_postprocess.h"

#include "turbo_ocr/common/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>

#include <opencv2/imgproc.hpp>

using turbo_ocr::engine::TrtEngine;

namespace turbo_ocr::detection {

bool PaddleDet::load_model(const std::string& model_path, const DetResizeParams& resize,
                           const DbParams& db) {
  engine_ = std::make_unique<TrtEngine>(model_path);
  if (!engine_->load())
    return false;
  return init_buffers(resize, db);
}

bool PaddleDet::init_buffers(const DetResizeParams& resize, const DbParams& db) {
  // Per-model resize policy with env overrides layered on (env wins). The
  // engine builder must size its TRT profile MAX off the same effective
  // max-side or the runtime and engine silently disagree.
  resize_ = read_det_resize(resize);
  kMaxSideLen_ = effective_det_max_side(resize_);

  size_t max_pixels = static_cast<size_t>(kMaxSideLen_) * kMaxSideLen_;

  d_input_ = CudaPtr<float>(max_pixels * 3);
  d_output_ = CudaPtr<float>(max_pixels);

  // Pre-allocate bitmap buffer (kMaxSideLen_ x kMaxSideLen_ uint8)
  d_bitmap_buf_ = CudaPtr<uint8_t>(max_pixels);

  // Pre-allocate batch buffers (kMaxBatchSize x max_pixels)
  d_batch_input_ = CudaPtr<float>(static_cast<size_t>(kMaxBatchSize) * max_pixels * 3);
  d_batch_output_ = CudaPtr<float>(static_cast<size_t>(kMaxBatchSize) * max_pixels);
  d_batch_bitmap_ = CudaPtr<uint8_t>(static_cast<size_t>(kMaxBatchSize) * max_pixels);

  // Device arrays for batched kernel launch parameters
  d_batch_src_ptrs_ = CudaPtr<void *>(kMaxBatchSize);
  d_batch_src_steps_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_src_heights_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_src_widths_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_dst_heights_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_dst_widths_ = CudaPtr<int>(kMaxBatchSize);
  // Pinned host staging for async copy (avoids pageable fallback)
  h_batch_src_ptrs_ = CudaHostPtr<void *>(kMaxBatchSize);
  h_batch_src_steps_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_src_heights_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_src_widths_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_dst_heights_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_dst_widths_ = CudaHostPtr<int>(kMaxBatchSize);

  // Bind I/O pointers once for single-image path (never change)
  engine_->bind_io(d_input_.get(), d_output_.get());

  // DB params: this model's base + env overrides (read_db_params), same
  // det_config.h source as the CPU detector. GPU_BOX_THRESH / GPU_UNCLIP_SCALE
  // remain as overrides layered on top.
  const DbParams eff_db = read_db_params(db);
  db_thresh_ = eff_db.thresh;
  box_thresh_ = eff_db.box_thresh;
  unclip_ratio_ = eff_db.unclip_ratio;

  // GPU CCL mode: 0=CPU contours, 1=GPU CCL+per-ROI findContours (default),
  // 2=all-GPU JFA per-component Euclidean unclip
  //
  // Empty/garbage env values keep the default instead of atoi/atof-ing to 0,
  // and parsed values are CLAMPED to sane ranges (a 0/negative box_thresh
  // floods spurious boxes; a 0 unclip_scale collapses every polygon; inf
  // disables detection entirely — same silent-failure class as GitHub #23).
  // The server also strict-validates these at boot with the same ranges; this
  // lenient layer covers CLI/tools.
  auto env_val = [](const char *name, auto cur, auto parse) {
    const char *env = std::getenv(name);
    if (!env || !*env) return cur;
    char *end = nullptr;
    auto v = parse(env, &end);
    if (end == env || *end != '\0' || std::isnan(static_cast<double>(v))) {
      std::cerr << "[PaddleDet] ignoring malformed " << name << "=\"" << env << "\"\n";
      return cur;
    }
    return static_cast<decltype(cur)>(v);
  };
  gpu_ccl_mode_ = std::clamp(
      env_val("GPU_CCL", gpu_ccl_mode_,
              [](const char *s, char **e) { return std::strtol(s, e, 10); }),
      0, 2);
  // box_thresh_ + unclip_scale_ apply to all three modes (0/1/2). Ranges match
  // the server-boot validation in server_config.h::from_env.
  box_thresh_ = std::clamp(env_val("GPU_BOX_THRESH", box_thresh_, std::strtof),
                           0.001f, 1.0f);
  unclip_scale_ = std::clamp(env_val("GPU_UNCLIP_SCALE", unclip_scale_, std::strtof),
                             0.1f, 10.0f);

  if (gpu_ccl_mode_ > 0) {
    // Pre-allocate ALL GPU CCL buffers (no per-request alloc)
    d_ccl_labels_ = CudaPtr<int>(max_pixels);
    d_ccl_compact_ids_ = CudaPtr<int>(max_pixels);
    d_ccl_id_counter_ = CudaPtr<int>(1);
    // 2x kMaxGpuComponents: first half for per-component bboxes, second half for filtered output
    d_ccl_bboxes_ = CudaPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents * 2);
    d_ccl_num_boxes_ = CudaPtr<int>(1);

    // Pinned host memory for result transfer
    h_ccl_boxes_ = CudaHostPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents);
    h_bitmap_ = CudaHostPtr<uint8_t>(max_pixels);
    h_bitmap_pixels_ = max_pixels;

    // JFA (Jump Flooding) per-component label expansion
    d_jfa_labels_ = CudaPtr<uint32_t>(max_pixels);
    d_jfa_seeds_ = CudaPtr<uint32_t>(max_pixels);
    d_jfa_seeds_alt_ = CudaPtr<uint32_t>(max_pixels);
    d_expand_per_comp_ = CudaPtr<float>(turbo_ocr::kernels::kMaxGpuComponents);
    d_perim_per_comp_ = CudaPtr<int>(turbo_ocr::kernels::kMaxGpuComponents);
    d_ccl_moments_ = CudaPtr<unsigned long long>(turbo_ocr::kernels::kMaxGpuComponents * 6);
    d_ccl_orient_ = CudaPtr<float>(turbo_ocr::kernels::kMaxGpuComponents * 6);
    h_exp_boxes_ = CudaHostPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents);
  }

  return true;
}

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

std::vector<Box>
PaddleDet::run(const GpuImage &gpu_img, int orig_h, int orig_w,
               cudaStream_t stream) {
  auto [resize_h, resize_w] = compute_det_resize(orig_h, orig_w, resize_);

  float *d_pred = d_output_.get();
  uint8_t *d_bitmap = d_bitmap_buf_.get();

  // 1+2. Fused resize + normalize + CHW (single kernel, no intermediate buffer)
  turbo_ocr::kernels::cuda_fused_resize_normalize_det(gpu_img, d_input_.get(), resize_w,
                                                resize_h, stream);

  // 3. Inference (dynamic H,W) -- I/O already bound in load_model
  nvinfer1::Dims4 input_dims{1, 3, resize_h, resize_w};
  if (!engine_->infer_dynamic(input_dims, stream)) {
    throw turbo_ocr::InferenceError("Detection TRT inference failed");
  }

  // 4. Threshold on GPU for bitmap
  turbo_ocr::kernels::cuda_threshold_to_u8(d_pred, d_bitmap, resize_w, resize_h, db_thresh_,
                                           stream);

  // 5. Choose contour extraction path
  if (gpu_ccl_mode_ == 2) {
    return run_gpu_ccl_fast(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream);
  } else if (gpu_ccl_mode_ == 1) {
    return run_gpu_ccl(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream);
  } else {
    return run_cpu_contours(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream);
  }
}

// ============================================================================
// Batched detection: process N images in a single TRT inference call.
// All images are resized to the same target dimensions (max of the batch).
// ============================================================================
std::vector<std::vector<Box>>
PaddleDet::run_batch(const std::vector<GpuImage> &gpu_imgs,
                     const std::vector<std::pair<int,int>> &orig_dims,
                     cudaStream_t stream) {
  const int n = static_cast<int>(gpu_imgs.size());
  if (n == 0)
    return {};

  // Single image → use the optimized single-image path.
  if (n == 1) {
    auto boxes = run(gpu_imgs[0], orig_dims[0].first, orig_dims[0].second, stream);
    return {std::move(boxes)};
  }

  int batch_size;
  int resize_h, resize_w;
  struct PerImgInfo {
    int orig_h, orig_w;
    int resize_h, resize_w;  // letterbox content dims inside the batch canvas
  };
  std::vector<PerImgInfo> infos;

  // Clamp to max batch size
  batch_size = std::min(n, kMaxBatchSize);

  // --- Compute unified canvas dimensions (max across batch, rounded to 32) ---
  // Each image keeps its OWN aspect-preserving resize dims and is letterboxed
  // into the canvas top-left; the remainder is padding. Stretching everything
  // to the canvas dims instead (the old behavior) distorted glyphs whenever a
  // batch mixed aspect ratios, silently degrading detection vs /ocr/raw.
  int max_resize_h = 0, max_resize_w = 0;
  infos.resize(batch_size);

  for (int i = 0; i < batch_size; i++) {
    int h = orig_dims[i].first;
    int w = orig_dims[i].second;
    auto [rh, rw] = compute_det_resize(h, w, resize_);
    infos[i] = {h, w, rh, rw};
    max_resize_h = std::max(max_resize_h, rh);
    max_resize_w = std::max(max_resize_w, rw);
  }

  // Use the unified (max) dimensions for all images in the batch
  resize_h = max_resize_h;
  resize_w = max_resize_w;
  const int pixels_per_image = resize_h * resize_w;

  // --- 1. Upload per-image metadata to device ---
  // Use pre-allocated pinned buffers for truly async transfers. batch_size ==
  // min(n, kMaxBatchSize) <= n and gpu_imgs.size() == n, so i is always in range.
  for (int i = 0; i < batch_size; i++) {
    h_batch_src_ptrs_.get()[i]    = gpu_imgs[i].data;
    h_batch_src_steps_.get()[i]   = static_cast<int>(gpu_imgs[i].step);
    h_batch_src_heights_.get()[i] = gpu_imgs[i].rows;
    h_batch_src_widths_.get()[i]  = gpu_imgs[i].cols;
    h_batch_dst_heights_.get()[i] = infos[i].resize_h;
    h_batch_dst_widths_.get()[i]  = infos[i].resize_w;
  }

  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_ptrs_.get(), h_batch_src_ptrs_.get(),
                              batch_size * sizeof(void *),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_steps_.get(), h_batch_src_steps_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_heights_.get(), h_batch_src_heights_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_widths_.get(), h_batch_src_widths_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_dst_heights_.get(), h_batch_dst_heights_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_dst_widths_.get(), h_batch_dst_widths_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));

  // --- 2. Batched fused resize + normalize + CHW (letterboxed per image) ---
  turbo_ocr::kernels::cuda_batch_fused_resize_normalize_det(
      (const void *const *)d_batch_src_ptrs_.get(), d_batch_src_steps_.get(),
      d_batch_src_heights_.get(), d_batch_src_widths_.get(),
      d_batch_dst_heights_.get(), d_batch_dst_widths_.get(),
      d_batch_input_.get(), resize_w, resize_h, batch_size, stream);

  // --- 3. Single TRT inference call with batch=N ---
  // Temporarily rebind I/O to batch buffers
  engine_->bind_io(d_batch_input_.get(), d_batch_output_.get());

  nvinfer1::Dims4 input_dims{batch_size, 3, resize_h, resize_w};
  if (!engine_->infer_dynamic(input_dims, stream)) {
    // Restore single-image binding before throwing
    engine_->bind_io(d_input_.get(), d_output_.get());
    throw turbo_ocr::InferenceError("Batched detection TRT inference failed");
  }

  // Restore single-image I/O binding for future single-image calls
  engine_->bind_io(d_input_.get(), d_output_.get());

  // --- 4. Batched threshold (all images at once) ---
  turbo_ocr::kernels::cuda_batch_threshold_to_u8(d_batch_output_.get(), d_batch_bitmap_.get(),
                                                 resize_w, resize_h, batch_size, db_thresh_,
                                                 stream);

  // --- 5. Per-image post-processing (GPU CCL fast / CPU contours) ---
  // Each image's probability map + bitmap slice is passed to the post-process
  // helper explicitly — no shared mutable member state to save/restore.
  const int real_n = batch_size;
  std::vector<std::vector<Box>> all_boxes(real_n);

  for (int i = 0; i < real_n; i++) {
    const int orig_h = infos[i].orig_h;
    const int orig_w = infos[i].orig_w;

    const size_t off = static_cast<size_t>(i) * pixels_per_image;
    const float *d_pred = d_batch_output_.get() + off;
    const uint8_t *d_bitmap = d_batch_bitmap_.get() + off;

    if (gpu_ccl_mode_ == 2) {
      all_boxes[i] = run_gpu_ccl_fast(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream,
                                      infos[i].resize_h, infos[i].resize_w);
    } else if (gpu_ccl_mode_ == 1) {
      all_boxes[i] = run_gpu_ccl(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream,
                                 infos[i].resize_h, infos[i].resize_w);
    } else {
      all_boxes[i] = run_cpu_contours(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream,
                                      infos[i].resize_h, infos[i].resize_w);
    }
  }

  // --- 6. Handle overflow: process remaining images via single-image path ---
  if (n > kMaxBatchSize) {
    all_boxes.resize(n);
    for (int i = kMaxBatchSize; i < n; i++) {
      all_boxes[i] = run(gpu_imgs[i], orig_dims[i].first, orig_dims[i].second, stream);
    }
  }

  return all_boxes;
}

} // namespace turbo_ocr::detection
