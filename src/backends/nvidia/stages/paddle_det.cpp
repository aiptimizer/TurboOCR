#include "nvidia/stages/paddle_det.h"
#include "turbo_ocr/analysis/detection/det_config.h"
#include "nvidia/kernels_cuda/kernels_cuda.h"
#include "turbo_ocr/analysis/detection/det_postprocess.h"

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/errors.h"

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
  // Empty/garbage env values keep the default instead of parsing to 0, and
  // parsed values are CLAMPED to sane ranges (a 0/negative box_thresh floods
  // spurious boxes; a 0 unclip_scale collapses every polygon; inf disables
  // detection entirely — same silent-failure class as GitHub #23). That is
  // exactly env::env_int / env::env_float's contract, so the hand-rolled
  // parse-and-clamp lambda that used to sit here is gone: it did the same work
  // without putting the three knobs into the startup inventory.
  //
  // Ranges match the server-boot strict validation in
  // server_config.h::from_env; this lenient layer covers CLI/tools.
  gpu_ccl_mode_ = env::env_int("GPU_CCL", gpu_ccl_mode_, 0, 2);
  // box_thresh_ + unclip_scale_ apply to all three modes (0/1/2).
  box_thresh_ = env::env_float("GPU_BOX_THRESH", box_thresh_, 0.001f, 1.0f);
  unclip_scale_ = env::env_float("GPU_UNCLIP_SCALE", unclip_scale_, 0.1f, 10.0f);

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

// SUBMIT ONLY — steps 1-4 (preprocess, TRT forward, threshold). Every one is an
// async launch on `stream`, so this returns with the work in flight and nothing
// synchronized. The host post-process that needs the result is collect_boxes().
//
// The split exists so NvDetector can offer the seam's two-phase detection
// (IDetector::enqueue/collect): the caller gets the host back while the device
// works. run() below is the two called back to back — the sequence is
// unchanged, only its seam is new.
void PaddleDet::submit_forward(const GpuImage &gpu_img, int orig_h, int orig_w,
                               cudaStream_t stream) {
  auto [resize_h, resize_w] = compute_det_resize(orig_h, orig_w, resize_);
  // collect_boxes() must read the SAME extents this submission used; it cannot
  // recompute them without re-deriving the resize policy, and a mismatch would
  // decode the probability map at the wrong stride.
  pending_resize_h_ = resize_h;
  pending_resize_w_ = resize_w;

  turbo_ocr::kernels::cuda_fused_resize_normalize_det(gpu_img, d_input_.get(),
                                                      resize_w, resize_h, stream);
  nvinfer1::Dims4 input_dims{1, 3, resize_h, resize_w};
  if (!engine_->infer_dynamic(input_dims, stream))
    throw turbo_ocr::InferenceError("Detection TRT inference failed");
  turbo_ocr::kernels::cuda_threshold_to_u8(d_output_.get(), d_bitmap_buf_.get(),
                                           resize_w, resize_h, db_thresh_, stream);
}

// COLLECT — step 5. Each path below synchronizes `stream` itself before reading
// the map, so this is where the host actually waits.
std::vector<Box> PaddleDet::collect_boxes(int orig_h, int orig_w,
                                          cudaStream_t stream) {
  float *d_pred = d_output_.get();
  uint8_t *d_bitmap = d_bitmap_buf_.get();
  const int resize_h = pending_resize_h_;
  const int resize_w = pending_resize_w_;
  if (gpu_ccl_mode_ == 2)
    return run_gpu_ccl_fast(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream);
  if (gpu_ccl_mode_ == 1)
    return run_gpu_ccl(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream);
  return run_cpu_contours(d_pred, d_bitmap, resize_h, resize_w, orig_h, orig_w, stream);
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
