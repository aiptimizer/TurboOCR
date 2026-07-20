#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/engine/trt/trt_engine.h"
#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/cuda/cuda_ptr.h"
#include "turbo_ocr/kernels/kernels.h"

namespace turbo_ocr::detection {

/// GPU text detector using TensorRT (DB post-processing).
class PaddleDet {
public:
  PaddleDet() = default;
  ~PaddleDet() noexcept = default; // RAII handles all GPU cleanup

  /// Load a TensorRT detection engine and allocate GPU buffers. resize/db are
  /// this model's official PaddleOCR detection config (server::DetInferConfig
  /// fields); both default to the kDetResizeDefault/kDbDefaults base so
  /// explicit-DET-override callers and tests keep working. Env vars layered on
  /// top by read_det_resize/read_db_params in init_buffers() always win.
  [[nodiscard]] bool load_model(const std::string &model_path,
                                const DetResizeParams &resize = kDetResizeDefault,
                                const DbParams &db = kDbDefaults);

  // Takes GpuImage directly - no double upload
  [[nodiscard]] std::vector<Box> run(const GpuImage &gpu_img, int orig_h, int orig_w,
                                     cudaStream_t stream = 0);

  // Batched detection: process N images in a single TRT inference call.
  // All images are resized to the same dimensions (max of the batch, rounded to 32).
  // Returns one vector<Box> per image.
  [[nodiscard]] std::vector<std::vector<Box>>
  run_batch(const std::vector<GpuImage> &gpu_imgs,
            const std::vector<std::pair<int,int>> &orig_dims,
            cudaStream_t stream = 0);

private:
  // Per-model resize policy (this model's official config + env overrides).
  // Set from read_det_resize(cfg) in init_buffers(); drives compute_det_resize()
  // at every resize site. Buffers size off effective_det_max_side(resize_).
  DetResizeParams resize_ = kDetResizeDefault;

  // DB post-processing parameters (PP-OCRv6 defaults). Set from
  // detection/det_config.h read_db_params() in init_buffers(); env-overridable
  // via DET_DB_THRESH/DET_BOX_THRESH/DET_UNCLIP.
  float db_thresh_ = kDbDefaults.thresh;
  float unclip_ratio_ = kDbDefaults.unclip_ratio;
  // Engine optimization-profile MAX side. Set to effective_det_max_side(resize_)
  // (resize_.max_side_limit, DET_MAX_SIDE env wins) in init_buffers(); sizes the
  // pinned buffers. Pre-init to the config default so the value is sane before
  // load_model runs.
  int kMaxSideLen_ = kDetResizeDefault.max_side_limit;
  static constexpr float kMinBoxSide = 3.0f;
  static constexpr float kMinUnclippedSide = 5.0f; // kMinBoxSide + 2

  // GPU CCL mode:
  //   0 = CPU contours fallback (OpenCV findContours)
  //   1 = GPU CCL + per-ROI findContours on CPU (default; produces rotated
  //       min-area-rects; F1 matches CPU baseline)
  //   2 = all-GPU JFA per-component Euclidean unclip (no pred_map download,
  //       no CPU contours; F1 within run-to-run noise of CCL=1; axis-aligned
  //       quads only)
  int gpu_ccl_mode_ = 1;
  // Set from read_db_params() in init_buffers(); GPU_BOX_THRESH/
  // GPU_UNCLIP_SCALE remain as overrides on top.
  float box_thresh_ = kDbDefaults.box_thresh;
  float unclip_scale_ = 1.0f;

  std::unique_ptr<engine::TrtEngine> engine_;

  // Maximum batch size for batched detection
  static constexpr int kMaxBatchSize = 8;

  // Pre-allocated GPU buffers (single-image, RAII)
  CudaPtr<float> d_input_;
  CudaPtr<float> d_output_;

  // Pre-allocated batch GPU buffers (kMaxBatchSize images, RAII)
  CudaPtr<float> d_batch_input_;
  CudaPtr<float> d_batch_output_;
  CudaPtr<uint8_t> d_batch_bitmap_;

  // Device-side arrays for batched kernel launch params (RAII).
  // dst_heights/dst_widths are each image's aspect-preserving letterbox
  // content dims inside the unified batch canvas.
  CudaPtr<void *> d_batch_src_ptrs_;
  CudaPtr<int> d_batch_src_steps_;
  CudaPtr<int> d_batch_src_heights_;
  CudaPtr<int> d_batch_src_widths_;
  CudaPtr<int> d_batch_dst_heights_;
  CudaPtr<int> d_batch_dst_widths_;

  // Pinned host staging for batch metadata (RAII)
  CudaHostPtr<void *> h_batch_src_ptrs_;
  CudaHostPtr<int> h_batch_src_steps_;
  CudaHostPtr<int> h_batch_src_heights_;
  CudaHostPtr<int> h_batch_src_widths_;
  CudaHostPtr<int> h_batch_dst_heights_;
  CudaHostPtr<int> h_batch_dst_widths_;

  // Pre-allocated bitmap buffer (RAII)
  CudaPtr<uint8_t> d_bitmap_buf_;

  // GPU CCL buffers (pre-allocated in load_model — NO per-request alloc, RAII)
  CudaPtr<int> d_ccl_labels_;
  CudaPtr<int> d_ccl_compact_ids_;     // [max_pixels] compact component IDs
  CudaPtr<int> d_ccl_id_counter_;      // [1] atomic counter for compact IDs
  CudaPtr<kernels::GpuDetBox> d_ccl_bboxes_;
  CudaPtr<int> d_ccl_num_boxes_;
  // Host-side result buffer for GPU CCL (pinned memory, RAII)
  CudaHostPtr<kernels::GpuDetBox> h_ccl_boxes_;
  // Pinned destination for the bitmap download in run_gpu_ccl — a pageable
  // cv::Mat there degrades the async copy to a ~47µs staged blocking copy
  // per image.
  CudaHostPtr<uint8_t> h_bitmap_;
  size_t h_bitmap_pixels_ = 0;

  // Reusable contour/mask buffers (avoid per-call heap allocation)
  std::vector<cv::Point> shifted_buf_;
  cv::Mat mask_buf_;
  std::vector<std::vector<cv::Point>> contours_buf_;
  std::vector<cv::Vec4i> hierarchy_buf_;

  // GPU CCL contour extraction buffers (reused per-component)
  std::vector<std::vector<cv::Point>> ccl_roi_contours_buf_;
  std::vector<cv::Point> ccl_contour_buf_;

  // JFA buffers for per-component Euclidean unclip on GPU (RAII).
  // Used by run_gpu_ccl_fast (GPU_CCL=2): all-GPU post-processing path that
  // matches CPU CCL=1 accuracy without downloading the prediction map.
  CudaPtr<uint32_t> d_jfa_labels_;     // [max_pixels] expanded label map
  CudaPtr<uint32_t> d_jfa_seeds_;      // [max_pixels] packed JFA nearest-seed coords (primary)
  CudaPtr<uint32_t> d_jfa_seeds_alt_;  // [max_pixels] JFA ping-pong buffer
  CudaPtr<float> d_expand_per_comp_;   // [kMaxGpuComponents] per-component expand
  CudaPtr<int> d_perim_per_comp_;      // [kMaxGpuComponents] per-component crack perimeter
  // Oriented min-area-rect scratch (mode-2): PCA second-moment sums (uint64)
  // and per-component axis + projection extents (float). [kMaxGpuComponents*6].
  CudaPtr<unsigned long long> d_ccl_moments_;
  CudaPtr<float> d_ccl_orient_;
  // Pinned host buffer for post-expand bboxes. Pre-allocated once so
  // run_gpu_ccl_fast doesn't cudaMallocHost on every request.
  CudaHostPtr<kernels::GpuDetBox> h_exp_boxes_;

  // Common buffer allocation. resize/db are the per-model config base; env
  // overrides (read_det_resize/read_db_params) are applied here so they win.
  [[nodiscard]] bool init_buffers(const DetResizeParams &resize,
                                  const DbParams &db);

  // Post-process helpers take the device probability map + bitmap for the image
  // slice explicitly (no hidden member state), so single-image and per-batch-slice
  // callers share one path re-usable across SEQUENTIAL slices. Not thread-safe:
  // the helpers still write shared instance scratch (h_ccl_boxes_, ccl_contour_buf_,
  // d_jfa_*, ...), so one instance serves one worker thread (the pool contract).

  // resize_h/resize_w are the probability-map extents (the batch canvas for
  // run_batch slices). content_h/content_w are the letterboxed extent the
  // image actually occupies inside that canvas — the box→original mapping
  // divides by content/orig, never canvas/orig. -1 (single-image callers)
  // means content == canvas.

  // GPU CCL path: returns boxes from GPU + per-ROI findContours (accurate)
  [[nodiscard]] std::vector<Box> run_gpu_ccl(const float *d_pred, const uint8_t *d_bitmap,
                                              int resize_h, int resize_w,
                                              int orig_h, int orig_w,
                                              cudaStream_t stream,
                                              int content_h = -1, int content_w = -1);

  // GPU CCL fast (GPU_CCL=2): all-GPU JFA per-component Euclidean unclip.
  // Matches CPU CCL=1 word-F1 within run-to-run noise (~0.900 vs 0.902 on
  // FUNSD), with tighter latency tail (no pred_map download, no findContours).
  [[nodiscard]] std::vector<Box> run_gpu_ccl_fast(const float *d_pred, const uint8_t *d_bitmap,
                                                    int resize_h, int resize_w,
                                                    int orig_h, int orig_w,
                                                    cudaStream_t stream,
                                                    int content_h = -1, int content_w = -1);

  // CPU fallback path (original findContours)
  [[nodiscard]] std::vector<Box> run_cpu_contours(const float *d_pred, const uint8_t *d_bitmap,
                                                   int resize_h, int resize_w,
                                                   int orig_h, int orig_w,
                                                   cudaStream_t stream,
                                                   int content_h = -1, int content_w = -1);

};

} // namespace turbo_ocr::detection
