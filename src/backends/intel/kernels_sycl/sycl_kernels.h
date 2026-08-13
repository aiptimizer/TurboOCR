#pragma once

// SyclKernels — the Intel backend's IKernels implementation (backend/kernels.h).
// SYCL device kernels (oneAPI/DPC++, Level Zero) that reproduce the semantics of
// src/backends/cpu/kernels_host/host_kernels.cpp (which is itself the portable statement of
// src/backends/nvidia/kernels_cuda/*.cu), so the Intel path stays device-resident around the OpenVINO
// forward passes exactly as the CUDA kernels do around TensorRT.
//
// NATIVE SYCL (caps() == true):
//   resize_normalize   <- fused_resize_normalize_chw_kernel (bilinear BGR + norm)
//   warp_crops         <- batch_roi_warp_kernel (perspective + bilinear + norm)
//   threshold          <- threshold_to_u8_kernel
//   argmax             <- argmax_kernel (per-timestep, lowest-index tie-break)
//   preprocess_region  <- cuda_fused_* region preprocessors
//
// DECLARED HOST FALLBACK (caps() == false) — the plan's graceful-degradation
// lever, and the RIGHT call rather than a hand-rolled approximation:
//
//   db_postprocess — DB connected components + Clipper unclip + min-area-rect
//     ordering. There is no portable SYCL primitive for CCL, and the shared
//     host implementation (detection::extract_boxes_from_bitmap) is the SAME
//     function the CPU and NVIDIA-contour paths use, so falling back costs zero
//     accuracy and inherits every future fix. This op is called ONCE per image
//     on a small (<= ~1280x1280 f32 + u8) map, so the D2H is a rounding error
//     next to det inference. Writing a bespoke SYCL union-find here would be a
//     second implementation of shared post-processing policy — precisely what
//     the dedup rules forbid — and would have to be re-validated separately.
//     A native version is a measured optimisation, not a correctness need.
//
//   decode_image — host OpenCV decode + H2D. No VAAPI/oneVPL path yet (the
//     nvJPEG analogue); see README bring-up item 5.
//
// Both fallbacks move only small buffers; the page image, crops and logits stay
// in USM for the whole pipeline.
//
// The public header carries no SYCL types (pImpl), so the device-neutral shared
// pipeline can hold a unique_ptr<IKernels> without a DPC++ toolchain in scope.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "turbo_ocr/backend/kernels.h"

namespace turbo_ocr::intel {

class L0Allocator;

class SyclKernels final : public backend::IKernels {
public:
  // `alloc` owns the USM context; every kernel enqueues on the CALLER's
  // DeviceQueue so device work stays on one ordered lane.
  explicit SyclKernels(std::shared_ptr<L0Allocator> alloc);
  ~SyclKernels() override;

  [[nodiscard]] backend::KernelCaps caps() const override;

  [[nodiscard]] backend::ImageView decode_image(const std::uint8_t *data,
                                                std::size_t len,
                                                backend::DeviceQueue &queue) override;

  void resize_normalize(const backend::ImageView &src, float *dst_chw, int dst_w,
                        int dst_h, const backend::NormParams &params,
                        backend::DeviceQueue &queue) override;

  void warp_crops(const backend::ImageView &src, const float *d_M_invs,
                  const int *d_crop_widths, float *d_dst_batch, int batch_size,
                  int dst_h, int dst_w, const backend::NormParams &params,
                  backend::DeviceQueue &queue) override;

  void threshold(const float *src, std::uint8_t *dst, int w, int h,
                 int batch_size, float thresh, backend::DeviceQueue &queue) override;

  [[nodiscard]] std::vector<turbo_ocr::Box>
  db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap, int w,
                 int h, const backend::DbPostParams &params,
                 backend::DeviceQueue &queue) override;

  void argmax(const float *input_probs, int *output_indices,
              float *output_scores, int batch_size, int seq_len,
              int num_classes, backend::DeviceQueue &queue) override;

  void preprocess_region(const backend::ImageView &src, const backend::Rect &rect,
                         backend::PreprocKind kind, float *dst_chw,
                         backend::DeviceQueue &queue) override;

  // Warmup hook: pre-size the host mirrors the two fallback ops need, so the
  // first request does not allocate. `max_map_pixels` is the largest det map
  // (resize_h * resize_w) the detector can produce.
  void reserve_host_fallback(std::size_t max_map_pixels) override;

  struct Impl;

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::intel
