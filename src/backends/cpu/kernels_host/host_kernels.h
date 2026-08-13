#pragma once

// HostKernels — the CpuBackend IKernels (backend/kernels.h), implemented with
// OpenCV on the host. Every op runs natively on the CPU (KernelCaps all true),
// because the host IS the fallback target the other backends copy to; there is
// nothing below it to degrade onto.
//
// The op set mirrors the existing device kernels exactly:
//   * warp_crops     — cv::warpPerspective per crop, /norm to CHW, zero-pad —
//                      bit-for-bit the crop+normalize OrtPaddleRec::preprocess_box
//                      performs (single inverse warp, BORDER_REPLICATE, RGB planes).
//   * resize_normalize — cv::resize + a folded (scale, shift) convertTo per
//                      channel, matching the det ImageNet-BGR normalization.
//   * threshold      — cv::threshold to a uint8 fg/bg bitmap.
//   * db_postprocess — delegates to the shared, already-proven
//                      detection::extract_boxes_from_bitmap (CCL + unclip).
//   * argmax         — the per-timestep CTC argmax loop.
//   * decode_image   — host JPEG/PNG decode into a kernels-owned BGR8 buffer.
//   * preprocess_region — fused table/layout region preprocessors.
//
// MEMORY: on the host every pointer/ImageView already lives in host RAM, so
// there is no copy-to-device; the DeviceQueue is a synchronous no-op.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/backend/kernels.h"

namespace turbo_ocr::cpu {

class HostKernels final : public backend::IKernels {
public:
  HostKernels() = default;
  ~HostKernels() override = default;

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
                 int batch_size, float thresh,
                 backend::DeviceQueue &queue) override;

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

private:
  // Owns the most recent decode_image() output so its ImageView stays valid
  // until the next decode (the seam's stated invalidation point).
  cv::Mat decoded_;

  // Reusable scratch for db_postprocess (extract_boxes_from_bitmap is
  // caller-scratch-threaded for thread-safety).
  std::vector<cv::Point> shifted_buf_;
  cv::Mat mask_buf_;
  std::vector<std::vector<cv::Point>> contours_buf_;
  std::vector<cv::Vec4i> hierarchy_buf_;
  cv::Mat bitmap_scratch_;
};

} // namespace turbo_ocr::cpu
