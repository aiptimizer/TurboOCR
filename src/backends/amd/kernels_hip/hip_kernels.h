#pragma once

// HipKernels — backend::IKernels over the hipified pre/post kernel library
// (amd/kernels_hip/*.hip). This is the AMD device pre/post op set: it keeps the
// detector's pred-map, bitmap, CCL scratch, rec/cls warp batches, and argmax all
// on the GPU, returning to the host only at the small DB-box / argmax boundary.
//
// caps() reports every op native EXCEPT:
//   * decode_image — no nvJPEG analog on ROCm. Host-fallback: OpenCV imdecode +
//     H2D into a pooled device buffer. This is reported HONESTLY as
//     caps().decode_image == false so the shared layer can account for the host
//     round-trip rather than believing decode is resident. (TODO: rocJPEG/VAAPI.)
// Everything else — resize_normalize, warp_crops, threshold, db_postprocess
// (CCL + JFA unclip), argmax, preprocess_region (all four PreprocKinds, ported
// in kernels_hip/table_kernels.hip) — runs natively via the .hip kernels.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "turbo_ocr/backend/kernels.h"

namespace turbo_ocr::amd {

using backend::DbPostParams;
using backend::DeviceQueue;
using backend::ImageView;
using backend::IKernels;
using backend::KernelCaps;
using backend::NormParams;
using backend::PreprocKind;
using backend::Rect;

class HipAllocator;

class HipKernels final : public IKernels {
public:
  // `alloc` supplies the pooled decode buffer + CCL scratch; shared with the
  // rest of the backend so allocations are validated against one allocator.
  explicit HipKernels(std::shared_ptr<HipAllocator> alloc);
  ~HipKernels() override;

  [[nodiscard]] KernelCaps caps() const override;

  [[nodiscard]] ImageView decode_image(const std::uint8_t *data,
                                       std::size_t len,
                                       DeviceQueue &queue) override;

  void resize_normalize(const ImageView &src, float *dst_chw, int dst_w,
                        int dst_h, const NormParams &params,
                        DeviceQueue &queue) override;

  void warp_crops(const ImageView &src, const float *d_M_invs,
                  const int *d_crop_widths, float *d_dst_batch, int batch_size,
                  int dst_h, int dst_w, const NormParams &params,
                  DeviceQueue &queue) override;

  void threshold(const float *src, std::uint8_t *dst, int w, int h,
                 int batch_size, float thresh, DeviceQueue &queue) override;

  [[nodiscard]] std::vector<turbo_ocr::Box>
  db_postprocess(const float *d_pred_map, const std::uint8_t *d_bitmap, int w,
                 int h, const DbPostParams &params, DeviceQueue &queue) override;

  void argmax(const float *input_probs, int *output_indices,
              float *output_scores, int batch_size, int seq_len,
              int num_classes, DeviceQueue &queue) override;

  void preprocess_region(const ImageView &src, const Rect &rect,
                         PreprocKind kind, float *dst_chw,
                         DeviceQueue &queue) override;

private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

} // namespace turbo_ocr::amd
