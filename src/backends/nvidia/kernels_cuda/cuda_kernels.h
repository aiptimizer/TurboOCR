#pragma once

// CudaKernels — the NVIDIA IKernels (backend/kernels.h). A 1:1 wrap of the free
// functions in src/backends/nvidia/kernels_cuda/kernels_cuda.h (the 9 CUDA pre/post
// kernels) plus the fused table/layout preprocessors. Every op forwards to the
// exact existing kernel, so output is bit-identical by construction; caps()
// reports every op native (device == Cuda).
//
//   decode_image       -> NvJpegDecoder (device-resident JPEG) + host tail
//   resize_normalize   -> cuda_fused_resize_normalize_det / _layout
//   warp_crops         -> cuda_batch_roi_warp
//   threshold          -> cuda_threshold_to_u8 / cuda_batch_threshold_to_u8
//   db_postprocess     -> cuda_gpu_ccl_detect + JFA unclip chain (mode-2 pure
//                         device path; the mode-1 per-ROI findContours variant
//                         stays inside PaddleDet, which owns its scratch)
//   argmax             -> cuda_argmax
//   preprocess_region  -> cuda_fused_resize_normalize_layout(sub-rect) /
//                         cuda_fused_table_cls_pre / cuda_fused_slanext_pre[_rgb]
//
// SCRATCH: db_postprocess needs the CCL/JFA scratch buffers PaddleDet
// pre-allocates. Here they are allocated lazily and grown to fit w*h so the hot
// path never allocates after the first call of a given size. All device memory
// is owned via CudaPtr/CudaHostPtr (RAII), matching the existing detector.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "turbo_ocr/backend/kernels.h"

// Forward-declare so this header stays CUDA-light; the .cpp/.cu pulls the real
// device decoder + CudaPtr definitions.
namespace turbo_ocr::decode {
class NvJpegDecoder;
}

namespace turbo_ocr::nvidia {

class CudaKernels final : public backend::IKernels {
public:
  CudaKernels();
  ~CudaKernels() override;

  [[nodiscard]] backend::KernelCaps caps() const override;

  // nvJPEG decodes JPEG only; every other container falls to the host tail, and
  // the caller must know that BEFORE it leases a replica (kernels.h).
  [[nodiscard]] bool can_decode_image(const std::uint8_t *data,
                                      std::size_t len) const override;

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

  void argmax(const float *input_probs, int *output_indices, float *output_scores,
              int batch_size, int seq_len, int num_classes,
              backend::DeviceQueue &queue) override;

  void preprocess_region(const backend::ImageView &src, const backend::Rect &rect,
                         backend::PreprocKind kind, float *dst_chw,
                         backend::DeviceQueue &queue) override;

private:
  // Lazy device decoder (constructed on first decode_image()).
  std::unique_ptr<decode::NvJpegDecoder> decoder_;

  // CCL/JFA scratch for db_postprocess, grown to fit; opaque pimpl so this
  // header need not pull CudaPtr / GpuDetBox.
  struct DbScratch;
  std::unique_ptr<DbScratch> db_;
  void ensure_db_scratch(int w, int h);
};

} // namespace turbo_ocr::nvidia
