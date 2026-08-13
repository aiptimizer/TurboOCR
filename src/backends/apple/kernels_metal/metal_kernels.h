#pragma once

// MetalKernels — the Apple IKernels (backend/kernels.h). Device pre/post ops as
// Metal compute shaders (shaders.metal), plus the host-fallback DB post-process.
//
// NATIVE on the GPU (caps() == true):
//   warp_crops       -> warp_crops        (tools/probes/apple/warp.metal, the residency win)
//   resize_normalize -> resize_normalize  (det/layout resize + normalize)
//   threshold        -> threshold_u8      (prob map -> bitmap for DB)
//   argmax           -> argmax            (raw [B,T,C] logits -> [B,T] idx/score;
//                       the recognizer's default path folds this into MPSGraph)
//   decode_image     -> host imdecode + resident RGBA8/BGR8 upload (MetalImage)
//
// HOST FALLBACK (caps() == false) — no Metal primitive yet:
//   db_postprocess   -> turbo_ocr::detection::extract_boxes_from_bitmap
//                       (CCL + unclip on the host; unified memory makes the
//                        D2H a coherent read, not a PCIe copy). TODO(apple-hardware): a Metal
//                        union-find CCL + JFA-unclip for full residency + a
//                        one-command-buffer detect (see README).
//   preprocess_region-> TODO(apple-hardware) table/layout fused preproc; logs unsupported.
//
// Every buffer arg lives in the queue's Metal space (a registered MTLBuffer's
// .contents); textures are recovered from ImageView via ensure_texture().

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "turbo_ocr/backend/kernels.h"

namespace turbo_ocr::apple {

class MetalImage;

class MetalKernels final : public backend::IKernels {
public:
  MetalKernels();
  ~MetalKernels() override;

  [[nodiscard]] backend::KernelCaps caps() const override;

  [[nodiscard]] backend::ImageView
  decode_image(const std::uint8_t *data, std::size_t len,
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
  // Owned MetalImage for the last decode_image() (valid until the next decode,
  // per the kernels.h decode contract). Opaque to keep this header ObjC-free.
  std::unique_ptr<MetalImage> decoded_;
};

} // namespace turbo_ocr::apple
