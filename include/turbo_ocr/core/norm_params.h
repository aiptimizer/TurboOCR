#pragma once

// THE canonical NormParams factories — one definition of every normalization
// the pipeline uses, shared by EVERY backend.
//
// WHY THIS FILE EXISTS (read before adding a constant anywhere else):
// detection got det_config.h and recognition got rec_geometry.h + rec_batching.h,
// and both stages are now provably identical across CPU/CUDA/HIP/SYCL/Metal.
// Normalization got no such header, so the same six magic floats were retyped in
// eight places (intel_stages.cpp, host_kernels.cpp, rocm_stages.cpp x3,
// mps_stages.mm, sycl_kernels.cpp x2) — and three separate backends then shipped
// the SAME bug: ImageNet mean/std fed to the text-line orientation classifier,
// which is trained on rec's (x/127.5 - 1). Wrong-distribution input to cls means
// mis-detected 180-degree lines means reversed text, on that backend only.
//
// RULE: a backend or stage NEVER writes mean/inv_std/inv_scale literals. It calls
// one of these factories. If a new model needs a new distribution, add a factory
// here so every backend picks it up at once.

#include "turbo_ocr/backend/kernels.h" // NormParams, ChannelOrder

namespace turbo_ocr::backend::norm {

// ---------------------------------------------------------------------------
// REC / CLS: (pixel/255 - 0.5) / 0.5  ==  pixel/127.5 - 1, RGB planes.
//
// Used by BOTH the recognizer and the text-line orientation classifier. The
// classifier looks like it "should" be ImageNet (PP-LCNet_x0_25 backbone), but
// the shipped export is NOT: see src/analysis/classification/ort_paddle_cls.cpp:33
// (convertTo(CV_32F, 1.0/127.5, -1.0)) and the MEASURED note in
// src/backends/nvidia/stages/paddle_cls.cpp:67-71 (rec norm 85.37% FUNSD vs ImageNet
// 85.30%). Do not "fix" cls to ImageNet — that regression has now been
// introduced three times (Intel, Apple variant, AMD).
//
// This is also the seam's DEFAULT NormParams, so `NormParams{}` == rec_norm().
[[nodiscard]] inline NormParams rec_norm() noexcept {
  NormParams p;
  p.inv_scale = 1.0f / 255.0f;
  p.mean[0] = p.mean[1] = p.mean[2] = 0.5f;
  p.inv_std[0] = p.inv_std[1] = p.inv_std[2] = 2.0f;
  p.order = ChannelOrder::RGB;
  p.letterbox = false;
  return p;
}

// The classifier's normalization IS rec's. Named separately so cls call sites
// read as a deliberate choice rather than a copy of rec's, and so a future
// re-export that genuinely wants ImageNet changes ONE line here.
[[nodiscard]] inline NormParams cls_norm() noexcept { return rec_norm(); }

// ---------------------------------------------------------------------------
// DET / TABLE / SLANeXt: ImageNet mean/std over pixel/255, applied POSITIONALLY
// (plane 0 gets 0.485 regardless of which source channel feeds it). `order`
// selects which source channel feeds plane 0.
[[nodiscard]] inline NormParams imagenet(ChannelOrder order) noexcept {
  NormParams p;
  p.inv_scale = 1.0f / 255.0f;
  p.mean[0] = 0.485f;
  p.mean[1] = 0.456f;
  p.mean[2] = 0.406f;
  p.inv_std[0] = 1.0f / 0.229f;
  p.inv_std[1] = 1.0f / 0.224f;
  p.inv_std[2] = 1.0f / 0.225f;
  p.order = order;
  p.letterbox = false;
  return p;
}

// The det/table convention: BGR-positional (plane 0 == B gets 0.485). This is
// what cuda_fused_resize_normalize_det bakes in, so every other backend must
// match it or R and B are swapped on that backend alone.
[[nodiscard]] inline NormParams imagenet_bgr() noexcept {
  return imagenet(ChannelOrder::BGR);
}

// The SLANeXt encoder-split export wants RGB planes with the same ImageNet stats.
[[nodiscard]] inline NormParams imagenet_rgb() noexcept {
  return imagenet(ChannelOrder::RGB);
}

// ---------------------------------------------------------------------------
// LAYOUT (PP-DocLayoutV3): pixel/255 only — its NormalizeImage is {mean:0,std:1}.
// BGR planes, stretched to the 800x800 canvas (letterbox=false is DELIBERATE:
// the layout stage's coordinate rescale is derived from the stretch; see the
// letterbox note in kernels.h).
[[nodiscard]] inline NormParams layout_norm() noexcept {
  NormParams p;
  p.inv_scale = 1.0f / 255.0f;
  p.mean[0] = p.mean[1] = p.mean[2] = 0.0f;
  p.inv_std[0] = p.inv_std[1] = p.inv_std[2] = 1.0f;
  p.order = ChannelOrder::BGR;
  p.letterbox = false;
  return p;
}

} // namespace turbo_ocr::backend::norm
