#pragma once

// THE shared text-line orientation (0/180) classification policy.
//
// WHY THIS FILE EXISTS: detection has det_config.h and recognition has
// rec_geometry.h + rec_batching.h, and both stages are now identical across
// every backend. Classification had NO shared header, and every surviving
// per-backend accuracy fork found by the duplication audit lived here:
//   * kClsImageH/W/kClsThresh retyped in FIVE places
//     (include/turbo_ocr/analysis/classification/ort_paddle_cls.h:38-40,
//      include/turbo_ocr/analysis/classification/ort_paddle_cls.h:30-32,
//      src/backends/amd/stages/rocm_stages.cpp,
//      src/backends/intel/stages/intel_stages.cpp,
//      src/backends/apple/stages/mps_stages.h)
//   * the 180-degree quad flip written FOUR times, two different ways
//     (a cyclic rebuild on AMD, two std::swaps everywhere else — equal only by
//      luck of the corner ordering)
//   * the flip comparison written as `>` on three backends and `>=` on AMD
//   * ImageNet normalization fed to the classifier on THREE backends
//     (Intel, an Apple variant, and AMD — the third occurrence of one bug)
//
// RULE: no backend re-derives any of this. Include the header, call the helpers.

#include <algorithm> // std::swap
#include <utility>

#include "turbo_ocr/backend/kernels.h"      // NormParams
#include "turbo_ocr/base/geometry/box.h"  // turbo_ocr::Box
#include "turbo_ocr/core/norm_params.h"   // norm::cls_norm

namespace turbo_ocr::classification {

// PP-OCRv5 text-line orientation classifier (PP-LCNet_x0_25) input geometry.
// The PP-OCRv4 shape (48x192) works ONLY on TRT (dynamic-shape profile); ORT,
// OpenVINO, MIGraphX and MPSGraph all reject it against the shipped export.
inline constexpr int kClsImageH = 80;
inline constexpr int kClsImageW = 160;

// Confidence floor for accepting a 180-degree flip. The canonical comparison is
// STRICT (`score > kClsThresh`) — see should_flip_180() below.
inline constexpr float kClsThresh = 0.9f;

// The classifier's output layout: [B, 2] = {p(0 deg), p(180 deg)} after Softmax.
inline constexpr int kClsNumClasses = 2;
inline constexpr int kClsIndex0 = 0;
inline constexpr int kClsIndex180 = 1;

// THE classifier normalization. It is REC's ((x/255 - 0.5)/0.5, i.e. x/127.5 - 1),
// NOT ImageNet — the PP-LCNet backbone is a red herring, the shipped export was
// trained with rec's transform. Authorities:
//   src/analysis/classification/ort_paddle_cls.cpp:33  convertTo(CV_32F, 1.0/127.5, -1.0)
//   src/backends/nvidia/stages/paddle_cls.cpp:67-71   MEASURED 85.37% (rec) vs 85.30%
//                                             (ImageNet) on FUNSD-50
// Three backends have independently "corrected" this to ImageNet and each time
// it produced mis-classified 180-degree lines (=> reversed text) on that backend
// alone. Do not repeat it.
[[nodiscard]] inline backend::NormParams cls_norm() noexcept {
  return backend::norm::cls_norm();
}

// THE 180-degree quad rotation. Corner order is [tl, tr, br, bl]; rotating the
// quad by 180 degrees maps it to [br, bl, tl, tr], i.e. swap the two diagonal
// pairs. AMD rebuilt the array cyclically ({q[2],q[3],q[0],q[1]}) and the others
// used two std::swaps; those happen to agree for this corner order, but only by
// luck — one shared definition removes the coincidence.
inline void flip_quad_180(turbo_ocr::Box &b) noexcept {
  std::swap(b.pts[0], b.pts[2]);
  std::swap(b.pts[1], b.pts[3]);
}

// THE flip decision, matching OrtPaddleCls::run / PaddleCls::run verbatim: flip
// only when the 180 class both WINS and clears the confidence threshold, with a
// STRICT comparison. AMD used `>=`, which flips a hair more lines than every
// other backend on exactly-0.9 scores.
[[nodiscard]] inline bool should_flip_180(float p0, float p180) noexcept {
  return p180 > p0 && p180 > kClsThresh;
}

// Same decision expressed for backends whose engine emits an argmax head
// (index + max value) instead of the raw [B,2] row — Apple's MPSGraph path.
// Equivalent because the export ends in Softmax, so the max value IS p180 when
// the argmax index is kClsIndex180.
[[nodiscard]] inline bool should_flip_180_argmax(int argmax_index,
                                                 float max_score) noexcept {
  return argmax_index == kClsIndex180 && max_score > kClsThresh;
}

} // namespace turbo_ocr::classification
