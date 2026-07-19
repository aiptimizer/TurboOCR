#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "turbo_ocr/common/types.h"

// Crop-width geometry shared by the TRT and ORT recognizers. The width math
// and its ceiling live here exactly once: kMaxRecWidth bounds every rec
// engine profile AND every crop computation, so a change in one place cannot
// desynchronize buffer sizes from bucket selection (an out-of-bounds read
// away when they drift).
namespace turbo_ocr::recognition {

// KNOWN RECALL CEILING: this cap horizontally compresses any line with
// aspect ratio beyond kMaxRecWidth/rec_image_h (4000/48 ≈ 83:1 — e.g. a
// full-width 2000px line under ~24px tall), squashing glyphs below the CTC
// receptive field. Inherent CRNN limit, rare on document lines at det scale;
// the mitigation (split over-long crops and stitch the transcripts) is an
// accuracy-gated experiment, not a local fix.
inline constexpr int kMaxRecWidth = 4000;

// Narrowest crop width fed to any recognizer. Tiny boxes (single characters,
// checkbox marks) keep their natural width down to this floor and are
// right-padded — never stretched to a wider canvas, which smears the glyph
// beyond recognition. Both the TRT and ORT paths must use this same floor or
// their outputs diverge on exactly those small boxes.
inline constexpr int kMinRecWidth = 32;

// Fixed bucket table for the TRT engine's optimization profiles. The last
// bucket must equal kMaxRecWidth so every clamped width has a bucket.
inline constexpr std::array kRecWidthBuckets = {320, 480, 800, 1200,
                                                1600, 2000, 2500, 3200, 4000};
static_assert(kRecWidthBuckets.back() == kMaxRecWidth,
              "bucket table must cover the full clamped width range");

// Aspect ratio of a (possibly rotated) detection quad: edge lengths, not AABB.
[[nodiscard]] inline float box_aspect(const Box &box) {
  const float w = std::sqrt(((box[0][0] - box[1][0]) * (box[0][0] - box[1][0])) +
                            ((box[0][1] - box[1][1]) * (box[0][1] - box[1][1])));
  const float h = std::sqrt(((box[0][0] - box[3][0]) * (box[0][0] - box[3][0])) +
                            ((box[0][1] - box[3][1]) * (box[0][1] - box[3][1])));
  return (h > 0) ? (w / h) : 0.0f;
}

// Natural crop width at the recognizer input height, clamped to
// [floor_w, kMaxRecWidth]. floor_w differs by backend: the TRT path allows
// down to 32px, the ORT path floors at its model input width.
[[nodiscard]] inline int natural_rec_width(float aspect, int rec_image_h,
                                           int floor_w) {
  const int w = std::min(static_cast<int>(std::ceil(rec_image_h * aspect)),
                         kMaxRecWidth);
  return std::max(w, floor_w);
}

// TRT policy: snap to the fixed bucket table (matches the engine profiles).
[[nodiscard]] inline int snap_width_bucket(int w) {
  return *std::lower_bound(kRecWidthBuckets.begin(), kRecWidthBuckets.end(),
                           std::min(w, kMaxRecWidth));
}

// ORT policy: snap UP to a step multiple so a batch only ever pads each crop
// by at most step-1 columns, clamped to the global ceiling but never below
// the crop's own content width.
[[nodiscard]] inline int snap_width_step(int target_w, int step) {
  int bucket_w = target_w;
  if (step > 1) bucket_w = ((target_w + step - 1) / step) * step;
  bucket_w = std::min(bucket_w, kMaxRecWidth);
  return std::max(bucket_w, target_w);
}

} // namespace turbo_ocr::recognition
