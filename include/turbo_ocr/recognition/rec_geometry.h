#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "turbo_ocr/common/geometry/perspective.h"
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

// Recognizer input width for a detection quad: the SWAPPED natural content
// width from compute_crop_transform — the same transform the warp itself
// evaluates — floored at kMinRecWidth (capped at kMaxRecWidth inside the
// transform). The single width source for BOTH the TRT and ORT recognizers.
// Deriving it from the raw edge aspect instead (box_aspect) under-sizes
// vertical boxes: their bucket then clamps the swapped width and the rendered
// text gets compressed horizontally.
[[nodiscard]] inline int rec_input_width(const turbo_ocr::Box &box,
                                         int rec_image_h) {
  const auto ct = turbo_ocr::compute_crop_transform(box, rec_image_h,
                                                    kMaxRecWidth);
  return std::max(ct.crop_width, kMinRecWidth);
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
