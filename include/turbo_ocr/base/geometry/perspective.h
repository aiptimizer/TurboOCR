#pragma once

#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/base/geometry/perspective_math.h"

#include <algorithm>
#include <cmath>

namespace turbo_ocr {

/// Result of computing a crop transform for a box.
struct CropTransform {
  float M_inv[9];  // 3x3 inverse perspective matrix
  int crop_width;  // width of the resized crop in pixels
  bool vertical;   // true if the box was rotated (vertical text)
};

/// Compute the inverse perspective transform that maps a destination rectangle
/// (0,0)-(resize_w, target_h) back to the quadrilateral defined by @p box in
/// the source image.  The destination width is clamped to [1, max_width].
///
/// The single source of truth for crop-transform logic: both recognizers
/// (TRT warp kernel, ORT cv::warpPerspective), both classifiers, and
/// rec_input_width all derive their geometry from this function.
inline CropTransform compute_crop_transform(const Box &box, int target_h,
                                            int max_width) {
  auto f = [](int v) { return static_cast<float>(v); };
  float bx0 = f(box[0][0]), by0 = f(box[0][1]);
  float bx1 = f(box[1][0]), by1 = f(box[1][1]);
  float bx2 = f(box[2][0]), by2 = f(box[2][1]);
  float bx3 = f(box[3][0]), by3 = f(box[3][1]);

  float crop_w =
      std::sqrt((bx0 - bx1) * (bx0 - bx1) + (by0 - by1) * (by0 - by1));
  float crop_h =
      std::sqrt((bx0 - bx3) * (bx0 - bx3) + (by0 - by3) * (by0 - by3));

  bool vertical = (crop_h >= crop_w * kVerticalAspectRatio);

  float src_f[8];
  if (vertical) {
    src_f[0] = bx3; src_f[1] = by3;
    src_f[2] = bx0; src_f[3] = by0;
    src_f[4] = bx1; src_f[5] = by1;
    src_f[6] = bx2; src_f[7] = by2;
    std::swap(crop_w, crop_h);
  } else {
    src_f[0] = bx0; src_f[1] = by0;
    src_f[2] = bx1; src_f[3] = by1;
    src_f[4] = bx2; src_f[5] = by2;
    src_f[6] = bx3; src_f[7] = by3;
  }

  float ar = (crop_h > 0) ? (crop_w / crop_h) : 0;
  int resize_w =
      std::min(static_cast<int>(std::ceil(target_h * ar)), max_width);
  resize_w = std::max(resize_w, 1);

  float rw = f(resize_w), rh = f(target_h);
  float dst_f[8] = {0, 0, rw, 0, rw, rh, 0, rh};

  CropTransform ct{};
  ct.crop_width = resize_w;
  ct.vertical = vertical;
  compute_perspective_inv(dst_f, src_f, ct.M_inv);
  return ct;
}

} // namespace turbo_ocr
