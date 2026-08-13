#include "turbo_ocr/pipeline/ocr_pipeline_detail.h"

#include <algorithm>
#include <climits>
#include <string>

#include "turbo_ocr/base/env_utils.h"


namespace turbo_ocr::pipeline::detail {

Box adjust_table_region(const Box &in,
                        const std::vector<OCRResultItem> &results) {
  // Margin is a fraction of the region, so [0,1] is its whole domain.
  static const float kCropMargin = env::env_float("TABLE_CROP_MARGIN", 0.03f, 0.0f, 1.0f);
  static const bool kDetUnion = env::env_or("TABLE_CROP_MODE", "") == "detunion";
  Box region = in;
  if (kDetUnion) {
    int lx1 = INT_MAX, ly1 = INT_MAX, lx2 = INT_MIN, ly2 = INT_MIN;
    for (const auto &p : region.pts) {
      lx1 = std::min(lx1, p[0]); ly1 = std::min(ly1, p[1]);
      lx2 = std::max(lx2, p[0]); ly2 = std::max(ly2, p[1]);
    }
    int ux1 = INT_MAX, uy1 = INT_MAX, ux2 = INT_MIN, uy2 = INT_MIN;
    bool any = false;
    for (const auto &r : results) {
      int cx = 0, cy = 0;
      for (const auto &p : r.box.pts) { cx += p[0]; cy += p[1]; }
      cx /= 4; cy /= 4;
      if (cx >= lx1 && cx <= lx2 && cy >= ly1 && cy <= ly2) {
        for (const auto &p : r.box.pts) {
          ux1 = std::min(ux1, p[0]); uy1 = std::min(uy1, p[1]);
          ux2 = std::max(ux2, p[0]); uy2 = std::max(uy2, p[1]);
        }
        any = true;
      }
    }
    if (any) {
      int nx1 = std::max(lx1, ux1), ny1 = std::max(ly1, uy1);
      int nx2 = std::min(lx2, ux2), ny2 = std::min(ly2, uy2);
      if (nx2 > nx1 && ny2 > ny1)
        region.pts = {{{nx1, ny1}, {nx2, ny1}, {nx2, ny2}, {nx1, ny2}}};
    }
  }
  if (kCropMargin > 0.0f) {
    int ax1 = INT_MAX, ay1 = INT_MAX, ax2 = INT_MIN, ay2 = INT_MIN;
    for (const auto &p : region.pts) {
      ax1 = std::min(ax1, p[0]); ay1 = std::min(ay1, p[1]);
      ax2 = std::max(ax2, p[0]); ay2 = std::max(ay2, p[1]);
    }
    const int mw = static_cast<int>((ax2 - ax1) * kCropMargin);
    const int mh = static_cast<int>((ay2 - ay1) * kCropMargin);
    ax1 -= mw; ay1 -= mh; ax2 += mw; ay2 += mh;  // backend clamps to image
    region.pts = {{{ax1, ay1}, {ax2, ay1}, {ax2, ay2}, {ax1, ay2}}};
  }
  return region;
}

} // namespace turbo_ocr::pipeline::detail
