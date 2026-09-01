#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/analysis/detection/det_postprocess.h"
#include "clipper/clipper.hpp"

#include "turbo_ocr/core/db_post_config.h" // detection::kMaxDbComponents

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <ranges>
#include <utility>
#include <vector>
#include <opencv2/imgproc.hpp>

using turbo_ocr::Box;

namespace turbo_ocr::detection {

float box_score_fast(const cv::Mat &pred_map,
                     const std::vector<cv::Point> &contour,
                     std::vector<cv::Point> &shifted_buf,
                     cv::Mat &mask_buf) {
  const int h = pred_map.rows;
  const int w = pred_map.cols;

  // Clip the contour's axis-aligned extent to the map. Contour points come
  // from findContours on the bitmap so they are already in [0,w-1]x[0,h-1];
  // the clamps only guard the GPU-CCL callers that shift ROI-local points.
  int xmin = w, xmax = 0, ymin = h, ymax = 0;
  for (const auto &pt : contour) {
    xmin = std::min(xmin, std::max(0, pt.x));
    xmax = std::max(xmax, std::min(w - 1, pt.x));
    ymin = std::min(ymin, std::max(0, pt.y));
    ymax = std::max(ymax, std::min(h - 1, pt.y));
  }
  if (xmax <= xmin || ymax <= ymin)
    return 0.0f;

  // Shift contour into local ROI coords using the caller-owned scratch buffer
  // (per-thread) so no per-call heap allocation occurs.
  shifted_buf.clear();
  shifted_buf.reserve(contour.size());
  for (const auto &pt : contour)
    shifted_buf.emplace_back(pt.x - xmin, pt.y - ymin);

  const int mask_w = xmax - xmin + 1;
  const int mask_h = ymax - ymin + 1;
  if (mask_buf.rows != mask_h || mask_buf.cols != mask_w ||
      mask_buf.type() != CV_8UC1)
    mask_buf.create(mask_h, mask_w, CV_8UC1);
  mask_buf.setTo(0);

  const cv::Point *pts_ptr = shifted_buf.data();
  const int npts = static_cast<int>(shifted_buf.size());
  cv::fillPoly(mask_buf, &pts_ptr, &npts, 1, cv::Scalar(1));

  const cv::Mat roi = pred_map(cv::Rect(xmin, ymin, mask_w, mask_h));
  return static_cast<float>(cv::mean(roi, mask_buf)[0]);
}

std::vector<cv::Point> unclip(const std::vector<cv::Point> &polygon,
                              float unclip_ratio) {
  const double perimeter = cv::arcLength(polygon, true);
  if (perimeter == 0.0)
    return polygon;

  // PaddleOCR DBPostProcess offset distance = area * ratio / perimeter,
  // expanded outward with a round join (pyclipper JT_ROUND / ET_CLOSEDPOLYGON).
  const double area = cv::contourArea(polygon);
  const double distance = area * unclip_ratio / perimeter;

  // Per-thread scratch: reusing the offsetter and its paths across the
  // thousands of per-candidate calls avoids repeated heap churn. Clear()
  // preserves the default MiterLimit(2.0)/ArcTolerance(0.25), so each run is
  // bit-identical to a freshly constructed ClipperOffset.
  thread_local ClipperLib::ClipperOffset co;
  thread_local ClipperLib::Path subj;
  thread_local ClipperLib::Paths solution;

  co.Clear();
  subj.clear();
  subj.reserve(polygon.size());
  for (const auto &pt : polygon)
    subj.emplace_back(pt.x, pt.y);

  co.AddPath(subj, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
  co.Execute(solution, distance);

  if (solution.empty())
    return polygon;

  // A concave/self-touching source can offset into several rings; keep the
  // largest by area (matches taking the single expanded ring in the common case).
  const ClipperLib::Path &best_path =
      (solution.size() == 1)
          ? solution.front()
          : *std::ranges::max_element(solution, {}, [](const ClipperLib::Path &p) {
              return ClipperLib::Area(p);
            });

  std::vector<cv::Point> result;
  result.reserve(best_path.size());
  for (const auto &p : best_path)
    result.emplace_back(static_cast<int>(p.X), static_cast<int>(p.Y));
  return result;
}

void order_quad_tl_tr_br_bl(float xs[4], float ys[4]) noexcept {
  // Stable insertion sort by x (strict '>' keeps ties in original order,
  // mirroring Python sorted()'s stability exactly).
  for (int i = 1; i < 4; ++i) {
    const float kx = xs[i], ky = ys[i];
    int j = i - 1;
    while (j >= 0 && xs[j] > kx) {
      xs[j + 1] = xs[j];
      ys[j + 1] = ys[j];
      --j;
    }
    xs[j + 1] = kx;
    ys[j + 1] = ky;
  }
  const int tl = (ys[0] < ys[1]) ? 0 : 1;
  const int bl = 1 - tl;
  const int tr = (ys[2] < ys[3]) ? 2 : 3;
  const int br = 5 - tr; // {2,3} sum to 5

  const float ox[4] = {xs[tl], xs[tr], xs[br], xs[bl]};
  const float oy[4] = {ys[tl], ys[tr], ys[br], ys[bl]};
  for (int k = 0; k < 4; ++k) { xs[k] = ox[k]; ys[k] = oy[k]; }
}

Box get_mini_boxes(const std::vector<cv::Point> &contour, float &min_side) {
  const cv::RotatedRect rect = cv::minAreaRect(contour);
  min_side = std::min(rect.size.width, rect.size.height);

  std::array<cv::Point2f, 4> pts;
  rect.points(pts.data());

  float xs[4], ys[4];
  for (int k = 0; k < 4; ++k) { xs[k] = pts[k].x; ys[k] = pts[k].y; }
  order_quad_tl_tr_br_bl(xs, ys);

  const auto ri = [](float v) { return static_cast<int>(std::round(v)); };
  Box box;
  for (int k = 0; k < 4; ++k) box[k] = {ri(xs[k]), ri(ys[k])};
  return box;
}

std::vector<Box> extract_boxes_from_bitmap(
    const cv::Mat &pred_map, cv::Mat &bitmap,
    int orig_h, int orig_w, int resize_h, int resize_w,
    float det_db_box_thresh, float det_db_unclip_ratio,
    float min_box_side, float min_unclipped_side,
    std::vector<cv::Point> &shifted_buf, cv::Mat &mask_buf,
    std::vector<std::vector<cv::Point>> &contours_buf,
    std::vector<cv::Vec4i> &hierarchy_buf) {

  const float ratio_h = static_cast<float>(resize_h) / orig_h;
  const float ratio_w = static_cast<float>(resize_w) / orig_w;

  std::vector<Box> boxes;
  boxes.reserve(256);

  contours_buf.clear();
  hierarchy_buf.clear();
  cv::findContours(bitmap, contours_buf, hierarchy_buf, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  // DB candidate budget — the SHARED detection::kMaxDbComponents, which the
  // vendor kernels' kMaxGpuComponents copies are static_assert-pinned to (see
  // cuda_kernels.cpp / hip_kernels.cpp). A local literal here used to be the
  // one un-pinned copy: raising the shared budget would have left this
  // reference path truncating at the old value with only a comment noticing.
  static constexpr int kMaxCandidates = detection::kMaxDbComponents;
  static_assert(kMaxCandidates == 3000,
                "shared DB budget changed — re-verify the GPU scratch sizing "
                "that the vendor static_asserts pin to it");
  const int total = static_cast<int>(contours_buf.size());

  // Spend the budget on MERIT, not scan order. PaddleOCR's DBPostProcess
  // slices the first max_candidates=1000 contours in findContours order and
  // silently drops the rest — on a dense page (a UI screenshot easily emits
  // 3000+ components) that discards whole spatial regions unexamined, since
  // scan order is position, not quality. Here the free rejects (degenerate
  // point counts, sub-3px bounding rects — the specks that dominate dense
  // pages) run over EVERY contour first, which on real pages usually brings
  // the survivors under budget with nothing dropped at all; only if genuine
  // candidates still exceed it are the LARGEST kept, so what goes is by
  // construction the least likely to be a text line. The bounding rect is
  // computed once per contour and reused as both the filter and the rank.
  std::vector<std::pair<long, int>> cand; // (bounding-rect area, contour idx)
  cand.reserve(static_cast<std::size_t>(total));
  for (int i = 0; i < total; ++i) {
    const auto &c = contours_buf[i];
    if (c.size() <= 2)
      continue;
    const cv::Rect br = cv::boundingRect(c);
    if (br.width < 3 || br.height < 3)
      continue;
    cand.emplace_back(static_cast<long>(br.width) * br.height, i);
  }
  if (static_cast<int>(cand.size()) > kMaxCandidates) {
    // Partial-select the largest; ties broken by index so the kept set is
    // deterministic. Restore index order afterwards — the filters below are
    // order-independent, but the returned boxes should not depend on the
    // selection algorithm's internal ordering.
    std::nth_element(cand.begin(), cand.begin() + kMaxCandidates, cand.end(),
                     [](const std::pair<long, int> &a,
                        const std::pair<long, int> &b) {
                       return a.first != b.first ? a.first > b.first
                                                 : a.second < b.second;
                     });
    // A note, not a warning: the dropped candidates are the smallest specks
    // on an unusually dense page, which the score/size filters below would
    // almost certainly have rejected anyway. Logged once per process.
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
      // Info: the code comment above already calls this "a note, not a
      // warning" — which is exactly why it must not go to stderr.
      TOCR_LOG_INFO("det: dense page, scoring the largest candidates only "
                    "(further dense pages not logged)",
                    "candidates", (long)cand.size(),
                    "scored", (long)kMaxCandidates);
    }
    cand.resize(static_cast<std::size_t>(kMaxCandidates));
    std::sort(cand.begin(), cand.end(),
              [](const std::pair<long, int> &a, const std::pair<long, int> &b) {
                return a.second < b.second;
              });
  }

  for (const auto &[area, idx] : cand) {
    (void)area;
    const auto &contour = contours_buf[idx];

    // Filters below are commutative (all reject via `continue`), so the kept
    // set is order-independent. The min-side test runs before the score to
    // match PaddleOCR's ordering and to skip the area-proportional
    // fillPoly+mean of box_score_fast on candidates too thin to survive.
    // (The free rejects — point count, sub-3px rect — already ran above,
    // before the candidate budget.)
    float ssid = 0.0f;
    (void)get_mini_boxes(contour, ssid);
    if (ssid < min_box_side)
      continue;

    const float score = box_score_fast(pred_map, contour, shifted_buf, mask_buf);
    if (score < det_db_box_thresh)
      continue;

    const auto unclipped = unclip(contour, det_db_unclip_ratio);
    if (unclipped.size() < 3)
      continue;

    float ssid2 = 0.0f;
    Box box = get_mini_boxes(unclipped, ssid2);
    if (ssid2 < min_unclipped_side)
      continue;

    for (int k = 0; k < 4; ++k) {
      box[k][0] = std::clamp(static_cast<int>(std::round(box[k][0] / ratio_w)), 0, orig_w - 1);
      box[k][1] = std::clamp(static_cast<int>(std::round(box[k][1] / ratio_h)), 0, orig_h - 1);
    }

    const int dx01 = box[0][0] - box[1][0], dy01 = box[0][1] - box[1][1];
    const int dx03 = box[0][0] - box[3][0], dy03 = box[0][1] - box[3][1];
    const int rw = static_cast<int>(std::sqrt(static_cast<double>(dx01 * dx01 + dy01 * dy01)));
    const int rh = static_cast<int>(std::sqrt(static_cast<double>(dx03 * dx03 + dy03 * dy03)));
    if (rw <= 3 || rh <= 3)
      continue;

    boxes.push_back(box);
  }

  return boxes;
}

} // namespace turbo_ocr::detection
