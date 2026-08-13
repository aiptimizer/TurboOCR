#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ranges>
#include <utility>
#include <vector>

namespace turbo_ocr {

// Stack-allocated bounding box: 4 corners [tl, tr, br, bl], each [x, y].
// Replaces std::vector<std::vector<int>> -- zero heap allocations.
struct Box {
  std::array<std::array<int, 2>, 4> pts; // [tl, tr, br, bl]

  constexpr auto &operator[](std::size_t i) noexcept { return pts[i]; }
  constexpr const auto &operator[](std::size_t i) const noexcept { return pts[i]; }

  // C++20 three-way comparison -- enables ==, !=, <, >, <=, >= automatically
  constexpr auto operator<=>(const Box &) const noexcept = default;
};

/// Vertical text threshold: crop_h >= crop_w * kVerticalAspectRatio.
/// Used consistently by box detection, classification, and recognition.
inline constexpr float kVerticalAspectRatio = 1.5f;

// Axis-aligned bounding rect over all 4 corners of a Box, as
// [x0, y0, x1, y1] with x0 ≤ x1 and y0 ≤ y1. Correct for rotated /
// slanted quads where corner[0] and corner[2] are NOT a diagonal.
[[nodiscard]] inline std::array<int, 4> aabb(const Box &b) noexcept {
  int x0 = b[0][0], x1 = b[0][0];
  int y0 = b[0][1], y1 = b[0][1];
  for (int k = 1; k < 4; ++k) {
    x0 = std::min(x0, b[k][0]); x1 = std::max(x1, b[k][0]);
    y0 = std::min(y0, b[k][1]); y1 = std::max(y1, b[k][1]);
  }
  return {x0, y0, x1, y1};
}

// Clamp a Box's axis-aligned rect to the [0,cols)×[0,rows) page bounds and
// return it as [x0, y0, w, h] with w,h >= 1. Shared crop-rect computation for
// the formula/table VLM crop sites (page D2H → per-region crop) so the
// clamp-and-size logic lives in exactly one place.
[[nodiscard]] inline std::array<int, 4>
clamped_crop_rect(const Box &b, int cols, int rows) noexcept {
  const auto r = aabb(b);
  const int x0 = std::clamp(r[0], 0, std::max(0, cols - 1));
  const int y0 = std::clamp(r[1], 0, std::max(0, rows - 1));
  const int x1 = std::clamp(r[2], x0, cols);
  const int y1 = std::clamp(r[3], y0, rows);
  return {x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0)};
}

// Check if a box is vertically oriented (height >= width * 1.5).
// Uses integer arithmetic to avoid floating-point precision issues.
[[nodiscard]] inline bool is_vertical_box(const Box &b) noexcept {
  int w = std::max(std::abs(b[1][0] - b[0][0]), std::abs(b[2][0] - b[3][0]));
  int h = std::max(std::abs(b[3][1] - b[0][1]), std::abs(b[2][1] - b[1][1]));
  return static_cast<int64_t>(h) * h >= static_cast<int64_t>(w) * w * 225 / 100;
}

// Sort boxes top-to-bottom, left-to-right (in-place, deterministic).
// Boxes whose tops are within kSameLineThreshold px belong to one line and sort
// by X within it.
//
// WHY NOT `y / kSameLineThreshold` (what this used to do): that quantizes into
// FIXED bands, so the effective tolerance depends on absolute position rather
// than on the distance between the boxes. Two boxes 1px apart land on opposite
// sides of any multiple of 10 — tops of 29 and 34 were treated as DIFFERENT
// lines while 30 and 34 were the same one. Real symptom: two words typeset on
// one line came back in the wrong order on one backend and the right order on
// another, purely because their detected tops straddled a band edge.
//
// Grouping first, then sorting by (line, x), also keeps the comparator a valid
// strict weak ordering. Comparing with a tolerance directly does not — it is
// not transitive (a~b, b~c, but a<c), which is undefined behaviour in
// std::sort, so the obvious "fix" for the above is its own, worse, bug.
inline void sorted_boxes(std::vector<Box> &dt_boxes) {
  static constexpr int kSameLineThreshold = 10;
  if (dt_boxes.size() < 2) return;

  // Pass 1: order by top edge so line members are adjacent.
  std::ranges::stable_sort(dt_boxes, [](const Box &a, const Box &b) {
    if (a[0][1] != b[0][1]) return a[0][1] < b[0][1];
    return a[0][0] < b[0][0];
  });

  // Pass 2: walk in Y order, opening a new line when the gap from the CURRENT
  // LINE'S TOP exceeds the threshold. Measuring from the line's own top (not
  // from the previous box) stops a run of boxes each 9px below the last from
  // chaining into one arbitrarily tall "line".
  std::vector<int> line(dt_boxes.size(), 0);
  int line_top = dt_boxes[0][0][1];
  for (std::size_t i = 1; i < dt_boxes.size(); ++i) {
    if (dt_boxes[i][0][1] - line_top > kSameLineThreshold) {
      line[i] = line[i - 1] + 1;
      line_top = dt_boxes[i][0][1];
    } else {
      line[i] = line[i - 1];
    }
  }

  // Pass 3: (line, x). Indices are permuted alongside so the grouping computed
  // above survives the reorder.
  std::vector<std::size_t> idx(dt_boxes.size());
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::ranges::stable_sort(idx, [&](std::size_t a, std::size_t b) {
    if (line[a] != line[b]) return line[a] < line[b];
    return dt_boxes[a][0][0] < dt_boxes[b][0][0];
  });
  std::vector<Box> out;
  out.reserve(dt_boxes.size());
  for (std::size_t i : idx) out.push_back(dt_boxes[i]);
  dt_boxes.swap(out);
}


// Centroid of a (possibly rotated) quad box.
[[nodiscard]] inline std::pair<float, float> quad_centroid(const Box &b) noexcept {
  float cx = 0.0f, cy = 0.0f;
  for (int k = 0; k < 4; ++k) {
    cx += static_cast<float>(b[k][0]);
    cy += static_cast<float>(b[k][1]);
  }
  return {cx * 0.25f, cy * 0.25f};
}

// Centroid-in-AABB ownership test — the single rule for "which layout cell
// owns this quad"; serializer, router, and markdown must agree on it.
[[nodiscard]] inline bool centroid_in_aabb(const Box &b,
                                           const std::array<int, 4> &r) noexcept {
  const auto [cx, cy] = quad_centroid(b);
  return cx >= static_cast<float>(r[0]) && cx <= static_cast<float>(r[2]) &&
         cy >= static_cast<float>(r[1]) && cy <= static_cast<float>(r[3]);
}

} // namespace turbo_ocr
