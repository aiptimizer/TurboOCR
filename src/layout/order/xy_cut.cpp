#include "turbo_ocr/layout/order/reading_order.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory_resource>
#include <numeric>

namespace turbo_ocr::layout {

namespace {

// Pull the start/end coordinates for `axis` from a rect [x0, y0, x1, y1].
// axis 0 → (x0, x1); axis 1 → (y0, y1).
inline std::pair<int, int> rect_extent(const std::array<int, 4> &r, int axis) {
  if (axis == 0) return {r[0], r[2]};
  return {r[1], r[3]};
}

// Maximum number of bins kept in a 1D projection histogram. When the page
// extent along the projected axis exceeds this, the histogram is built at
// reduced resolution (bin size = ceil(max_end / kProjectionMaxBins)). For
// adversarial detection output (many boxes spread across a 10kx10k page)
// this caps the per-call allocation at ~16 KB instead of growing with the
// page size. XY-cut splits on whitespace gaps measured in bins, so a
// pixel-accurate histogram is unnecessary — 4096 bins still resolves the
// gutter between two columns of even the densest layout.
constexpr int kProjectionMaxBins = 4096;

// Build a downsampling factor (bin width in pixels) for a histogram of
// extent `max_end`. Always >= 1; equals 1 when no downsampling is needed.
inline int projection_scale_for(int max_end) {
  if (max_end <= kProjectionMaxBins) return 1;
  return (max_end + kProjectionMaxBins - 1) / kProjectionMaxBins;
}

} // namespace

std::vector<int>
projection_by_bboxes(const std::vector<std::array<int, 4>> &rects, int axis) {
  if (rects.empty() || (axis != 0 && axis != 1)) return {};

  // OCR detection emits coordinates in image space, which is always
  // non-negative. Clamp starts at 0 and size the projection by max_end.
  // (PaddleX's reference adds a mirror branch for negative starts, but
  // the math there truncates mixed-sign ranges and inverts purely
  // negative rects — kept out here because no caller in this codebase
  // produces negatives.)
  int max_end = 0;
  for (const auto &r : rects) {
    max_end = std::max(max_end, rect_extent(r, axis).second);
  }

  if (max_end <= 0) return {};

  // Cap histogram size: at most kProjectionMaxBins bins. Each pixel
  // [a, b) contributes to bins [a/scale, ceil(b/scale)).
  const int scale = projection_scale_for(max_end);
  const int n_bins =
      (scale == 1) ? max_end : (max_end + scale - 1) / scale;

  std::vector<int> projection(static_cast<size_t>(n_bins), 0);
  for (const auto &r : rects) {
    auto [s, e] = rect_extent(r, axis);
    const int a = std::max(s, 0);
    const int b = std::min(e, max_end);
    if (a >= b) continue;
    const int a_bin = a / scale;
    const int b_bin = (b + scale - 1) / scale;
    for (int i = a_bin; i < b_bin; ++i) projection[static_cast<size_t>(i)] += 1;
  }
  return projection;
}

std::vector<ProjectionSegment>
split_projection_profile(const std::vector<int> &projection, int min_value,
                         int min_gap) {
  // Match the reference exactly: collect all indices where the profile
  // strictly exceeds min_value, then split where the index gap exceeds
  // min_gap. Segment ends are exclusive (last_significant + 1).
  std::vector<int> sig;
  sig.reserve(projection.size());
  for (size_t i = 0; i < projection.size(); ++i) {
    if (projection[i] > min_value) sig.push_back(static_cast<int>(i));
  }
  if (sig.empty()) return {};

  std::vector<ProjectionSegment> segments;
  int seg_start = sig.front();
  for (size_t i = 1; i < sig.size(); ++i) {
    if (sig[i] - sig[i - 1] > min_gap) {
      segments.push_back({seg_start, sig[i - 1] + 1});
      seg_start = sig[i];
    }
  }
  segments.push_back({seg_start, sig.back() + 1});
  return segments;
}

namespace {

// Compute the per-axis page extent over a subset of rects. Used to derive
// the projection downsampling factor *before* building the histogram, so
// segments produced by split_projection_profile can be scaled back to
// pixel coordinates for the index-selection step in recursive_xy_cut.
inline int max_end_for(const std::pmr::vector<std::array<int, 4>> &rects,
                       int axis) {
  int max_end = 0;
  for (const auto &r : rects) {
    max_end = std::max(max_end, (axis == 0) ? r[2] : r[3]);
  }
  return max_end;
}

// Build a 1D projection histogram over a pmr::vector of rects, capped at
// kProjectionMaxBins bins. Returned histogram and out-param `scale` are
// related by `bin_index * scale ≈ pixel_coord`. Pure pmr variant — avoids
// the heap allocation that the public projection_by_bboxes would incur on
// every recursion frame.
std::pmr::vector<int>
projection_by_bboxes_pmr(const std::pmr::vector<std::array<int, 4>> &rects,
                         int axis, int &scale, std::pmr::memory_resource *mr) {
  scale = 1;
  std::pmr::vector<int> projection(mr);
  if (rects.empty() || (axis != 0 && axis != 1)) return projection;

  const int max_end = max_end_for(rects, axis);
  if (max_end <= 0) return projection;

  scale = projection_scale_for(max_end);
  const int n_bins =
      (scale == 1) ? max_end : (max_end + scale - 1) / scale;
  projection.assign(static_cast<size_t>(n_bins), 0);
  for (const auto &r : rects) {
    const int s = (axis == 0) ? r[0] : r[1];
    const int e = (axis == 0) ? r[2] : r[3];
    const int a = std::max(s, 0);
    const int b = std::min(e, max_end);
    if (a >= b) continue;
    const int a_bin = a / scale;
    const int b_bin = (b + scale - 1) / scale;
    for (int i = a_bin; i < b_bin; ++i) projection[static_cast<size_t>(i)] += 1;
  }
  return projection;
}

// pmr variant of split_projection_profile. Same semantics as the public
// function but allocates from `mr`.
std::pmr::vector<ProjectionSegment>
split_projection_profile_pmr(const std::pmr::vector<int> &projection,
                             int min_value, int min_gap,
                             std::pmr::memory_resource *mr) {
  std::pmr::vector<int> sig(mr);
  sig.reserve(projection.size());
  for (size_t i = 0; i < projection.size(); ++i) {
    if (projection[i] > min_value) sig.push_back(static_cast<int>(i));
  }
  std::pmr::vector<ProjectionSegment> segments(mr);
  if (sig.empty()) return segments;
  int seg_start = sig.front();
  for (size_t i = 1; i < sig.size(); ++i) {
    if (sig[i] - sig[i - 1] > min_gap) {
      segments.push_back({seg_start, sig[i - 1] + 1});
      seg_start = sig[i];
    }
  }
  segments.push_back({seg_start, sig.back() + 1});
  return segments;
}

// Internal recursion using pmr-backed scratch vectors. All transient
// vectors (subset, sort orders, projections, segment lists) allocate from
// the shared `mr` (a monotonic_buffer_resource owned by the public
// recursive_xy_cut entry point). This collapses ~5 heap allocations per
// frame down to amortised pool growth.
void recursive_xy_cut_impl(const std::vector<std::array<int, 4>> &rects,
                           const std::pmr::vector<int> &indices,
                           std::vector<int> &res, int min_gap,
                           std::pmr::memory_resource *mr, int depth = 0) {
  if (indices.empty()) return;
  if (indices.size() == 1) { res.push_back(indices.front()); return; }

  // Deep alternating H/V nesting would grow the monotonic pool super-linearly
  // (each frame keeps its scratch until the top-level call returns) and
  // recurse unbounded. Real layouts nest a handful of levels; past the cap,
  // emit the remaining boxes in their current (x-then-y sorted) order — a
  // stable fallback, same spirit as the "no progress" bail below.
  constexpr int kMaxXyCutDepth = 64;
  if (depth >= kMaxXyCutDepth) {
    for (int idx : indices) res.push_back(idx);
    return;
  }

  // Build the local view (rects subset for this recursion frame).
  std::pmr::vector<std::array<int, 4>> subset(mr);
  subset.reserve(indices.size());
  for (int idx : indices) subset.push_back(rects[static_cast<size_t>(idx)]);

  // 1. Sort by x_min (ascending) for X-axis projection.
  std::pmr::vector<int> order(indices.size(), mr);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return subset[a][0] < subset[b][0]; });

  std::pmr::vector<std::array<int, 4>> x_sorted_rects(mr);
  std::pmr::vector<int> x_sorted_indices(mr);
  x_sorted_rects.reserve(order.size());
  x_sorted_indices.reserve(order.size());
  for (int p : order) {
    x_sorted_rects.push_back(subset[p]);
    x_sorted_indices.push_back(indices[static_cast<size_t>(p)]);
  }

  int x_scale = 1;
  auto x_proj = projection_by_bboxes_pmr(x_sorted_rects, 0, x_scale, mr);
  // X-axis splits use pixel min_gap of 1; convert to bin space (>=1).
  const int x_min_gap_bins = (x_scale > 1) ? std::max(1, 1 / x_scale) : 1;
  auto x_intervals =
      split_projection_profile_pmr(x_proj, 0, x_min_gap_bins, mr);
  if (x_intervals.empty()) {
    // Degenerate case (e.g. all zero-area boxes): emit in current order
    // so callers still see every box.
    for (int idx : x_sorted_indices) res.push_back(idx);
    return;
  }

  // Lift segments back to pixel space for the index-selection compares.
  if (x_scale > 1) {
    for (auto &xi : x_intervals) {
      xi.start *= x_scale;
      xi.end *= x_scale;
    }
  }

  // PaddleX flips the X intervals when any x_min is negative (RTL pages).
  if (!x_sorted_rects.empty() && x_sorted_rects.front()[0] < 0) {
    std::reverse(x_intervals.begin(), x_intervals.end());
  }

  for (const auto &xi : x_intervals) {
    // 2. Select rects whose |x_min| falls into the current X interval.
    std::pmr::vector<std::array<int, 4>> col_rects(mr);
    std::pmr::vector<int> col_indices(mr);
    for (size_t k = 0; k < x_sorted_rects.size(); ++k) {
      int x_min_abs = std::abs(x_sorted_rects[k][0]);
      if (xi.start <= x_min_abs && x_min_abs < xi.end) {
        col_rects.push_back(x_sorted_rects[k]);
        col_indices.push_back(x_sorted_indices[k]);
      }
    }
    if (col_rects.empty()) continue;

    // 3. Sort the column by y_min, project onto Y-axis.
    std::pmr::vector<int> y_order(col_rects.size(), mr);
    std::iota(y_order.begin(), y_order.end(), 0);
    std::stable_sort(
        y_order.begin(), y_order.end(),
        [&](int a, int b) { return col_rects[a][1] < col_rects[b][1]; });

    std::pmr::vector<std::array<int, 4>> y_sorted_rects(mr);
    std::pmr::vector<int> y_sorted_indices(mr);
    y_sorted_rects.reserve(y_order.size());
    y_sorted_indices.reserve(y_order.size());
    for (int p : y_order) {
      y_sorted_rects.push_back(col_rects[static_cast<size_t>(p)]);
      y_sorted_indices.push_back(col_indices[static_cast<size_t>(p)]);
    }

    int y_scale = 1;
    auto y_proj = projection_by_bboxes_pmr(y_sorted_rects, 1, y_scale, mr);
    const int y_min_gap_bins =
        (y_scale > 1) ? std::max(1, min_gap / y_scale) : min_gap;
    auto y_intervals =
        split_projection_profile_pmr(y_proj, 0, y_min_gap_bins, mr);
    if (y_intervals.empty()) {
      for (int idx : y_sorted_indices) res.push_back(idx);
      continue;
    }

    // 4. If the Y projection is a single segment, no further splitting:
    // emit current sequence as-is.
    if (y_intervals.size() == 1) {
      for (int idx : y_sorted_indices) res.push_back(idx);
      continue;
    }

    // Lift Y segments back to pixel space for compares below.
    if (y_scale > 1) {
      for (auto &yi : y_intervals) {
        yi.start *= y_scale;
        yi.end *= y_scale;
      }
    }

    // 5. Otherwise recurse on each Y segment.
    for (const auto &yi : y_intervals) {
      std::pmr::vector<int> row_indices(mr);
      for (size_t k = 0; k < y_sorted_rects.size(); ++k) {
        int y_min = y_sorted_rects[k][1];
        if (yi.start <= y_min && y_min < yi.end) {
          row_indices.push_back(y_sorted_indices[k]);
        }
      }
      if (row_indices.empty()) continue;
      if (row_indices.size() == y_sorted_indices.size()) {
        // No progress (this Y segment contains every box) — bail to
        // avoid infinite recursion on degenerate inputs.
        for (int idx : row_indices) res.push_back(idx);
        continue;
      }
      recursive_xy_cut_impl(rects, row_indices, res, min_gap, mr, depth + 1);
    }
  }
}

} // namespace

void recursive_xy_cut(const std::vector<std::array<int, 4>> &rects,
                      const std::vector<int> &indices,
                      std::vector<int> &res, int min_gap) {
  if (indices.empty()) return;

  // Stack-resident initial buffer for the monotonic pool: large enough to
  // service the typical body of a page (~1600 frames * a few KB scratch)
  // before the pool falls back to heap-backed chunk growth. Chunks are
  // freed only when the resource is destroyed, so the pool's lifetime is
  // bounded to this single top-level call.
  alignas(std::max_align_t) std::byte stack_buf[64 * 1024];
  std::pmr::monotonic_buffer_resource pool(stack_buf, sizeof(stack_buf));

  std::pmr::vector<int> pmr_indices(&pool);
  pmr_indices.reserve(indices.size());
  for (int i : indices) pmr_indices.push_back(i);
  recursive_xy_cut_impl(rects, pmr_indices, res, min_gap, &pool);
}

} // namespace turbo_ocr::layout
