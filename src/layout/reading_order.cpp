#include "turbo_ocr/layout/reading_order.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdlib>
#include <memory_resource>
#include <numeric>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/layout/child_blocks.h"
#include "turbo_ocr/layout/match_unsorted.h"
#include "turbo_ocr/layout/text_line_cluster.h"

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

// XY-cut over a subset of layout indices. Helper extracted so callers can
// reuse it on each priority bucket without duplicating the AABB build.
static void
xy_cut_subset(const std::vector<LayoutBox> &layout,
              const std::vector<int> &subset,
              std::vector<int> &out, int min_gap) {
  if (subset.empty()) return;
  std::vector<std::array<int, 4>> rects;
  rects.reserve(subset.size());
  for (int idx : subset) {
    auto [x0, y0, x1, y1] = aabb(layout[static_cast<size_t>(idx)].box);
    rects.push_back({x0, y0, x1, y1});
  }
  std::vector<int> local(subset.size());
  std::iota(local.begin(), local.end(), 0);
  std::vector<int> local_order;
  local_order.reserve(subset.size());
  recursive_xy_cut(rects, local, local_order, min_gap);

  // Defense in depth: degenerate inputs (overlapping AABBs, zero areas)
  // can drop indices from the recursion. Append any missed in input
  // order so callers always see a complete permutation of the subset.
  std::vector<char> seen(subset.size(), 0);
  for (int li : local_order) {
    if (li >= 0 && static_cast<size_t>(li) < seen.size()) seen[li] = 1;
  }
  for (size_t k = 0; k < local.size(); ++k) {
    if (!seen[k]) local_order.push_back(static_cast<int>(k));
  }

  for (int li : local_order) out.push_back(subset[static_cast<size_t>(li)]);
}

std::vector<int>
assign_reading_order(const std::vector<LayoutBox> &layout, int min_gap) {
  std::vector<int> result;
  if (layout.empty()) return result;

  // Class-aware bucketing: header → body → footer/reference. Each bucket
  // gets its own XY-cut so multi-line headers and reference lists keep
  // their internal order. PaddleX's xycut_enhanced does the same — page
  // furniture should not interleave with the body.
  std::array<std::vector<int>, 3> buckets;
  for (size_t i = 0; i < layout.size(); ++i) {
    int b = reading_priority_bucket(layout[i].class_id);
    buckets[static_cast<size_t>(b)].push_back(static_cast<int>(i));
  }

  result.reserve(layout.size());
  for (auto &subset : buckets) xy_cut_subset(layout, subset, result, min_gap);
  return result;
}

std::vector<int>
assign_reading_order_for_results(const std::vector<OCRResultItem> &results,
                                 std::vector<LayoutBox> &layout,
                                 int min_gap) {
  std::vector<int> out;
  out.reserve(results.size());
  if (results.empty()) return out;

  // Shared "no usable layout signal" exit: sort every result by
  // (y_center, x_center) and emit. Decorate-sort-undecorate keeps the key
  // a pure per-result computation instead of recomputing it inside the
  // comparator. Used by both fall-back branches below.
  const auto emit_yx_fallback = [&results, &out]() {
    struct K { int y4, x4, idx; };
    std::vector<K> keys;
    keys.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
      int sx = 0, sy = 0;
      for (int k = 0; k < 4; ++k) {
        sx += results[i].box[k][0];
        sy += results[i].box[k][1];
      }
      keys.push_back({sy, sx, static_cast<int>(i)});
    }
    std::stable_sort(keys.begin(), keys.end(), [](const K &a, const K &b) {
      if (a.y4 != b.y4) return a.y4 < b.y4;
      return a.x4 < b.x4;
    });
    for (const auto &k : keys) out.push_back(k.idx);
  };

  // Layout-empty fast path: text-line detection boxes are line-level,
  // not paragraph-level. Running XY-cut on raw detection boxes can
  // over-split a one-column document into spurious "columns" the moment
  // there's a horizontal gap between two short lines. Without any layout
  // signal there's nothing better than y-then-x.
  if (layout.empty()) {
    emit_yx_fallback();
    return out;
  }

  // ----- Class-aware bucketing + augmented XY-cut over (layout ∪ orphans) ----
  //
  // The PP-DocLayoutV3 layout model emits 25 classes. PaddleX's
  // xycut_enhanced pipeline doesn't run all of them through one flat
  // XY-cut: page furniture (header / footer / footnote / reference /
  // vision_footnote) is hoisted out into top/bottom strata so it doesn't
  // interleave with the body, and the body proper runs through XY-cut.
  // A 'reference' block geometrically placed mid-page in a malformed
  // document still belongs at the end of the reading order.
  //
  // Inside each bucket we still need the orphan handling: results whose
  // centroid falls outside every layout region (page numbers the layout
  // model missed, OCR detections in the gutter, etc.) get a synthetic
  // XY-cut entry from their detection AABB so they land in their natural
  // geometric position rather than trailing the bucket.
  //
  // Orphans always go into the BODY bucket. They could in theory fall
  // inside the header/footer band of the page, but with no class signal
  // we bias toward the safer placement (let XY-cut decide their y/x slot
  // within the body). Headers and footers themselves are explicit layout
  // regions, not orphans.
  //
  // Tagged-rect kinds:
  //   0 = real layout region; payload = layout index
  //   1 = orphan result;       payload = result index
  struct AugRect {
    std::array<int, 4> aabb;
    int kind;
    int payload;
  };

  // Pre-compute layout AABBs (used by both the bucket sort and the XY-cut).
  std::vector<std::array<int, 4>> layout_aabb(layout.size());
  for (size_t li = 0; li < layout.size(); ++li) {
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[li].box);
    layout_aabb[li] = {x0, y0, x1, y1};
  }

  // Group results by their layout_id and pre-sort each group into row
  // order. The naive (y_center, x_center) sort fails on tables: cells in
  // the same row routinely have 1-3 pixels of y-jitter from text-line
  // detection, so a strict y-tiebreak interleaves columns. We bucket
  // y-centroids by a row tolerance derived from the median text-line
  // height of the group: tol = max(4, median_height/3). This is scale
  // invariant — works at 100 dpi and at 600 dpi alike. `floor(cy / tol)`
  // is a function of one input, giving the strict weak ordering
  // std::stable_sort requires.
  std::vector<std::vector<int>> by_layout(layout.size());
  for (size_t ri = 0; ri < results.size(); ++ri) {
    int lid = results[ri].layout_id;
    if (lid >= 0 && static_cast<size_t>(lid) < by_layout.size())
      by_layout[lid].push_back(static_cast<int>(ri));
  }
  // Per-group row tolerance from median bbox height.
  std::vector<int> group_row_tol(layout.size(), 8);
  for (size_t li = 0; li < by_layout.size(); ++li) {
    const auto &v = by_layout[li];
    if (v.size() < 2) continue;
    std::vector<int> heights;
    heights.reserve(v.size());
    for (int ri : v) {
      int y_min = INT_MAX, y_max = INT_MIN;
      for (int k = 0; k < 4; ++k) {
        y_min = std::min(y_min, results[ri].box[k][1]);
        y_max = std::max(y_max, results[ri].box[k][1]);
      }
      heights.push_back(std::max(1, y_max - y_min));
    }
    auto mid = heights.begin() + heights.size() / 2;
    std::nth_element(heights.begin(), mid, heights.end());
    int median_h = *mid;
    group_row_tol[li] = std::max(4, median_h / 3);
  }
  // Decorate-sort-undecorate: the (row, x) key is a pure function of each
  // result, so materialise it once per element rather than recomputing 8
  // corner sums per comparison. Keys are built in ascending result-index
  // order and stable_sort preserves that on ties, so the emitted
  // permutation is byte-identical to the in-comparator version. Groups of
  // size < 2 need no sort. The key buffer is reused across groups.
  struct RowKey { int row, xsum, ri; };
  std::vector<RowKey> row_keys;
  for (size_t li = 0; li < by_layout.size(); ++li) {
    auto &v = by_layout[li];
    if (v.size() < 2) continue;
    const int tol = group_row_tol[li];
    row_keys.clear();
    row_keys.reserve(v.size());
    for (int ri : v) {
      int sy = 0, sx = 0;
      for (int k = 0; k < 4; ++k) {
        sy += results[ri].box[k][1];
        sx += results[ri].box[k][0];
      }
      row_keys.push_back({(sy / 4) / tol, sx, ri});
    }
    std::stable_sort(row_keys.begin(), row_keys.end(),
                     [](const RowKey &a, const RowKey &b) {
                       if (a.row != b.row) return a.row < b.row;
                       return a.xsum < b.xsum;
                     });
    for (size_t i = 0; i < v.size(); ++i) v[i] = row_keys[i].ri;
  }

  // Mostly-orphans fast path: if very few results matched a layout box
  // (e.g. layout model nearly missed the page) the body bucket would
  // run XY-cut over LINE-level detection boxes, which can spuriously
  // split a single column into "columns" the moment two short lines
  // have a horizontal gap. Layout AABBs typically dwarf line AABBs,
  // so a single matched layout containing 99 orphan rects also hits
  // the recursion's "no progress" bail-out at recursive_xy_cut(). Fall
  // back to the y-then-x sort whenever the matched fraction is below
  // 5% — that's well into "layout missed it" territory.
  size_t matched_count = 0;
  for (size_t ri = 0; ri < results.size(); ++ri) {
    int lid = results[ri].layout_id;
    if (lid < 0 || static_cast<size_t>(lid) >= layout.size()) continue;
    if (layout[static_cast<size_t>(lid)].class_id ==
        kSupplementaryRegionClassId) continue;
    ++matched_count;
  }
  // 5% threshold: fewer than ceil(0.05 * N) matches means "almost no
  // signal from layout" — better to fall back than risk the regression.
  size_t min_matches = std::max<size_t>(1, (results.size() * 5 + 99) / 100);
  if (matched_count < min_matches) {
    emit_yx_fallback();
    return out;
  }

  // Cluster the OCR detection boxes into per-cell TextLines. This
  // populates each LayoutBox with direction, num_of_lines,
  // text_line_height, text_line_width, and seg_*_coordinate — which
  // feed the label-aware insertion (weighted_distance_insert
  // disperse term + get_seg_flag look-ahead) and the child-block
  // detection (real proximity threshold per block instead of the
  // height-over-text_line_height approximation).
  cluster_text_lines(results, layout);

  // Page-level direction (majority vote across text-class cells).
  // Drives axis selection in weighted_distance_insert and the
  // bucket-level sort key.
  const Direction page_direction = infer_page_direction(layout);

  // Page-level text_line_width / text_line_height as means across
  // text-class cells — the disperse-term scale for doc_title in
  // weighted_distance_insert, and the cross-cell proximity threshold
  // for child-block detection. Falls back to 0 when no text cell got
  // any clustered lines.
  int text_line_width = 0;
  int text_line_height = 0;
  {
    long long sum_w = 0, sum_h = 0;
    int n = 0;
    for (const auto &lb : layout) {
      if (lb.class_id != 22 /*text*/) continue;
      if (lb.text_line_height <= 0) continue;
      sum_w += lb.text_line_width;
      sum_h += lb.text_line_height;
      ++n;
    }
    if (n > 0) {
      text_line_width = static_cast<int>(sum_w / n);
      text_line_height = static_cast<int>(sum_h / n);
    }
  }

  // Detect parent → children relationships once for the page; the
  // sidecar links survive across buckets so vision_footnote can stay
  // glued to its (body-bucket) parent vision block even if the bucket
  // sweep would otherwise emit them in different strata.
  const auto child_links = detect_child_blocks(layout, text_line_height);

  // Set of layout indices that ARE children of some parent. These
  // never participate in bucket collection or XY-cut directly — they
  // emit under their parent's slot via the emit loop's child splice.
  std::vector<char> is_child(layout.size(), 0);
  for (const auto &cl : child_links) {
    for (int ci : cl.child_indices) {
      if (ci >= 0 && static_cast<size_t>(ci) < is_child.size())
        is_child[static_cast<size_t>(ci)] = 1;
    }
  }

  // Process each bucket in TOP→BODY→BOTTOM order. Each bucket splits
  // layout boxes into:
  //   - regulars (kBody) → run through XY-cut
  //   - unsorted (titles, captions, vision, cross-refs, unordered) →
  //     inserted via match_unsorted_blocks AFTER the XY-cut so each
  //     uses the strategy keyed by its order label (weighted distance
  //     for titles/vision, manhattan for unordered, reference for
  //     cross-refs).
  // Orphan results (synthetic SupplementaryRegion members) are inlined
  // into the body bucket as XY-cut entries so they keep their
  // geometric placement.
  std::vector<char> emitted(results.size(), 0);
  auto run_bucket = [&](int bucket) {
    std::vector<AugRect> aug;
    std::vector<UnsortedBlock> unsorted;
    aug.reserve(layout.size());
    for (size_t li = 0; li < layout.size(); ++li) {
      if (layout[li].class_id == kSupplementaryRegionClassId) continue;
      if (is_child[li]) continue;  // emitted under its parent
      if (reading_priority_bucket(layout[li].class_id) != bucket) continue;
      const OrderLabel ol = order_label_for(layout[li].class_id);
      if (ol == OrderLabel::kBody) {
        aug.push_back({layout_aabb[li], 0, static_cast<int>(li)});
      } else {
        unsorted.push_back({static_cast<int>(li), layout_aabb[li], ol,
                            layout[li].class_id});
      }
    }
    if (bucket == 1) {
      for (size_t ri = 0; ri < results.size(); ++ri) {
        int lid = results[ri].layout_id;
        const bool is_orphan =
            (lid < 0) ||
            (static_cast<size_t>(lid) >= layout.size()) ||
            (layout[static_cast<size_t>(lid)].class_id ==
             kSupplementaryRegionClassId);
        if (is_orphan) {
          auto [x0, y0, x1, y1] = turbo_ocr::aabb(results[ri].box);
          aug.push_back({{x0, y0, x1, y1}, 1, static_cast<int>(ri)});
        }
      }
    }
    if (aug.empty() && unsorted.empty()) return;

    // 1. XY-cut over the regulars. For vertical-direction pages we
    //    mirror x coordinates around max_x BEFORE the cut so the
    //    algorithm's left-to-right behaviour produces a right-to-left
    //    column order (CJK tategaki). Coordinates are restored only
    //    in the final layout-idx mapping so callers see the original
    //    bbox.
    std::vector<std::array<int, 4>> rects;
    rects.reserve(aug.size());
    int mirror_x = 0;
    if (page_direction == Direction::kVertical) {
      for (const auto &a : aug) {
        mirror_x = std::max(mirror_x, a.aabb[2]);
      }
    }
    for (const auto &a : aug) {
      if (page_direction == Direction::kVertical) {
        // Reflect: new_x0 = mirror_x - old_x1, new_x1 = mirror_x - old_x0.
        // Keeps width identical; flips order so the rightmost column
        // becomes the leftmost in the cut input.
        rects.push_back({mirror_x - a.aabb[2], a.aabb[1],
                         mirror_x - a.aabb[0], a.aabb[3]});
      } else {
        rects.push_back(a.aabb);
      }
    }
    std::vector<int> aug_indices(aug.size());
    std::iota(aug_indices.begin(), aug_indices.end(), 0);
    std::vector<int> aug_order;
    aug_order.reserve(aug.size());
    recursive_xy_cut(rects, aug_indices, aug_order, min_gap);
    std::vector<char> seen(aug.size(), 0);
    for (int ai : aug_order) {
      if (ai >= 0 && static_cast<size_t>(ai) < seen.size()) seen[ai] = 1;
    }
    for (size_t k = 0; k < aug.size(); ++k) {
      if (!seen[k]) aug_order.push_back(static_cast<int>(k));
    }

    // 2. Convert XY-cut output to a list of (UnsortedBlock + emit
    //    payload) so match_unsorted_blocks can mutate ordering. Layout
    //    AABB carries the original aug payload (kind, idx) via the
    //    `class_id` field's sign trick: positive class_id = real
    //    layout idx into outer layout vector; -1 sentinel is reserved
    //    for orphan rects (kind == 1) which we encode via order_label.
    std::vector<UnsortedBlock> sorted_blocks;
    sorted_blocks.reserve(aug.size());
    for (int ai : aug_order) {
      const auto &a = aug[static_cast<size_t>(ai)];
      if (a.kind == 0) {
        sorted_blocks.push_back({a.payload, a.aabb, OrderLabel::kBody,
                                  layout[static_cast<size_t>(a.payload)].class_id});
      } else {
        // Orphan rect: payload is a result index. Encode via negative
        // layout_idx so the emitter can recover it (-2 - ri).
        sorted_blocks.push_back({-2 - a.payload, a.aabb,
                                  OrderLabel::kBody, -1});
      }
    }

    // 3. Insert label-aware unsorted blocks.
    if (!unsorted.empty()) {
      match_unsorted_blocks(sorted_blocks, unsorted, text_line_width,
                              page_direction, layout);
    }

    // 4. Emit results. For each sorted layout entry, also splice in
    //    the layout's descendants (children, grandchildren, …) in
    //    top-down geometric order — mirrors PaddleX's
    //    insert_child_blocks. Descendants come from
    //    flatten_descendants which handles arbitrary tree depth and
    //    is cycle-safe.
    auto emit_layout = [&](int layout_idx) {
      for (int ri : by_layout[static_cast<size_t>(layout_idx)]) {
        if (!emitted[static_cast<size_t>(ri)]) {
          out.push_back(ri);
          emitted[static_cast<size_t>(ri)] = 1;
        }
      }
    };
    for (const auto &sb : sorted_blocks) {
      if (sb.layout_idx >= 0) {
        emit_layout(sb.layout_idx);
        const auto descendants =
            flatten_descendants(sb.layout_idx, child_links, layout);
        for (int ci : descendants) emit_layout(ci);
      } else {
        const int ri = -2 - sb.layout_idx;
        if (ri >= 0 && static_cast<size_t>(ri) < results.size() &&
            !emitted[static_cast<size_t>(ri)]) {
          out.push_back(ri);
          emitted[static_cast<size_t>(ri)] = 1;
        }
      }
    }
  };
  run_bucket(0);  // headers
  run_bucket(1);  // body (with orphans)
  run_bucket(2);  // footers / footnotes / references

  // Final defense in depth: a result whose layout_id pointed at a region
  // that somehow yielded no XY-cut entry across all three buckets.
  for (size_t ri = 0; ri < results.size(); ++ri) {
    if (!emitted[ri]) out.push_back(static_cast<int>(ri));
  }
  return out;
}

} // namespace turbo_ocr::layout
