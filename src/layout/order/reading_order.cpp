#include "turbo_ocr/layout/order/reading_order.h"

#include <algorithm>
#include <array>
#include <climits>
#include <numeric>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/layout/blocks/child_blocks.h"
#include "turbo_ocr/layout/blocks/match_unsorted.h"
#include "turbo_ocr/layout/blocks/text_line_cluster.h"

namespace turbo_ocr::layout {

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
