#pragma once

#include <algorithm>
#include <cstddef>   // std::size_t
#include <string>
#include <tuple>
#include <vector>

#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/core/layout_types.h"

namespace turbo_ocr::layout {

// How nested layout boxes are reconciled. The model intentionally emits some
// classes (formulas, tables, titles, footnotes) nested inside a larger region;
// the mode decides which copy survives.
//   kKeepAll   ("all")   = keep every box, outer and nested. Default: nothing
//                          the model emitted is dropped.
//   kKeepOuter ("outer") = keep the outer/larger region, drop boxes nested in
//                          it. On forms (every field is a box inside an outer
//                          frame) this collapses the page to a few containers.
//   kKeepInner ("inner") = keep the innermost boxes, drop the pure containers.
enum class MergeMode { kKeepAll, kKeepOuter, kKeepInner };

// Canonical strings are "all"/"outer"/"inner". The old "union"/"large"/"small"
// names are still accepted as deprecated aliases so existing configs keep
// working.
inline MergeMode layout_merge_mode() {
  static const MergeMode mode = [] {
    const std::string s = env::env_or("LAYOUT_MERGE_MODE", "all");
    if (s == "outer" || s == "large") return MergeMode::kKeepOuter;
    if (s == "inner" || s == "small") return MergeMode::kKeepInner;
    return MergeMode::kKeepAll; // "all" / "union" / unset
  }();
  return mode;
}

// display_formula / inline_formula are kept even when nested in a text or table
// region, so standalone math is never swallowed by its surrounding block.
constexpr bool is_formula_class(int cls) {
  return cls == 5 /*display_formula*/ || cls == 15 /*inline_formula*/;
}

// Pin the magic class indices used here so a future relabel of kLayoutLabels
// can't silently break the formula guard. (kImageClassId, the nestable-child
// set, and the rest of the pins live in layout_types.h.)
static_assert(kLayoutLabels[5] == "display_formula" &&
                  kLayoutLabels[15] == "inline_formula",
              "layout class indices drifted — update layout_postfilter.h");

// Axis-aligned corner tuple (x0,y0,x1,y1) of a layout box. PP-DocLayoutV3
// emits axis-aligned regions, so corner 0 is top-left and corner 2 is
// bottom-right.
inline std::tuple<int, int, int, int>
layout_box_coords(const LayoutBox &lb) noexcept {
  return {lb.box[0][0], lb.box[0][1], lb.box[2][0], lb.box[2][1]};
}

// Exact pixel area. Kept in int64 (not the float the `inside` ratio uses) so
// the parent-ranking order below is exact for large regions — a float product
// loses integer precision past 2^24, which on a 300-dpi page is a plausible
// region area.
inline long long layout_box_area(const LayoutBox &lb) noexcept {
  auto [x0, y0, x1, y1] = layout_box_coords(lb);
  return static_cast<long long>(x1 - x0) * static_cast<long long>(y1 - y0);
}

// A is "inside" B when >=90% of A's area overlaps B.
//
// THE single containment predicate. The parent/child hierarchy
// (containment_parents) and the LAYOUT_MERGE_MODE drop rule both call this,
// so a box the merge logic drops for being nested is by construction a box
// the hierarchy also saw as contained — they cannot drift apart.
inline bool layout_box_inside(const LayoutBox &a, const LayoutBox &b) noexcept {
  auto [ax0, ay0, ax1, ay1] = layout_box_coords(a);
  auto [bx0, by0, bx1, by1] = layout_box_coords(b);
  int ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
  int ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
  float inter = std::max(0, ix1 - ix0) * std::max(0, iy1 - iy0);
  float area_a = static_cast<float>(ax1 - ax0) * (ay1 - ay0);
  return area_a > 0 && (inter / area_a) >= 0.9f;
}

// Containment parent of every box: the index of the SMALLEST box that
// contains it, or -1 when nothing does. A caption inside a figure inside a
// content block therefore gets the figure, not the block.
//
// CYCLES ARE STRUCTURALLY IMPOSSIBLE. A parent must rank strictly above its
// child in the total order (area DESC, index ASC) — bigger area wins, and on
// an exact area tie the earlier box wins (the vector arrives NMS-sorted by
// score, so "earlier" means "higher confidence"). Following parent links
// therefore walks strictly up a total order over a finite set and must
// terminate. This is what resolves the near-duplicate pair NMS did not catch:
// two boxes each >=90% inside the other cannot point at each other, because
// only one of them outranks the other; the loser becomes the child and the
// winner gets parent -1.
//
// Note the >=90%-of-A's-area rule means a "container" can be slightly SMALLER
// than the box it contains (down to 0.9x). The rank order deliberately makes
// the larger box the parent in that case rather than trusting the predicate's
// direction.
inline std::vector<int>
containment_parents(const std::vector<LayoutBox> &boxes) {
  const std::size_t n = boxes.size();
  std::vector<int> parent(n, -1);
  std::vector<long long> area(n);
  for (std::size_t i = 0; i < n; ++i) area[i] = layout_box_area(boxes[i]);

  // Strict total order: true when j ranks above i (j is "more of a parent").
  auto outranks = [&](std::size_t j, std::size_t i) {
    if (area[j] != area[i]) return area[j] > area[i];
    return j < i;
  };

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j || !outranks(j, i)) continue;
      if (!layout_box_inside(boxes[i], boxes[j])) continue;
      // Keep the LOWEST-ranked (smallest) container seen so far.
      if (parent[i] < 0 || outranks(static_cast<std::size_t>(parent[i]), j))
        parent[i] = static_cast<int>(j);
    }
  }
  return parent;
}

// OPT-IN: when LAYOUT_KEEP_NESTED_CHILDREN is set, the nested-box
// reconciliation also protects the model's legitimate child classes
// (figure_title, footnote, formula_number, paragraph_title — see
// is_nestable_class) from being dropped inside a parent region, the same way
// formulas are always protected. Default (unset) leaves every merge mode
// byte-identical to before: only formulas survive nesting.
inline bool layout_keep_nested_children() {
  static const bool on = [] {
    const std::string v = env::env_or("LAYOUT_KEEP_NESTED_CHILDREN", "");
    return !v.empty() && v != "0";
  }();
  return on;
}

// Shared post-decode cleanup for the GPU and CPU layout paths: NMS, oversized
// "image" drop, containment hierarchy (parent_id), then nested-box
// reconciliation per LAYOUT_MERGE_MODE.
// `mode` defaults to the process-wide LAYOUT_MERGE_MODE. It is a parameter
// only so the unit tests can exercise all three modes in one binary —
// layout_merge_mode() caches its getenv in a function-local static, so a test
// that setenv'd between cases would silently get whichever mode ran first.
inline std::vector<LayoutBox>
postfilter_layout_boxes(std::vector<LayoutBox> out, int orig_h, int orig_w,
                        MergeMode mode = layout_merge_mode()) {
  // 1. NMS: same-class IoU >= 0.6 or cross-class IoU >= 0.98 suppresses the
  //    lower-scoring box.
  std::sort(out.begin(), out.end(),
            [](const LayoutBox &a, const LayoutBox &b) {
              return a.score > b.score;
            });
  auto compute_iou = [&](const LayoutBox &a, const LayoutBox &b) -> float {
    auto [ax0, ay0, ax1, ay1] = layout_box_coords(a);
    auto [bx0, by0, bx1, by1] = layout_box_coords(b);
    int ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    int ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
    float inter = std::max(0, ix1 - ix0 + 1) * std::max(0, iy1 - iy0 + 1);
    float area_a = static_cast<float>(ax1 - ax0 + 1) * (ay1 - ay0 + 1);
    float area_b = static_cast<float>(bx1 - bx0 + 1) * (by1 - by0 + 1);
    float union_area = area_a + area_b - inter;
    return union_area > 0 ? inter / union_area : 0.0f;
  };
  constexpr float kIoUSame = 0.6f;
  constexpr float kIoUDiff = 0.98f;
  std::vector<LayoutBox> nms_out;
  nms_out.reserve(out.size());
  std::vector<bool> suppressed(out.size(), false);
  for (size_t i = 0; i < out.size(); ++i) {
    if (suppressed[i]) continue;
    nms_out.push_back(out[i]);
    for (size_t j = i + 1; j < out.size(); ++j) {
      if (suppressed[j]) continue;
      float thresh = (out[i].class_id == out[j].class_id) ? kIoUSame : kIoUDiff;
      if (compute_iou(out[i], out[j]) >= thresh) suppressed[j] = true;
    }
  }

  // 2. Drop "image" detections covering >82% (portrait) / >93% (landscape) of
  //    the page — a full-page "image" box is a detector artefact, not content.
  const float img_area = static_cast<float>(orig_w) * orig_h;
  const float area_thresh = (orig_h > orig_w) ? 0.82f : 0.93f;
  constexpr int kImageClassId = 14; // "image" in kLayoutLabels
  std::erase_if(nms_out, [&](const LayoutBox &lb) {
    if (lb.class_id != kImageClassId) return false;
    float box_area = static_cast<float>(lb.box[2][0] - lb.box[0][0]) *
                     (lb.box[2][1] - lb.box[0][1]);
    return box_area > area_thresh * img_area;
  });

  // 3. Containment hierarchy over the survivors of steps 1-2.
  const size_t n = nms_out.size();
  const std::vector<int> parent = containment_parents(nms_out);

  // 4. Reconcile nested boxes per LAYOUT_MERGE_MODE. "all" keeps everything.
  //    "outer" drops boxes nested in another (keep containers). "inner" drops
  //    the pure containers (keep innermost). Same `layout_box_inside`
  //    predicate as the hierarchy above; the ONE difference is policy, not
  //    geometry: formula boxes are never *counted* as inside a non-formula box
  //    so standalone math is never dropped. The hierarchy still records that
  //    containment — a display_formula sitting in a text block genuinely
  //    belongs to it, and suppressing the link would throw information away
  //    without protecting anything.
  std::vector<bool> keep(n, true);
  if (mode != MergeMode::kKeepAll) {
    std::vector<bool> contains_other(n, false), inside_other(n, false);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        if (is_formula_class(nms_out[i].class_id) &&
            !is_formula_class(nms_out[j].class_id))
          continue;
        if (layout_box_inside(nms_out[i], nms_out[j])) {
          inside_other[i] = true;
          contains_other[j] = true;
        }
      }
    }
    for (size_t i = 0; i < n; ++i)
      keep[i] = (mode == MergeMode::kKeepOuter)
                    ? !inside_other[i]
                    : (!contains_other[i] || inside_other[i]);
  }

  // 5. Compact + reparent. parent_id must never dangle, so a survivor whose
  //    parent was dropped by the merge mode inherits the nearest SURVIVING
  //    ancestor (walking the chain, which terminates because parent links
  //    strictly ascend the rank order), or -1 if the whole chain went. The
  //    walk runs in every mode — in "all" nothing is dropped so it is a
  //    single no-op hop, and parent_id survives unchanged.
  std::vector<int> new_index(n, -1);
  int next_index = 0;
  for (size_t i = 0; i < n; ++i)
    if (keep[i]) new_index[i] = next_index++;

  std::vector<LayoutBox> kept;
  kept.reserve(static_cast<size_t>(next_index));
  for (size_t i = 0; i < n; ++i) {
    if (!keep[i]) continue;
    int p = parent[i];
    while (p >= 0 && !keep[static_cast<size_t>(p)])
      p = parent[static_cast<size_t>(p)];
    nms_out[i].parent_id = (p >= 0) ? new_index[static_cast<size_t>(p)] : -1;
    kept.push_back(std::move(nms_out[i]));
  }
  return kept;
}

} // namespace turbo_ocr::layout
