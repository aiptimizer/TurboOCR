#include "turbo_ocr/router/cua_router.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace turbo_ocr::router {

namespace {

[[nodiscard]] inline std::array<int, 4>
aabb_of_layout(const turbo_ocr::layout::LayoutBox &lb) noexcept {
  return turbo_ocr::aabb(lb.box);
}


[[nodiscard]] inline bool gate_overlap_pass(
    const std::vector<turbo_ocr::layout::LayoutBox> &layout) noexcept {
  for (const auto &lb : layout) {
    switch (lb.class_id) {
      case 3:  // chart
      case 5:  // display_formula
      case 14: // image
      case 15: // inline_formula
      case 21: // table
        return true;
      default:
        break;
    }
  }
  return false;
}

} // namespace


// -- CuaRouter -----------------------------------------------------------------

CuaRouter::CuaRouter() noexcept {
  // Reserve thread-local scratch up to a typical page working set
  // (kMaxDetections=300 per plan 05 §10). Avoids any allocation on the
  // hot path for normal inputs.
  decisions_.reserve(64);
  layout_aabbs_.reserve(64);
}

CuaRouter::~CuaRouter() noexcept = default;

void CuaRouter::classify(const std::vector<turbo_ocr::Box> &det_boxes,
                         const std::vector<turbo_ocr::layout::LayoutBox> &layout,
                         RoutingPlan &plan) const {
  plan.clear();
  plan.rec_suppress.assign(det_boxes.size(), 0);

  if (layout.empty()) {
    plan.text_indices.reserve(det_boxes.size());
    plan.text_to_layout_id.reserve(det_boxes.size());
    for (int i = 0; i < static_cast<int>(det_boxes.size()); ++i) {
      plan.text_indices.push_back(i);
      plan.text_to_layout_id.push_back(-1);
    }
    return;
  }

  // 1. Precompute layout AABBs.
  layout_aabbs_.clear();
  layout_aabbs_.reserve(layout.size());
  for (const auto &lb : layout) {
    layout_aabbs_.push_back(aabb_of_layout(lb));
  }

  // 2. Lazy OverlapStats — only when at least one image/chart/table/
  //    formula class exists on the page.
  const bool need_overlap = gate_overlap_pass(layout);
  overlap_.clear();
  if (need_overlap) {
    build_overlap_stats(layout, det_boxes, layout_aabbs_, overlap_);
  }

  // 3. PageStats — only the bits we actually use downstream.
  page_stats_ = PageStats{};
  for (const auto &lb : layout) {
    if ((lb.class_id == 5 || lb.class_id == 15) &&
        lb.score >= cfg_.tau_trust[lb.class_id]) {
      page_stats_.has_confident_formula = true;
      break;
    }
  }

  // 4. Per-layout routing decisions.
  decisions_.clear();
  decisions_.reserve(layout.size());
  for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
    decisions_.push_back(route(i, layout[i], overlap_, page_stats_, cfg_));
  }

  // 5. Tie-breakers (containment > IoU class-priority > score > id).
  resolve_tie_breakers(decisions_, layout, layout_aabbs_);

  // 6. Bucket per-layout decisions for downstream dispatch.
  plan.table_layout_ids.reserve(layout.size());
  plan.formula_layout_ids.reserve(layout.size());
  plan.skip_layout_ids.reserve(layout.size());
  for (const auto &d : decisions_) {
    switch (d.dest) {
      case Destination::Table:
        plan.table_layout_ids.push_back(d.layout_idx);
        break;
      case Destination::Formula:
        plan.formula_layout_ids.push_back(d.layout_idx);
        break;
      case Destination::Skip:
        plan.skip_layout_ids.push_back(d.layout_idx);
        break;
      case Destination::Text:
        break;
    }
  }

  // 7. Det-box bucketing.
  //    A det box's owning layout is its centroid-in-AABB match (same
  //    rule as assign_layout_ids). When the owning cell routes to
  //    Table or Formula AND that cell is NOT in dual-routing salvage
  //    mode, the det box is suppressed from rec to avoid double-OCR.
  plan.text_indices.reserve(det_boxes.size());
  plan.text_to_layout_id.reserve(det_boxes.size());

  for (std::size_t i = 0; i < det_boxes.size(); ++i) {
    int owner = -1;
    for (std::size_t li = 0; li < layout_aabbs_.size(); ++li) {
      if (turbo_ocr::centroid_in_aabb(det_boxes[i], layout_aabbs_[li])) {
        owner = static_cast<int>(li);
        break;
      }
    }

    bool suppress = false;
    if (owner >= 0) {
      const auto &d = decisions_[owner];
      if ((d.dest == Destination::Table ||
           d.dest == Destination::Formula) && !d.also_text) {
        suppress = true;
      }
    }

    if (suppress) {
      plan.rec_suppress[i] = 1;
    } else {
      plan.text_indices.push_back(static_cast<int>(i));
      plan.text_to_layout_id.push_back(owner);
    }
  }
}

} // namespace turbo_ocr::router
