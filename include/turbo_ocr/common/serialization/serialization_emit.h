#pragma once

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/serialization/serialization_blocks.h"
#include "turbo_ocr/common/serialization/serialization_items.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/router/router_types.h"

namespace turbo_ocr {

// Back-compat: text-only response. Existing non-layout code paths keep
// calling this signature unchanged.
[[nodiscard]] inline std::string
results_to_json(const std::vector<OCRResultItem> &results) {
  std::string j;
  j.reserve(results.size() * 200);
  j += '{';
  detail::append_results_array(j, results);
  j += '}';
  return j;
}

// Assign stable numeric IDs to every text item and every layout item, and
// cross-reference each text item to the layout region containing its box
// center (via `layout_id`). No-op when `layout` is empty — in that case
// text items keep their default id=-1 / layout_id=-1 and the serializer
// omits the fields entirely (so responses without layout stay byte-
// identical to pre-layout clients).
//
// Matching rule: a text item's `layout_id` is the id of the first layout
// region whose axis-aligned bbox contains the text item's bounding-box
// center. If no layout region contains the center, the item is added to
// a synthesised "SupplementaryRegion" block whose bbox is the minimum
// enclosing rectangle of the unmatched items — mirroring PaddleX's
// pipeline_v2 fallback so every result keeps a valid layout_id.
inline void assign_layout_ids(std::vector<OCRResultItem> &results,
                              std::vector<layout::LayoutBox> &layout) {
  // Backward-compat: when the caller didn't request layout (empty input)
  // we don't synthesise anything — the serializer then omits the layout
  // key + per-result layout_id keys, keeping responses byte-identical to
  // pre-layout clients.
  if (layout.empty()) return;

  // Idempotent short-circuit: pipelines run this before reading-order so
  // assign_reading_order_for_results can read layout_id; serialization
  // calls it again to be defensive when invoked directly. Detect a prior
  // run by the side-effect we leave behind — layout[0].id transitions
  // from -1 (default) to 0 once assigned.
  if (layout.front().id == 0) return;

  // 1. Assign IDs to layout boxes and cache the axis-aligned bbox of
  //    each 4-corner Box. aabb() lives in common/box.h so the same
  //    min/max logic is shared with the auto_verified /ocr/pdf path.
  struct LRect { int x0, y0, x1, y1; };
  std::vector<LRect> lrects;
  lrects.reserve(layout.size());
  for (size_t i = 0; i < layout.size(); ++i) {
    layout[i].id = static_cast<int>(i);
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[i].box);
    lrects.push_back({x0, y0, x1, y1});
  }

  // 2. Assign IDs to text items and resolve layout_id by center-in-rect.
  //    Text boxes may be rotated quads (detection output) so we use their
  //    centroid rather than any corner.
  for (size_t i = 0; i < results.size(); ++i) {
    auto &it = results[i];
    it.id = static_cast<int>(i);
    const auto [cx, cy] = quad_centroid(it.box);
    for (size_t j = 0; j < lrects.size(); ++j) {
      const auto &r = lrects[j];
      if (cx >= static_cast<float>(r.x0) && cx <= static_cast<float>(r.x1) &&
          cy >= static_cast<float>(r.y0) && cy <= static_cast<float>(r.y1)) {
        it.layout_id = static_cast<int>(j);
        break;
      }
    }
  }

  // 3. Supplementary region for orphans. Walk the results once: any item
  //    still at layout_id == -1 contributes its AABB to a running
  //    minimum-enclosing rectangle. If at least one orphan exists, append
  //    a synthetic LayoutBox covering them all and rebind their
  //    layout_ids to the synthetic block.
  int supp_x0 = INT_MAX, supp_y0 = INT_MAX;
  int supp_x1 = INT_MIN, supp_y1 = INT_MIN;
  bool has_orphan = false;
  for (const auto &it : results) {
    if (it.layout_id >= 0) continue;
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(it.box);
    supp_x0 = std::min(supp_x0, x0);
    supp_y0 = std::min(supp_y0, y0);
    supp_x1 = std::max(supp_x1, x1);
    supp_y1 = std::max(supp_y1, y1);
    has_orphan = true;
  }
  if (!has_orphan) return;

  layout::LayoutBox supp;
  supp.class_id = layout::kSupplementaryRegionClassId;
  supp.score = 1.0f;
  supp.box[0] = {supp_x0, supp_y0};
  supp.box[1] = {supp_x1, supp_y0};
  supp.box[2] = {supp_x1, supp_y1};
  supp.box[3] = {supp_x0, supp_y1};
  const int supp_idx = static_cast<int>(layout.size());
  supp.id = supp_idx;
  layout.push_back(supp);

  for (auto &it : results) {
    if (it.layout_id < 0) it.layout_id = supp_idx;
  }
}

// Text + optional layout response. When `layout` is empty the "layout"
// key is omitted entirely (not emitted as []) so clients that don't know
// about layout see zero diff in the response body. When layout is non-
// empty, both vectors are mutated in place to carry numeric IDs and
// text→layout cross-references.
[[nodiscard]] inline std::string
results_to_json(std::vector<OCRResultItem> &results,
                std::vector<layout::LayoutBox> &layout) {
  assign_layout_ids(results, layout);
  std::string j;
  j.reserve(results.size() * 200 + layout.size() * 120);
  j += '{';
  detail::append_results_array(j, results);
  if (!layout.empty()) {
    j += ',';
    detail::append_layout_array(j, layout);
  }
  j += '}';
  return j;
}

// Full response with optional reading_order. When `reading_order` is
// empty the key is omitted entirely (no `"reading_order"`), keeping the
// output byte-identical to the layout-only overload above. The existing
// two-arg overloads remain unchanged so callers that don't know about
// reading-order keep working.
[[nodiscard]] inline std::string
results_with_reading_order(
    std::vector<OCRResultItem> &results,
    std::vector<layout::LayoutBox> &layout,
    const std::vector<int> &reading_order) {
  assign_layout_ids(results, layout);
  std::string j;
  j.reserve(results.size() * 200 + layout.size() * 120);
  j += '{';
  detail::append_results_array(j, results);
  if (!layout.empty()) {
    j += ',';
    detail::append_layout_array(j, layout);
  }
  if (!reading_order.empty()) {
    j += ',';
    detail::append_reading_order_array(j, reading_order);
  }
  j += '}';
  return j;
}

// Full response with optional `blocks` aggregate. Same layout/reading-
// order plumbing as the variant above, plus a `blocks` array (one entry
// per non-empty layout cell, in reading order, with joined content).
// Callers ask for this by setting want_blocks=true at the route level
// (mapped from `?as_blocks=1` on HTTP / `as_blocks=true` in proto).
//
// When `reading_order` or `layout` is empty, `blocks` is also omitted —
// aggregation requires both to be present.
[[nodiscard]] inline std::string
results_with_blocks(
    std::vector<OCRResultItem> &results,
    std::vector<layout::LayoutBox> &layout,
    const std::vector<int> &reading_order) {
  assign_layout_ids(results, layout);
  std::string j;
  j.reserve(results.size() * 220 + layout.size() * 200);
  j += '{';
  detail::append_results_array(j, results);
  if (!layout.empty()) {
    j += ',';
    detail::append_layout_array(j, layout);
  }
  if (!reading_order.empty()) {
    j += ',';
    detail::append_reading_order_array(j, reading_order);
  }
  if (!reading_order.empty() && !layout.empty()) {
    j += ',';
    detail::append_blocks_array(j, results, layout, reading_order);
  }
  j += '}';
  return j;
}

// Dispatch helper: emit either `results_with_blocks` or
// `results_with_reading_order` depending on the route-level flag.
// Lets every call site stay one-liner without sprinkling branches.
[[nodiscard]] inline std::string
emit_results_json(std::vector<OCRResultItem> &results,
                  std::vector<layout::LayoutBox> &layout,
                  const std::vector<int> &reading_order,
                  bool want_blocks) {
  return want_blocks
             ? results_with_blocks(results, layout, reading_order)
             : results_with_reading_order(results, layout, reading_order);
}

// OcrPipelineResult emitter. Conditionally appends `tables` and `formulas`
// when populated by the CUA router stages. Text-only pages where the
// router never fired produce a response byte-identical to
// emit_results_json above — both vectors are empty and their keys are
// omitted entirely. Reuses the same `assign_layout_ids` mutation so the
// `id`/`layout_id` fields are consistent with the legacy emitters.
[[nodiscard]] inline std::string
emit_pipeline_result_json(pipeline::OcrPipelineResult &out,
                          bool want_blocks) {
  assign_layout_ids(out.results, out.layout);
  std::string j;
  j.reserve(out.results.size() * 220 +
            out.layout.size() * 200 +
            out.tables.size() * 256 +
            out.formulas.size() * 192);
  j += '{';
  detail::append_results_array(j, out.results);
  if (!out.layout.empty()) {
    j += ',';
    detail::append_layout_array(j, out.layout);
  }
  if (!out.reading_order.empty()) {
    j += ',';
    detail::append_reading_order_array(j, out.reading_order);
  }
  if (want_blocks && !out.reading_order.empty() && !out.layout.empty()) {
    j += ',';
    detail::append_blocks_array(j, out.results, out.layout, out.reading_order);
  }
  if (!out.tables.empty()) {
    j += ',';
    detail::append_tables_array(j, out.tables);
  }
  if (!out.formulas.empty()) {
    j += ',';
    detail::append_formulas_array(j, out.formulas);
  }
  // Additive degradation signal: present only when the formula stage actually
  // failed a region (backend error, not empty input). Omitted on the clean
  // path so existing responses stay byte-identical.
  if (out.formula_degraded) {
    j += ",\"formula_degraded\":true";
    if (!out.formula_warning.empty()) {
      j += ",\"formula_warning\":\"";
      detail::append_escaped_string(j, out.formula_warning);
      j += '"';
    }
  }
  // Same additive contract for the table stage (see formula_degraded above).
  if (out.table_degraded) {
    j += ",\"table_degraded\":true";
    if (!out.table_warning.empty()) {
      j += ",\"table_warning\":\"";
      detail::append_escaped_string(j, out.table_warning);
      j += '"';
    }
  }
  // Same additive contract for the base OCR/recognition stage: detection found
  // text regions but recognition produced no usable text (see text_degraded).
  if (out.text_degraded) {
    j += ",\"text_degraded\":true";
    if (!out.text_warning.empty()) {
      j += ",\"text_warning\":\"";
      detail::append_escaped_string(j, out.text_warning);
      j += '"';
    }
  }
  j += '}';
  return j;
}

} // namespace turbo_ocr
