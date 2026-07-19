#pragma once

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/router/router_types.h"
#include <climits>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace turbo_ocr {

namespace detail {

// Full JSON-string escape with a UTF-8 backstop (definition below). Forward-
// declared so every text-bearing writer above its definition can route
// through the ONE escaper instead of a weaker inline copy that would ship
// RFC-8259-invalid bytes on malformed UTF-8. Caller writes the quotes.
inline void append_escaped_string(std::string &j, const std::string &s);

// Serialize a confidence/score as JSON. snprintf("%.5g", NaN|Inf) emits the
// bare tokens `nan`/`inf`, which are NOT valid JSON and break the entire
// response for a strict client — a single degraded region must not do that.
// Non-finite -> 0.
inline void append_score(std::string &j, double v) {
  char buf[16];
  if (std::isfinite(v))
    std::snprintf(buf, sizeof(buf), "%.5g", v);
  else
    std::snprintf(buf, sizeof(buf), "0");
  j += buf;
}

// Append `[[x,y],[x,y],[x,y],[x,y]]` to j — shared by text + layout writers.
inline void append_box(std::string &j, const Box &box) {
  j += '[';
  for (int k = 0; k < 4; ++k) {
    if (k > 0) j += ',';
    j += '[';
    j += std::to_string(box[k][0]);
    j += ',';
    j += std::to_string(box[k][1]);
    j += ']';
  }
  j += ']';
}

// Append one OCR text item (without enclosing braces). Caller wraps with {}.
// When `item.source` is non-empty and not "ocr", a "source" field is also
// emitted — this is how /ocr/pdf's `auto_verified` / `geometric` / `auto`
// modes tell clients which path produced each item. For every other code
// path `source` is empty and we stay byte-identical to the pre-feature
// response.
inline void append_ocr_item(std::string &j, const OCRResultItem &item) {
  j += '{';
  if (item.id >= 0) {
    j += "\"id\":";
    j += std::to_string(item.id);
    j += ',';
  }
  j += "\"text\":\"";
  append_escaped_string(j, item.text);
  j += "\",\"confidence\":";
  append_score(j, item.confidence);
  j += ",\"bounding_box\":";
  append_box(j, item.box);
  if (!item.source.empty() && item.source != "ocr") {
    // INVARIANT: item.source is only ever set from internal string literals
    // (e.g. "ocr", "pdf", "geometric", "auto", "auto_verified") — never from
    // user input. Minimal escaping suffices. If that ever changes, route it
    // through the text-escape loop above.
    j += ",\"source\":\"";
    for (char c : item.source) {
      if (c == '"' || c == '\\') j += '\\';
      j += c;
    }
    j += '"';
  }
  if (item.layout_id >= 0) {
    j += ",\"layout_id\":";
    j += std::to_string(item.layout_id);
  }
  j += '}';
}

// Append one layout item. Class label is emitted both as the human-readable
// string (`class`) and as the raw integer (`class_id`).
inline void append_layout_item(std::string &j, const layout::LayoutBox &lb) {
  j += '{';
  if (lb.id >= 0) {
    j += "\"id\":";
    j += std::to_string(lb.id);
    j += ',';
  }
  j += "\"class\":\"";
  auto name = layout::label_name(lb.class_id);
  for (char c : name) j += c;   // labels are ASCII, no escaping needed
  j += "\",\"class_id\":";
  j += std::to_string(lb.class_id);
  j += ",\"confidence\":";
  append_score(j, lb.score);
  j += ",\"bounding_box\":";
  append_box(j, lb.box);
  j += '}';
}

// Append `"results":[ ... ]` (no enclosing braces). Callers compose the
// outer object envelope themselves so PDF per-page blocks can share this.
inline void append_results_array(std::string &j,
                                  const std::vector<OCRResultItem> &results) {
  j += "\"results\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    if (i > 0) j += ',';
    append_ocr_item(j, results[i]);
  }
  j += ']';
}

inline void append_layout_array(std::string &j,
                                 const std::vector<layout::LayoutBox> &layout) {
  j += "\"layout\":[";
  for (size_t i = 0; i < layout.size(); ++i) {
    if (i > 0) j += ',';
    append_layout_item(j, layout[i]);
  }
  j += ']';
}

inline void append_reading_order_array(std::string &j,
                                       const std::vector<int> &order) {
  j += "\"reading_order\":[";
  for (size_t i = 0; i < order.size(); ++i) {
    if (i > 0) j += ',';
    j += std::to_string(order[i]);
  }
  j += ']';
}

// Full JSON-string escape. Tables emit HTML (quotes, ampersands) and
// formulas emit LaTeX (backslashes, braces) — both need every escape
// the OCR text branch uses. Caller writes the surrounding quotes.
inline void append_escaped_string(std::string &j, const std::string &s) {
  const size_t n = s.size();
  for (size_t i = 0; i < n;) {
    const auto uc = static_cast<unsigned char>(s[i]);
    if (uc < 0x80) {  // ASCII: JSON-escape control/special chars, pass the rest through
      const char c = s[i++];
      switch (c) {
        case '"':  j += "\\\""; break;
        case '\\': j += "\\\\"; break;
        case '\b': j += "\\b";  break;
        case '\f': j += "\\f";  break;
        case '\n': j += "\\n";  break;
        case '\r': j += "\\r";  break;
        case '\t': j += "\\t";  break;
        default:
          if (uc < 0x20) {
            char buf[7];
            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(uc));
            j += buf;
          } else {
            j += c;
          }
      }
      continue;
    }
    // Multi-byte: copy a well-formed UTF-8 sequence verbatim, else emit U+FFFD. A backstop so
    // no producer can ever ship RFC-8259-invalid bytes that a strict JSON client would reject
    // (valid output is byte-identical to before).
    const int len = (uc >> 5) == 0x6 ? 2 : (uc >> 4) == 0xE ? 3 : (uc >> 3) == 0x1E ? 4 : 0;
    bool ok = len >= 2 && i + static_cast<size_t>(len) <= n;
    for (int k = 1; k < len && ok; ++k)
      ok = (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
    if (ok) { j.append(s, i, static_cast<size_t>(len)); i += static_cast<size_t>(len); }
    else { j += "\xEF\xBF\xBD"; ++i; }
  }
}

inline void append_tables_array(std::string &j,
                                 const std::vector<router::TableResult> &tables) {
  j += "\"tables\":[";
  for (size_t i = 0; i < tables.size(); ++i) {
    if (i > 0) j += ',';
    const auto &t = tables[i];
    j += "{\"layout_id\":";
    j += std::to_string(t.layout_id);
    j += ",\"html\":\"";
    append_escaped_string(j, t.html);
    j += "\",\"confidence\":";
    append_score(j, t.score);
    j += ",\"bounding_box\":";
    append_box(j, t.box);
    j += '}';
  }
  j += ']';
}

inline void append_formulas_array(std::string &j,
                                   const std::vector<router::FormulaResult> &formulas) {
  j += "\"formulas\":[";
  for (size_t i = 0; i < formulas.size(); ++i) {
    if (i > 0) j += ',';
    const auto &f = formulas[i];
    j += "{\"layout_id\":";
    j += std::to_string(f.layout_id);
    j += ",\"latex\":\"";
    append_escaped_string(j, f.latex);
    j += "\",\"confidence\":";
    append_score(j, f.score);
    j += ",\"bounding_box\":";
    append_box(j, f.box);
    j += '}';
  }
  j += ']';
}

// Append `"blocks":[ ... ]` — paragraph-level aggregate, one entry per
// non-empty layout cell, in reading order. Mirrors the granularity
// PaddleX's PP-StructureV3 emits via parsing_res_list (one LayoutBlock
// per layout region with joined `content`).
//
// Each entry: {id, layout_id, class, bounding_box, content, order_index}
// `content` joins child texts with ' ' on the same line and '\n' on a
// new line. Line boundaries are detected from y-jump > text_line_height
// (or 8px when text_line_height is 0). The first non-empty layout cell
// in reading order gets order_index=0.
inline void append_blocks_array(std::string &j,
                                 const std::vector<OCRResultItem> &results,
                                 const std::vector<layout::LayoutBox> &layout,
                                 const std::vector<int> &reading_order) {
  j += "\"blocks\":[";
  if (reading_order.empty() || layout.empty()) {
    j += ']';
    return;
  }

  // Group results in reading order by layout_id, preserving first-
  // encounter order so the emitted blocks are themselves in reading
  // order. We walk reading_order and bucket each result.
  std::vector<int> block_order_of_layout(layout.size(), -1);
  std::vector<std::vector<int>> grouped(layout.size());
  int next_block_idx = 0;
  for (int ri : reading_order) {
    if (ri < 0 || static_cast<size_t>(ri) >= results.size()) continue;
    const int lid = results[ri].layout_id;
    if (lid < 0 || static_cast<size_t>(lid) >= layout.size()) continue;
    if (block_order_of_layout[lid] < 0) {
      block_order_of_layout[lid] = next_block_idx++;
    }
    grouped[lid].push_back(ri);
  }

  // Iterate layouts in their reading-order rank.
  std::vector<int> layout_emit_order(layout.size(), -1);
  for (size_t li = 0; li < layout.size(); ++li) {
    if (block_order_of_layout[li] >= 0)
      layout_emit_order[block_order_of_layout[li]] = static_cast<int>(li);
  }

  bool first = true;
  for (int oi = 0; oi < next_block_idx; ++oi) {
    const int li = layout_emit_order[oi];
    if (li < 0) continue;
    const auto &cell = layout[li];
    const auto &members = grouped[li];
    if (members.empty()) continue;

    if (!first) j += ',';
    first = false;
    j += '{';
    j += "\"id\":";
    j += std::to_string(oi);
    j += ",\"layout_id\":";
    j += std::to_string(li);
    j += ",\"class\":\"";
    auto name = layout::label_name(cell.class_id);
    for (char c : name) j += c;
    j += "\",\"bounding_box\":";
    append_box(j, cell.box);
    j += ",\"content\":\"";

    // Build the content with smart line-joining:
    //  - Same y-band as previous member  → ' '   (intra-line text run)
    //  - Different y-band, prev line ends NEAR the cell's right margin
    //    (within line_h of cell.x1)        → ' '   (paragraph wrap; the
    //                                              previous line ran out
    //                                              of horizontal space and
    //                                              continues here)
    //  - Different y-band, prev line ends MID-CELL                → '\n'
    //                                              (real paragraph break)
    //
    // Result: a multi-line paragraph emits as one flowing string ('the
    // quick brown fox jumps over the lazy dog and continues here'), but
    // distinct paragraphs in the same layout cell stay separated by
    // newlines. This is the PaddleX `format_line` strategy applied at
    // serialization time using only the per-cell bbox + text_line_height
    // already populated by cluster_text_lines.
    auto [cell_x0, cell_y0, cell_x1, cell_y1] = turbo_ocr::aabb(cell.box);
    const int line_h = cell.text_line_height > 0
                          ? cell.text_line_height
                          : 8;
    const int wrap_tol = std::max(8, line_h);  // pixels from cell right edge
    int prev_cy = INT_MIN;
    int prev_x1 = INT_MIN;
    for (size_t mi = 0; mi < members.size(); ++mi) {
      const auto &it = results[members[mi]];
      int sy = 0;
      int max_x = INT_MIN;
      for (int k = 0; k < 4; ++k) {
        sy += it.box[k][1];
        max_x = std::max(max_x, it.box[k][0]);
      }
      const int cy = sy / 4;
      if (mi > 0) {
        const int dy = std::abs(cy - prev_cy);
        if (dy <= line_h / 2) {
          // Same y-band — separate text spans on the same line.
          j += ' ';
        } else {
          // Different y-band — wrapped vs hard break.
          const bool prev_reached_right_margin =
              prev_x1 >= cell_x1 - wrap_tol;
          if (prev_reached_right_margin) j += ' ';
          else                            j += "\\n";
        }
      }
      prev_cy = cy;
      prev_x1 = max_x;
      // Same JSON escaping + UTF-8 backstop as every other text field.
      append_escaped_string(j, it.text);
    }
    j += "\",\"order_index\":";
    j += std::to_string(oi);
    j += '}';
  }
  j += ']';
}

} // namespace detail

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
    float cx = 0.0f, cy = 0.0f;
    for (int k = 0; k < 4; ++k) {
      cx += static_cast<float>(it.box[k][0]);
      cy += static_cast<float>(it.box[k][1]);
    }
    cx *= 0.25f;
    cy *= 0.25f;
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
