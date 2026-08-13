#pragma once

#include <string>
#include <vector>

#include "turbo_ocr/serialization/serialization_primitives.h"
#include "turbo_ocr/core/types.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/core/router_types.h"

namespace turbo_ocr {

namespace detail {

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
//
// `parent_id` (the containing region's id, e.g. the display_formula a
// formula_number sits in) is emitted only when the post-filter found a
// container. A page whose regions are all top-level therefore serialises
// byte-identically to the pre-hierarchy response.
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
  if (lb.parent_id >= 0) {
    j += ",\"parent_id\":";
    j += std::to_string(lb.parent_id);
  }
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

// Append one table cell. `row`/`col`/`rowspan`/`colspan` are emitted only when
// the grid walk actually placed the cell (row >= 0) — an unplaceable cell says
// nothing about its position rather than claiming (0,0).
inline void append_table_cell(std::string &j, const router::TableCell &c) {
  j += "{\"text\":\"";
  append_escaped_string(j, c.text);
  j += "\",\"bounding_box\":";
  append_box(j, c.box);
  if (c.row >= 0) {
    j += ",\"row\":";
    j += std::to_string(c.row);
    j += ",\"col\":";
    j += std::to_string(c.col);
    j += ",\"rowspan\":";
    j += std::to_string(c.rowspan);
    j += ",\"colspan\":";
    j += std::to_string(c.colspan);
  }
  j += '}';
}

// The layout_id/html/confidence/bounding_box fields are byte-identical to the
// pre-cells response; "cells" is purely additive (empty array for backends that
// return HTML without geometry, e.g. a remote VLM).
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
    j += ",\"cells\":[";
    for (size_t k = 0; k < t.cells.size(); ++k) {
      if (k > 0) j += ',';
      append_table_cell(j, t.cells[k]);
    }
    j += "]}";
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

} // namespace detail

} // namespace turbo_ocr
