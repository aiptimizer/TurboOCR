#pragma once

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include <cstdio>
#include <string>
#include <vector>

namespace turbo_ocr {

namespace detail {

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
  for (char c : item.text) {
    // Compare against unsigned so UTF-8 continuation bytes (0x80+) don't
    // sign-extend to negative and get mis-escaped as \u00xx. JSON allows
    // raw UTF-8 in strings; only control chars (< 0x20) need \u escaping.
    auto uc = static_cast<unsigned char>(c);
    switch (c) {
      case '"':  j += "\\\""; break;
      case '\\': j += "\\\\"; break;
      case '\n': j += "\\n"; break;
      case '\r': j += "\\r"; break;
      case '\t': j += "\\t"; break;
      default:
        if (uc < 0x20) {
          char buf[7];
          snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(uc));
          j += buf;
        } else {
          j += c;
        }
    }
  }
  j += "\",\"confidence\":";
  char conf_str[16];
  snprintf(conf_str, sizeof(conf_str), "%.5g", item.confidence);
  j += conf_str;
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
  char conf_str[16];
  snprintf(conf_str, sizeof(conf_str), "%.5g", lb.score);
  j += conf_str;
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
// center. If no layout region contains the center, layout_id stays -1.
inline void assign_layout_ids(std::vector<OCRResultItem> &results,
                              std::vector<layout::LayoutBox> &layout) {
  if (layout.empty()) return;

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

} // namespace turbo_ocr
