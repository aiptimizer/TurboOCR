#pragma once

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/serialization/serialization_primitives.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr {

namespace detail {

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

} // namespace turbo_ocr
