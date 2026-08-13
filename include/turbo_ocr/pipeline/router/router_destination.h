#pragma once

#include <cstdint>

namespace turbo_ocr::router {

enum class Destination : uint8_t {
  Text = 0,
  Table = 1,
  Formula = 2,
  Skip = 3,
};

// Static class-id → default Destination LUT (25 PP-DocLayoutV3 labels
// from layout_types.h, plus the SupplementaryRegion sentinel). The
// `score` parameter is part of the signature so callers carrying a
// confidence can pass it without an extra hop; the static default
// itself depends only on class_id. Image/chart/seal default to Skip —
// the Text-fallback for figure-embedded captions lives in `route()`
// because it depends on OverlapStats which this header doesn't see.
[[nodiscard]] inline constexpr Destination
destination_for(int class_id, float /*score*/) noexcept {
  switch (class_id) {
    case  3: return Destination::Skip;     // chart
    case  5: return Destination::Formula;  // display_formula
    case  9: return Destination::Skip;     // footer_image
    case 13: return Destination::Skip;     // header_image
    case 14: return Destination::Skip;     // image
    case 15: return Destination::Formula;  // inline_formula
    case 20: return Destination::Skip;     // seal
    case 21: return Destination::Table;    // table
    case  0:                               // abstract
    case  1:                               // algorithm
    case  2:                               // aside_text
    case  4:                               // content
    case  6:                               // doc_title
    case  7:                               // figure_title
    case  8:                               // footer
    case 10:                               // footnote
    case 11:                               // formula_number
    case 12:                               // header
    case 16:                               // number
    case 17:                               // paragraph_title
    case 18:                               // reference
    case 19:                               // reference_content
    case 22:                               // text
    case 23:                               // vertical_text
    case 24:                               // vision_footnote
    case -1:                               // SupplementaryRegion
      return Destination::Text;
    default:
      return Destination::Text;
  }
}

} // namespace turbo_ocr::router
