#pragma once

// CUDA-free types for PP-DocLayoutV3 layout detection results. Lives in a
// separate header from paddle_layout.h so that the JSON serializer — and
// unit tests that don't link against CUDA — can include this without
// dragging in <cuda_runtime.h> through cuda_ptr.h.

#include <array>
#include <string_view>

#include "turbo_ocr/common/geometry/box.h"

namespace turbo_ocr::layout {

// 25 class labels emitted by PP-DocLayoutV3. Order matches
// ~/.paddlex/official_models/PP-DocLayoutV3/inference.yml label_list.
// Size is deduced via std::to_array so a mismatch between the comment and
// the data can't silently leave a trailing empty slot.
inline constexpr auto kLayoutLabels = std::to_array<std::string_view>({
    "abstract",        "algorithm",       "aside_text",    "chart",
    "content",         "display_formula", "doc_title",     "figure_title",
    "footer",          "footer_image",    "footnote",      "formula_number",
    "header",          "header_image",    "image",         "inline_formula",
    "number",          "paragraph_title", "reference",     "reference_content",
    "seal",            "table",           "text",          "vertical_text",
    "vision_footnote",
});

// Sentinel class_id used for layout boxes synthesised after the fact —
// e.g. minimum-enclosing region for OCR results that didn't land inside
// any detected layout box. Mirrors PaddleX's "SupplementaryRegion"
// fallback: every result is guaranteed a layout_id pointing into the
// layout array, even when the layout model missed its region.
inline constexpr int kSupplementaryRegionClassId = -1;
inline constexpr std::string_view kSupplementaryRegionLabel =
    "SupplementaryRegion";

inline constexpr std::string_view label_name(int class_id) noexcept {
  if (class_id == kSupplementaryRegionClassId)
    return kSupplementaryRegionLabel;
  if (class_id >= 0 && class_id < static_cast<int>(kLayoutLabels.size()))
    return kLayoutLabels[class_id];
  return {};
}

// Reading-order priority bucket per layout class.
//
// PaddleX's xycut_enhanced pipeline partitions layout regions into three
// strata before running the spatial sort, so common page furniture lands
// in the right slot regardless of where the layout model placed it:
//
//   0  TOP    — header, header_image. Read first.
//   1  BODY   — every other class (text, paragraph_title, doc_title,
//                table, image, abstract, formulas, figures, charts,
//                seals, captions, asides, AND `number` because page
//                numbers can appear at the top OR the bottom of a page).
//                Run through XY-cut.
//   2  BOTTOM — footer, footer_image, footnote, reference,
//                reference_content, vision_footnote. Read last.
//
// Within each bucket we run XY-cut so the natural left-to-right /
// top-to-bottom order still applies to multi-line headers, footers,
// reference lists, etc. Static-asserts below pin the class IDs to the
// label-array slots so a future PaddleX label re-shuffle can't silently
// misroute a class.
inline constexpr int reading_priority_bucket(int class_id) noexcept {
  // Only the unambiguous header/footer/reference classes get bucketed.
  // 'number' (page number) can sit at the top OR the bottom of a page
  // depending on style, so we leave it in BODY and let XY-cut place it
  // by geometric position.
  switch (class_id) {
    case 12: // header
    case 13: // header_image
      return 0;
    case 8:  // footer
    case 9:  // footer_image
    case 10: // footnote
    case 18: // reference
    case 19: // reference_content
    case 24: // vision_footnote
      return 2;
    default:
      return 1;
  }
}

// "image" class id, referenced by the oversized-image filter in
// layout_postfilter.h. Pinned to its label slot by the static_assert below.
inline constexpr int kImageClassId = 14; // "image"

// Classes the layout model intentionally emits *nested* inside a larger
// region — they are children, not duplicate subsets (paragraph_title in
// content, figure_title in image, footnote in footer, formula_number in
// display_formula). Used only by the OPT-IN nested-child protection in
// layout_postfilter.h (LAYOUT_KEEP_NESTED_CHILDREN=1); the default merge
// behaviour is unchanged. `text` (class 22) is deliberately excluded:
// keeping a text region nested in a table region makes the table recognizer's
// cell matcher extract the same text twice (once as a table cell, once as
// loose OCR).
inline constexpr bool is_nestable_class(int class_id) noexcept {
  switch (class_id) {
    case 5:  // display_formula
    case 7:  // figure_title
    case 10: // footnote
    case 11: // formula_number
    case 15: // inline_formula
    case 17: // paragraph_title
      return true;
    default:
      return false;
  }
}

// If PaddleX ever reshuffles the label list, fail at build time rather than
// silently misrouting classes through reading_priority_bucket,
// is_nestable_class, or the oversized-image filter. The full 25-label list is
// pinned (not just the classes the post-processor references) so ANY relabel
// or reorder is caught, not only the ones currently consumed.
static_assert(kLayoutLabels.size() == 25, "PP-DocLayoutV3 has 25 layout classes");
static_assert(kLayoutLabels[0]  == "abstract",          "class_id 0 must be 'abstract'");
static_assert(kLayoutLabels[1]  == "algorithm",         "class_id 1 must be 'algorithm'");
static_assert(kLayoutLabels[2]  == "aside_text",        "class_id 2 must be 'aside_text'");
static_assert(kLayoutLabels[3]  == "chart",             "class_id 3 must be 'chart'");
static_assert(kLayoutLabels[4]  == "content",           "class_id 4 must be 'content'");
static_assert(kLayoutLabels[5]  == "display_formula",   "class_id 5 must be 'display_formula'");
static_assert(kLayoutLabels[6]  == "doc_title",         "class_id 6 must be 'doc_title'");
static_assert(kLayoutLabels[7]  == "figure_title",      "class_id 7 must be 'figure_title'");
static_assert(kLayoutLabels[8]  == "footer",            "class_id 8 must be 'footer'");
static_assert(kLayoutLabels[9]  == "footer_image",      "class_id 9 must be 'footer_image'");
static_assert(kLayoutLabels[10] == "footnote",          "class_id 10 must be 'footnote'");
static_assert(kLayoutLabels[11] == "formula_number",    "class_id 11 must be 'formula_number'");
static_assert(kLayoutLabels[12] == "header",            "class_id 12 must be 'header'");
static_assert(kLayoutLabels[13] == "header_image",      "class_id 13 must be 'header_image'");
static_assert(kLayoutLabels[14] == "image",             "class_id 14 must be 'image'");
static_assert(kLayoutLabels[15] == "inline_formula",    "class_id 15 must be 'inline_formula'");
static_assert(kLayoutLabels[16] == "number",            "class_id 16 must be 'number'");
static_assert(kLayoutLabels[17] == "paragraph_title",   "class_id 17 must be 'paragraph_title'");
static_assert(kLayoutLabels[18] == "reference",         "class_id 18 must be 'reference'");
static_assert(kLayoutLabels[19] == "reference_content", "class_id 19 must be 'reference_content'");
static_assert(kLayoutLabels[20] == "seal",              "class_id 20 must be 'seal'");
static_assert(kLayoutLabels[21] == "table",             "class_id 21 must be 'table'");
static_assert(kLayoutLabels[22] == "text",              "class_id 22 must be 'text'");
static_assert(kLayoutLabels[23] == "vertical_text",     "class_id 23 must be 'vertical_text'");
static_assert(kLayoutLabels[24] == "vision_footnote",   "class_id 24 must be 'vision_footnote'");

// Reading direction for a layout cell or for the page as a whole.
// Mirrors PaddleX's "horizontal" / "vertical" string. Vertical is used
// for CJK tategaki layouts (right-to-left columns, top-to-bottom within
// a column).
enum class Direction : int {
  kHorizontal = 0,
  kVertical = 1,
};

struct LayoutBox {
  int class_id = 0;
  float score = 0.0f;
  // Axis-aligned 4-corner box in the ORIGINAL input image's coordinate
  // system. PP-DocLayoutV3 applies im_shape + scale_factor internally, so
  // the model's output is already in original coordinates.
  Box box{};
  // Cross-reference ID emitted when layout detection is enabled. Default
  // -1 means "not assigned" and the serializer omits the field.
  int id = -1;
  // Read-order index emitted natively by PP-DocLayoutV3 (column 6 of each
  // detection row); -1 when not provided. Kept straight from the model and
  // NOT serialised — reading order is still derived geometrically by
  // reading_order.cpp (class-bucketed XY-cut), so this only preserves the
  // model's own ordering signal for any future consumer.
  int read_order = -1;

  // Text-line metadata populated by cluster_text_lines (a per-cell pre-
  // pass that groups the OCR detection boxes whose layout_id maps to
  // this cell). Default-zero values are the fallback when no result
  // landed in this cell. None of these fields are serialised — they
  // exist only to drive reading-order placement and child-block
  // detection internally.
  Direction direction = Direction::kHorizontal;
  int num_of_lines = 0;
  // Mean line height / width across the clustered TextLines. Useful as
  // a proximity threshold scale (PaddleX's `2 * text_line_height`).
  int text_line_height = 0;
  int text_line_width = 0;
  // Bbox coordinates of the FIRST and LAST text lines along the cell's
  // primary direction. Used by get_seg_flag to detect paragraph
  // continuation: a multi-line cell whose last line ends near the
  // right margin (horizontal) or bottom margin (vertical) is
  // continuing across a hard split.
  int seg_start_coordinate = 0;  // left edge of first line (horizontal) / top of first line (vertical)
  int seg_end_coordinate = 0;    // right edge of last line (horizontal) / bottom of last line (vertical)
};

} // namespace turbo_ocr::layout
