#pragma once

// Per-cell text-line clustering pre-pass — port of PaddleX's
// LayoutBlock.group_boxes_into_lines + calculate_text_line_direction
// (paddlex/inference/pipelines/layout_parsing/layout_objects.py).
//
// For each layout cell, group the OCR detection boxes whose centre
// falls inside it into TextLine groups (boxes whose perpendicular
// projection overlaps by >= LINE_HEIGHT_IOU_THRESHOLD). For each
// cell:
//   - direction is set by the majority-vote of its text boxes
//     (width*1.5 >= height ⇒ horizontal)
//   - num_of_lines = TextLine count
//   - text_line_height / text_line_width = means across the lines
//   - seg_start_coordinate = left/top edge of the first line
//   - seg_end_coordinate   = right/bottom edge of the last line
//
// These values feed match_unsorted (weighted_distance_insert disperse
// term, get_seg_flag look-ahead) and child_blocks (real proximity
// threshold instead of approx_num_lines).
//
// Page-level direction inference (majority vote across text-class
// cells) is exposed by infer_page_direction.

#include <vector>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::layout {

// Threshold used by group_boxes_into_lines for declaring two boxes
// to be on the same text line — projection overlap (small-area mode)
// must exceed this value. Mirrors LINE_SETTINGS["line_height_iou_threshold"].
inline constexpr float kLineHeightIouThreshold = 0.6f;

// Per-cell direction-vote ratio: a single box is "horizontal" iff
// width * kTextLineDirectionRatio >= height. PaddleX uses 1.5 in
// calculate_text_line_direction.
inline constexpr float kTextLineDirectionRatio = 1.5f;

// Cluster the results into per-cell TextLines. Mutates each LayoutBox
// in `layout` to populate direction / num_of_lines / text_line_height /
// text_line_width / seg_start_coordinate / seg_end_coordinate.
//
// Caller MUST run assign_layout_ids first so each result has a
// resolved layout_id (the SupplementaryRegion synthetic block also
// gets a real layout_id, so its members participate normally).
void cluster_text_lines(const std::vector<OCRResultItem> &results,
                        std::vector<LayoutBox> &layout);

// Infer the dominant page direction from text-class cells (majority
// vote of their per-cell direction). Returns kHorizontal for empty
// inputs and for ties. Run AFTER cluster_text_lines.
[[nodiscard]] Direction
infer_page_direction(const std::vector<LayoutBox> &layout);

} // namespace turbo_ocr::layout
