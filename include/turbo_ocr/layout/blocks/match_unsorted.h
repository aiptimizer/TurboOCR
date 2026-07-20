#pragma once

// Layer 2 of the PaddleX layout reading-order pipeline: label-aware
// insertion of "unsorted" blocks (titles, captions, references) into a
// list already sorted by XY-cut. Mirrors paddlex/inference/pipelines/
// layout_parsing/xycut_enhanced/utils.py functions:
//
//   - reference_insert
//   - manhattan_insert
//   - euclidean_insert
//   - weighted_distance_insert
//   - match_unsorted_blocks (dispatch)
//
// Differences vs the Python reference:
//   - No LayoutRegion abstraction — text_line_width / text_line_height
//     and direction are passed through as parameters. Per-block
//     metadata (num_of_lines, seg_*_coordinate, direction) lives on
//     LayoutBox itself and is populated by cluster_text_lines.
//   - tolerance_len for non-doc_title blocks is fixed at 2 px
//     (PaddleX's default in XYCUT_SETTINGS["edge_distance_compare_
//     tolerance_len"]); doc_title scales by text_line_width.
//   - get_seg_flag uses the cluster pre-pass output (num_of_lines +
//     seg_start/end_coordinate) directly. Cells with no clustered
//     lines (vision blocks, empty cells) yield the conservative
//     default (paragraph start = true, paragraph end = true).

#include <array>
#include <vector>

#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::layout {

// Coarse-grained role used by the label-aware insertion strategies.
// Mirrors PaddleX's BLOCK_LABEL_MAP partitioning.
enum class OrderLabel : int {
  kBody = 0,         // text, formula, algorithm — sorted by XY-cut directly
  kDocTitle,         // doc_title — pinned to position 0
  kParagraphTitle,   // paragraph_title etc. — weighted_distance_insert
  kVisionTitle,      // figure_title, table_title — weighted_distance_insert
  kVision,           // image, table, chart, figure — weighted_distance_insert
  kCrossReference,   // reference, footnote, vision_footnote — reference_insert
  kUnordered,        // aside_text, seal, page number, formula_number — manhattan_insert
};

// Per-block paragraph-segmentation flags. Computed via get_seg_flag
// from the text-line cluster pre-pass output (num_of_lines + seg_*_
// coordinate). Both default to true for blocks that aren't part of a
// continuing paragraph.
struct SegFlag {
  bool seg_start_flag = true;  // true when block starts a new paragraph
  bool seg_end_flag = true;    // true when block ends a paragraph
};

// Map a PP-DocLayoutV3 class_id to an OrderLabel. Class IDs not in any
// special bucket fall through to kBody (which means "use whatever the
// XY-cut bucketing decides").
[[nodiscard]] OrderLabel order_label_for(int class_id) noexcept;

// Insert `block` into `sorted_blocks` based on the strategy keyed by
// `block`'s order label. Mirrors match_unsorted_blocks.
//
// `sorted_blocks` carries indices into the caller's layout vector.
// Each entry is paired with its AABB so we don't have to refetch on
// every comparison. Mutates sorted_blocks by inserting block.
struct UnsortedBlock {
  int layout_idx;             // index into the outer `layout` vector
  std::array<int, 4> aabb;    // x0, y0, x1, y1
  OrderLabel order_label;
  int class_id;
};

// Compute the (seg_start_flag, seg_end_flag) pair for `current`
// relative to its predecessor `prev`. Mirrors PaddleX's get_seg_flag:
//   - seg_start_flag = false ⟺ prev's last line ends near the right
//     margin AND current's first line starts at the left margin AND
//     prev has more than one line. That's the "current continues
//     prev's paragraph" case.
//   - seg_end_flag = false ⟺ same, but checked from current's last
//     line to a hypothetical successor margin (here: current itself
//     hits the right margin → it's still mid-paragraph at the end).
//
// `direction` selects which axis is "primary" (start = left for
// horizontal, top for vertical).
[[nodiscard]] SegFlag
get_seg_flag(const LayoutBox &current,
             const LayoutBox &prev,
             Direction direction);

void match_unsorted_block(const UnsortedBlock &block,
                          std::vector<UnsortedBlock> &sorted_blocks,
                          int text_line_width,
                          Direction direction,
                          const std::vector<LayoutBox> &layout);

// Convenience: insert every block in `unsorted` into `sorted` using its
// per-block strategy. Iteration order matches PaddleX:
//   - kDocTitle blocks first (so the "pin to 0" rule lands them at top)
//   - everything else after
void match_unsorted_blocks(std::vector<UnsortedBlock> &sorted,
                           std::vector<UnsortedBlock> &unsorted,
                           int text_line_width,
                           Direction direction,
                           const std::vector<LayoutBox> &layout);

} // namespace turbo_ocr::layout
