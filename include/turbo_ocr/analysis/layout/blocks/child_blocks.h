#pragma once

// Child-block detection + splicing — port of PaddleX's hierarchical
// reading-order helpers (paddlex/inference/pipelines/layout_parsing/
// xycut_enhanced/utils.py: insert_child_blocks, sort_child_blocks,
// update_doc_title_child_blocks, update_paragraph_title_child_blocks,
// update_vision_child_blocks, update_region_child_blocks).
//
// PP-DocLayoutV3 emits no parent/child info. PaddleX derives
// hierarchy from spatial heuristics (proximity, overlap, dimension
// ratios). We do the same and attach the resulting parent→children
// edges to a sidecar `ChildLinks` array.
//
// After XY-cut + match_unsorted_blocks produces the flat reading
// order, splice_child_blocks walks the parents in that order, sorts
// each parent with its children top-down, and replaces the parent
// with the contiguous parent-then-children run. Child blocks that
// also appeared in the flat sequence are skipped on emission so they
// don't double-emit.

#include <vector>

#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/analysis/layout/blocks/match_unsorted.h"

namespace turbo_ocr::layout {

struct ChildLinks {
  // Indices into the outer `layout` vector. Empty for blocks that
  // didn't accrue children. Sorted in detection order; final reading
  // order is recomputed by sort_child_blocks during the splice.
  std::vector<int> child_indices;
};

// Build ChildLinks for every layout entry. `text_line_height` is the
// median height of `text`-class blocks on the page (used as the
// proximity threshold). Empty input → empty output.
[[nodiscard]] std::vector<ChildLinks>
detect_child_blocks(const std::vector<LayoutBox> &layout,
                    int text_line_height);

// Walk `sorted` in order; for each entry that has child indices in
// `links`, pull the children, sort the (parent, children...) sequence
// top-down by y-then-x, and splice that run in place of the parent.
// Any child that ALSO appears in `sorted` independently is removed
// from its standalone position so it only emits once (under its
// parent's slot).
void splice_child_blocks(std::vector<UnsortedBlock> &sorted,
                         const std::vector<ChildLinks> &links,
                         const std::vector<LayoutBox> &layout);

// Return the descendants of `parent_idx` in emission order. Each
// level is sorted top-down by y0 then x0 (matching PaddleX's
// sort_child_blocks for horizontal direction). Recursion handles
// arbitrary tree depth: a child's own children are emitted right
// after the child, before its sibling.
//
// Cycle / self-loop protection: every visited index is recorded in a
// `visited` set; revisits are skipped silently. The walk also bounds
// recursion depth via the layout vector size — the deepest legitimate
// chain is `layout.size()-1` long, so anything deeper indicates a
// pathological cycle and the walk halts.
//
// Empty links → empty output. parent_idx not in [0, layout.size()) →
// empty output.
[[nodiscard]] std::vector<int>
flatten_descendants(int parent_idx,
                    const std::vector<ChildLinks> &links,
                    const std::vector<LayoutBox> &layout);

} // namespace turbo_ocr::layout
