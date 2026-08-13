#pragma once

#include <array>
#include <string>

namespace turbo_ocr::docassembly {

// CUDA-free, per-region parsing block — the unit the document-assembly
// stage operates on. Mirrors PaddleX's PaddleOCRVLBlock so the ported
// cross-page / title-level / repetition logic stays faithful, but carries
// only the fields those algorithms actually read.
//
// `bbox` is axis-aligned [x0, y0, x1, y1] in page pixels (use turbo_ocr::aabb
// to derive it from a rotated Box). `content` is the recognized payload:
// plain text for text-bound labels, an HTML table for `table`, LaTeX for
// formulas. `label` is a PP-DocLayoutV3 class name (see layout_types.h).
struct ParsingBlock {
  std::string label;
  std::array<int, 4> bbox{0, 0, 0, 0};
  std::string content;

  int page_index = 0;

  // Identity used by cross-page table merge: every block gets a unique
  // global_block_id; global_group_id points at the block it was folded
  // into (== global_block_id when it stands alone). Mirrors PaddleX.
  int global_block_id = 0;
  int global_group_id = 0;

  // Heading depth assigned by title leveling. 0 == doc_title, 1..4 ==
  // paragraph_title depth, -1 == not a heading / unset.
  int title_level = -1;
};

} // namespace turbo_ocr::docassembly
