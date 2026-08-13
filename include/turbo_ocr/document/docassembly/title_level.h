#pragma once

#include <string>
#include <vector>

#include "turbo_ocr/document/docassembly/parsing_block.h"

namespace turbo_ocr::docassembly {

// (numbering style, semantic depth) extracted from a heading string.
// `symbol` is empty and `level` is -1 when no numbering is recognized.
// Mirrors PaddleX title_level.get_symbol_and_level.
struct SymbolLevel {
  std::string symbol;
  int level = -1;
};

SymbolLevel get_symbol_and_level(const std::string& content);

// Assign `title_level` to every `paragraph_title` block in place, voting
// among semantic numbering, relative numbering order, and a font-size
// (height) cluster. Direct port of assign_levels_to_parsing_res; the
// height clustering is an independent deterministic 1-D k-means (not
// sklearn-bit-identical) but yields the same monotonic size→level mapping,
// which is all the downstream vote consumes. Non-heading blocks untouched.
void assign_title_levels(std::vector<std::vector<ParsingBlock>>& blocks_by_page);

} // namespace turbo_ocr::docassembly
