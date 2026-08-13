#pragma once

#include <string>
#include <vector>

#include "turbo_ocr/document/docassembly/parsing_block.h"

namespace turbo_ocr::docassembly {

// Stitch a table that spills across a page boundary back into one table.
// Port of PaddleX merge_tables_across_pages (merge_table.py, itself adapted
// from MinerU). For each adjacent page pair (walked last→first), if the last
// table on page i-1 and the first table on page i have matching column
// structure, compatible widths, and only page furniture (headers/footers/
// page numbers/seals) between them, the second table's data rows are appended
// to the first and the second is emptied.
//
// Mutates `blocks_by_page` in place: assigns global_block_id / global_group_id
// (flattened page order) and rewrites merged table content. A merged-away
// table ends with empty content and global_group_id pointing at its host.
void merge_tables_across_pages(std::vector<std::vector<ParsingBlock>>& blocks_by_page);

// Exposed for tests: full-width ASCII (U+FF01..U+FF5E) → half-width.
std::string full_to_half(const std::string& text);

} // namespace turbo_ocr::docassembly
