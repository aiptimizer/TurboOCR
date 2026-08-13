#pragma once

#include <array>
#include <string>
#include <vector>

#include "turbo_ocr/core/router_types.h"  // router::TableCell
#include "turbo_ocr/analysis/table/cell_matcher.h"   // MatchedCell

namespace turbo_ocr::table {

// Build the client-facing cell list for ONE table region from the same three
// inputs reconstruct_html consumes, so the JSON `cells` array and the HTML
// describe the same table by construction:
//
//   structure  — the SLANeXt token stream (wrapped <html><body><table> …).
//   quads      — per <td>-family slot, the cell quad in PAGE pixels
//                (region-local model output already shifted by the crop origin).
//   matched    — per slot, the indices into `ocr_texts` matched to it, after the
//                per-cell crop-OCR backfill has appended its recoveries.
//   ocr_texts  — the region's text pool (page-OCR lines + crop-OCR recoveries).
//
// quads/matched/ocr_texts index alignment is the caller's invariant; this
// function only reads what it is given and tolerates short/ragged inputs.
//
// Grid position (row/col/rowspan/colspan) is DERIVED, not guessed: the token
// stream carries <tr> boundaries and the literal colspan/rowspan attribute
// tokens, which is exactly what the HTML table grid algorithm needs. A slot the
// walk cannot place (e.g. a <td> before any <tr>) keeps row = col = -1 and is
// serialized without those fields.
//
// Returns one entry per element of `quads`, in <td> order — including
// zero-area placeholder slots, so cells[i] is the i-th <td> of the html.
[[nodiscard]] std::vector<router::TableCell> build_table_cells(
    const std::vector<std::string>& structure,
    const std::vector<std::array<int, 8>>& quads,
    const std::vector<MatchedCell>& matched,
    const std::vector<std::string>& ocr_texts);

} // namespace turbo_ocr::table
