#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/core/types.h"          // OCRResultItem
#include "turbo_ocr/core/router_types.h"   // router::TableResult
#include "turbo_ocr/analysis/table/slanext/slanext_dict.h"
#include "turbo_ocr/analysis/table/table_types.h"

namespace turbo_ocr::table {

// Decode one SLANeXt sample.
//
// `structure_probs`: row-major (T, V) softmax probabilities.
// `loc_preds`:       row-major (T, 8) quad slice (normalized model space).
// `dict`:            SLANeXt vocab (post merge_no_span_structure=True).
// `padded_w/h`:      Network input padding shape (488/488).
// `ori_w/h`:         Original region size in pixels (before resize+pad).
StructureResult decode_structure(
    const float* structure_probs,
    const float* loc_preds,
    std::size_t t,
    std::size_t v,
    const CharDict& dict,
    int padded_w,
    int padded_h,
    int ori_w,
    int ori_h);

// Batched recognition over the empty-cell crops (page-coordinate quads). The
// callable closes over the caller's device page + stream/queue + recognizer;
// pass {} when no cell recognizer is configured.
using SlanextCellRecFn = std::function<std::vector<std::pair<std::string, float>>(
    const std::vector<Box> &empty_cells)>;

// THE host-side SLANeXt region postprocess — the ONE copy of the table policy
// that turns a decoded structure + the page's text OCR into a TableResult:
//   gather in-region OCR lines -> shift region-local cell quads to page coords
//   -> match_cells_to_ocr -> crop-OCR backfill of empty cells (min 4px cell,
//   0.5 confidence floor) -> reconstruct_html + build_table_cells.
//
// SHARED on purpose (project rule: generic policy is never fixed per backend).
// This body briefly existed as verbatim copies inside the TRT and CPU
// recognizers, and they had already drifted cosmetically; a threshold tune or
// a matching fix in one copy would silently not reach the other. The only
// device-specific steps — running the structure model and recognizing crops —
// enter through `sr` and `cell_rec`.
//
// `layout_id` is left at -1; the caller stamps it (input order preserved).
[[nodiscard]] router::TableResult slanext_postprocess_region(
    const StructureResult &sr,
    const std::vector<OCRResultItem> &page_ocr,
    const Box &region,
    const SlanextCellRecFn &cell_rec);

} // namespace turbo_ocr::table
