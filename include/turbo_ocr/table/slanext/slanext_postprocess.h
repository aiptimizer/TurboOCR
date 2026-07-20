#pragma once

#include <cstddef>

#include "turbo_ocr/table/slanext/slanext_dict.h"
#include "turbo_ocr/table/table_types.h"

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

} // namespace turbo_ocr::table
