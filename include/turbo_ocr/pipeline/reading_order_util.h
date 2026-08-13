#pragma once

#include <vector>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/analysis/layout/order/reading_order.h"

namespace turbo_ocr::pipeline {

// Shared assign-ids + XY-cut pair used by the GPU single/batch paths, the CPU
// pipeline, and the PDF job — one definition so the orphan-result contract
// (synthetic XY-cut entries for unmatched detections) cannot drift.
inline void maybe_assign_reading_order(bool want,
                                       std::vector<OCRResultItem> &results,
                                       std::vector<layout::LayoutBox> &layout,
                                       std::vector<int> &reading_order) {
  if (!want || layout.empty()) return;
  turbo_ocr::assign_layout_ids(results, layout);
  reading_order =
      layout::assign_reading_order_for_results(results, layout);
}

} // namespace turbo_ocr::pipeline
