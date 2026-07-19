#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/common/types.h"        // OCRResultItem, Box, kDropScore
#include "turbo_ocr/pipeline/pipeline_result.h"  // OcrPipelineResult, finalize_deferred

namespace turbo_ocr::pipeline::detail {

// No-silent-failure guard for the base OCR/recognition stage. Detection found
// `num_boxes` text regions but recognition produced no usable text — flag the
// result degraded so the response can never be a clean empty 200 that looks
// identical to a genuinely text-free page. A page with zero detections is NOT
// degraded (correctly text-free). Mirrors the formula/table degraded contract.
inline void flag_text_degraded(OcrPipelineResult &out, std::size_t num_boxes) {
  if (num_boxes > 0 && out.results.empty()) {
    out.text_degraded = true;
    out.text_warning =
        "text stage degraded: detection found " + std::to_string(num_boxes) +
        " text region(s) but recognition produced no usable text "
        "(all crops decoded empty/blank; not a genuinely blank page)";
  }
}

// Partial recognition drops (engine output exceeded the decode buffers, see
// PaddleRec::last_dropped_crops) surface as text_degraded even when the rest
// of the page decoded fine — a thinner page must never read as a clean one.
inline void flag_dropped_crops(OcrPipelineResult &out, int dropped) {
  if (dropped <= 0) return;
  out.text_degraded = true;
  const std::string w =
      "text stage degraded: recognition dropped " + std::to_string(dropped) +
      " crop(s) (engine output exceeded decode buffers)";
  out.text_warning = out.text_warning.empty() ? w : out.text_warning + "; " + w;
}

// The single combine step every pipeline path ends with: pair recognition
// output with its boxes, drop empty/below-kDropScore results, then apply the
// text-degraded guard. One implementation so the filter semantics can never
// drift between the cv::Mat, GpuImage and batch paths again.
inline void combine_recognition(OcrPipelineResult &out,
                                const std::vector<Box> &boxes,
                                std::vector<std::pair<std::string, float>> &rec_results) {
  out.results.reserve(out.results.size() + boxes.size());
  const std::size_t n = std::min(boxes.size(), rec_results.size());
  for (std::size_t i = 0; i < n; ++i) {
    if (rec_results[i].second < turbo_ocr::kDropScore) continue;
    if (rec_results[i].first.empty()) continue;
    out.results.push_back({
        .text = std::move(rec_results[i].first),
        .confidence = rec_results[i].second,
        .box = boxes[i],
    });
  }
  flag_text_degraded(out, boxes.size());
}

// Backend-independent table-region adjustment (kept out of the recognizers so
// the env knobs live in one place and backends receive an already-adjusted
// box):
//   TABLE_CROP_MODE=detunion — snap to the tight AABB of the det text boxes
//     inside the layout box (so the region can only tighten).
//   TABLE_CROP_MARGIN — expand by this fraction per side (default 0.03, the
//     measured best on the 117-table set; layout boxes tend to clip border
//     rows/cols and structure-TEDS is sensitive to missing edge cells).
Box adjust_table_region(const Box &in,
                        const std::vector<OCRResultItem> &results);

} // namespace turbo_ocr::pipeline::detail
