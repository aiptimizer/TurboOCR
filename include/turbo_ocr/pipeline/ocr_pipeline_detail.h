#pragma once

// Device-free OCR result policy. It lives at the shared
// include/turbo_ocr/pipeline/ level and its .cpp is compiled into
// turbo_ocr_common, so the policy resolves exactly once in every configure. It
// used to be a private header inside the CUDA pipeline back when that directory
// was called "legacy", and being filed under a name that announced deletion is
// precisely why the unified pipeline copied it instead of including it.
//
// The CUDA-native orchestration that was the second consumer is gone;
// UnifiedOcrPipeline is now the only caller, and it calls THESE — the
// anonymous-namespace fork it carried (which had drifted: different warning text
// for the same condition, a different combine_recognition arity, and
// assign-instead-of-append on the degraded warning) is deleted, and so is the
// later, byte-identical copy of adjust_table_region that left the definition
// below with zero callers.
//
// DO NOT re-fork this. Generic policy is shared, never per pipeline — see the
// comment on combine_recognition below for what that cost last time. That a
// single orchestration remains is not a reason to inline these back into it:
// the fork appeared the last time this policy lived next to one caller.

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "turbo_ocr/core/types.h"        // OCRResultItem, Box, kDropScore
#include "turbo_ocr/pipeline/pipeline_result.h"  // OcrPipelineResult, finalize_deferred

namespace turbo_ocr::pipeline::detail {

// No-silent-failure guard for the base OCR/recognition stage. Detection found
// `num_boxes` text regions but recognition produced no usable text — flag the
// result degraded so the response can never be a clean empty 200 that looks
// identical to a genuinely text-free page. A page with zero detections is NOT
// degraded (correctly text-free). Mirrors the formula/table degraded contract.
// APPENDS. It must never assign: this fires *after* combine_recognition's
// under-return check, so assigning would replace a correct diagnosis
// ("N of M region(s) were not recognized") with a FALSE one — the crops did
// not decode empty, the recognizer never ran. That defect was live in both
// orchestrations; it is what tests/cpp/pipeline/test_pipeline_detail.cpp's
// "appends, never overwrites" case pins.
inline void flag_text_degraded(OcrPipelineResult &out, std::size_t num_boxes) {
  if (num_boxes > 0 && out.results.empty()) {
    out.text_degraded = true;
    const std::string w =
        "text stage degraded: detection found " + std::to_string(num_boxes) +
        " text region(s) but recognition produced no usable text "
        "(all crops decoded empty/blank; not a genuinely blank page)";
    out.text_warning = out.text_warning.empty() ? w : out.text_warning + "; " + w;
  }
}

// The generic per-stage degradation writer, shared by the text stage here and by
// the table/formula/layout stages. APPENDS, same reason. Previously forked three
// ways — a private copy in the unified pipeline, another in the CUDA pipeline's
// dispatch, and absent here — and both copies ASSIGNED.
inline void set_stage_degraded(bool &degraded, std::string &warning,
                               const char *stage, std::size_t failed,
                               std::size_t total, const char *why) {
  degraded = true;
  const std::string w = std::string(stage) + " stage degraded: " +
                        std::to_string(failed) + " of " + std::to_string(total) +
                        " region(s) " + why;
  warning = warning.empty() ? w : warning + "; " + w;
}

// Partial recognition drops (engine output exceeded the decode buffers, see
// PaddleRec::last_dropped_crops) surface as text_degraded even when the rest
// of the page decoded fine — a thinner page must never read as a clean one.
inline void flag_dropped_crops(OcrPipelineResult &out, int dropped) {
  if (dropped <= 0) return;
  out.text_degraded = true;
  // Wording is the GENERAL cause, not NVIDIA's. Every vendor reports drops
  // through IRecognizer::last_dropped_crops(): Intel's `if (!rec_ran) continue`,
  // Apple's `chunk_ok[ci] = 0`, AMD/NVIDIA's dropped_crops_. "engine output
  // exceeded decode buffers" described only the CUDA path.
  const std::string w =
      "text stage degraded: recognition dropped " + std::to_string(dropped) +
      " crop(s) (the recognizer failed on them; they are not blank)";
  out.text_warning = out.text_warning.empty() ? w : out.text_warning + "; " + w;
}

// The single combine step every pipeline path ends with: pair recognition
// output with its boxes, drop empty/below-kDropScore results, then apply the
// text-degraded guard. One implementation so the filter semantics can never
// drift between the cv::Mat, GpuImage and batch paths again.
// `dropped_crops` is DEFAULTED so the CUDA pipeline's existing 3-arg call sites
// keep compiling unchanged (they call flag_dropped_crops separately on the next
// line, which is idempotent — a second call with 0 returns immediately). Folding
// it in is what stops a future call site from forgetting the accounting, which
// is exactly how the unified pipeline lost it once already.
inline void combine_recognition(OcrPipelineResult &out,
                                const std::vector<Box> &boxes,
                                std::vector<std::pair<std::string, float>> &rec_results,
                                int dropped_crops = 0) {
  out.results.reserve(out.results.size() + boxes.size());
  const std::size_t n = std::min(boxes.size(), rec_results.size());
  // A recognizer that returns FEWER results than detection found boxes had its
  // tail SILENTLY truncated here, with flag_text_degraded firing only when the
  // whole page came back empty. The unified pipeline accounts for it; this path
  // (still built by the CUDA configure) did not, so the same partial failure
  // was a visible warning on one server and a silently thin page on the other.
  // Generic policy is shared, never per pipeline.
  //
  // This is the SIZE case only; a recognizer that pre-sizes its output and
  // leaves failed chunks empty returns the full length, which is what
  // flag_dropped_crops(rec_->last_dropped_crops()) covers at the call sites.
  if (n < boxes.size())
    set_stage_degraded(out.text_degraded, out.text_warning, "text",
                       boxes.size() - n, boxes.size(),
                       "were not recognized (recognizer under-returned: it "
                       "produced fewer results than detection found boxes)");
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
  flag_dropped_crops(out, dropped_crops);
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
