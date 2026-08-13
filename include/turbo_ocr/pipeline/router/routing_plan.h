#pragma once

#include <cstdint>
#include <vector>

namespace turbo_ocr::router {

// Per-page result of CuaRouter::classify. POD-ish: trivially clearable,
// no allocations on the hot path once the inner vectors have been
// reserved by the owning thread-local CuaRouter instance.
struct RoutingPlan {
  // Indices into OcrPipeline `boxes` (det boxes) that should run through
  // the normal text recognition path.
  std::vector<int> text_indices;

  // Indices into OcrPipelineResult `layout` that route to the table /
  // formula pipelines respectively. Skip is recorded only for telemetry
  // / bench attribution — the det boxes inside a Skip region still flow
  // through the text path via `text_indices`.
  std::vector<int> table_layout_ids;
  std::vector<int> formula_layout_ids;
  std::vector<int> skip_layout_ids;

  // For every entry in `text_indices`, the layout cell it landed in, or
  // -1 if it was an orphan. Same length as `text_indices`.
  std::vector<int> text_to_layout_id;

  // Mask aligned to the full `boxes.size()` (NOT to text_indices). True
  // means the det box's center falls inside a Table or Formula region,
  // so the rec engine must not recognize it independently — the table
  // or formula pipeline will produce that text.
  std::vector<uint8_t> rec_suppress;

  void clear() noexcept;
};

} // namespace turbo_ocr::router
