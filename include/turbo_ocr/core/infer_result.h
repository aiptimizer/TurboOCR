#pragma once

#include <string>
#include <vector>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/core/types.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/core/router_types.h"

namespace turbo_ocr::server {

/// Combined result of one inference: text OCR results + optional layout +
/// optional table/formula structure. Every image route on the unified
/// server (/ocr, /ocr/raw, /ocr/batch, /ocr/pixels) and the gRPC image RPCs
/// consume it through the shared InferFunc seam;
/// emit_infer_result_json serializes these keys. (The pre-merge GPU server's
/// separate image_routes.cpp emitter is gone.)
struct InferResult {
  std::vector<OCRResultItem>            results;
  std::vector<layout::LayoutBox>        layout;
  std::vector<int>                      reading_order;
  std::vector<router::TableResult>      tables;
  std::vector<router::FormulaResult>    formulas;
  bool                                  formula_degraded = false;
  std::string                           formula_warning;
  bool                                  table_degraded = false;
  std::string                           table_warning;
  // Detection found text regions but recognition produced nothing usable — the no-silent-
  // failure signal for the TEXT stage (the pipeline computes it via flag_text_degraded; it
  // was previously dropped here, so /ocr + CPU /ocr/raw returned a clean empty 200).
  bool                                  text_degraded = false;
  std::string                           text_warning;
};

/// Move an InferResult's fields into the equivalent OcrPipelineResult. The
/// single conversion site between the two shapes — the GPU infer lambda and
/// the JSON emitter below both go through it.
[[nodiscard]] inline pipeline::OcrPipelineResult
to_pipeline_result(InferResult &&inf) {
  pipeline::OcrPipelineResult out;
  out.results = std::move(inf.results);
  out.layout = std::move(inf.layout);
  out.reading_order = std::move(inf.reading_order);
  out.tables = std::move(inf.tables);
  out.formulas = std::move(inf.formulas);
  out.formula_degraded = inf.formula_degraded;
  out.formula_warning = std::move(inf.formula_warning);
  out.table_degraded = inf.table_degraded;
  out.table_warning = std::move(inf.table_warning);
  out.text_degraded = inf.text_degraded;
  out.text_warning = std::move(inf.text_warning);
  return out;
}

/// Inverse of to_pipeline_result: adopt a pipeline result on the InferFunc
/// boundary. Forwards every degradation signal — dropping one would make a
/// degraded backend look byte-identical to a clean empty result.
[[nodiscard]] inline InferResult
from_pipeline_result(pipeline::OcrPipelineResult &&out) {
  return InferResult{
      .results          = std::move(out.results),
      .layout           = std::move(out.layout),
      .reading_order    = std::move(out.reading_order),
      .tables           = std::move(out.tables),
      .formulas         = std::move(out.formulas),
      .formula_degraded = out.formula_degraded,
      .formula_warning  = std::move(out.formula_warning),
      .table_degraded   = out.table_degraded,
      .table_warning    = std::move(out.table_warning),
      .text_degraded    = out.text_degraded,
      .text_warning     = std::move(out.text_warning),
  };
}

// Serialize an InferResult, emitting `tables`/`formulas` (+ degraded signals)
// when present. Reuses the shared OcrPipelineResult emitter so the CPU `/ocr`
// + `/ocr/raw` responses are byte-identical to the GPU server's structure JSON.
// On a text-only result the structure vectors are empty and their keys are
// omitted — byte-identical to the legacy emit_results_json output.
[[nodiscard]] inline std::string
emit_infer_result_json(InferResult &inf, bool want_blocks) {
  if (inf.tables.empty() && inf.formulas.empty() && !inf.formula_degraded &&
      !inf.table_degraded && !inf.text_degraded) {
    return turbo_ocr::emit_results_json(inf.results, inf.layout,
                                        inf.reading_order, want_blocks);
  }
  auto out = to_pipeline_result(std::move(inf));
  return turbo_ocr::emit_pipeline_result_json(out, want_blocks);
}

} // namespace turbo_ocr::server
