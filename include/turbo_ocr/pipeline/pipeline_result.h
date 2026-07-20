#pragma once

#include <functional>
#include <future>
#include <string>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/router/router_types.h"

namespace turbo_ocr::pipeline {

// Async-decouple carrier (highly-async serving path). When a remote backend's
// run is deferred (defer_external + supports_async), the GPU pipeline worker
// submits crops non-blocking and stashes one future + its target metadata per
// region HERE, then returns immediately — freeing its GPU slot. finalize_
// deferred() awaits + parses + assembles OFF the GPU worker (on the work-pool
// thread). Empty on the synchronous path, so text-only / local-backend / gRPC
// / PDF / batch results are byte-identical.
struct PendingCrop {
  int                      layout_id = -1;
  turbo_ocr::Box           box{};
  float                    score = 0.0f;
  std::future<std::string> fut;   // raw endpoint response (pre-parse)
};
struct PendingExternal {
  // Self-contained parser snapshots captured at dispatch time, ON the GPU
  // worker while the recognizer is still alive (recognizer->async_result_
  // parser()). They copy only value-state (e.g. the parser enum) and hold NO
  // pointer back into the recognizer, so finalize_deferred() — which runs later
  // on the work-pool thread and can block on a slow VLM future — stays valid
  // even if the owning pipeline is recycled (its recognizers freed) meanwhile.
  // Storing the raw recognizer pointer here was a cross-thread use-after-free.
  using AsyncParser = std::function<std::string(const std::string &)>;
  AsyncParser formula_parse;
  AsyncParser table_parse;
  std::vector<PendingCrop> formula;
  std::vector<PendingCrop> table;
  [[nodiscard]] bool empty() const noexcept {
    return formula.empty() && table.empty();
  }
};

/// Bundles text OCR results + optional layout detections from a single
/// pipeline run. `layout` is empty when layout was not requested or the
/// pipeline has no layout model. `reading_order` is filled only when
/// reading-order assignment was requested. `tables` / `formulas` are
/// populated by the CUA router stages — empty by default so the
/// back-compat serializer emits a byte-identical response on text-only
/// pages.
struct OcrPipelineResult {
  std::vector<OCRResultItem>             results;
  std::vector<layout::LayoutBox>         layout;
  std::vector<int>                       reading_order;
  std::vector<router::TableResult>       tables;
  std::vector<router::FormulaResult>     formulas;
  PendingExternal                        pending;  // deferred async work (empty on sync path)

  // Set when a configured formula region failed to recognize (engine result
  // ok==false, e.g. sidecar RPC crash/timeout) — distinct from a region that
  // legitimately had no formula. Surfaced additively in the JSON response so a
  // degraded formula stage never looks byte-identical to a clean empty result.
  bool                                   formula_degraded = false;
  // Human-readable detail for the degradation (empty when not degraded).
  std::string                            formula_warning;

  // Same contract for the table stage: set when a configured table region
  // produced no structure (async backend transport failure / empty response,
  // or local decode that yielded empty HTML). Without this, a remote-VLM
  // timeout on the default async table backend would be byte-identical to a
  // page that simply had no table.
  bool                                   table_degraded = false;
  std::string                            table_warning;

  // Text/recognition stage degradation: set when detection found text regions
  // (boxes > 0) but recognition yielded no usable text (all crops decoded to
  // empty/blank). Without this, a recognition-stage failure on a text page is
  // byte-identical to a genuinely blank page — the no-silent-failure contract
  // that the formula/table stages already honour, extended to base OCR.
  bool                                   text_degraded = false;
  std::string                            text_warning;
};

// Await + parse + assemble any deferred external (VLM) work into out.tables /
// out.formulas, then clear out.pending. Called on the work-pool / HTTP thread
// (NOT a GPU pipeline worker) after the deferred run_with_layout returns. A
// no-op when out.pending is empty, so it is always safe to call unconditionally.
void finalize_deferred(OcrPipelineResult &out);

} // namespace turbo_ocr::pipeline
