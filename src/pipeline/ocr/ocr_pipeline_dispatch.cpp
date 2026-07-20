// dispatch_router_ and its table / formula halves: region classification via
// the CUA router, then per-stage dispatch to the configured recognizer with
// the shared no-silent-failure degraded accounting.

#include "turbo_ocr/pipeline/ocr/ocr_pipeline.h"
#include <unordered_map>
#include "infer_one.h"
#include "ocr_pipeline_detail.h"
#include "recognizer_registry.h"
#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/cuda/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/log/timing.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/formula/routing/auto_cjk_formula.h"
#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/router/cua_router.h"
#include "turbo_ocr/backend_routing/routing_config.h"
#include "turbo_ocr/table/table_recognizer.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"
#include "turbo_ocr/table/slanext/slanext_enc_split.h"
#include "turbo_ocr/table/table_types.h"
#include "turbo_ocr/engine/trt/onnx_to_trt.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>

#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::pipeline;
using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::GpuImage;
using turbo_ocr::PipelineTimer;
// is_vertical_box / sorted_boxes are called unqualified below; ADL resolves them
// to turbo_ocr:: from their Box / vector<Box> arguments, so no using-decl needed.
using turbo_ocr::detection::PaddleDet;
using turbo_ocr::classification::PaddleCls;
using turbo_ocr::recognition::PaddleRec;
using turbo_ocr::layout::PaddleLayout;
using turbo_ocr::pipeline::OcrPipelineResult;

using turbo_ocr::pipeline::detail::adjust_table_region;
namespace {
// Shared no-silent-failure accounting for the four under-return/empty-result
// guards below: flag the stage degraded and compose the uniform
// "<stage> stage degraded: N of M region(s) <why>" warning.
void set_stage_degraded(bool &degraded, std::string &warning, const char *stage,
                        std::size_t failed, std::size_t total,
                        const char *why) {
  degraded = true;
  warning = std::string(stage) + " stage degraded: " + std::to_string(failed) +
            " of " + std::to_string(total) + " region(s) " + why;
}
} // namespace

void OcrPipeline::dispatch_router_(OcrPipelineResult &out,
                                   const GpuImage &gpu_img,
                                   const std::vector<Box> &boxes,
                                   PipelineTimer &timer,
                                   const backend_routing::RequestRouting &routing,
                                   bool defer_external,
                                   bool want_tables,
                                   bool want_formulas) {
  // text-only short-circuits — every one of these MUST bail BEFORE any
  // new CUDA API call (plan 04 §7 invariants).
  if (!router_) return;
  if (out.layout.empty()) return;

  timer.cpu_start("router_classify");
  plan_.clear();
  router_->classify(boxes, out.layout, plan_);
  timer.cpu_stop();

  // Per-request override (Tier-A): pick the named registry backend for this
  // request's table/formula regions, else the route default. Pointer pick only
  // — no per-request model construction on the hot path.
  table::ITableRecognizer   *table_rec   = pick_table_recognizer_(routing.table);
  formula::IFormulaRecognizer *formula_rec = pick_formula_recognizer_(routing.formula);

  // Strict opt-in: a configured backend is necessary but not sufficient — the
  // request must explicitly ask (?tables=1 / ?formulas=1). Layout alone never
  // triggers them.
  const bool has_table   = want_tables   && !plan_.table_layout_ids.empty()   && table_rec;
  const bool has_formula = want_formulas && !plan_.formula_layout_ids.empty() && formula_rec;
  if (!has_table && !has_formula) return;          // routed-all-text bail

  // A CUDA fault in the table/formula stage is caught here so a recoverable one
  // self-heals: the detection guard above only recycles on a deadline overrun,
  // so without this every later request to this worker would re-hit the wedge.
  try {
  if (has_table) dispatch_tables_(out, gpu_img, timer, table_rec, defer_external);
  if (has_formula)
    dispatch_formulas_(out, gpu_img, timer, formula_rec, defer_external);

  } catch (const turbo_ocr::CudaError &) {
    // Sticky -> the context is poisoned, fail-fast for a pod restart (same
    // policy as the detection guard). Recoverable -> clear the error and flag
    // this worker's pipeline for rebuild so the next request self-heals; a
    // rebuild also drops any stale table/formula done-event from a throw that
    // skipped its cudaEventRecord. Re-throw so this request still fails loud.
    turbo_ocr::abort_on_sticky_cuda_fault("dispatch_router_ table/formula");
    cudaGetLastError();  // clear the recoverable error before the rebuild
    recycle_requested_.store(true, std::memory_order_relaxed);
    throw;
  }
}

// Tables: dispatch every detected table region to the configured backend
// (TABLE_BACKEND -> slanext|vlm) behind one ITableRecognizer. Crop
// margin/detunion region adjustment stays here (backend-independent); per-
// backend cell-fill + HTML assembly live inside run(). The recognizer
// returns one TableResult per region in input order; we stamp layout_id.
void OcrPipeline::dispatch_tables_(OcrPipelineResult &out,
                                   const GpuImage &gpu_img,
                                   PipelineTimer &timer,
                                   table::ITableRecognizer *table_rec,
                                   bool defer_external) {
  timer.gpu_start("table_dispatch");
  CUDA_CHECK(cudaStreamWaitEvent(table_stream_, det_only_event_, 0));
  std::vector<Box> tboxes;
  std::vector<int> tlids;
  tboxes.reserve(plan_.table_layout_ids.size());
  tlids.reserve(plan_.table_layout_ids.size());
  for (int lid : plan_.table_layout_ids) {
    if (lid < 0 || static_cast<std::size_t>(lid) >= out.layout.size()) continue;
    tlids.push_back(lid);
    tboxes.push_back(adjust_table_region(out.layout[lid].box, out.results));
  }
  if (defer_external && table_rec->supports_async()) {
    // Non-blocking submit on the GPU worker; await + OTSL->HTML happens in
    // finalize_deferred() off the worker. Local structure backends (SLANeXt)
    // report supports_async()==false and never reach here, so the
    // cell-fill-from-page_ocr path is unchanged for them.
    auto futs = table_rec->submit_async(gpu_img, tboxes, table_stream_);
    out.pending.table_parse = table_rec->async_result_parser();
    out.pending.table.reserve(futs.size());
    for (std::size_t i = 0; i < futs.size() && i < tlids.size(); ++i)
      out.pending.table.push_back(
          {tlids[i], tboxes[i], out.layout[tlids[i]].score, std::move(futs[i])});
    // Async submit can under-return (e.g. an empty vector on a transient
    // page-D2H failure): every dispatched region that got no future is then
    // silently dropped, and finalize_deferred (which keys off pe.table.size())
    // would never see it. Flag the gap loud here so the stage can't return a
    // clean 200 with no table — same no-silent-failure contract the sync path
    // enforces.
    if (futs.size() < tlids.size())
      set_stage_degraded(
          out.table_degraded, out.table_warning, "table",
          tlids.size() - futs.size(), tlids.size(),
          "were not dispatched (async submit returned no future: "
          "page D2H copy or backend transport failure, not empty input)");
  } else {
    // Local structure backends fill empty grid cells via per-cell crop OCR.
    table_rec->set_cell_recognizer(rec_.get());
    out.tables = table_rec->run(gpu_img, tboxes, out.results, table_stream_);
    std::size_t degraded_tables = 0;
    for (std::size_t i = 0; i < out.tables.size() && i < tlids.size(); ++i) {
      out.tables[i].layout_id = tlids[i];
      if (out.tables[i].html.empty()) ++degraded_tables;  // decode produced nothing
    }
    // Backend under-return: regions past out.tables.size() got NO result at
    // all — count them too, or a short return is a silent drop (the async
    // table path and the sync formula path already guard exactly this).
    if (out.tables.size() < tlids.size())
      degraded_tables += tlids.size() - out.tables.size();
    if (degraded_tables > 0)
      set_stage_degraded(
          out.table_degraded, out.table_warning, "table", degraded_tables,
          tlids.size(),
          "produced no HTML (structure decode failed or backend "
          "under-returned, not empty input)");
  }
  timer.gpu_stop();
  CUDA_CHECK(cudaEventRecord(table_done_event_, table_stream_));
}

// Formulas: FormulaNet::run takes a Box list (sub-rects) and self-
// syncs on `formula_stream_` per sub-batch. Wait on det_only_event_
// before dispatch so gpu_img is safe to read.
void OcrPipeline::dispatch_formulas_(OcrPipelineResult &out,
                                     const GpuImage &gpu_img,
                                     PipelineTimer &timer,
                                     formula::IFormulaRecognizer *formula_rec,
                                     bool defer_external) {
  CUDA_CHECK(cudaStreamWaitEvent(formula_stream_, det_only_event_, 0));

  // Per-page routing hint for a composite (auto) formula backend: is the
  // page's recognized TEXT substantially CJK? A CJK page routes all its
  // formulas to the Chinese-capable model. THRESHOLDED (>=3 CJK chars AND
  // >=1% of text) so a single stray CJK glyph from an OCR misrecognition on a
  // math-heavy EN page does NOT escalate the whole page — measured: EN
  // false-positives have 1 CJK in ~1700 chars (0.06%), real Chinese pages
  // 23-88%. Gated on wants_context_hint(): single-model backends discard the
  // hint, so we skip the O(page-text) scan for them entirely.
  if (formula_rec->wants_context_hint()) {
    int cjk = 0, total = 0;
    for (const auto &r : out.results) {
      const auto st = formula::cjk_stats(r.text);
      cjk += st.cjk;
      total += st.total;
    }
    const bool page_has_cjk = cjk >= 3 && total > 0 && cjk * 100 >= total;
    formula_rec->set_context_hint(page_has_cjk);
  }

  std::vector<Box> fboxes;
  std::vector<int> flids;
  fboxes.reserve(plan_.formula_layout_ids.size());
  flids.reserve(plan_.formula_layout_ids.size());
  for (int lid : plan_.formula_layout_ids) {
    if (lid >= 0 && static_cast<std::size_t>(lid) < out.layout.size()) {
      flids.push_back(lid);
      fboxes.push_back(out.layout[lid].box);
    }
  }
  timer.gpu_start("formula_dispatch");
  if (defer_external && formula_rec->supports_async()) {
    // Non-blocking submit on the GPU worker; await + LaTeX-parse happens in
    // finalize_deferred() off the worker. Local engines (FormulaNet, PP-
    // FormulaNet-S) report supports_async()==false and keep the sync path.
    auto futs = formula_rec->submit_async(gpu_img, fboxes, formula_stream_);
    out.pending.formula_parse = formula_rec->async_result_parser();
    out.pending.formula.reserve(futs.size());
    for (std::size_t i = 0; i < futs.size() && i < flids.size(); ++i) {
      const int lid = flids[i];
      out.pending.formula.push_back(
          {lid, out.layout[lid].box, out.layout[lid].score, std::move(futs[i])});
    }
    // Under-return guard (mirrors the sync path's eng_res.size()<flids.size()
    // check + the table branch above): a dropped region must degrade loud,
    // never silently vanish behind a clean 200.
    if (futs.size() < flids.size())
      set_stage_degraded(
          out.formula_degraded, out.formula_warning, "formula",
          flids.size() - futs.size(), flids.size(),
          "were not dispatched (async submit returned no future: "
          "page D2H copy or backend transport failure, not empty input)");
  } else {
    auto eng_res = formula_rec->run(gpu_img, fboxes, formula_stream_);
    out.formulas.reserve(eng_res.size());
    std::size_t degraded_regions = 0;
    for (std::size_t i = 0; i < eng_res.size() && i < flids.size(); ++i) {
      const int lid = flids[i];
      // A dispatched formula region that yields no LaTeX is degraded — whether
      // the backend flagged ok==false (sidecar RPC crash) or simply returned
      // empty (a VLM transport failure resolves to "" with ok left default).
      // These regions were classified as formula, so empty == recognition
      // failure, never a clean "no formula here". Mirrors the async
      // finalize_deferred check so sync and async degrade identically.
      if (!eng_res[i].ok || eng_res[i].latex.empty()) ++degraded_regions;
      router::FormulaResult fr;
      fr.layout_id = lid;
      fr.latex     = std::move(eng_res[i].latex);
      fr.score     = out.layout[lid].score;      // proxy until engine surfaces one
      fr.box       = out.layout[lid].box;
      out.formulas.push_back(std::move(fr));
    }
    // Backend under-returned (e.g. an empty vector on a transient page-D2H /
    // stream-sync failure): every dispatched region with no result is degraded.
    if (eng_res.size() < flids.size())
      degraded_regions += flids.size() - eng_res.size();
    if (degraded_regions > 0)
      set_stage_degraded(
          out.formula_degraded, out.formula_warning, "formula",
          degraded_regions, flids.size(),
          "produced no LaTeX (backend error or empty result, not "
          "empty input)");
  }
  timer.gpu_stop();
  CUDA_CHECK(cudaEventRecord(formula_done_event_, formula_stream_));
}
