// UnifiedOcrPipeline — router + table/formula dispatch.
//
// Everything downstream of detection that decides WHICH recognizer a region
// goes to, and drives the table and formula stages. Device-free: one
// DeviceQueue rather than a stream+event per modality, and no fault-recycle
// path — a backend surfaces a device fault through its own exception.
//
// Split out of unified_ocr_pipeline.cpp, which had reached 1000 lines — over the
// 900-line ceiling tools/checks/architecture.sh enforces. The seams were already
// named by the banner comments in that file; this is those seams made physical,
// not a new decomposition. All four TUs define members of the SAME class, so the
// header is unchanged and nothing outside this directory can tell the difference.

#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/backend/routing_config.h"
#include "turbo_ocr/analysis/formula/cjk_stats.h"                // cjk_stats
#include "turbo_ocr/base/log/logger.h"               // TOCR_LOG_*
#include "turbo_ocr/base/geometry/box.h"             // Box, sorted_boxes, is_vertical_box
#include "turbo_ocr/core/types.h"                    // OCRResultItem, kDropScore
#include "turbo_ocr/pipeline/ocr_pipeline_detail.h"    // THE shared OCR result policy
#include "turbo_ocr/core/router_types.h"             // TableResult / FormulaResult

namespace turbo_ocr::pipeline {


using ::turbo_ocr::Box;
using ::turbo_ocr::OCRResultItem;

// Shared OCR result policy — pipeline::detail::*. This used to be an
// anonymous-namespace COPY here, which drifted from the original: different
// warning text for the same condition, a different combine_recognition arity,
// and assign-instead-of-append (which silently replaced a correct diagnosis with
// a false one). One implementation now, in turbo_ocr_common. Generic policy is
// shared, never per pipeline.
//
// Only what this TU CALLS is named. flag_text_degraded/flag_dropped_crops are
// deliberately absent: combine_recognition calls them itself, so a using-decl
// for them here would be scaffolding that reads as a call site.
using turbo_ocr::pipeline::detail::adjust_table_region;
using turbo_ocr::pipeline::detail::set_stage_degraded;

void UnifiedOcrPipeline::dispatch_router_(
    OcrPipelineResult &out, const backend::ImageView &view,
    const std::vector<Box> &boxes, const RunFlags &flags,
    const backend_routing::RequestRouting &routing, bool defer_external) {
  if (!router_) return;
  if (out.layout.empty()) return;

  plan_.clear();
  router_->classify(boxes, out.layout, plan_);

  backend::ITableRecognizer *table_rec = pick_table_recognizer_(routing.table);
  backend::IFormulaRecognizer *formula_rec =
      pick_formula_recognizer_(routing.formula);

  const bool has_table =
      flags.tables && !plan_.table_layout_ids.empty() && table_rec;
  const bool has_formula =
      flags.formulas && !plan_.formula_layout_ids.empty() && formula_rec;
  if (!has_table && !has_formula) return;

  if (has_table) dispatch_tables_(out, view, table_rec, defer_external);
  if (has_formula) dispatch_formulas_(out, view, formula_rec, defer_external);
}

void UnifiedOcrPipeline::dispatch_tables_(OcrPipelineResult &out,
                                          const backend::ImageView &view,
                                          backend::ITableRecognizer *table_rec,
                                          bool defer_external) {
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
    auto futs = table_rec->submit_async(view, tboxes, *queue_);
    out.pending.table_parse = table_rec->async_result_parser();
    out.pending.table.reserve(futs.size());
    for (std::size_t i = 0; i < futs.size() && i < tlids.size(); ++i)
      out.pending.table.push_back(
          {tlids[i], tboxes[i], out.layout[tlids[i]].score, std::move(futs[i])});
    if (futs.size() < tlids.size())
      set_stage_degraded(
          out.table_degraded, out.table_warning, "table",
          tlids.size() - futs.size(), tlids.size(),
          "were not dispatched (async submit returned no future: page D2H copy "
          "or backend transport failure, not empty input)");
  } else {
    // Local structure backends fill empty grid cells via per-cell crop OCR.
    table_rec->set_cell_recognizer(rec_.get());
    out.tables = table_rec->run(view, tboxes, out.results, *queue_);
    std::size_t degraded_tables = 0;
    for (std::size_t i = 0; i < out.tables.size() && i < tlids.size(); ++i) {
      out.tables[i].layout_id = tlids[i];
      if (out.tables[i].html.empty()) ++degraded_tables;
    }
    if (out.tables.size() < tlids.size())
      degraded_tables += tlids.size() - out.tables.size();
    if (degraded_tables > 0)
      set_stage_degraded(
          out.table_degraded, out.table_warning, "table", degraded_tables,
          tlids.size(),
          "produced no HTML (structure decode failed or backend under-returned, "
          "not empty input)");
  }
}

void UnifiedOcrPipeline::dispatch_formulas_(
    OcrPipelineResult &out, const backend::ImageView &view,
    backend::IFormulaRecognizer *formula_rec, bool defer_external) {
  // Per-page CJK routing hint for a composite (auto) formula backend. Gated on
  // wants_context_hint() so single-model backends skip the O(page-text) scan.
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
  // PP-FormulaNet is MARGIN-SENSITIVE: it was trained on crops with
  // whitespace context, and an exactly-tight layout box pushes its AR decoder
  // out of distribution — tokens scramble ("erg }cdot\mathrmmathrm{ K K}"
  // for a region that decodes to a clean "\boxed{10^{-16}\mathrm{erg\cdot
  // K}^{-1}}" with a 4 px margin; measured on omnidocbench ...60403612_179).
  // Expand HERE, in the one dispatch every backend's recognizer shares, so
  // CPU/CUDA/HIP/Metal all see the same margined region; the boxes REPORTED
  // in the response stay the layout model's own (out.layout[lid].box below).
  // Too much margin pulls neighbouring glyphs in (8 px started emitting
  // \begin{array} noise), so the default stays small; FORMULA_CROP_PAD tunes.
  static const int formula_pad = env::env_int("FORMULA_CROP_PAD", 4, 0, 64);
  const auto pad_box = [&](const Box &b) {
    auto r = aabb(b);  // [x0, y0, x1, y1], rotation-safe
    const int x0 = std::max(0, r[0] - formula_pad);
    const int y0 = std::max(0, r[1] - formula_pad);
    const int x1 = std::min(view.cols - 1, r[2] + formula_pad);
    const int y1 = std::min(view.rows - 1, r[3] + formula_pad);
    Box p;
    p.pts = {{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}};
    return p;
  };
  for (int lid : plan_.formula_layout_ids) {
    if (lid >= 0 && static_cast<std::size_t>(lid) < out.layout.size()) {
      flids.push_back(lid);
      fboxes.push_back(pad_box(out.layout[lid].box));
    }
  }
  if (defer_external && formula_rec->supports_async()) {
    auto futs = formula_rec->submit_async(view, fboxes, *queue_);
    out.pending.formula_parse = formula_rec->async_result_parser();
    out.pending.formula.reserve(futs.size());
    for (std::size_t i = 0; i < futs.size() && i < flids.size(); ++i) {
      const int lid = flids[i];
      out.pending.formula.push_back(
          {lid, out.layout[lid].box, out.layout[lid].score, std::move(futs[i])});
    }
    if (futs.size() < flids.size())
      set_stage_degraded(
          out.formula_degraded, out.formula_warning, "formula",
          flids.size() - futs.size(), flids.size(),
          "were not dispatched (async submit returned no future: page D2H copy "
          "or backend transport failure, not empty input)");
  } else {
    auto eng_res = formula_rec->run(view, fboxes, *queue_);
    out.formulas.reserve(eng_res.size());
    std::size_t degraded_regions = 0;
    for (std::size_t i = 0; i < eng_res.size() && i < flids.size(); ++i) {
      const int lid = flids[i];
      if (!eng_res[i].ok || eng_res[i].latex.empty()) ++degraded_regions;
      router::FormulaResult fr;
      fr.layout_id = lid;
      fr.latex = std::move(eng_res[i].latex);
      fr.score = out.layout[lid].score;
      fr.box = out.layout[lid].box;
      out.formulas.push_back(std::move(fr));
    }
    if (eng_res.size() < flids.size())
      degraded_regions += flids.size() - eng_res.size();
    if (degraded_regions > 0)
      set_stage_degraded(
          out.formula_degraded, out.formula_warning, "formula", degraded_regions,
          flids.size(),
          "produced no LaTeX (backend error or empty result, not empty input)");
  }
}

backend::ITableRecognizer *
UnifiedOcrPipeline::pick_table_recognizer_(const std::string &name) const {
  if (!name.empty()) {
    auto it = table_registry_.find(name);
    if (it != table_registry_.end()) return it->second.get();
  }
  return table_recognizer_;
}

backend::IFormulaRecognizer *
UnifiedOcrPipeline::pick_formula_recognizer_(const std::string &name) const {
  if (!name.empty()) {
    auto it = formula_registry_.find(name);
    if (it != formula_registry_.end()) return it->second.get();
  }
  return formula_;
}


} // namespace turbo_ocr::pipeline
