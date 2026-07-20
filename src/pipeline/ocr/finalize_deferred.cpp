// finalize_deferred — drains OcrPipelineResult::pending: joins the deferred
// external-backend futures (VLM table/formula crops) under one shared
// deadline, with per-crop exception containment and degraded accounting.
// Declared in turbo_ocr/pipeline/pipeline_result.h next to the field it
// drains.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <utility>

#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/router/router_types.h"
#include "turbo_ocr/table/table_recognizer.h"

namespace turbo_ocr::pipeline {

namespace {

// Await + parse + assemble ONE deferred stage (formula OR table) under the
// shared finalize deadline, moving each parsed payload into `out_vec`. Both
// stages are byte-for-byte the same control flow, differing only in the result
// type and two label strings, so they live here once — the deadline honouring,
// no-silent-failure counting, and exception containment cannot drift apart.
//
// * Bounded total wait: wait_until(deadline) — NOT wait_for — means that once
//   the deadline has passed, every remaining future resolves instantly (ready
//   ones are still harvested; only genuinely-pending ones are tagged degraded),
//   so the whole stage can never block longer than the single shared backstop
//   and table crops never wait again behind already-resolved formula crops.
// * No future is left un-awaited or double-awaited: each is inspected exactly
//   once; a timed-out promise-backed future is safe to abandon (non-blocking
//   dtor) and its shared state is freed when the crop vector is cleared.
// * No silent failure: any region whose payload is empty — timeout, transport
//   failure ("" from the pool), a broken promise, or a throwing parser —
//   increments `degraded`, so a remote failure can never look byte-identical to
//   a page that genuinely had no table/formula.
// * Exception containment: get()/parse throws are caught per-crop (std or not),
//   so one bad future can never drop its siblings and nothing escapes to leave
//   `pending` un-cleared.
template <class Result, class MakeResult>
void finalize_stage(std::vector<PendingCrop> &crops,
                    const PendingExternal::AsyncParser &parse,
                    std::vector<Result> &out_vec,
                    const std::chrono::steady_clock::time_point deadline,
                    const long timeout_ms, const char *stage,
                    const char *artifact, bool &degraded_flag,
                    std::string &warning, MakeResult make_result) {
  out_vec.reserve(out_vec.size() + crops.size());
  std::size_t degraded = 0;
  for (auto &pc : crops) {
    std::string payload;
    try {
      if (pc.fut.wait_until(deadline) == std::future_status::ready) {
        payload = parse(pc.fut.get());  // moves the raw response into the parser
      } else {
        std::cerr << "[finalize_deferred] " << stage
                  << " future timed out after " << timeout_ms
                  << " ms (crop-pool worker stalled?); tagging degraded\n";
      }
    } catch (const std::exception &e) {
      std::cerr << "[finalize_deferred] " << stage
                << " future threw: " << e.what() << '\n';
    } catch (...) {
      // A parser / shared-state throw that is not std::exception-derived still
      // must not drop the sibling crops nor escape and leave pending un-cleared.
      std::cerr << "[finalize_deferred] " << stage
                << " future threw a non-standard exception; tagging degraded\n";
    }
    if (payload.empty()) ++degraded;
    out_vec.push_back(make_result(pc, std::move(payload)));
  }
  if (degraded > 0) {
    degraded_flag = true;
    warning = std::string(stage) + " stage degraded: " +
              std::to_string(degraded) + " of " + std::to_string(crops.size()) +
              " region(s) produced no " + artifact +
              " (async backend transport failure or empty response, not empty "
              "input)";
  }
}

} // namespace

// Await + parse + assemble deferred external (VLM) crops, OFF the GPU worker.
// See pipeline_result.h. No-op when out.pending is empty (sync path).
void finalize_deferred(OcrPipelineResult &out) {
  auto &pe = out.pending;
  // Backstop the future joins: the crop pool resolves every promise (success or
  // its own per-crop timeout), so this only fires if a pool worker itself died/
  // wedged. The promise-backed futures are safe to abandon (no blocking dtor),
  // so a timed-out crop is left empty -> tagged degraded rather than hanging the
  // GPU worker. Configurable via FINALIZE_DEFERRED_TIMEOUT_MS (default 120000).
  static const long kFinalizeTimeoutMs = [] {
    const char *e = std::getenv("FINALIZE_DEFERRED_TIMEOUT_MS");
    long v = e ? std::atol(e) : 120000;
    if (v <= 0) v = 120000;
    // Cap at REQUEST_TIMEOUT_MS (same env the server reads; default 60s,
    // 0 == unbounded so the cap is skipped). PRECISE GUARANTEE: this bounds
    // the finalize stage to one request-timeout window measured from FINALIZE
    // ENTRY — it is a relative backstop, not the request's absolute deadline
    // (pre-finalize pipeline time is not subtracted), so a wedged worker can
    // still hold this GPU worker up to REQUEST_TIMEOUT_MS after the client's
    // 504. Threading the request's absolute deadline through PendingExternal
    // would tighten that; in practice the crop pool's own per-crop timeouts
    // resolve the futures long before this backstop binds.
    const char *rt = std::getenv("REQUEST_TIMEOUT_MS");
    const long req = rt ? std::atol(rt) : 60000;
    if (req > 0 && v > req) v = req;
    return v;
  }();
  // ONE shared deadline across BOTH the formula and table joins — not a per-future
  // timeout. A per-future wait_for could block up to (M+N)×kFinalizeTimeoutMs,
  // blowing past the per-request cap the value above was clamped to; a single
  // wait_until(deadline) bounds the total backstop wait to kFinalizeTimeoutMs and
  // stops table crops from waiting again behind already-resolved formula crops.
  const auto finalize_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kFinalizeTimeoutMs);
  // The deferred (async) path is what the DEFAULT vlm backend uses. Mirror the
  // synchronous path's no-silent-failure contract: every region here was a
  // table/formula crop dispatched to the backend, so an empty parse result —
  // whether from an exhausted transport failure (the pool resolves the future
  // with "" on a 4xx/exhausted-retry) or a broken-promise throw — means the
  // stage produced nothing for a region it was asked to handle. Count those and
  // surface a degradation flag so a remote timeout can never look byte-identical
  // to a page that genuinely had no table/formula.
  if (pe.formula_parse && !pe.formula.empty()) {
    finalize_stage<router::FormulaResult>(
        pe.formula, pe.formula_parse, out.formulas, finalize_deadline,
        kFinalizeTimeoutMs, "formula", "LaTeX", out.formula_degraded,
        out.formula_warning, [](const PendingCrop &pc, std::string latex) {
          router::FormulaResult fr;
          fr.layout_id = pc.layout_id;
          fr.latex     = std::move(latex);
          fr.score     = pc.score;
          fr.box       = pc.box;
          return fr;
        });
  }
  if (pe.table_parse && !pe.table.empty()) {
    finalize_stage<router::TableResult>(
        pe.table, pe.table_parse, out.tables, finalize_deadline,
        kFinalizeTimeoutMs, "table", "HTML", out.table_degraded,
        out.table_warning, [](const PendingCrop &pc, std::string html) {
          router::TableResult tr;
          tr.layout_id = pc.layout_id;
          tr.html      = std::move(html);
          tr.score     = pc.score;
          tr.box       = pc.box;
          return tr;
        });
  }
  pe = PendingExternal{};
}

} // namespace turbo_ocr::pipeline
