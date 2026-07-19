#include "turbo_ocr/pipeline/ocr_pipeline.h"
#include <unordered_map>
#include "infer_one.h"
#include "ocr_pipeline_detail.h"
#include "recognizer_registry.h"
#include "turbo_ocr/classification/cls_options.h"
#include "turbo_ocr/common/backend_error.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/common/timing.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/formula/auto_cjk_formula.h"
#include "turbo_ocr/formula/formula_recognizer.h"
#include "turbo_ocr/layout/reading_order.h"
#include "turbo_ocr/router/cua_router.h"
#include "turbo_ocr/routing/routing_config.h"
#include "turbo_ocr/table/table_recognizer.h"
#include "turbo_ocr/table/cell_matcher.h"
#include "turbo_ocr/table/html_reconstruct.h"
#include "turbo_ocr/table/slanext_enc_split.h"
#include "turbo_ocr/table/table_types.h"
#include "turbo_ocr/engine/onnx_to_trt.h"

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
// Single fault taxonomy for the upload+detection stage, shared by the cv::Mat
// and GpuImage entry points. Returns true when the fault is attributable to a
// degenerate INPUT (genuinely no text -> empty result is the correct answer);
// throws InferenceError for everything else so a real fault — crucially OOM
// under VRAM pressure — is never a silent blank page. Callers run this only
// after abort_on_sticky_cuda_fault() has ruled out a poisoned context; the
// error is cleared either way so it does not poison subsequent requests.
[[nodiscard]] bool det_fault_is_degenerate_input(const turbo_ocr::CudaError &e,
                                                 int cols, int rows,
                                                 const char *where) {
  const cudaError_t err = cudaGetLastError();
  // InvalidPitchValue / InvalidConfiguration fire for dims no GPU op can
  // launch on (1x1 pages, corrupt-decoded Mats with broken pitch). The
  // generic InvalidValue is accepted as degenerate ONLY for genuinely tiny
  // inputs — on a full-size image it means a real launch bug and must be
  // loud, not an empty 200.
  const bool tiny = cols < 8 || rows < 8;
  const bool degenerate = err == cudaErrorInvalidPitchValue ||
                          err == cudaErrorInvalidConfiguration ||
                          (err == cudaErrorInvalidValue && tiny);
  if (degenerate) {
    TOCR_LOG_WARN_RL("degenerate input, returning empty result", "cols", cols,
                     "rows", rows, "where", where, "error", e.what(),
                     "cuda", cudaGetErrorString(err));
    return true;
  }
  TOCR_LOG_ERROR_RL("detection GPU fault, surfacing as inference error",
                    "cols", cols, "rows", rows, "where", where,
                    "error", e.what(), "cuda", cudaGetErrorString(err));
  throw turbo_ocr::InferenceError(std::string("detection GPU fault: ") + e.what());
}
} // namespace

void OcrPipeline::dispatch_router_(OcrPipelineResult &out,
                                   const GpuImage &gpu_img,
                                   const std::vector<Box> &boxes,
                                   PipelineTimer &timer,
                                   const routing::RequestRouting &routing,
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

  // Tables: dispatch every detected table region to the configured backend
  // (TABLE_BACKEND -> slanext|vlm) behind one ITableRecognizer. Crop
  // margin/detunion region adjustment stays here (backend-independent); per-
  // backend cell-fill + HTML assembly live inside run(). The recognizer
  // returns one TableResult per region in input order; we stamp layout_id.
  if (has_table) {
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
      if (futs.size() < tlids.size()) {
        out.table_degraded = true;
        out.table_warning =
            "table stage degraded: " + std::to_string(tlids.size() - futs.size()) +
            " of " + std::to_string(tlids.size()) +
            " region(s) were not dispatched (async submit returned no future: "
            "page D2H copy or backend transport failure, not empty input)";
      }
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
      if (degraded_tables > 0) {
        out.table_degraded = true;
        out.table_warning =
            "table stage degraded: " + std::to_string(degraded_tables) + " of " +
            std::to_string(tlids.size()) +
            " region(s) produced no HTML (structure decode failed or backend "
            "under-returned, not empty input)";
      }
    }
    timer.gpu_stop();
    CUDA_CHECK(cudaEventRecord(table_done_event_, table_stream_));
  }

  // Formulas: FormulaNet::run takes a Box list (sub-rects) and self-
  // syncs on `formula_stream_` per sub-batch. Wait on det_only_event_
  // before dispatch so gpu_img is safe to read.
  if (has_formula) {
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
      if (futs.size() < flids.size()) {
        out.formula_degraded = true;
        out.formula_warning =
            "formula stage degraded: " + std::to_string(flids.size() - futs.size()) +
            " of " + std::to_string(flids.size()) +
            " region(s) were not dispatched (async submit returned no future: "
            "page D2H copy or backend transport failure, not empty input)";
      }
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
      if (degraded_regions > 0) {
        out.formula_degraded = true;
        out.formula_warning =
            "formula stage degraded: " + std::to_string(degraded_regions) +
            " of " + std::to_string(flids.size()) +
            " region(s) produced no LaTeX (backend error or empty result, not "
            "empty input)";
      }
    }
    timer.gpu_stop();
    CUDA_CHECK(cudaEventRecord(formula_done_event_, formula_stream_));
  }

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

std::vector<OCRResultItem> OcrPipeline::run(const cv::Mat &img,
                                            cudaStream_t stream) {
  return run_with_layout(img, stream).results;
}

OcrPipelineResult OcrPipeline::run_with_layout(const cv::Mat &img,
                                               cudaStream_t stream,
                                               bool want_layout,
                                               bool want_reading_order,
                                               const routing::RequestRouting &routing,
                                               bool defer_external,
                                               bool want_tables,
                                               bool want_formulas) {
  UseGuard _ug{in_use_, "run_with_layout"};
  const bool layout_active = use_layout_ && want_layout;
  if (img.empty()) [[unlikely]] return OcrPipelineResult{};

  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  // Upload + detection wrapped: degenerate inputs (e.g. 1×1, corrupt-
  // decoded Mats with zero-aligned pitch) trip CUDA "invalid pitch" in
  // cudaMemcpy2DAsync or in the resize kernel. Reset the stream and
  // return an empty result instead of bubbling up a 500 — there is no
  // text to detect and the request shouldn't poison subsequent ones.
  GpuImage gpu_img;
  std::vector<Box> boxes;
  try {
    gpu_img = upload_image(img, stream, timer);
    timer.gpu_start("detection_inference");
    boxes = det_->run(gpu_img, img.rows, img.cols, stream);
    timer.gpu_stop();
  } catch (const turbo_ocr::CudaError &e) {
    // Surface any async fault, then: a STICKY fault poisons the context for every future
    // request, so fail fast and let the orchestrator restart a healthy pod (std::_Exit, not
    // std::abort — skip atexit/dtors that would issue poisoned CUDA calls and hang).
    cudaStreamSynchronize(stream);
    turbo_ocr::abort_on_sticky_cuda_fault("run_with_layout/upload+det");
    if (det_fault_is_degenerate_input(e, img.cols, img.rows,
                                      "run_with_layout/upload+det"))
      return OcrPipelineResult{};
  }

  // Sort boxes top-to-bottom, left-to-right (in-place)
  timer.cpu_start("box_postprocessing");
  sorted_boxes(boxes);
  timer.cpu_stop();

  // Optional layout detection — dispatched on a dedicated layout_stream_
  // that waits only on det (via det_only_event_), so layout TRT execute
  // overlaps with cls on `stream` AND with rec on `rec_stream_`. The
  // host-side decode happens in collect() at the very end of run().
  // collect() must only run when enqueue succeeded: on an enqueue failure
  // it would sync the stale d2h_event_ of the LAST successful execute and
  // hand this page the previous page's boxes.
  bool layout_enqueued = false;
  if (layout_active) {
    CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
    CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));
    timer.gpu_start("layout_enqueue");
    layout_enqueued = layout_->enqueue(gpu_img, img.rows, img.cols,
                                       layout_stream_);
    timer.gpu_stop();
  }

  // Optional angle classification (CLS_ALL_BOXES / vertical-only gate).
  classify_angles_(gpu_img, boxes, stream, &timer);

  // Recognition — launch on dedicated rec_stream_ so the caller's stream is
  // free for the next image's upload+detection (pipeline parallelism).
  // Record det_event_ on the caller's stream after det+cls, then make
  // rec_stream_ wait on it before launching recognition.
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));

  timer.gpu_start("recognition_inference");
  auto rec_results = rec_->run(gpu_img, boxes, rec_stream_);
  timer.gpu_stop();

  // Record rec_event_ so the NEXT run() can wait for this recognition to
  // finish before reusing the image buffer. Note: rec_->run() syncs
  // rec_stream_ internally (for D2H + CTC decode), so by the time we get
  // here rec_stream_ is idle and this event is immediately "done". The event
  // is still useful as a correctness guard and for future async recognition.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  // Combine (filter by drop_score, matching Python's behavior)
  OcrPipelineResult out;
  detail::combine_recognition(out, boxes, rec_results);
  detail::flag_dropped_crops(out, rec_->last_dropped_crops());

  // Layout collect waits on d2h_event_ recorded on layout_stream_. Because
  // layout and rec run on separate streams, total wall-clock is bounded by
  // max(layout, cls+rec); on typical pages rec dominates so the wait is a
  // no-op.
  if (layout_enqueued) {
    out.layout = layout_->collect();
  }

  // CUA router + table/formula dispatch. No-op on text-only pages (see
  // dispatch_router_'s short-circuits — plan 04 §7).
  dispatch_router_(out, gpu_img, boxes, timer, routing, defer_external,
                   want_tables, want_formulas);

  // Reading-order over layout regions, with synthetic XY-cut entries
  // for orphan results so unmatched detections (page numbers, headers
  // the layout model missed) land in their natural position instead of
  // trailing the entire document. Helper is shared with cpu_ocr_pipeline.
  if (want_reading_order && !out.layout.empty()) {
    turbo_ocr::assign_layout_ids(out.results, out.layout);
    out.reading_order =
        turbo_ocr::layout::assign_reading_order_for_results(out.results, out.layout);
  }

  timer.print_total();

  return out;
}

OcrPipelineResult OcrPipeline::run_layout_only(const cv::Mat &img,
                                                cudaStream_t stream) {
  UseGuard _ug{in_use_, "run_layout_only"};
  OcrPipelineResult out;
  // Fast no-op if layout isn't loaded: skip the upload entirely. Callers
  // in /ocr/pdf geometric/auto modes never need OCR text here — they fill
  // results from the PDFium text layer — so returning empty is correct.
  if (!use_layout_ || !layout_) return out;

  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  auto gpu_img = upload_image(img, stream, timer);

  // Layout runs on layout_stream_ for the same reason run_with_layout
  // does: overlap with whatever else the caller has in flight. We still
  // record det_only_event_ on `stream` so layout_stream_ waits for the
  // upload before reading the image buffer.
  CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));

  timer.gpu_start("layout_only");
  if (layout_->enqueue(gpu_img, img.rows, img.cols, layout_stream_))
    out.layout = layout_->collect();
  timer.gpu_stop();

  // Mirror the event bookkeeping of run_with_layout so the next
  // run_with_layout()/run_layout_only() call can sync correctly on its
  // turn. rec_event_ is recorded on rec_stream_ — we didn't touch
  // rec_stream_ at all, so record an already-completed event by pushing
  // a no-op into rec_stream_ after layout_stream_ is known to be done.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  timer.print_total();
  return out;
}

OcrPipelineResult OcrPipeline::run_layout_and_structure(
    const cv::Mat &img, cudaStream_t stream,
    std::vector<OCRResultItem> text_results, bool want_tables,
    bool want_formulas, const routing::RequestRouting &routing) {
  UseGuard _ug{in_use_, "run_layout_and_structure"};
  OcrPipelineResult out;
  out.results = std::move(text_results);
  // No layout model -> nothing to route on; return the text unchanged. The
  // router also no-ops without table/formula backends (dispatch_router_ bails).
  if (!use_layout_ || !layout_) return out;

  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  auto gpu_img = upload_image(img, stream, timer);

  // Record on `stream` so both layout_stream_ and the router's table/formula
  // streams (which wait on det_only_event_) see the upload complete before
  // reading gpu_img — mirrors run_layout_only / run_with_layout.
  CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));

  timer.gpu_start("layout_only");
  if (layout_->enqueue(gpu_img, img.rows, img.cols, layout_stream_))
    out.layout = layout_->collect();
  timer.gpu_stop();

  // The router classifies regions and fills table cells using the text boxes as
  // the "detections" — here the text-layer boxes. Same call the OCR path makes.
  std::vector<Box> boxes;
  boxes.reserve(out.results.size());
  for (const auto &r : out.results) boxes.push_back(r.box);

  dispatch_router_(out, gpu_img, boxes, timer, routing, /*defer_external=*/false,
                   want_tables, want_formulas);

  // Event bookkeeping parity: we never touched rec_stream_, so record an
  // already-complete rec_event_ for the next run()'s wait.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));
  timer.print_total();
  return out;
}

std::vector<OCRResultItem> OcrPipeline::run(GpuImage gpu_img,
                                            cudaStream_t stream) {
  return run_with_layout(gpu_img, stream).results;
}

OcrPipelineResult OcrPipeline::run_with_layout(GpuImage gpu_img,
                                               cudaStream_t stream,
                                               bool want_layout,
                                               bool want_reading_order,
                                               const routing::RequestRouting &routing,
                                               bool defer_external,
                                               bool want_tables,
                                               bool want_formulas) {
  UseGuard _ug{in_use_, "run_with_layout(GpuImage)"};
  const bool layout_active = use_layout_ && want_layout;
  PipelineTimer timer;
  timer.init(stream);
  timer.reset();

  // No image_upload stage — the image is already on the GPU.
  // Wait for any previous consumers that might still be reading their source
  // image. For caller-owned GpuImage this is a correctness guard only.
  wait_prior_readers_();

  // Detection — same fault taxonomy as the cv::Mat entry point: degenerate
  // input -> empty result, everything else loud.
  std::vector<Box> boxes;
  try {
    timer.gpu_start("detection_inference");
    boxes = det_->run(gpu_img, gpu_img.rows, gpu_img.cols, stream);
    timer.gpu_stop();
  } catch (const turbo_ocr::CudaError &e) {
    cudaStreamSynchronize(stream);
    turbo_ocr::abort_on_sticky_cuda_fault("run_with_layout(GpuImage)/det");
    if (det_fault_is_degenerate_input(e, gpu_img.cols, gpu_img.rows,
                                      "run_with_layout(GpuImage)/det"))
      return OcrPipelineResult{};
  }

  // Sort boxes top-to-bottom, left-to-right (in-place)
  timer.cpu_start("box_postprocessing");
  sorted_boxes(boxes);
  timer.cpu_stop();

  // Optional layout detection (see run(cv::Mat, stream) for rationale,
  // including why collect() is gated on the enqueue result).
  bool layout_enqueued = false;
  if (layout_active) {
    CUDA_CHECK(cudaEventRecord(det_only_event_, stream));
    CUDA_CHECK(cudaStreamWaitEvent(layout_stream_, det_only_event_, 0));
    timer.gpu_start("layout_enqueue");
    layout_enqueued = layout_->enqueue(gpu_img, gpu_img.rows, gpu_img.cols,
                                       layout_stream_);
    timer.gpu_stop();
  }

  // Optional angle classification (CLS_ALL_BOXES / vertical-only gate) —
  // was vertical-only here while the cv::Mat path honored CLS_ALL_BOXES;
  // the shared helper ends that drift.
  classify_angles_(gpu_img, boxes, stream, &timer);

  // Recognition — use det_event_ for det→rec stream handoff.
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));

  timer.gpu_start("recognition_inference");
  auto rec_results = rec_->run(gpu_img, boxes, rec_stream_);
  timer.gpu_stop();

  // Record rec_event_ for the next run() to wait on.
  CUDA_CHECK(cudaEventRecord(rec_event_, rec_stream_));

  // Combine (filter by drop_score)
  OcrPipelineResult out;
  detail::combine_recognition(out, boxes, rec_results);
  detail::flag_dropped_crops(out, rec_->last_dropped_crops());

  // Layout collect — see run(cv::Mat, stream) above.
  if (layout_enqueued) {
    out.layout = layout_->collect();
  }

  // CUA router + table/formula dispatch (text-only path bails inside).
  dispatch_router_(out, gpu_img, boxes, timer, routing, defer_external,
                   want_tables, want_formulas);

  // Reading-order — see run(...) above for the contract; helper handles
  // orphan results (missing layout match) via synthetic XY-cut entries.
  if (want_reading_order && !out.layout.empty()) {
    turbo_ocr::assign_layout_ids(out.results, out.layout);
    out.reading_order =
        turbo_ocr::layout::assign_reading_order_for_results(out.results, out.layout);
  }

  timer.print_total();

  return out;
}

std::vector<std::vector<OCRResultItem>> OcrPipeline::run_batch(
    const std::vector<cv::Mat> &imgs, cudaStream_t stream) {
  auto outs = run_batch_with_layout(imgs, stream,
                                    /*want_layout=*/false,
                                    /*want_reading_order=*/false);
  std::vector<std::vector<OCRResultItem>> results;
  results.reserve(outs.size());
  for (auto &out : outs)
    results.push_back(std::move(out.results));
  return results;
}

std::vector<OcrPipelineResult> OcrPipeline::run_batch_with_layout(
    const std::vector<cv::Mat> &imgs, cudaStream_t stream,
    bool want_layout, bool want_reading_order,
    bool want_tables, bool want_formulas) {
  if (imgs.empty())
    return {};

  // If only one image, just use single-image path
  if (imgs.size() == 1) {
    std::vector<OcrPipelineResult> single;
    single.push_back(run_with_layout(imgs[0], stream, want_layout,
                                     want_reading_order, /*routing=*/{},
                                     /*defer_external=*/false,
                                     want_tables, want_formulas));
    return single;
  }

  // Guard AFTER the n==1 delegation above — run_with_layout self-guards, and a
  // guard here would trip on that nested call.
  UseGuard _ug{in_use_, "run_batch_with_layout"};

  const int n = static_cast<int>(imgs.size());

  // Guard against exceeding pre-allocated batch buffer capacity.
  // Callers should chunk at kMaxBatchImages before calling this method.
  if (n > kMaxBatchImages) [[unlikely]] {
    TOCR_LOG_WARN_RL("run_batch over batch capacity, truncating", "requested", n,
                     "max", kMaxBatchImages);
  }
  const int batch_n = std::min(n, kMaxBatchImages);

  // Batch buffers and the shared pinned staging buffer are reused across
  // requests — same reuse contract as upload_image().
  wait_prior_readers_();

  // --- Phase 1: Upload all images to GPU, run batched detection + cls ---
  // We need all images alive on GPU simultaneously for batched recognition.
  struct PerImage {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int rows = 0, cols = 0;
    std::vector<Box> boxes;
  };
  std::vector<PerImage> per_img(batch_n);

  // Upload all images to GPU first
  for (int i = 0; i < batch_n; i++) {
    const auto &img = imgs[i];
    auto &pi = per_img[i];
    pi.rows = img.rows;
    pi.cols = img.cols;

    // Use pre-allocated GPU buffer (grow-only, avoids cudaMalloc per batch)
    auto &bbuf = batch_img_bufs_[i];
    if (img.rows > bbuf.cap_rows || img.cols > bbuf.cap_cols) [[unlikely]] {
      if (bbuf.d_buf) cudaFree(bbuf.d_buf);
      bbuf.d_buf = nullptr;
      // Zero cap before alloc so an OOM throw leaves {nullptr, cap 0}
      // (see upload_image) — never a stale-cap null buffer next request.
      bbuf.cap_rows = 0;
      bbuf.cap_cols = 0;
      CUDA_CHECK(cudaMallocPitch(&bbuf.d_buf, &bbuf.pitch, img.cols * 3, img.rows));
      bbuf.cap_rows = img.rows;
      bbuf.cap_cols = img.cols;
    }
    pi.d_buf = bbuf.d_buf;
    pi.pitch = bbuf.pitch;

    // Upload via the shared pinned staging buffer
    auto needed = static_cast<size_t>(img.rows) * img.step;
    if (needed > h_pinned_size_) [[unlikely]] {
      if (h_pinned_buf_) {
        cudaFreeHost(h_pinned_buf_);
        h_pinned_buf_ = nullptr;
      }
      // Zero before alloc — if cudaHostAlloc throws (pinned OOM) the buffer
      // stays {nullptr, size 0} so the next batch re-enters this realloc branch
      // instead of memcpy'ing into a null buffer under a stale (larger) size.
      h_pinned_size_ = 0;
      // Upload-only pinned buffer: CPU writes (memcpy) once and the GPU DMAs
      // it. Write-combined uncached memory is ~10-15% faster for this access
      // pattern (no read-back from CPU).
      CUDA_CHECK(cudaHostAlloc(&h_pinned_buf_, needed, cudaHostAllocWriteCombined));
      h_pinned_size_ = needed;
    }
    std::memcpy(h_pinned_buf_, img.data, needed);
    CUDA_CHECK(cudaMemcpy2DAsync(pi.d_buf, pi.pitch, h_pinned_buf_, img.step,
                                  img.cols * 3, img.rows,
                                  cudaMemcpyHostToDevice, stream));
    // Sync before next iteration: h_pinned_buf_ is shared and will be
    // overwritten, so the async copy must complete first.
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Batched detection across the whole batch (one TRT call per <=8-image
  // chunk via the batch-capable det profile) instead of batch-1 per image.
  // det run_batch resizes all images to the batch-max shape and maps boxes
  // back per image — ~40% cheaper det/image at batch 8 (measured). Only the
  // batch path uses this; single /ocr/raw still takes the batch-1 fast path.
  std::vector<std::vector<Box>> all_det_boxes(batch_n);
  {
    std::vector<GpuImage> gpu_imgs;
    std::vector<std::pair<int, int>> dims;
    gpu_imgs.reserve(batch_n);
    dims.reserve(batch_n);
    for (int i = 0; i < batch_n; i++) {
      gpu_imgs.push_back({per_img[i].d_buf, per_img[i].pitch,
                          per_img[i].rows, per_img[i].cols});
      dims.emplace_back(per_img[i].rows, per_img[i].cols);
    }
    try {
      all_det_boxes = det_->run_batch(gpu_imgs, dims, stream);
    } catch (const turbo_ocr::CudaError &e) {
      // A batched det fault cannot be attributed to one member image, so the
      // degenerate-input downgrade of the single paths does not apply: sticky
      // check, then loud for the whole batch (never a silent empty 200 x N).
      cudaStreamSynchronize(stream);
      turbo_ocr::abort_on_sticky_cuda_fault("run_batch_with_layout/det");
      cudaGetLastError();
      throw turbo_ocr::InferenceError(
          std::string("batched detection GPU fault: ") + e.what());
    }
  }

  // Assign detection results and run angle classification per-image
  // (CLS_ALL_BOXES / vertical-only gate, same policy as the single paths —
  // the batch path previously ignored CLS_ALL_BOXES entirely).
  for (int i = 0; i < batch_n; i++) {
    per_img[i].boxes = std::move(all_det_boxes[i]);
    sorted_boxes(per_img[i].boxes);
    GpuImage gpu_img{per_img[i].d_buf, per_img[i].pitch,
                     per_img[i].rows, per_img[i].cols};
    classify_angles_(gpu_img, per_img[i].boxes, stream, nullptr);
  }

  // --- Phase 2: Batched recognition across ALL images ---
  std::vector<PaddleRec::ImageCrops> image_crops(batch_n);
  for (int i = 0; i < batch_n; i++) {
    image_crops[i].img = GpuImage{per_img[i].d_buf, per_img[i].pitch,
                                  per_img[i].rows, per_img[i].cols};
    image_crops[i].boxes = std::move(per_img[i].boxes);
  }

  // Launch batched recognition on rec_stream_ (pipeline parallelism)
  CUDA_CHECK(cudaEventRecord(det_event_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(rec_stream_, det_event_, 0));
  auto all_rec_results = rec_->run_multi(image_crops, rec_stream_);
  // Note: rec_->run_multi() syncs rec_stream_ internally for D2H + CTC decode,
  // so no additional cudaStreamSynchronize needed here.

  // --- Phase 3: Combine results and filter by drop_score ---
  std::vector<OcrPipelineResult> all_results(batch_n);
  const auto &dropped = rec_->last_dropped_per_image();
  for (int i = 0; i < batch_n; i++) {
    detail::combine_recognition(all_results[i], image_crops[i].boxes,
                                all_rec_results[i]);
    if (static_cast<size_t>(i) < dropped.size())
      detail::flag_dropped_crops(all_results[i], dropped[i]);
  }

  // --- Phase 4 (opt-in): layout + CUA router (table/formula) per page ---
  // Engages only when the caller asked for layout AND the operator loaded
  // a layout model — text-only batches pay zero layout/router cost.
  if (want_layout && use_layout_ && layout_)
    run_batch_layout_stage_(image_crops, want_reading_order, stream,
                            all_results, want_tables, want_formulas);

  // No cleanup needed — batch_img_bufs_ are pre-allocated and reused

  return all_results;
}

void OcrPipeline::run_batch_layout_stage_(
    const std::vector<PaddleRec::ImageCrops> &image_crops,
    bool want_reading_order, cudaStream_t stream,
    std::vector<OcrPipelineResult> &outs,
    bool want_tables, bool want_formulas) {
  const int batch_n = static_cast<int>(image_crops.size());

  // All det/rec GPU work is host-synced by this point (run_multi syncs
  // internally), but table/formula streams gate on det_only_event_, so
  // record it on the caller's stream to give them a valid ordering point.
  CUDA_CHECK(cudaEventRecord(det_only_event_, stream));

  for (int i = 0; i < batch_n; i++) {
    const GpuImage &gpu_img = image_crops[i].img;
    // collect() only after a successful enqueue — on failure it would
    // sync the stale d2h_event_ of the last successful execute and hand
    // this page the PREVIOUS page's boxes (which the router would then
    // crop at wrong coordinates on the wrong image).
    if (!layout_->enqueue(gpu_img, gpu_img.rows, gpu_img.cols,
                          layout_stream_))
      continue;
    outs[i].layout = layout_->collect();
    PipelineTimer t;
    dispatch_router_(outs[i], gpu_img, image_crops[i].boxes, t,
                     /*routing=*/{}, /*defer_external=*/false,
                     want_tables, want_formulas);
  }

  // Reading order — same contract as run_with_layout: helper handles
  // orphan results (missing layout match) via synthetic XY-cut entries.
  if (!want_reading_order)
    return;
  for (auto &out : outs) {
    if (out.layout.empty()) continue;
    turbo_ocr::assign_layout_ids(out.results, out.layout);
    out.reading_order =
        turbo_ocr::layout::assign_reading_order_for_results(out.results,
                                                            out.layout);
  }
}
