// The PDF job: the sequential streamed-render driver and the run_pdf_job
// entry point over an InferFunc. Page prep (text-layer open, per-page mode
// resolution, the shared serializer) lives in pdf_job_pages.cpp.
//
// There is no second half. This was "the CPU half", opposite a pdf_job_gpu.cpp
// driving the CUDA dispatcher, with a pdf_job_sink.cpp store layer between
// them; all of that is gone, and so is the second server binary the banner
// used to name.

#include "turbo_ocr/pipeline/job/pdf_job.h"

#include "turbo_ocr/pdf/text/font_match.h"
#include "turbo_ocr/pdf/text/region_extract.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/analysis/classification/doc_orientation_common.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/core/types.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/analysis/layout/order/reading_order.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
// The two LEAVES this TU actually needs, not the server_types.h umbrella:
// InferResult (what the InferFunc returns) and InferOptions (what it takes).
// InferFunc / OrientFunc already arrive via pdf_job.h -> server/service_fns.h.
// The umbrella pulled Drogon, JsonCpp, the metrics registry and the whole
// validation layer into a TU that never sees an HTTP request — the exact
// coupling include/turbo_ocr/core/infer_options.h was split out to end.
#include "turbo_ocr/core/infer_result.h"
#include "turbo_ocr/core/infer_options.h"

#include "pdf_job_internal.h"
#include "turbo_ocr/pipeline/reading_order_util.h"

namespace turbo_ocr::pipeline {

namespace detail {

// Streamed render driver. Pages are decoded + OCR'd by a small worker pool
// (PdfJobOptions::page_workers / TURBO_PDF_PAGE_WORKERS; 1 = strictly
// sequential): each infer() takes its own pipeline-pool lease, so a
// multi-page document uses the idle replicas the old inline callback left on
// the table. mode==Ocr means we never visited prepopulate_pages, so resolved
// mode is pinned here. Geometric pages keep their layer-derived text and
// rescale point->pixel coords. Returns num_pages; reports unreadable rendered
// PPMs via `decode_failures` (a server-side fault). Every callback body is
// fully wrapped in try/catch: the enqueue callback runs INSIDE
// render_streamed's poll loop on the thread that owns the still-joinable
// render thread, so an escaping throw would unwind past that std::thread and
// std::terminate the process.
//
// `opts.mode` HERE IS THE RESOLVED MODE, not the client's requested one. This
// used to take a separate `pdf::PdfMode mode` parameter beside an `opts` that
// HAS a `.mode` member meaning something different, with only a comment keeping
// the two apart — and `opts.mode` was one token away and compiled clean. The
// failure that enables is concrete: a PDF whose text layer will not open is
// downgraded to Ocr, and an edit that read the requested mode would re-enter the
// geometric branch, skip OCR for prose, and return a 200 with empty `results` on
// every page. run_pdf_job now hands us ONE options value with `mode` already
// resolved, so the two cannot disagree by construction.
// PDF text-layer boxes arrive in points (the PDF unit, 72/inch); every box the
// rest of the pipeline handles is in the render's pixel space. This is the one
// conversion between them.
//
// It is the last survivor of a PdfPageSink store layer that the deleted CUDA
// dispatcher used to drive — nine helpers over a struct nothing constructs any
// more. The other eight went with it (see git history for
// src/pipeline/job/pdf_job_sink.cpp); this one had a real caller, right below.
void rescale_boxes_pt_to_px(std::vector<OCRResultItem> &results, int dpi) {
  const float pt_to_px = static_cast<float>(dpi) / 72.0f;
  for (auto &item : results)
    for (int k = 0; k < 4; ++k) {
      item.box[k][0] = static_cast<int>(std::round(item.box[k][0] * pt_to_px));
      item.box[k][1] = static_cast<int>(std::round(item.box[k][1] * pt_to_px));
    }
}

// Claim ONE finished page for PdfJobOptions::on_page_ready, at most once.
//
// The ledger (state.emitted) is what makes "at most once" true across the two
// places a page can become final: the render callback, and run_pdf_job for the
// pages that are never rasterised at all. Without it the streaming routes either
// double-send a page or — the failure this fixes — send none.
//
// SPLIT into claim (under the caller's state lock) + invoke (OUTSIDE it):
// on_page_ready ends in a wire write — gRPC's ServerWriter::Write blocks on
// the client's flow-control window — and the single emit_page used to run the
// callback while HOLDING state_mu, so one slow streaming client parked every
// page worker (each blocked on state_mu while holding a pipeline-pool lease)
// and the document serialized to the client's read rate. The COPY (not move)
// is still deliberate: the non-streaming routes read the same slot afterwards.
[[nodiscard]] std::optional<PdfPageResult>
claim_page_for_emit(PdfStreamRenderState &state, const PdfJobOptions &opts,
                    int page_idx) {
  if (!opts.on_page_ready) return std::nullopt;
  const auto i = static_cast<size_t>(page_idx);
  if (i >= state.page_results.size()) return std::nullopt;
  if (state.emitted.size() < state.page_results.size())
    state.emitted.resize(state.page_results.size(), 0);
  if (state.emitted[i]) return std::nullopt;
  state.emitted[i] = 1;
  return PdfPageResult(state.page_results[i]);
}

// Single-threaded convenience for the non-rasterised pages in run_pdf_job —
// no state_mu exists there, so claiming and invoking in one step is fine.
void emit_page(PdfStreamRenderState &state, const PdfJobOptions &opts,
               int page_idx) {
  if (auto pg = claim_page_for_emit(state, opts, page_idx))
    opts.on_page_ready(page_idx, std::move(*pg));
}

// The whole per-page body, shared by run_streamed_render_cpu's sequential path
// and its worker pool. It works on a LOCAL PdfPageResult moved out of (and back
// into) its slot under state_mu: another page's arrival can resize
// page_results, so a worker must never hold a reference into it across its own
// inference.
void process_rendered_page(const server::InferFunc &infer,
                           const PdfJobOptions &opts,
                           const server::OrientFunc &orient_fn,
                           pdf::PdfMode mode, PdfStreamRenderState &state,
                           std::mutex &state_mu, int page_idx,
                           const std::string &ppm_path) noexcept {
       try {
        cv::Mat img = render::PdfRenderer::decode_ppm(ppm_path);
        if (img.empty()) {
          TOCR_LOG_ERROR("Failed to decode PPM for page",
                         "route", "/ocr/pdf", "page", page_idx);
          std::lock_guard<std::mutex> lk(state_mu);
          ++state.decode_failures;
          return;
        }

        PdfPageResult pg;
        {
          std::lock_guard<std::mutex> lk(state_mu);
          if (page_idx >= static_cast<int>(state.page_results.size()))
            state.page_results.resize(page_idx + 1);
          pg = std::move(state.page_results[static_cast<size_t>(page_idx)]);
        }

        server::InferOptions inf_opts;
        inf_opts.want_layout = opts.want_layout;
        inf_opts.want_reading_order = opts.want_reading_order;
        inf_opts.want_tables = opts.want_tables;
        inf_opts.want_formulas = opts.want_formulas;
        if (mode == pdf::PdfMode::Ocr)
          pg.resolved_mode = pdf::PdfMode::Ocr;

        // OCR pages: de-rotate upright (autorotate=1) BEFORE encode + infer.
        // Geometric pages are born-digital/upright with pt-space text — skip.
        if (pg.resolved_mode != pdf::PdfMode::Geometric && opts.autorotate &&
            orient_fn) {
          int orient = orient_fn(img);
          if (orient) classification::rotate_upright(img, orient);
          pg.orientation_deg = orient;
        }

        if (opts.image_mode == PdfImageMode::Inline)
          pg.encoded_image = pdf::encode_page_image(img, opts.encode_opts);

        if (pg.resolved_mode == pdf::PdfMode::Geometric) {
          // Geometric NEVER OCRs prose: pg.results is the PDF text layer (empty
          // for an image-only page — mode=auto/ocr are the modes that OCR). We
          // still run layout + table/formula structure on the rendered image
          // when requested. NOTE (CPU vs GPU): the CPU InferFunc always runs
          // det/rec, so we call it for the structure and DISCARD its OCR text
          // (inf.results) to honor the no-OCR contract — the GPU path skips
          // det/rec entirely via run_layout_and_structure. Less efficient here,
          // but same output; fixing needs a want_text=false CPU structure mode.
          if (opts.want_layout) {
            auto inf = infer(img, inf_opts);
            pg.layout = std::move(inf.layout);
            // Structure + its degradation apply to geometric pages (the router
            // runs on the rendered image); the OCR'd inf.results is discarded.
            pg.tables = std::move(inf.tables);
            pg.formulas = std::move(inf.formulas);
            pg.formula_degraded = inf.formula_degraded;
            pg.formula_warning = std::move(inf.formula_warning);
            pg.table_degraded = inf.table_degraded;
            pg.table_warning = std::move(inf.table_warning);
          }
          pg.width = img.cols;
          pg.height = img.rows;
          pg.effective_dpi = opts.dpi;
          // pg.results is the pt-space text layer (empty for image pages).
          rescale_boxes_pt_to_px(pg.results, opts.dpi);
          maybe_assign_reading_order(opts.want_reading_order, pg.results,
                                     pg.layout, pg.reading_order);
        } else {
          auto inf = infer(img, inf_opts);
          move_pipeline_fields(pg, std::move(inf));
          pg.width = img.cols;
          pg.height = img.rows;
          pg.effective_dpi = opts.dpi;
          for (auto &item : pg.results) item.source = "ocr";
        }
        // The four raster-alive hooks, all on the `pg` reference bound above.
        // They used to re-test `page_idx < state.page_results.size()` and re-bind
        // the SAME element under four different names. The guard is
        // unconditionally true after the resize at the top of this callback, and
        // nothing here resizes state.page_results — which the code already
        // assumes, since `pg` is used freely across the infer() calls above. Two
        // readings of one invariant is worse than either: it sends the next
        // reader hunting for an invalidation that does not exist, and invites a
        // "fix" (a resize between the hooks) that would actually break `pg`.
        //
        // Per-page Markdown while the page bitmap is still alive (parity with
        // the GPU maybe_render_page_markdown call).
        if (opts.render_page_markdown)
          pg.markdown = opts.render_page_markdown(pg, img);
        // Fillable-field proposals — same reason as markdown: the detectors
        // need the raster, and this is the last point at which it is alive.
        if (opts.detect_page_fields)
          pg.fields = opts.detect_page_fields(pg, img);
        // The type each line is set in — same reason again: this is the last
        // point at which the raster the boxes refer to still exists.
        if (opts.want_line_styles) {
          pg.line_styles = pdf::measure_page_line_styles(pg.results, img);
          pg.font_match = pdf::match_page_family(pg.results, img, pg.line_styles);
        }
        // Figures lifted out of the raster, so they can be moved later.
        if (opts.want_movable_regions) {
          pg.region_images = pdf::extract_movable_regions(img, pg.layout);
          pg.rule_shapes = pdf::extract_rules(img);
          pg.block_shapes = pdf::extract_blocks(img);
        }
        // The page is FINAL here — every hook above has run and the raster is
        // about to go away — so the slot gets its content back and /ocr/stream
        // and gRPC RecognizeStream learn about it, both under state_mu.
        // Emitting is a COPY, not a move: the non-streaming routes read the
        // same slot out of `page_results` afterwards, so moving the page out
        // would answer /ocr/pdf with a blank one whenever a stream happened to
        // be attached.
        std::optional<PdfPageResult> to_emit;
        {
          std::lock_guard<std::mutex> lk(state_mu);
          state.page_results[static_cast<size_t>(page_idx)] = std::move(pg);
          to_emit = claim_page_for_emit(state, opts, page_idx);
        }
        // The wire write happens OUTSIDE state_mu (see claim_page_for_emit's
        // rationale); the ledger was already marked under the lock, so
        // exactly-once holds regardless of how long this blocks.
        if (to_emit) opts.on_page_ready(page_idx, std::move(*to_emit));
       } catch (const std::exception &e) {
        {
          std::lock_guard<std::mutex> lk(state_mu);
          ++state.page_failures;
        }
        if (opts.on_page_failed) opts.on_page_failed(page_idx);
        TOCR_LOG_ERROR("PDF page inference error", "route", "/ocr/pdf",
                       "page", page_idx, "error", std::string_view(e.what()));
       } catch (...) {
        {
          std::lock_guard<std::mutex> lk(state_mu);
          ++state.page_failures;
        }
        if (opts.on_page_failed) opts.on_page_failed(page_idx);
        TOCR_LOG_ERROR("PDF page inference error (unknown)",
                       "route", "/ocr/pdf", "page", page_idx);
       }
}

int run_streamed_render_cpu(const server::InferFunc &infer,
                            render::PdfRenderer &pdf_renderer,
                            const uint8_t *pdf_data, size_t pdf_len,
                            const PdfJobOptions &opts,
                            const server::OrientFunc &orient_fn,
                            PdfStreamRenderState &state) {
  const pdf::PdfMode mode = opts.mode; // resolved; see above

  // Guards every read/write of `state` (slots, counters, the emitted ledger)
  // once pages are processed by more than one worker. Exactly-once emission is
  // kept by claiming the ledger entry under this lock (claim_page_for_emit);
  // the on_page_ready wire write itself runs OUTSIDE it, so a slow consumer
  // never stalls the other page workers.
  std::mutex state_mu;

  // Per-page body (decode -> infer -> hooks -> emit) — process_rendered_page
  // above, shared verbatim by the sequential path and the worker pool.
  auto process_page = [&](int page_idx, const std::string &ppm_path) noexcept {
    process_rendered_page(infer, opts, orient_fn, mode, state, state_mu,
                          page_idx, ppm_path);
  };

  // A page that never rasterises (mode!=ocr, need_render=0) is skipped before
  // it costs a queue slot or a decode.
  auto page_needs_work = [&](int page_idx) {
    return !(mode != pdf::PdfMode::Ocr &&
             page_idx < static_cast<int>(state.need_render.size()) &&
             !state.need_render[static_cast<size_t>(page_idx)]);
  };

  const int workers = opts.page_workers > 0
                          ? opts.page_workers
                          : env::env_int("TURBO_PDF_PAGE_WORKERS", 3, 1, 64);
  if (workers <= 1) {
    auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len,
        opts.dpi, [&](int page_idx, const std::string &ppm_path) noexcept {
          if (page_needs_work(page_idx)) process_page(page_idx, ppm_path);
        });
    return stream_handle.num_pages;
  }

  // PAGE-PARALLEL. The renderer's poll thread only enqueues; `workers`
  // threads decode + infer concurrently, each infer taking its own
  // pipeline-pool lease — so pages use the idle replicas the sequential path
  // left on the table, and the pool's bounded acquire is the real
  // backpressure. The queue is bounded at 2x workers: a full queue blocks the
  // poll thread, not memory (the queue holds paths; each worker holds at most
  // one decoded raster).
  std::mutex q_mu;
  std::condition_variable q_cv, space_cv;
  std::deque<std::pair<int, std::string>> q;
  bool render_done = false;

  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers));
  for (int i = 0; i < workers; ++i) {
    pool.emplace_back([&] {
      for (;;) {
        std::pair<int, std::string> job;
        {
          std::unique_lock<std::mutex> lk(q_mu);
          q_cv.wait(lk, [&] { return !q.empty() || render_done; });
          if (q.empty()) return;
          job = std::move(q.front());
          q.pop_front();
        }
        space_cv.notify_one();
        process_page(job.first, job.second);
      }
    });
  }

  // Drain on EVERY path out of this function. A worker pool that is only
  // joined on the success path turns any render_streamed throw into
  // std::terminate ("terminate called without an active exception" — a
  // joinable std::thread destroyed), taking the whole server down for one
  // bad document. Found the hard way: the first /ocr/pdf on the NVIDIA box
  // threw in the renderer and killed the process instead of answering 500.
  auto drain = [&] {
    {
      std::lock_guard<std::mutex> lk(q_mu);
      render_done = true;
    }
    q_cv.notify_all();
    for (auto &t : pool)
      if (t.joinable()) t.join();
  };

  try {
    auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len,
        opts.dpi, [&](int page_idx, const std::string &ppm_path) noexcept {
         try {
          if (!page_needs_work(page_idx)) return;
          std::unique_lock<std::mutex> lk(q_mu);
          space_cv.wait(lk,
                        [&] { return q.size() < static_cast<size_t>(workers) * 2; });
          q.emplace_back(page_idx, ppm_path);
          q_cv.notify_one();
         } catch (...) {
          // Queueing cannot plausibly throw, but this callback runs under the
          // renderer's still-joinable poll thread (see the function comment) —
          // never let anything unwind through it.
         }
        });
    // Drain BEFORE stream_handle dies: it owns the tmpdir the queued PPM
    // paths point into, and a worker may still be decoding one.
    drain();
    return stream_handle.num_pages;
  } catch (...) {
    drain(); // no handle on this path: render_streamed threw before making one
    throw;   // surfaces to the route as a normal request failure
  }
}

} // namespace detail

// The PDF job. Sequential page OCR via the synchronous InferFunc.
//
// AUTO_VERIFIED IS RESOLVED HERE, not by the caller. The verified path
// cross-checked every OCR detection against the text layer and belonged to the
// CUDA orchestration; without it, auto_verified can only mean auto. Three of
// the four transports aliased it themselves and the fourth (HTTP /ocr/stream)
// did not — so the same `mode=auto_verified` request took a different path
// depending on which door it came through, and the one that did not alias ran
// pages tagged `resolved_mode: auto_verified` with nothing verified. Resolving
// it at the single point every transport funnels through is what makes the four
// agree by construction rather than by four people remembering.
[[nodiscard]] PdfJobResult run_pdf_job(
    const server::InferFunc &infer, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, const PdfJobOptions &opts,
    const server::OrientFunc &orient_fn) {
  PdfJobResult job;

  pdf::PdfMode mode = opts.mode;
  if (mode == pdf::PdfMode::AutoVerified) mode = pdf::PdfMode::Auto;
  std::unique_ptr<pdf::PdfDocument> pdf_doc;
  std::vector<pdf::PdfPageText> page_text_cache;
  open_pdf_for_text_layer(pdf_data, pdf_len, mode, pdf_doc, page_text_cache);

  std::vector<PdfPageResult> page_results;
  std::vector<uint8_t> need_render;

  if (mode != pdf::PdfMode::Ocr)
    prepopulate_pages(mode, opts.want_layout, page_text_cache, page_results,
                      need_render, /*any_need_render=*/nullptr,
                      opts.image_mode == PdfImageMode::Inline);

  int decode_failures = 0;
  int page_failures = 0;
  // The streaming ledger, shared with the render callback. See
  // PdfStreamRenderState::emitted: a page becomes final in one of three places,
  // and each of them has to be able to tell whether the others got there first.
  std::vector<uint8_t> emitted(page_results.size(), 0);
  detail::PdfStreamRenderState state{page_results, need_render, decode_failures,
                                     page_failures, emitted};
  try {
    bool any_need_render = (mode == pdf::PdfMode::Ocr) ||
        std::any_of(need_render.begin(), need_render.end(),
                    [](uint8_t v) { return v != 0; });
    // ONE options value carrying the RESOLVED mode. Copying PdfJobOptions once
    // per job is free next to a PDF render, and it removes the hazard of two
    // mode values being in scope inside the driver.
    PdfJobOptions eff = opts;
    eff.mode = mode;

    // A text-layer page in mode=auto is never rasterised — prepopulate_pages
    // already wrote everything it will ever have, and the render callback
    // returns immediately for it. It is final NOW, so a streaming client should
    // have it now rather than after every OCR'd page in the document has been
    // through the pipeline.
    for (size_t i = 0; i < need_render.size(); ++i)
      if (!need_render[i]) detail::emit_page(state, eff, static_cast<int>(i));

    if (any_need_render) {
      // `mode` (not opts.mode): open_pdf_for_text_layer above may have
      // downgraded it to Ocr when the text layer could not be opened.
      int num_pages = detail::run_streamed_render_cpu(
          infer, pdf_renderer, pdf_data, pdf_len, eff, orient_fn, state);
      if (static_cast<int>(page_results.size()) < num_pages)
        page_results.resize(num_pages);
    }

    // Anything the two passes above did not cover — most often a slot created
    // by the resize just now, when the PDF has more pages than were rendered.
    // A client that was told `pages: N` in the meta event must receive N page
    // events or its own loop never terminates, so an empty page is still a page.
    for (size_t i = 0; i < page_results.size(); ++i)
      detail::emit_page(state, eff, static_cast<int>(i));
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR("PDF render failed", "route", "/ocr/pdf",
                   "error", std::string_view(e.what()));
    job.status = PdfJobStatus::RenderFailed;
    return job;
  }

  if (page_results.empty()) { job.status = PdfJobStatus::EmptyPdf; return job; }

  if (decode_failures > 0) {
    job.status = PdfJobStatus::DecodeFailed;
    job.decode_failures = decode_failures;
    job.num_pages = static_cast<int>(page_results.size());
    return job;
  }

  if (page_failures > 0) {
    job.status = PdfJobStatus::PageFailed;
    job.page_failures = page_failures;
    job.num_pages = static_cast<int>(page_results.size());
    return job;
  }

  job.num_pages = static_cast<int>(page_results.size());
  job.pages = std::move(page_results);
  return job;
}

} // namespace turbo_ocr::pipeline
