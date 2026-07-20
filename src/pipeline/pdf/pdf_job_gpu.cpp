// GPU half of the PDF job: dispatcher-submitted page tasks, the streamed
// render driver, and the GPU run_pdf_job overload. GPU target only — this TU
// is not compiled into turboocr-cpu-server.

#include "turbo_ocr/pipeline/pdf/pdf_job.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/classification/doc_orientation_common.h"
#include "turbo_ocr/common/log/logger.h"
#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/layout/order/reading_order.h"
#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"

#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#include "pdf_job_internal.h"
#include "turbo_ocr/pipeline/reading_order_util.h"

namespace turbo_ocr::pipeline {

namespace detail {

// Geometric page WITH structure requested: keep the exact PDF text layer but
// recognize tables (→ HTML) and formulas (→ LaTeX) on the rendered image. The
// slot already holds the pt-space text-layer results (prepopulate_pages); we
// rescale them to this render's pixel space, run layout + the CUA router over
// them (no det/rec), and write layout/tables/formulas back alongside the layer
// text. This is what makes born-digital STEM PDFs export exact prose AND
// structured math/tables.
void store_geometric_structure_page(PdfPageSink &sink, int page_idx,
                                           GpuPipelineEntry &e,
                                           const cv::Mat &img,
                                           bool want_reading_order,
                                           std::vector<uint8_t> encoded_image) {
  std::vector<OCRResultItem> px_text;
  {
    std::lock_guard<std::mutex> lk(sink.results_mutex);
    px_text = sink.page_results[static_cast<size_t>(page_idx)].results;
  }
  rescale_boxes_pt_to_px(px_text, sink.dpi);

  auto out = e.pipeline->run_layout_and_structure(
      img, e.stream, std::move(px_text), sink.want_tables, sink.want_formulas);
  for (auto &it : out.results) it.source = "pdf";
  maybe_assign_reading_order(want_reading_order, out.results, out.layout,
                             out.reading_order);

  std::lock_guard<std::mutex> lk(sink.results_mutex);
  auto &slot = sink.page_results[static_cast<size_t>(page_idx)];
  move_pipeline_fields(slot, std::move(out));
  slot.width = img.cols;
  slot.height = img.rows;
  slot.effective_dpi = sink.dpi;
  slot.encoded_image = std::move(encoded_image);
  // resolved_mode stays Geometric (set by prepopulate_pages).
}

// Single-page GPU task: decode the PPM and run layout-only (Geometric) or the
// full pipeline (everything else). Runs on a dispatcher worker.
void ocr_single_page(GpuPipelineEntry &e, PdfPageSink &sink,
                            bool layout_enabled, bool want_reading_order,
                            int page_idx, const std::string &path) {
  cv::Mat img = render::PdfRenderer::decode_ppm(path);
  if (img.empty()) {
    TOCR_LOG_ERROR("Failed to decode PPM for page",
                   "route", "/ocr/pdf", "page", page_idx);
    sink.decode_failures.fetch_add(1, std::memory_order_relaxed);
    if (sink.on_page_failed) sink.on_page_failed(page_idx);
    return;
  }

  try {
    // Geometric mode NEVER runs prose OCR: the page text comes from the PDF's
    // embedded text layer. An image-only page has no layer, so its prose comes
    // back empty — by design. mode=auto is the mode that OCRs such pages;
    // mode=ocr always OCRs. Tables/formulas are still recognized on the rendered
    // image when requested (they are vision-recognized for born-digital pages
    // too — there is no "table text layer"), keeping the exact prose layer.
    const bool structure_requested = sink.want_tables || sink.want_formulas;
    if (page_mode_of(sink, page_idx) == pdf::PdfMode::Geometric) {
      // Geometric pages are born-digital/upright with pt-space text boxes;
      // autorotate does not apply (rotating would desync the boxes).
      auto encoded = maybe_encode_page(sink, img);
      if (structure_requested && layout_enabled) {
        store_geometric_structure_page(sink, page_idx, e, img,
                                       want_reading_order, std::move(encoded));
      } else {
        auto layout = layout_enabled
            ? e.pipeline->run_layout_only(img, e.stream).layout
            : std::vector<layout::LayoutBox>{};
        store_geometric_page(sink, page_idx, std::move(layout),
                             img.cols, img.rows, want_reading_order,
                             std::move(encoded));
      }
      maybe_render_page_markdown(sink, page_idx, img);
      if (sink.on_page_ready) sink.emit_page_ready(page_idx);
    } else {
      // OCR page: de-rotate upright FIRST (autorotate=1) so det/rec, the boxes,
      // and the encoded image all share one upright frame.
      int orient = sink.autorotate
          ? e.pipeline->detect_orientation(img, e.stream) : 0;
      if (orient) classification::rotate_upright(img, orient);
      auto encoded = maybe_encode_page(sink, img);
      // ?text=0: layout-only (or image-only when layout is off) — no det/rec.
      auto out = sink.want_text
          ? e.pipeline->run_with_layout(img, e.stream, layout_enabled,
                                        want_reading_order, /*routing=*/{},
                                        /*defer_external=*/false,
                                        sink.want_tables, sink.want_formulas)
          : (layout_enabled ? e.pipeline->run_layout_only(img, e.stream)
                            : OcrPipelineResult{});
      store_ocr_page(sink, page_idx, std::move(out), img.cols, img.rows,
                     std::move(encoded), orient);
      maybe_render_page_markdown(sink, page_idx, img);
      if (sink.on_page_ready) sink.emit_page_ready(page_idx);
    }
  } catch (const std::exception &ex) {
    // Leave the slot default (empty); the orchestrator turns any page_failures
    // into a job-level PAGE_FAILED rather than a 200 with a blank page.
    sink.page_failures.fetch_add(1, std::memory_order_relaxed);
    if (sink.on_page_failed) sink.on_page_failed(page_idx);
    TOCR_LOG_ERROR("PDF page inference error", "route", "/ocr/pdf",
                   "page", page_idx, "error", std::string_view(ex.what()));
  }
}

// GPU streamed render: per rendered page, submit a single-page OCR task
// chunk or submit a single-page task directly onto the dispatcher (H3 — no
// per-page OS thread). `sink` and `stream_handle` are shared_ptrs captured BY
// VALUE into every task so an abandoned (timed-out) task safely outlives this
// call: the sink and the scratch tmpdir live until the last task drops them.
// Populates *stream_handle with the render result and returns its num_pages.
[[nodiscard]] int run_streamed_render_gpu(
    PipelineDispatcher &dispatcher, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, bool layout_enabled,
    bool want_reading_order, pdf::PdfMode mode,
    const std::shared_ptr<PdfPageSink> &sink,
    const std::shared_ptr<render::PdfRenderer::StreamHandle> &stream_handle,
    std::vector<uint8_t> &need_render,
    std::vector<std::future<void>> &page_futures, std::mutex &futures_mutex,
    std::vector<int> &dropped_pages) {
  auto handle = pdf_renderer.render_streamed(pdf_data, pdf_len, sink->dpi,
      [&](int page_idx, std::string ppm_path) {
        bool is_geometric = false;
        {
          std::lock_guard<std::mutex> rlock(sink->results_mutex);
          auto &page_results = sink->page_results;
          if (page_idx >= static_cast<int>(page_results.size())) {
            page_results.resize(page_idx + 1);
            if (mode != pdf::PdfMode::Ocr &&
                page_idx >= static_cast<int>(need_render.size()))
              need_render.resize(page_idx + 1, 1);
          }
          if (mode != pdf::PdfMode::Ocr &&
              page_idx < static_cast<int>(need_render.size()) &&
              !need_render[page_idx])
            return;
          is_geometric = mode != pdf::PdfMode::Ocr &&
              page_results[page_idx].resolved_mode == pdf::PdfMode::Geometric;
        }

        (void)is_geometric;
        std::future<void> fut;
        try {
          fut = dispatcher.submit(
              [sink, stream_handle, layout_enabled, want_reading_order, page_idx,
               path = std::move(ppm_path)](auto &e) {
                ocr_single_page(e, *sink, layout_enabled, want_reading_order,
                                page_idx, path);
              });
        } catch (const turbo_ocr::PoolExhaustedError &) {
          TOCR_LOG_WARN("GPU queue full, dropping page", "route", "/ocr/pdf",
                        "page", page_idx);
          dropped_pages.push_back(page_idx);
          return;
        }
        std::lock_guard lock(futures_mutex);
        page_futures.push_back(std::move(fut));
      });

  // Hand the tmpdir-owning handle to the shared object the tasks co-own; the
  // tmpdir now lives until the last task (including any abandoned one) drops it.
  *stream_handle = std::move(handle);
  return stream_handle->num_pages;
}

} // namespace detail

// GPU PDF job. Submits page work directly onto the dispatcher (H3). Page tasks
// co-own the sink + StreamHandle by shared_ptr, so a task abandoned when its
// future overruns the job-wide deadline (request_timeout_ms) is memory-safe:
// the shared state outlives this call until that task finishes. Backpressure is
// the dispatcher queue depth: PoolExhaustedError mid-stream sets status=Dropped
// (caller -> SERVER_BUSY).
[[nodiscard]] PdfJobResult run_pdf_job(
    PipelineDispatcher &dispatcher, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, const PdfJobOptions &opts) {
  PdfJobResult job;

  // The sink OWNS every piece of state a page task touches (results, mutex,
  // text cache, document), heap-allocated and co-owned by the tasks, so an
  // abandoned task never dereferences a dead run_pdf_job local.
  auto sink = std::make_shared<detail::PdfPageSink>();
  sink->dpi = opts.dpi;
  sink->image_mode = opts.image_mode;
  sink->encode_opts = opts.encode_opts;
  sink->autorotate = opts.autorotate;
  sink->want_tables = opts.want_tables;
  sink->want_formulas = opts.want_formulas;
  sink->want_text = opts.want_text;
  sink->on_page_ready = opts.on_page_ready;
  sink->on_page_failed = opts.on_page_failed;
  sink->render_page_markdown = opts.render_page_markdown;

  pdf::PdfMode mode = opts.mode;
  open_pdf_for_text_layer(pdf_data, pdf_len, mode, sink->pdf_doc,
                          sink->page_text_cache);

  std::vector<uint8_t> need_render;
  bool any_need_render = (mode == pdf::PdfMode::Ocr);

  if (mode != pdf::PdfMode::Ocr) {
    prepopulate_pages(mode, opts.want_layout, sink->page_text_cache,
                      sink->page_results, need_render, &any_need_render,
                      opts.image_mode == PdfImageMode::Inline);
    // ?text=0: the caller asked for no text at all — drop what the text
    // layer prefilled so geometric pages honor the contract too.
    if (!opts.want_text)
      for (auto &pg : sink->page_results) pg.results.clear();
    // Streaming: render-skipped pages (trusted text layer, no pixels needed)
    // are final right now — emit them before the render even starts.
    if (opts.on_page_ready)
      for (size_t p = 0; p < need_render.size(); ++p)
        if (!need_render[p]) sink->emit_page_ready(static_cast<int>(p));
  }

  std::vector<int> dropped;
  auto stream_handle = std::make_shared<render::PdfRenderer::StreamHandle>();

  std::mutex futures_mutex;
  std::vector<std::future<void>> page_futures;
  int num_pages = 0;

  if (any_need_render) {
    try {
      num_pages = detail::run_streamed_render_gpu(
          dispatcher, pdf_renderer, pdf_data, pdf_len, opts.want_layout,
          opts.want_reading_order, mode, sink, stream_handle, need_render,
          page_futures, futures_mutex, dropped);
    } catch (const std::exception &e) {
      // No drain needed: every in-flight page task co-owns the sink/handle via
      // shared_ptr, so abandoning the futures here is memory-safe (a still-running
      // task can't dereference a dead local). An unbounded f.get() drain would
      // instead let one wedged worker hang the whole render-failure response.
      TOCR_LOG_ERROR("PDF render failed", "route", "/ocr/pdf",
                     "error", std::string_view(e.what()));
      job.status = PdfJobStatus::RenderFailed;
      return job;
    }
  } else {
    num_pages = sink->pdf_doc ? sink->pdf_doc->page_count() : 0;
  }

  {
    std::lock_guard<std::mutex> rlock(sink->results_mutex);
    if (static_cast<int>(sink->page_results.size()) < num_pages)
      sink->page_results.resize(num_pages);
  }

  // Bounded join against ONE job-wide deadline: a worker wedged on a single page
  // can't hang the whole request. The deadline is SCALED by page count — each
  // page gets request_timeout_ms, the whole job num_pages * that — so a flat 60s
  // budget can't kill a big-but-healthy PDF. On a per-future throw the future is
  // abandoned (the task keeps its shared_ptr copies, so this is safe) and the page
  // is counted as failed; the catch also backstops any UNEXPECTED escape (the
  // worker bodies catch their own) so nothing slips through as a silently-empty
  // page in a 200. A pure deadline overrun (TimeoutError) is NOT an inference page
  // failure: it surfaces as a distinct 504 / DEADLINE_EXCEEDED.
  const long per_page_budget = opts.request_timeout_ms;  // 0 == unbounded
  const long long job_budget =
      per_page_budget > 0
          ? static_cast<long long>(per_page_budget) * std::max(1, num_pages)
          : 0;
  const auto job_deadline = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(job_budget);
  bool deadline_exceeded = false;
  for (auto &f : page_futures) {
    try {
      if (job_budget > 0) {
        const auto now = std::chrono::steady_clock::now();
        long long remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                job_deadline - now).count();
        if (remaining < 1) remaining = 1;  // already past: near-instant abandon
        get_with_timeout(f, static_cast<long>(remaining));
      } else {
        f.get();
      }
    } catch (const turbo_ocr::TimeoutError &) {
      // Job-wide deadline overrun, not a page-level inference failure. Abandon the
      // remaining futures (their tasks co-own the sink) and report a 504.
      deadline_exceeded = true;
      break;
    } catch (const std::exception &e) {
      sink->page_failures.fetch_add(1, std::memory_order_relaxed);
      TOCR_LOG_ERROR("PDF page error", "route", "/ocr/pdf",
                     "error", std::string_view(e.what()));
      continue;
    }
  }

  job.num_pages = num_pages;
  if (num_pages == 0) { job.status = PdfJobStatus::EmptyPdf; return job; }

  if (deadline_exceeded) { job.status = PdfJobStatus::TimedOut; return job; }

  if (!dropped.empty()) {
    job.status = PdfJobStatus::Dropped;
    job.dropped_pages = static_cast<int>(dropped.size());
    job.first_dropped = dropped.front();
    return job;
  }

  if (const int failed = sink->decode_failures.load(std::memory_order_relaxed);
      failed > 0) {
    job.status = PdfJobStatus::DecodeFailed;
    job.decode_failures = failed;
    return job;
  }

  if (const int failed = sink->page_failures.load(std::memory_order_relaxed);
      failed > 0) {
    job.status = PdfJobStatus::PageFailed;
    job.page_failures = failed;
    return job;
  }

  // num_pages may have grown the vector under page_futures completion; trim to
  // the renderer-reported count.
  {
    std::lock_guard<std::mutex> rlock(sink->results_mutex);
    job.pages.reserve(num_pages);
    for (int i = 0; i < num_pages && i < static_cast<int>(sink->page_results.size()); ++i)
      job.pages.push_back(std::move(sink->page_results[i]));
  }
  return job;
}

} // namespace turbo_ocr::pipeline
