// CPU half of the PDF job: the sequential streamed-render driver and the
// InferFunc run_pdf_job overload. Compiled into BOTH servers (the GPU build
// exposes the CPU overload too); the GPU overload lives in pdf_job_gpu.cpp,
// page prep in pdf_job_pages.cpp, the sink store layer in pdf_job_sink.cpp.

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

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#endif
#include "pdf_job_internal.h"
#include "turbo_ocr/pipeline/reading_order_util.h"

namespace turbo_ocr::pipeline {

namespace detail {

// CPU streamed render. Sequential: decode PPM, run the InferFunc inline.
// mode==Ocr means we never visited prepopulate_pages, so resolved mode is
// pinned here. Geometric pages keep their layer-derived text and rescale
// point->pixel coords. Returns num_pages; reports unreadable rendered PPMs via
// `decode_failures` (a server-side fault). The callback body is fully wrapped
// in try/catch: it runs INSIDE render_streamed's poll loop on the thread that
// owns the still-joinable render thread, so an escaping throw would unwind past
// that std::thread and std::terminate the process.
int run_streamed_render_cpu(
    const server::InferFunc &infer, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, int dpi, bool want_layout,
    bool want_reading_order, pdf::PdfMode mode,
    std::vector<PdfPageResult> &page_results,
    const std::vector<uint8_t> &need_render, int &decode_failures,
    int &page_failures, PdfImageMode image_mode,
    const pdf::EncodeOptions &encode_opts,
    bool autorotate, const server::OrientFunc &orient_fn,
    bool want_tables, bool want_formulas,
    const std::function<std::string(PdfPageResult &, const cv::Mat &)>
        &render_page_markdown) {
  auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len, dpi,
      [&](int page_idx, const std::string &ppm_path) noexcept {
       try {
        if (mode != pdf::PdfMode::Ocr &&
            page_idx < static_cast<int>(need_render.size()) &&
            !need_render[static_cast<size_t>(page_idx)])
          return;

        cv::Mat img = render::PdfRenderer::decode_ppm(ppm_path);
        if (img.empty()) {
          TOCR_LOG_ERROR("Failed to decode PPM for page",
                         "route", "/ocr/pdf", "page", page_idx);
          ++decode_failures;
          return;
        }

        if (page_idx >= static_cast<int>(page_results.size()))
          page_results.resize(page_idx + 1);
        auto &pg = page_results[static_cast<size_t>(page_idx)];

        server::InferOptions inf_opts;
        inf_opts.want_layout = want_layout;
        inf_opts.want_reading_order = want_reading_order;
        inf_opts.want_tables = want_tables;
        inf_opts.want_formulas = want_formulas;
        if (mode == pdf::PdfMode::Ocr)
          pg.resolved_mode = pdf::PdfMode::Ocr;

        // OCR pages: de-rotate upright (autorotate=1) BEFORE encode + infer.
        // Geometric pages are born-digital/upright with pt-space text — skip.
        if (pg.resolved_mode != pdf::PdfMode::Geometric && autorotate && orient_fn) {
          int orient = orient_fn(img);
          if (orient) classification::rotate_upright(img, orient);
          pg.orientation_deg = orient;
        }

        if (image_mode == PdfImageMode::Inline)
          pg.encoded_image = pdf::encode_page_image(img, encode_opts);

        if (pg.resolved_mode == pdf::PdfMode::Geometric) {
          // Geometric NEVER OCRs prose: pg.results is the PDF text layer (empty
          // for an image-only page — mode=auto/ocr are the modes that OCR). We
          // still run layout + table/formula structure on the rendered image
          // when requested. NOTE (CPU vs GPU): the CPU InferFunc always runs
          // det/rec, so we call it for the structure and DISCARD its OCR text
          // (inf.results) to honor the no-OCR contract — the GPU path skips
          // det/rec entirely via run_layout_and_structure. Less efficient here,
          // but same output; fixing needs a want_text=false CPU structure mode.
          if (want_layout) {
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
          pg.effective_dpi = dpi;
          // pg.results is the pt-space text layer (empty for image pages).
          rescale_boxes_pt_to_px(pg.results, dpi);
          maybe_assign_reading_order(want_reading_order, pg.results, pg.layout,
                                     pg.reading_order);
        } else {
          auto inf = infer(img, inf_opts);
          move_pipeline_fields(pg, std::move(inf));
          pg.width = img.cols;
          pg.height = img.rows;
          pg.effective_dpi = dpi;
          for (auto &item : pg.results) item.source = "ocr";
        }
        // Per-page Markdown while the page bitmap is still alive (parity with
        // the GPU maybe_render_page_markdown call).
        if (render_page_markdown &&
            page_idx < static_cast<int>(page_results.size())) {
          auto &pg_md = page_results[static_cast<size_t>(page_idx)];
          pg_md.markdown = render_page_markdown(pg_md, img);
        }
       } catch (const std::exception &e) {
        ++page_failures;
        TOCR_LOG_ERROR("PDF page inference error", "route", "/ocr/pdf",
                       "page", page_idx, "error", std::string_view(e.what()));
       } catch (...) {
        ++page_failures;
        TOCR_LOG_ERROR("PDF page inference error (unknown)",
                       "route", "/ocr/pdf", "page", page_idx);
       }
      });
  return stream_handle.num_pages;
}

} // namespace detail

// CPU PDF job. Sequential page OCR via the synchronous InferFunc. AutoVerified
// is GPU-only, so the caller aliases it to Auto before invoking.
[[nodiscard]] PdfJobResult run_pdf_job(
    const server::InferFunc &infer, render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len, const PdfJobOptions &opts,
    const server::OrientFunc &orient_fn) {
  PdfJobResult job;

  pdf::PdfMode mode = opts.mode;
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
  try {
    bool any_need_render = (mode == pdf::PdfMode::Ocr) ||
        std::any_of(need_render.begin(), need_render.end(),
                    [](uint8_t v) { return v != 0; });
    if (any_need_render) {
      int num_pages = detail::run_streamed_render_cpu(
          infer, pdf_renderer, pdf_data, pdf_len, opts.dpi, opts.want_layout,
          opts.want_reading_order, mode, page_results, need_render,
          decode_failures, page_failures, opts.image_mode, opts.encode_opts,
          opts.autorotate, orient_fn, opts.want_tables, opts.want_formulas,
          opts.render_page_markdown);
      if (static_cast<int>(page_results.size()) < num_pages)
        page_results.resize(num_pages);
    }
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
