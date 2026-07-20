#include "turbo_ocr/http/pdf_routes.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <format>

#include "turbo_ocr/common/log/logger.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#endif

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/common/serialization/serialization.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/markdown/markdown_export.h"
#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pdf/pdf_job.h"
#include "simdutf.h"
#include "turbo_ocr/common/env_utils.h"
#include "turbo_ocr/validation/request_gate.h"
#include "turbo_ocr/server/server_types.h"

using turbo_ocr::pipeline::PdfImageMode;
using turbo_ocr::pipeline::PdfJobOptions;
using turbo_ocr::pipeline::PdfJobResult;
using turbo_ocr::pipeline::PdfJobStatus;
using turbo_ocr::pipeline::PdfPageResult;

#include "pdf_internal.h"

namespace turbo_ocr::routes {
using namespace pdfdetail;

#ifndef USE_CPU_ONLY
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        bool table_available,
                        bool formula_available,
                        int default_dpi,
                        int max_pdf_pages,
                        bool doc_ori_available) {

  // Availability from the warmed pipeline (single source of truth), threaded
  // from main(); not re-derived from config.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &dispatcher, &pdf_renderer, default_pdf_mode, layout_available,
       table_avail, formula_avail,
       default_dpi, max_pdf_pages, doc_ori_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    // Extract PDF bytes (lightweight, on event loop)
    auto pdf_buf = std::make_shared<std::string>();
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, *pdf_buf, pdf_ptr, pdf_len, callback))
      return;

    PdfRequestParams p;
    if (!parse_pdf_request(req, callback, layout_available, table_avail,
                           formula_avail, doc_ori_available, default_dpi,
                           default_pdf_mode, /*allow_image_only=*/true, p))
      return;

    const bool layout_enabled = p.opts.want_layout;
    const bool want_reading_order = p.opts.want_reading_order;
    const bool want_blocks = p.opts.want_blocks;
    const bool want_tables = p.opts.want_tables;
    const bool want_formulas = p.opts.want_formulas;
    const bool want_text = p.opts.want_text;
    const bool want_markdown = p.want_markdown;
    const bool md_as_pages = p.md_as_pages;
    const bool autorotate = p.autorotate;
    const int dpi = p.dpi;
    const pdf::PdfMode req_mode = p.mode;
    const PdfImageMode image_mode = p.image_mode;
    const pdf::EncodeOptions encode_opts = p.encode_opts;

    // For raw body case, pdf_ptr points into req->body() — copy into pdf_buf
    if (pdf_buf->empty())
      pdf_buf->assign(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &dispatcher, &pdf_renderer, // req dropped: unused, kept the full request (raw body) resident alongside pdf_buf
         layout_enabled, want_reading_order, want_blocks, want_tables, want_formulas,
         want_text, want_markdown, md_as_pages,
         dpi, req_mode, image_mode,
         encode_opts, max_pdf_pages, autorotate](server::DrogonCallback &cb) {
     // Wrap the whole body: post-render work (emit_pdf_response's multi-GB
     // reserve under images=inline) can throw bad_alloc, which the WorkPool
     // worker would otherwise swallow — leaving the client hung with no
     // response. run_with_error_handling turns it into 500.
     server::run_with_error_handling(cb, "/ocr/pdf", [&] {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, max_pdf_pages, cb)) return;

      PdfJobOptions job_opts;
      job_opts.dpi = dpi;
      job_opts.mode = req_mode;
      job_opts.want_layout = layout_enabled;
      job_opts.want_reading_order = want_reading_order;
      job_opts.want_blocks = want_blocks;
      job_opts.want_tables = want_tables;
      job_opts.want_formulas = want_formulas;
      job_opts.want_text = want_text;
      job_opts.autorotate = autorotate;
      job_opts.image_mode = image_mode;
      job_opts.encode_opts = encode_opts;
      // Bound the per-page future join with the configured request deadline so a
      // wedged page can't hang the whole PDF request (same value the dispatcher
      // applies to single-image submits).
      job_opts.request_timeout_ms = dispatcher.request_timeout_ms();

      if (want_markdown)
        job_opts.render_page_markdown = make_pdf_page_markdown_renderer();

      auto job = pipeline::run_pdf_job(dispatcher, pdf_renderer, pdf_data,
                                       pdf_len_local, job_opts);
      if (emit_job_error(job, cb)) return;

      if (want_markdown) {
        cb(emit_pdf_markdown_response(job.pages, md_as_pages));
        return;
      }

      cb(server::json_response(emit_pdf_response(job.pages, dpi, want_blocks,
                                                  image_mode, encode_opts,
                                                  autorotate)));
     });
    });
  }, {drogon::Post});
}
#endif // !USE_CPU_ONLY

// --- CPU overload: sequential page OCR via InferFunc ---
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        bool table_available,
                        bool formula_available,
                        int max_pdf_pages,
                        server::OrientFunc orient_fn) {
  const bool doc_ori_available = static_cast<bool>(orient_fn);
  // Availability passed in (CPU: env-derived = what actually loaded), not
  // routing-derived — the CPU pipeline loads table/formula from env, not routing.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &infer, &pdf_renderer, default_pdf_mode, layout_available,
       table_avail, formula_avail,
       max_pdf_pages, orient_fn, doc_ori_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    std::string decoded_buf;
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, decoded_buf, pdf_ptr, pdf_len, callback))
      return;

    PdfRequestParams p;
    if (!parse_pdf_request(req, callback, layout_available, table_avail,
                           formula_avail, doc_ori_available, kCpuDefaultDpi,
                           default_pdf_mode, /*allow_image_only=*/false, p))
      return;

    const bool want_layout = p.opts.want_layout;
    const bool want_reading_order = p.opts.want_reading_order;
    const bool want_blocks = p.opts.want_blocks;
    const bool want_tables = p.opts.want_tables;
    const bool want_formulas = p.opts.want_formulas;
    const bool want_text = p.opts.want_text;
    const bool want_markdown = p.want_markdown;
    const bool md_as_pages = p.md_as_pages;
    const bool autorotate = p.autorotate;
    const int dpi = p.dpi;
    const pdf::PdfMode req_mode = p.mode;
    const PdfImageMode image_mode = p.image_mode;
    const pdf::EncodeOptions encode_opts = p.encode_opts;

    auto pdf_buf = std::make_shared<std::string>(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &infer, &pdf_renderer, want_layout,
         want_reading_order, want_blocks, want_tables, want_formulas, want_text,
         dpi, want_markdown, md_as_pages,
         req_mode, image_mode, encode_opts, max_pdf_pages,
         autorotate, orient_fn](server::DrogonCallback &cb) {
     // See GPU route: wrap the body so a post-render bad_alloc returns 500
     // instead of being swallowed by the WorkPool (client hang).
     server::run_with_error_handling(cb, "/ocr/pdf", [&] {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, max_pdf_pages, cb)) return;

      // CPU server runs sequentially; the GPU AutoVerified path cross-checks
      // every OCR detection against the text layer in parallel. Doing the same
      // on CPU would require an extra pdfium text_in_rect call per detection
      // per page, doubling latency on a single-thread pipeline. Honest
      // behavior: alias auto_verified to auto on CPU and emit the actually-
      // resolved per-page mode in the response, so clients who set
      // auto_verified get auto's text-layer fast-path without us claiming
      // verification we didn't perform.
      pdf::PdfMode mode = req_mode;
      if (mode == pdf::PdfMode::AutoVerified) mode = pdf::PdfMode::Auto;

      PdfJobOptions job_opts;
      job_opts.dpi = dpi;
      job_opts.mode = mode;
      job_opts.want_layout = want_layout;
      job_opts.want_reading_order = want_reading_order;
      job_opts.want_blocks = want_blocks;
      job_opts.want_tables = want_tables;
      job_opts.want_formulas = want_formulas;
      // Parity with the GPU handler: CPU run_pdf_job ignores want_text today
      // (text=0 is rejected at parse time on this build), but the option must
      // not silently drift the moment that changes.
      job_opts.want_text = want_text;
      job_opts.autorotate = autorotate;
      job_opts.image_mode = image_mode;
      job_opts.encode_opts = encode_opts;

      if (want_markdown)
        job_opts.render_page_markdown = make_pdf_page_markdown_renderer();

      auto job = pipeline::run_pdf_job(infer, pdf_renderer, pdf_data,
                                       pdf_len_local, job_opts, orient_fn);
      if (emit_job_error(job, cb)) return;

      if (want_markdown) {
        cb(emit_pdf_markdown_response(job.pages, md_as_pages));
        return;
      }

      cb(server::json_response(emit_pdf_response(job.pages, dpi, want_blocks,
                                                  image_mode, encode_opts,
                                                  autorotate)));
     });
    });
  }, {drogon::Post});
}


} // namespace turbo_ocr::routes
