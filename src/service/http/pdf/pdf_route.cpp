#include "turbo_ocr/service/http/pdf_routes.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <format>

#include "turbo_ocr/base/log/logger.h"

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pipeline/job/pdf_job.h"
#include "simdutf.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/service/validation/request_gate.h"
#include "turbo_ocr/service/server/server_types.h"

using turbo_ocr::pipeline::PdfImageMode;
using turbo_ocr::pipeline::PdfJobOptions;
using turbo_ocr::pipeline::PdfJobResult;
using turbo_ocr::pipeline::PdfJobStatus;
using turbo_ocr::pipeline::PdfPageResult;

#include "pdf_internal.h"

namespace turbo_ocr::routes {
using namespace pdfdetail;

// (The register_pdf_route(PipelineDispatcher&, ...) overload lived here. It was
// dead AND unbuildable: PipelineDispatcher is only forward-declared — the type
// went with the CUDA pipeline — so the body called request_timeout_ms() on an
// incomplete type and invoked run_pdf_job(PipelineDispatcher&, ...), which
// pdf_job.h documents as deleted. Nothing registered it: server_main.cpp calls
// the InferFunc overload below, on every backend. Only the USE_CPU_ONLY=OFF
// configure ever compiled it, which is why no local build caught it.)

// --- CPU overload: sequential page OCR via InferFunc ---
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        const capability::CapabilityMask &loaded,
                        int max_pdf_pages,
                        server::OrientFunc orient_fn,
                        int default_dpi) {
  // Availability passed in (CPU: env-derived = what actually loaded), not
  // routing-derived — the CPU pipeline loads table/formula from env, not routing.
  //
  // ROUTE-LOCAL NARROWING: this path applies autorotate through `orient_fn`, so
  // a server that loaded the doc-orientation model but reached this registrar
  // without an OrientFunc genuinely cannot honour autorotate=1 here. Clearing
  // the bit makes the gate reject it with AUTOROTATE_DISABLED instead of
  // accepting the request and silently not rotating.
  capability::CapabilityMask pdf_loaded = loaded;
  if (!orient_fn)
    pdf_loaded.set(capability::CapabilityId::DocOrientation, false);

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &infer, &pdf_renderer, default_pdf_mode, pdf_loaded,
       max_pdf_pages, orient_fn, default_dpi](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    std::string decoded_buf;
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, decoded_buf, pdf_ptr, pdf_len, callback))
      return;

    PdfRequestParams p;
    if (!parse_pdf_request(req, callback, pdf_loaded, default_dpi,
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
    const bool want_searchable_pdf = p.want_searchable_pdf;
    const float min_confidence = p.min_confidence;
    const bool want_editable = p.want_editable;
    const bool want_movable = p.want_movable;
    const bool want_mark_regions = p.want_mark_regions;
    const bool want_fields = p.want_fields;
    const bool autorotate = p.opts.want_autorotate;
    const int dpi = p.dpi;
    const pdf::PdfMode req_mode = p.mode;
    const PdfImageMode image_mode = p.image_mode;
    const pdf::EncodeOptions encode_opts = p.encode_opts;

    auto pdf_buf = std::make_shared<std::string>(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &infer, &pdf_renderer, want_layout,
         want_reading_order, want_blocks, want_tables, want_formulas, want_text,
         dpi, want_markdown, md_as_pages, want_searchable_pdf, want_editable, want_movable, want_mark_regions, min_confidence,
         want_fields, req_mode, image_mode, encode_opts, max_pdf_pages,
         autorotate, orient_fn](server::DrogonCallback &cb) {
     // See GPU route: wrap the body so a post-render bad_alloc returns 500
     // instead of being swallowed by the WorkPool (client hang).
     server::run_with_error_handling(cb, "/ocr/pdf", [&] {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, max_pdf_pages, cb)) return;

      // auto_verified resolves to auto inside run_pdf_job — one place, so all
      // four transports agree. The response still reports the actually-resolved
      // per-page mode, so a client that asked for auto_verified is never told
      // verification happened when it did not.
      const pdf::PdfMode mode = req_mode;

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
      // Opt-in: the hook stays null unless asked, so the default request never
      // pays for the page-raster morphology the detectors do.
      if (want_fields)
        job_opts.detect_page_fields = make_pdf_page_field_detector();
      // Same bargain: measuring the typeface of every line is only worth doing
      // when something is going to be drawn in it.
      job_opts.want_line_styles = want_editable;
      job_opts.want_movable_regions = want_movable;

      auto job = pipeline::run_pdf_job(infer, pdf_renderer, pdf_data,
                                       pdf_len_local, job_opts, orient_fn);
      if (emit_job_error(job, cb)) return;

      if (want_markdown) {
        cb(emit_pdf_markdown_response(job.pages, md_as_pages));
        return;
      }

      if (want_searchable_pdf) {
        cb(emit_searchable_pdf_response(
            job.pages, pdf_data, pdf_len_local,
            SearchablePdfOptions{.min_confidence = min_confidence,
                                 .editable = want_editable,
                                 .movable = want_movable,
                                 .mark_regions = want_mark_regions}));
        return;
      }

      cb(server::json_response(emit_pdf_response(job.pages, dpi, want_blocks,
                                                  image_mode, encode_opts,
                                                  autorotate, want_fields)));
     });
    });
  }, {drogon::Post});
}


} // namespace turbo_ocr::routes
