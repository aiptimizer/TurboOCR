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

#ifndef USE_CPU_ONLY
namespace turbo_ocr::routes {
using namespace pdfdetail;

// --- /ocr/stream: NDJSON streaming — one endpoint for PDFs and images -------
//
// The response is application/x-ndjson: one JSON object per line, flushed as
// produced. Protocol:
//   {"event":"meta","kind":"pdf|image","pages":N,"dpi":D,"mode":"ocr|..."}
//   {"event":"page", <same shape as an /ocr/pdf pages[] element>}   (xN, AS
//        EACH PAGE COMPLETES — out of order; page_index identifies the page)
//   {"event":"page_error","page_index":i}                           (failures)
//   {"event":"error","code":"..."}                                  (job-level)
//   {"event":"end","pages":N,"failed":k}
// Errors detected BEFORE the first byte are normal HTTP 4xx; once streaming
// has begun they arrive as error events (the 200 status is already on the
// wire — chunked transfer has no second status line).


namespace {

// Serialized writer around Drogon's async stream: page events arrive
// concurrently from dispatcher workers, and trantor's AsyncStream must not be
// interleaved. send_line() returns false once the client is gone so producers
// can stop serializing.
struct NdjsonStream {
  std::mutex mu;
  drogon::ResponseStreamPtr stream;
  bool closed = false;

  bool send_line(std::string line) {
    line += '\n';
    std::lock_guard<std::mutex> lk(mu);
    if (closed || !stream) return false;
    if (!stream->send(line)) {
      closed = true;
      return false;
    }
    return true;
  }
  void finish() {
    std::lock_guard<std::mutex> lk(mu);
    if (stream && !closed) stream->close();
    closed = true;
  }
};

// {"event":"page",...} from a page object: splice the event key into the
// shared per-page JSON so the page shape stays byte-identical to /ocr/pdf.
[[nodiscard]] std::string ndjson_page_event(PdfPageResult &pg, size_t idx,
                                            int request_dpi, bool want_blocks,
                                            PdfImageMode image_mode,
                                            const pdf::EncodeOptions &encode_opts,
                                            bool want_orientation) {
  std::string page_json;
  append_pdf_page_json(page_json, pg, idx, request_dpi, want_blocks,
                       image_mode, encode_opts, want_orientation);
  std::string line = "{\"event\":\"page\",";
  line.append(page_json.data() + 1, page_json.size() - 1);
  return line;
}


} // namespace

void register_ocr_stream_route_gpu(server::WorkPool &pool,
                                   pipeline::PipelineDispatcher &dispatcher,
                                   render::PdfRenderer &pdf_renderer,
                                   const server::ImageDecoder &decode,
                                   pdf::PdfMode default_pdf_mode,
                                   bool layout_available,
                                   bool table_available,
                                   bool formula_available,
                                   int default_dpi,
                                   int max_pdf_pages,
                                   bool doc_ori_available) {
  const bool table_avail = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/stream",
      [&pool, &dispatcher, &pdf_renderer, &decode, default_pdf_mode,
       layout_available, table_avail, formula_avail, default_dpi,
       max_pdf_pages, doc_ori_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
    // ---- Everything below, up to the stream response, runs on the event
    // loop and produces REAL HTTP 4xx errors (nothing streamed yet). ----
    if (req->body().empty()) {
      callback(server::error_response(drogon::k400BadRequest, "EMPTY_BODY",
                                       "Empty body"));
      return;
    }
    // Drogon's async chunked stream needs a keep-alive connection: with
    // `Connection: close` the stream is torn down before the first chunk and
    // the client sees an EMPTY 200 body. Fail loud instead (python urllib is
    // the common offender; use requests/httpx or drop the header).
    {
      std::string conn(req->getHeader("connection"));
      for (char &c : conn)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (conn == "close") {
        callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
            "/ocr/stream requires a keep-alive connection (chunked streaming); "
            "remove the 'Connection: close' request header"));
        return;
      }
    }

    server::InferOptions opts;
    server::EndpointSpec spec;
    spec.pdf_options = true;
    spec.routing_unsupported_reason = server::kRoutingUnsupportedPdf;
    if (!server::validate_request(req, spec, layout_available, table_avail,
                                  formula_avail, /*valid_route_table=*/{},
                                  /*valid_route_formula=*/{}, &opts, callback,
                                  /*allow_image_only=*/true))
      return;

    auto dpi_str = req->getParameter("dpi");
    int dpi = dpi_str.empty() ? default_dpi : query_int(std::string(dpi_str), -1);
    if (dpi < kMinPdfDpi || dpi > kMaxPdfDpi) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI",
          std::format("DPI must be between {} and {}", kMinPdfDpi, kMaxPdfDpi)));
      return;
    }

    pdf::PdfMode req_mode = default_pdf_mode;
    auto mode_str = req->getParameter("mode");
    if (!mode_str.empty()) {
      // Same contract as /ocr/pdf (parse_pdf_request): an explicit but
      // unrecognized mode is a 400, never a silent fall-back to the default.
      if (!pdf::is_valid_pdf_mode(mode_str)) {
        callback(server::error_response(
            drogon::k400BadRequest, "INVALID_PARAMETER",
            "mode must be one of ocr, geometric, auto, auto_verified"));
        return;
      }
      req_mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);
    }

    PdfImageMode image_mode;
    pdf::EncodeOptions encode_opts;
    if (auto err = parse_image_query_params(req, image_mode, encode_opts);
        !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       "INVALID_PARAMETER", err));
      return;
    }

    bool autorotate = false;
    if (auto err = server::parse_bool_query(req, "autorotate", &autorotate);
        !err.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       "INVALID_PARAMETER", err));
      return;
    }
    if (autorotate && !doc_ori_available) {
      callback(server::error_response(drogon::k400BadRequest, "AUTOROTATE_DISABLED",
          "autorotate=1 requires the doc-orientation model (models/doc_ori.onnx); "
          "it was not found at startup"));
      return;
    }

    // Content sniff: %PDF magic -> PDF job; anything else -> single image.
    auto body = std::make_shared<std::string>(req->body());
    const bool is_pdf = body->size() >= 4 && body->compare(0, 4, "%PDF") == 0;

    if (is_pdf) {
      if (!opts.want_text && !opts.want_layout && image_mode != PdfImageMode::Inline) {
        callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
            "text=0 without layout=1 or images=inline would stream empty pages"));
        return;
      }
      // Cheap page-count guard + the count for the meta event.
      pdf::PdfDocument check_doc(
          reinterpret_cast<const uint8_t *>(body->data()), body->size());
      if (!check_doc.ok()) {
        callback(server::error_response(drogon::k400BadRequest, "INVALID_PDF",
                                         "Failed to open PDF"));
        return;
      }
      const int np = check_doc.page_count();
      if (np > max_pdf_pages) {
        callback(server::error_response(drogon::k400BadRequest, "PDF_TOO_LARGE",
            std::format("PDF has {} pages, maximum is {} (set MAX_PDF_PAGES to increase)",
                        np, max_pdf_pages)));
        return;
      }
      if (np == 0) {
        callback(server::error_response(drogon::k400BadRequest, "EMPTY_PDF",
                                         "PDF has no pages"));
        return;
      }

      auto state = std::make_shared<NdjsonStream>();
      const bool want_layout = opts.want_layout;
      const bool want_reading_order = opts.want_reading_order;
      const bool want_blocks = opts.want_blocks;
      const bool want_tables = opts.want_tables;
      const bool want_formulas = opts.want_formulas;
      const bool want_text = opts.want_text;
      auto resp = drogon::HttpResponse::newAsyncStreamResponse(
          [state, &pool, &dispatcher, &pdf_renderer, body, np, dpi, req_mode,
           image_mode, encode_opts, autorotate, want_layout, want_reading_order,
           want_blocks, want_tables, want_formulas, want_text](
              drogon::ResponseStreamPtr stream) {
        {
          std::lock_guard<std::mutex> lk(state->mu);
          state->stream = std::move(stream);
        }
        try {
          pool.submit([state, &dispatcher, &pdf_renderer, body, np, dpi,
                       req_mode, image_mode, encode_opts, autorotate,
                       want_layout, want_reading_order, want_blocks,
                       want_tables, want_formulas, want_text] {
            state->send_line(std::format(
                "{{\"event\":\"meta\",\"kind\":\"pdf\",\"pages\":{},\"dpi\":{},"
                "\"mode\":\"{}\"}}", np, dpi, pdf::mode_name(req_mode)));

            PdfJobOptions job_opts;
            job_opts.dpi = dpi;
            job_opts.mode = req_mode;
            job_opts.want_layout = want_layout;
            job_opts.want_reading_order = want_reading_order;
            job_opts.want_blocks = want_blocks;
            job_opts.want_tables = want_tables;
            job_opts.want_formulas = want_formulas;
            job_opts.want_text = want_text;
            job_opts.autorotate = autorotate;
            job_opts.image_mode = image_mode;
            job_opts.encode_opts = encode_opts;
            job_opts.request_timeout_ms = dispatcher.request_timeout_ms();
            job_opts.on_page_ready =
                [state, dpi, want_blocks, image_mode, encode_opts, autorotate](
                    int idx, PdfPageResult &&pg) {
              state->send_line(ndjson_page_event(
                  pg, static_cast<size_t>(idx), dpi, want_blocks, image_mode,
                  encode_opts, autorotate));
            };
            job_opts.on_page_failed = [state](int idx) {
              state->send_line(std::format(
                  "{{\"event\":\"page_error\",\"page_index\":{}}}", idx));
            };

            pipeline::PdfJobResult job;
            try {
              job = pipeline::run_pdf_job(
                  dispatcher, pdf_renderer,
                  reinterpret_cast<const uint8_t *>(body->data()),
                  body->size(), job_opts);
            } catch (const std::exception &ex) {
              TOCR_LOG_ERROR("stream PDF job error", "route", "/ocr/stream",
                             "error", std::string_view(ex.what()));
              state->send_line("{\"event\":\"error\",\"code\":\"INFERENCE_ERROR\"}");
              state->finish();
              return;
            }
            if (job.status != PdfJobStatus::Ok) {
              const char *code =
                  job.status == PdfJobStatus::RenderFailed ? "PDF_RENDER_FAILED"
                  : job.status == PdfJobStatus::Dropped    ? "SERVER_BUSY"
                  : job.status == PdfJobStatus::TimedOut   ? "INFERENCE_TIMEOUT"
                  : job.status == PdfJobStatus::EmptyPdf   ? "EMPTY_PDF"
                  : job.status == PdfJobStatus::DecodeFailed
                      ? "PAGE_DECODE_FAILED"
                      : "PAGE_FAILED";
              state->send_line(std::format(
                  "{{\"event\":\"error\",\"code\":\"{}\"}}", code));
            }
            state->send_line(std::format(
                "{{\"event\":\"end\",\"pages\":{},\"failed\":{}}}",
                job.num_pages, job.page_failures + job.decode_failures));
            state->finish();
          });
        } catch (const std::exception &) {
          // WorkPool full: the stream is already open, so say so in-band.
          state->send_line("{\"event\":\"error\",\"code\":\"SERVER_BUSY\"}");
          state->finish();
        }
      }, /*disableKickoffTimeout=*/true);
      resp->setContentTypeString("application/x-ndjson");
      callback(resp);
      return;
    }

    // ---- Single image (any container the decoder handles) ----
    if (!opts.want_text && !opts.want_layout) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
          "text=0 without layout=1 would stream an empty page"));
      return;
    }
    auto state = std::make_shared<NdjsonStream>();
    const server::InferOptions opts_v = opts;
    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [state, &pool, &dispatcher, &decode, body, opts_v](
            drogon::ResponseStreamPtr stream) {
      {
        std::lock_guard<std::mutex> lk(state->mu);
        state->stream = std::move(stream);
      }
      try {
        pool.submit([state, &dispatcher, &decode, body, opts_v] {
          state->send_line("{\"event\":\"meta\",\"kind\":\"image\",\"pages\":1}");
          cv::Mat img = decode(
              reinterpret_cast<const unsigned char *>(body->data()), body->size());
          const int kMaxImageDim = decode::max_image_dim();
          if (img.empty() || img.cols > kMaxImageDim || img.rows > kMaxImageDim ||
              decode::exceeds_pixel_cap(img.cols, img.rows)) {
            // Same three-way split as every other image route: decode failure,
            // per-side dimension cap, and pixel-AREA cap each keep their own
            // code so clients can tell which limit they hit.
            const char *code =
                img.empty() ? "IMAGE_DECODE_FAILED"
                : (img.cols > kMaxImageDim || img.rows > kMaxImageDim)
                    ? "DIMENSIONS_TOO_LARGE"
                    : "PIXELS_TOO_LARGE";
            state->send_line(std::format(
                "{{\"event\":\"error\",\"code\":\"{}\"}}", code));
            state->finish();
            return;
          }
          try {
            auto out = dispatcher.submit_for_default([img, opts_v](auto &e) {
              if (!opts_v.want_text)
                return e.pipeline->run_layout_only(img, e.stream);
              return e.pipeline->run_with_layout(img, e.stream,
                                                 opts_v.want_layout,
                                                 opts_v.want_reading_order,
                                                 opts_v.routing_override,
                                                 /*defer_external=*/true,
                                                 opts_v.want_tables,
                                                 opts_v.want_formulas);
            });
            pipeline::finalize_deferred(out);
            std::string inner = emit_pipeline_result_json(out, opts_v.want_blocks);
            std::string line = std::format(
                "{{\"event\":\"page\",\"page\":1,\"page_index\":0,"
                "\"width\":{},\"height\":{},", img.cols, img.rows);
            line.append(inner.data() + 1, inner.size() - 1);
            state->send_line(std::move(line));
            state->send_line("{\"event\":\"end\",\"pages\":1,\"failed\":0}");
          } catch (const turbo_ocr::TimeoutError &) {
            state->send_line("{\"event\":\"error\",\"code\":\"INFERENCE_TIMEOUT\"}");
          } catch (const turbo_ocr::PoolExhaustedError &) {
            // GPU-queue backpressure is retryable — emit SERVER_BUSY (parity
            // with the PDF-stream and every non-stream route), not the
            // terminal INFERENCE_ERROR the generic handler below would send.
            state->send_line("{\"event\":\"error\",\"code\":\"SERVER_BUSY\"}");
          } catch (const std::exception &ex) {
            TOCR_LOG_ERROR("stream image error", "route", "/ocr/stream",
                           "error", std::string_view(ex.what()));
            state->send_line("{\"event\":\"error\",\"code\":\"INFERENCE_ERROR\"}");
          }
          state->finish();
        });
      } catch (const std::exception &) {
        state->send_line("{\"event\":\"error\",\"code\":\"SERVER_BUSY\"}");
        state->finish();
      }
    }, /*disableKickoffTimeout=*/true);
    resp->setContentTypeString("application/x-ndjson");
    callback(resp);
  }, {drogon::Post});
}

} // namespace turbo_ocr::routes
#endif // !USE_CPU_ONLY
