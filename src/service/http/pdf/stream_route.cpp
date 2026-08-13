// /ocr/stream — NDJSON streaming, one endpoint for PDFs and single images.
//
// RESTORED. Deleted with src/cuda/ (as src/cuda/http/pdf_stream_route_gpu.cpp)
// when the duplicate CUDA-native HTTP layer was removed, and never ported — so
// the transport built for streaming had no streaming endpoint on any backend.
// Its supporting machinery survived and had been sitting consumerless ever
// since: PdfJobOptions::on_page_ready / on_page_failed (pdf_job.h, "set only by
// /ocr/stream"), PdfPageSink, pdf_job_sink.cpp, and the /ocr/stream bucket in
// metrics.h that could never fire. That orphaned scaffolding is what should have
// flagged the loss.
//
// The response is application/x-ndjson: one JSON object per line, flushed as
// produced. Protocol:
//   {"event":"meta","kind":"pdf|image","pages":N,"dpi":D,"mode":"ocr|..."}
//   {"event":"page", <same shape as an /ocr/pdf pages[] element>}   (xN, AS
//        EACH PAGE COMPLETES — out of order; page_index identifies the page)
//   {"event":"page_error","page_index":i}                           (failures)
//   {"event":"error","code":"..."}                                  (job-level)
//   {"event":"end","pages":N,"failed":k}
// Errors detected BEFORE the first byte are normal HTTP 4xx; once streaming has
// begun they arrive as error events (the 200 status is already on the wire —
// chunked transfer has no second status line).
#include "turbo_ocr/service/http/pdf_routes.h"

#include <cctype>
#include <format>
#include <mutex>
#include <optional>

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/base/log/logger.h"
#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/image/image_dims.h" // decode::peek_image_dimensions
#include "turbo_ocr/image/size_classify.h" // shared size verdict + wire codes
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"  // pdf::PdfDocument (page count guard)
#include "turbo_ocr/analysis/classification/doc_orientation_common.h" // rotate_upright
#include "turbo_ocr/pipeline/job/pdf_job.h"
#include "turbo_ocr/service/server/server_types.h"
#include "turbo_ocr/service/validation/request_gate.h"

using turbo_ocr::pipeline::PdfImageMode;
using turbo_ocr::pipeline::PdfJobOptions;
using turbo_ocr::pipeline::PdfJobResult;
using turbo_ocr::pipeline::PdfJobStatus;
using turbo_ocr::pipeline::PdfPageResult;

#include "pdf_internal.h"

namespace turbo_ocr::routes {
using namespace pdfdetail;

namespace {

// Serialized writer around Drogon's async stream: page events arrive
// concurrently from pipeline workers, and trantor's AsyncStream must not be
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

// {"event":"page",...} from a page object: splice the event key into the shared
// per-page JSON so the page shape stays byte-identical to /ocr/pdf.
[[nodiscard]] std::string ndjson_page_event(PdfPageResult &pg, size_t idx,
                                            int request_dpi, bool want_blocks,
                                            PdfImageMode image_mode,
                                            const pdf::EncodeOptions &encode_opts,
                                            bool want_orientation) {
  std::string page_json;
  append_pdf_page_json(page_json, pg, idx, request_dpi, want_blocks, image_mode,
                       encode_opts, want_orientation);
  std::string line = "{\"event\":\"page\",";
  line.append(page_json.data() + 1, page_json.size() - 1);
  return line;
}

// EVERYTHING /ocr/stream MUST AGREE ON BEFORE IT PROMISES A STREAM.
//
// Split out because the two are different kinds of code with different failure
// modes: every rejection below is a real HTTP 400 on a connection that has sent
// no bytes yet, while once the handler starts streaming NDJSON the status line
// is already gone and a failure can only be reported as a record inside the
// stream. Keeping the admission rules in one named place is what makes that
// boundary visible — it used to be the first 75 lines of a 300-line lambda.
//
// Returns nullopt having ALREADY answered `callback` with the error.
struct StreamRequest {
  server::InferOptions opts;
  int dpi = 0;
  pdf::PdfMode mode{};
  PdfImageMode image_mode{};
  pdf::EncodeOptions encode_opts;
  bool autorotate = false;
};

std::optional<StreamRequest>
admit_stream_request(const drogon::HttpRequestPtr &req,
                     const capability::CapabilityMask &loaded, int default_dpi,
                     pdf::PdfMode default_pdf_mode,
                     server::DrogonCallback &callback) {
  if (req->body().empty()) {
    callback(server::error_response(server::ErrorCode::kEmptyBody, "Empty body"));
    return std::nullopt;
  }
  // Drogon's async chunked stream needs a keep-alive connection: with
  // `Connection: close` the stream is torn down before the first chunk
  // and the client sees an EMPTY 200 body. Fail loud instead (python
  // urllib is the common offender; use requests/httpx or drop the header).
  {
    std::string conn(req->getHeader("connection"));
    for (char &c : conn)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (conn == "close") {
      callback(server::error_response(server::ErrorCode::kInvalidParameter,
                                      "/ocr/stream requires a keep-alive connection (chunked "
                                      "streaming); remove the 'Connection: close' request header"));
      return std::nullopt;
    }
  }

  server::InferOptions opts;
  server::EndpointSpec spec;
  spec.pdf_options = true;
  // NOT pdf_doc_params: this endpoint streams NDJSON and never honours
  // output=/min_confidence=/fields=/editable= — accepting them would be a
  // silent no-op, so they stay category-2 (ignored + header / strict 400).
  spec.acts_on = capability::CapabilityMask::all(); // autorotate applies
  spec.routing_unsupported_reason = server::kRoutingUnsupportedPdf;
  if (!server::validate_request(req, spec, loaded,
                                /*valid_route_table=*/{},
                                /*valid_route_formula=*/{}, &opts,
                                callback, /*allow_image_only=*/true))
    return std::nullopt;

  auto dpi_str = req->getParameter("dpi");
  int dpi =
      dpi_str.empty() ? default_dpi : query_int(std::string(dpi_str), -1);
  if (dpi < kMinPdfDpi || dpi > kMaxPdfDpi) {
    callback(server::error_response(server::ErrorCode::kInvalidDpi,
                                    std::format("DPI must be between {} and {}", kMinPdfDpi,
                                                kMaxPdfDpi)));
    return std::nullopt;
  }

  pdf::PdfMode req_mode = default_pdf_mode;
  auto mode_str = req->getParameter("mode");
  if (!mode_str.empty()) {
    // Same contract as /ocr/pdf (parse_pdf_request): an explicit but
    // unrecognized mode is a 400, never a silent fall-back to the default.
    if (!pdf::is_valid_pdf_mode(mode_str)) {
      callback(server::error_response(server::ErrorCode::kInvalidParameter,
                                      "mode must be one of ocr, geometric, auto, auto_verified"));
      return std::nullopt;
    }
    req_mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);
  }

  PdfImageMode image_mode;
  pdf::EncodeOptions encode_opts;
  if (auto err = parse_image_query_params(req, image_mode, encode_opts);
      !err.empty()) {
    callback(server::error_response(server::ErrorCode::kInvalidParameter, err));
    return std::nullopt;
  }

  // autorotate: parsed + availability-gated by validate_request above
  // (this endpoint's acts_on includes DocOrientation) — a bad value or a
  // missing model was already a 400 before anything streamed.
  const bool autorotate = opts.want_autorotate;
  return StreamRequest{std::move(opts), dpi,        req_mode,
                       image_mode,      encode_opts, autorotate};
}


} // namespace

void register_ocr_stream_route(server::WorkPool &pool,
                               const server::InferFunc &infer,
                               render::PdfRenderer &pdf_renderer,
                               const server::ImageDecoder &decode,
                               pdf::PdfMode default_pdf_mode,
                               const capability::CapabilityMask &loaded,
                               int default_dpi, int max_pdf_pages,
                               server::OrientFunc orient_fn) {
  drogon::app().registerHandler(
      "/ocr/stream",
      [&pool, &infer, &pdf_renderer, &decode, default_pdf_mode, loaded,
       default_dpi, max_pdf_pages, orient_fn](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        // Admission first: every 4xx it can produce happens here, on the event
        // loop, before a single byte of the stream is promised.
        auto admitted = admit_stream_request(req, loaded, default_dpi,
                                             default_pdf_mode, callback);
        if (!admitted) return;
        server::InferOptions &opts = admitted->opts;
        const int dpi = admitted->dpi;
        const pdf::PdfMode req_mode = admitted->mode;
        const PdfImageMode image_mode = admitted->image_mode;
        const pdf::EncodeOptions &encode_opts = admitted->encode_opts;
        const bool autorotate = admitted->autorotate;

        // Content sniff: %PDF magic -> PDF job; anything else -> single image.
        auto body = std::make_shared<std::string>(req->body());
        const bool is_pdf =
            body->size() >= 4 && body->compare(0, 4, "%PDF") == 0;

        if (is_pdf) {
          if (!opts.want_text && !opts.want_layout &&
              image_mode != PdfImageMode::Inline) {
            callback(server::error_response(server::ErrorCode::kInvalidParameter,
                "text=0 without layout=1 or images=inline would stream empty "
                                            "pages"));
            return;
          }
          // The page-count guard needs PDFium, and PDFium work must NEVER run
          // on the Drogon IO thread: PdfDocument's constructor, page_count()
          // and destructor each take the PROCESS-WIDE pdfium mutex, so a long
          // holder elsewhere (a 300-page ?output=pdf&editable=1 stamping pass)
          // would park every IO thread right here and the server would stop
          // answering everything — including /health. The check runs on the
          // WorkPool; submit_work marshals the 4xx answers back and turns a
          // full queue into SERVER_BUSY on its own.
          const server::InferOptions opts_pdf = opts;
          server::submit_work(
              pool, std::move(callback),
              [&pool, &infer, &pdf_renderer, body, max_pdf_pages, dpi, req_mode,
               image_mode, encode_opts, autorotate, opts_pdf,
               orient_fn](server::DrogonCallback &cb) {
          // The SHARED page-count guard (pdf_internal.h) — this used to be a
          // hand-rolled copy of it; out_pages feeds the meta event with the
          // same single PDFium open.
          int np = 0;
          if (reject_if_too_many_pages(
                  reinterpret_cast<const uint8_t *>(body->data()),
                  body->size(), max_pdf_pages, cb, &np))
            return;
          if (np < 0) {
            cb(server::error_response(server::ErrorCode::kInvalidPdf, "Failed to open PDF"));
            return;
          }
          if (np == 0) {
            cb(server::error_response(server::ErrorCode::kEmptyPdf, "PDF has no pages"));
            return;
          }

          auto state = std::make_shared<NdjsonStream>();
          const server::InferOptions &opts_v = opts_pdf;
          auto resp = drogon::HttpResponse::newAsyncStreamResponse(
              [state, &pool, &infer, &pdf_renderer, body, np, dpi, req_mode,
               image_mode, encode_opts, autorotate, opts_v,
               orient_fn](drogon::ResponseStreamPtr stream) {
                {
                  std::lock_guard<std::mutex> lk(state->mu);
                  state->stream = std::move(stream);
                }
                try {
                  pool.submit([state, &infer, &pdf_renderer, body, np, dpi,
                               req_mode, image_mode, encode_opts, autorotate,
                               opts_v, orient_fn] {
                    state->send_line(std::format(
                        "{{\"event\":\"meta\",\"kind\":\"pdf\",\"pages\":{},"
                        "\"dpi\":{},\"mode\":\"{}\"}}",
                        np, dpi, pdf::mode_name(req_mode)));

                    PdfJobOptions job_opts;
                    job_opts.dpi = dpi;
                    job_opts.mode = req_mode;
                    job_opts.want_layout = opts_v.want_layout;
                    job_opts.want_reading_order = opts_v.want_reading_order;
                    job_opts.want_blocks = opts_v.want_blocks;
                    job_opts.want_tables = opts_v.want_tables;
                    job_opts.want_formulas = opts_v.want_formulas;
                    job_opts.want_text = opts_v.want_text;
                    job_opts.autorotate = autorotate;
                    job_opts.image_mode = image_mode;
                    job_opts.encode_opts = encode_opts;
                    // THE hooks this endpoint exists for. They are why
                    // PdfJobOptions carries them at all, and they had no setter
                    // in the tree between this route's deletion and its return.
                    job_opts.on_page_ready = [state, dpi, opts_v, image_mode,
                                              encode_opts, autorotate](
                                                 int idx, PdfPageResult &&pg) {
                      state->send_line(ndjson_page_event(
                          pg, static_cast<size_t>(idx), dpi, opts_v.want_blocks,
                          image_mode, encode_opts, autorotate));
                    };
                    job_opts.on_page_failed = [state](int idx) {
                      state->send_line(std::format(
                          "{{\"event\":\"page_error\",\"page_index\":{}}}", idx));
                    };

                    PdfJobResult job;
                    try {
                      job = pipeline::run_pdf_job(
                          infer, pdf_renderer,
                          reinterpret_cast<const uint8_t *>(body->data()),
                          body->size(), job_opts, orient_fn);
                    } catch (const std::exception &ex) {
                      TOCR_LOG_ERROR("stream PDF job error", "route",
                                     "/ocr/stream", "error",
                                     std::string_view(ex.what()));
                      state->send_line(
                          "{\"event\":\"error\",\"code\":\"INFERENCE_ERROR\"}");
                      state->finish();
                      return;
                    }
                    if (job.status != PdfJobStatus::Ok) {
                      state->send_line(std::format(
                          "{{\"event\":\"error\",\"code\":\"{}\"}}",
                          pipeline::wire_code(job.status)));
                    }
                    state->send_line(std::format(
                        "{{\"event\":\"end\",\"pages\":{},\"failed\":{}}}",
                        job.num_pages,
                        job.page_failures + job.decode_failures));
                    state->finish();
                  });
                } catch (const std::exception &) {
                  // WorkPool full: the stream is already open, so say so
                  // in-band.
                  state->send_line(
                      "{\"event\":\"error\",\"code\":\"SERVER_BUSY\"}");
                  state->finish();
                }
              },
              /*disableKickoffTimeout=*/true);
          resp->setContentTypeString("application/x-ndjson");
          cb(resp);
              });
          return;
        }

        // ---- Single image (any container the decoder handles) ----
        if (!opts.want_text) {
          // The pre-restoration route served this through the pipeline's
          // run_layout_only. The device-agnostic InferFunc has no layout-only
          // entry point — make_infer_func always calls run_with_layout and does
          // not read opts.want_text — so honouring text=0 here would silently
          // run FULL OCR and return recognized text the caller asked not to
          // have. Reject instead; a wrong answer is worse than a refusal.
          //
          // Reachable only on a build whose shared gate admits text=0 (the
          // CPU/unified build rejects it in parse_options_core), so this costs
          // nothing today and cannot become a silent regression tomorrow.
          callback(server::error_response(server::ErrorCode::kInvalidParameter,
              "text=0 (layout-only) is not supported for single images on "
                                          "/ocr/stream; send a PDF, or use /ocr/raw?text=0"));
          return;
        }
        auto state = std::make_shared<NdjsonStream>();
        const server::InferOptions opts_v = opts;
        auto resp = drogon::HttpResponse::newAsyncStreamResponse(
            [state, &pool, &infer, &decode, body, opts_v, autorotate,
             orient_fn](drogon::ResponseStreamPtr stream) {
              {
                std::lock_guard<std::mutex> lk(state->mu);
                state->stream = std::move(stream);
              }
              try {
                pool.submit([state, &infer, &decode, body, opts_v, autorotate,
                             orient_fn] {
                  state->send_line(
                      "{\"event\":\"meta\",\"kind\":\"image\",\"pages\":1}");
                  // STAGE 1 — pre-decode header sniff (size_guards.h's shared
                  // two-stage rule; this route implemented only stage 2): a
                  // 2 KB JPEG declaring 60000x60000 must be refused WITHOUT
                  // calling the decoder, or cv::imdecode attempts the ~10 GB
                  // allocation before the post-decode check can run. Formats
                  // the sniff cannot parse (BMP/PNM) fall through to stage 2.
                  // The SHARED verdict (size_classify.h) for BOTH stages —
                  // this used to re-derive the dimension-vs-area split by
                  // hand, so the next cap change would have reached five call
                  // sites and missed this one.
                  if (auto d = decode::peek_image_dimensions(
                          reinterpret_cast<const unsigned char *>(body->data()),
                          body->size())) {
                    const auto v = decode::classify_image_size(d->width, d->height);
                    if (v != decode::ImageSizeVerdict::kOk) {
                      state->send_line(std::format(
                          "{{\"event\":\"error\",\"code\":\"{}\"}}",
                          decode::image_size_error_code(v)));
                      state->finish();
                      return;
                    }
                  }
                  cv::Mat img = decode(
                      reinterpret_cast<const unsigned char *>(body->data()),
                      body->size());
                  const auto post_v = img.empty()
                                          ? decode::ImageSizeVerdict::kOk
                                          : decode::classify_image_size(img.cols, img.rows);
                  if (img.empty() || post_v != decode::ImageSizeVerdict::kOk) {
                    // Same three-way split as every other image route: decode
                    // failure, per-side dimension cap, and pixel-AREA cap each
                    // keep their own code so clients can tell which limit they
                    // hit.
                    const char *code =
                        img.empty() ? "IMAGE_DECODE_FAILED"
                                    : decode::image_size_error_code(post_v);
                    state->send_line(std::format(
                        "{{\"event\":\"error\",\"code\":\"{}\"}}", code));
                    state->finish();
                    return;
                  }
                  // autorotate=1: de-rotate BEFORE inference, exactly like the
                  // PDF branch (which threads orient_fn into run_pdf_job).
                  // This branch used to accept + availability-gate the flag in
                  // admission and then silently ignore it.
                  if (autorotate && orient_fn) {
                    if (const int deg = orient_fn(img))
                      classification::rotate_upright(img, deg);
                  }
                  try {
                    // make_infer_func already runs with defer_external=true and
                    // calls finalize_deferred, so the external-recognizer drain
                    // the dispatcher version did by hand is covered.
                    auto out = server::to_pipeline_result(infer(img, opts_v));
                    std::string inner =
                        emit_pipeline_result_json(out, opts_v.want_blocks);
                    std::string line = std::format(
                        "{{\"event\":\"page\",\"page\":1,\"page_index\":0,"
                        "\"width\":{},\"height\":{},",
                        img.cols, img.rows);
                    line.append(inner.data() + 1, inner.size() - 1);
                    state->send_line(std::move(line));
                    state->send_line(
                        "{\"event\":\"end\",\"pages\":1,\"failed\":0}");
                  } catch (const turbo_ocr::TimeoutError &) {
                    state->send_line(
                        "{\"event\":\"error\",\"code\":\"INFERENCE_TIMEOUT\"}");
                  } catch (const turbo_ocr::PoolExhaustedError &) {
                    // Queue backpressure is retryable — emit SERVER_BUSY
                    // (parity with the PDF-stream and every non-stream route),
                    // not the terminal INFERENCE_ERROR the generic handler
                    // below would send.
                    state->send_line(
                        "{\"event\":\"error\",\"code\":\"SERVER_BUSY\"}");
                  } catch (const std::exception &ex) {
                    TOCR_LOG_ERROR("stream image error", "route", "/ocr/stream",
                                   "error", std::string_view(ex.what()));
                    state->send_line(
                        "{\"event\":\"error\",\"code\":\"INFERENCE_ERROR\"}");
                  }
                  state->finish();
                });
              } catch (const std::exception &) {
                state->send_line(
                    "{\"event\":\"error\",\"code\":\"SERVER_BUSY\"}");
                state->finish();
              }
            },
            /*disableKickoffTimeout=*/true);
        resp->setContentTypeString("application/x-ndjson");
        callback(resp);
      },
      {drogon::Post});
}

} // namespace turbo_ocr::routes
