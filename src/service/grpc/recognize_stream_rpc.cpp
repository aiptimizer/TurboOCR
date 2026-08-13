// gRPC RecognizeStream — the server-streaming twin of HTTP /ocr/stream.
//
// Until now the proto declared no streaming method at all, so the transport
// designed for streaming was the only one that could not stream. The event
// shape mirrors the NDJSON protocol exactly (meta -> page* -> [error] -> end),
// so a client ported between transports reads the same state machine, and the
// per-page body is the SAME serialize_page_results output the HTTP route emits.
#include "turbo_ocr/service/grpc/grpc_service.h"

#include <mutex>

#include "turbo_ocr/analysis/classification/doc_orientation_common.h" // rotate_upright
#include "turbo_ocr/image/size_classify.h"

namespace turbo_ocr::server {
namespace {

// grpc::ServerWriter is NOT thread-safe, and pages complete on pipeline worker
// threads: run_pdf_job's on_page_ready fires from whichever worker finished.
// One mutex around every Write is what makes the concurrent producers legal —
// the HTTP route needs the identical guard around its AsyncStream, for the same
// reason and with the same failure mode if omitted (interleaved frames).
struct EventWriter {
  std::mutex mu;
  grpc::ServerWriter<ocr::OCRStreamEvent> *w;
  bool closed = false;

  bool send(const ocr::OCRStreamEvent &ev) {
    std::lock_guard<std::mutex> lk(mu);
    if (closed || !w) return false;
    if (!w->Write(ev)) {  // client went away
      closed = true;
      return false;
    }
    return true;
  }
};

[[nodiscard]] ocr::OCRStreamEvent make_error_event(const char *code) {
  ocr::OCRStreamEvent ev;
  ev.set_event("error");
  ev.set_code(code);
  return ev;
}

// EVERYTHING RecognizeStream must agree on before it writes the first event.
//
// gRPC hands over the whole request in one message, unlike HTTP's chunked
// body, so — unlike admit_stream_request in stream_route.cpp — the content
// sniff (%PDF vs image) CAN happen here rather than after admission: the
// image branch's decode and dimension check, and the PDF branch's dpi/mode/
// page-count checks, are all still plain validation that runs before this
// RPC's "no bytes on the wire yet" moment, which for a stream is the first
// out.send() rather than an HTTP status line. Past that moment a failure can
// only be reported as an in-band event (see the two catch blocks left in the
// handler below, which do exactly that and are why they are NOT folded into
// guarded_infer).
//
// Returns nullopt having ALREADY produced the error in `out_status`.
struct StreamAdmitted {
  InferOptions opts;
  bool is_pdf = false;
  // Populated when !is_pdf: decoded and dimension-checked here because a
  // decode failure IS an admission failure (IMAGE_DECODE_FAILED), and doing
  // it twice would be wasted work, not a smaller diff.
  cv::Mat img;
  // Populated when is_pdf.
  int dpi = 0;
  pdf::PdfMode mode{};
  int num_pages = 0;
};

std::optional<StreamAdmitted>
admit_stream_request(grpc::ServerContext *ctx,
                     const ocr::OCRStreamRequest *request,
                     const capability::CapabilityMask &loaded,
                     bool have_pdf_renderer, int default_pdf_dpi,
                     pdf::PdfMode default_pdf_mode, int max_pdf_pages,
                     grpc::Status &out_status) {
  if (request->data().empty()) {
    out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            "EMPTY_BODY", "Empty data");
    return std::nullopt;
  }

  // The SHARED gate, exactly as the other RPCs use it. layout_only is not
  // offered on this RPC: over HTTP text=0 streams page images, which this
  // transport has no field for, so making it unrepresentable beats accepting
  // and then ignoring it.
  InferOptions opts;
  if (auto r = parse_proto_options(*request, /*layout_only=*/false, loaded,
                                   &opts, /*allow_image_only=*/false,
                                   capability::CapabilityMask::all());
      !r.error.empty()) {
    out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            r.error_code.c_str(), r.error);
    return std::nullopt;
  }

  // Content sniff: %PDF magic -> PDF job; anything else -> single image. Same
  // rule as the HTTP endpoint, so one request shape covers both there and here.
  const std::string &data = request->data();
  const bool is_pdf = data.size() >= 4 && data.compare(0, 4, "%PDF") == 0;

  StreamAdmitted out;
  out.is_pdf = is_pdf;

  if (!is_pdf) {
    if (auto err = grpc_pre_decode_dim_check(ctx, data); err) {
      out_status = *err;
      return std::nullopt;
    }
    cv::Mat img = grpc_decode_image(data);
    if (img.empty()) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "IMAGE_DECODE_FAILED", "Decode failed");
      return std::nullopt;
    }
    if (auto st = grpc_check_image_size(ctx, img.cols, img.rows)) {
      out_status = *st;
      return std::nullopt;
    }
    out.opts = std::move(opts);
    out.img = std::move(img);
    return out;
  }

  // ---- PDF ----
  if (!have_pdf_renderer) {
    out_status = grpc_error(ctx, grpc::StatusCode::UNIMPLEMENTED,
                            "PDF_NOT_AVAILABLE",
                            "PDF rendering not available on this server");
    return std::nullopt;
  }

  int dpi = request->dpi();
  if (dpi == 0) dpi = default_pdf_dpi;
  if (dpi < pipeline::kMinPdfDpi || dpi > pipeline::kMaxPdfDpi) {
    out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "INVALID_DPI",
                            std::format("DPI must be between {} and {}",
                                        pipeline::kMinPdfDpi, pipeline::kMaxPdfDpi));
    return std::nullopt;
  }

  pdf::PdfMode req_mode = default_pdf_mode;
  if (!request->mode().empty()) {
    // An explicit but unrecognized mode is an error, never a silent fall-back —
    // same contract as /ocr/pdf and /ocr/stream.
    if (!pdf::is_valid_pdf_mode(request->mode())) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER",
                              "mode must be one of ocr, geometric, auto, "
                              "auto_verified");
      return std::nullopt;
    }
    req_mode = pdf::parse_pdf_mode(request->mode(), default_pdf_mode);
  }

  int np = 0;
  {
    // Page-count guard before any render work — the same MAX_PDF_PAGES limit
    // RecognizePDF applies, read from the same member.
    pdf::PdfDocument probe(reinterpret_cast<const uint8_t *>(data.data()),
                           data.size());
    if (!probe.ok()) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PDF", "Failed to open PDF");
      return std::nullopt;
    }
    np = probe.page_count();
    if (np > max_pdf_pages) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "PDF_TOO_LARGE",
                              std::format("PDF has {} pages, maximum is {} (set "
                                          "MAX_PDF_PAGES to increase)",
                                          np, max_pdf_pages));
      return std::nullopt;
    }
    if (np == 0) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "EMPTY_PDF",
                              "PDF has no pages");
      return std::nullopt;
    }
  }

  out.opts = std::move(opts);
  out.dpi = dpi;
  out.mode = req_mode;
  out.num_pages = np;
  return out;
}

} // namespace

grpc::Status OCRServiceImpl::RecognizeStream(
    grpc::ServerContext *ctx, const ocr::OCRStreamRequest *request,
    grpc::ServerWriter<ocr::OCRStreamEvent> *writer) {
  // Admission first: everything above is a plain grpc::Status returned before
  // a single event reaches the client. After this point the RPC's success is
  // effectively committed — a failure can only travel as an in-band event
  // (the two catch blocks below, and the PDF job-status check), never as a
  // second status.
  grpc::Status admit_err;
  auto admitted = admit_stream_request(ctx, request, loaded_,
                                       pdf_renderer_ != nullptr && render::PdfRenderer::can_render(),
                                       default_pdf_dpi_, default_pdf_mode_,
                                       max_pdf_pages_, admit_err);
  if (!admitted) return admit_err;
  InferOptions &opts = admitted->opts;

  EventWriter out{.w = writer};

  if (!admitted->is_pdf) {
    // ---- Single image: one page event, then end. ----
    cv::Mat &img = admitted->img;
    // autorotate=1: de-rotate BEFORE inference — the same fix the HTTP twin
    // (stream_route.cpp) got, which this transport missed: the flag was
    // parsed and availability-gated in admission (acts_on = all()) and then
    // silently ignored, so a gRPC client got unrotated garbage OCR for a
    // request the server reported as honoured.
    if (opts.want_autorotate && orient_fn_) {
      if (const int deg = orient_fn_(img))
        classification::rotate_upright(img, deg);
    }
    {
      ocr::OCRStreamEvent meta;
      meta.set_event("meta");
      meta.set_kind("image");
      meta.set_pages(1);
      out.send(meta);
    }
    try {
      auto res = run_infer(img, opts.want_layout, opts.want_reading_order,
                           opts.want_tables, opts.want_formulas,
                           opts.routing_override);
      ocr::OCRStreamEvent ev;
      ev.set_event("page");
      ev.set_page_index(0);
      ev.set_width(img.cols);
      ev.set_height(img.rows);
      // The same serializer every other surface uses, so the page body cannot
      // drift between transports.
      ev.set_json_response(
          turbo_ocr::emit_pipeline_result_json(res, opts.want_blocks));
      out.send(ev);
    } catch (const turbo_ocr::PoolExhaustedError &) {
      out.send(make_error_event("SERVER_BUSY"));
    } catch (const turbo_ocr::TimeoutError &) {
      out.send(make_error_event("INFERENCE_TIMEOUT"));
    } catch (const std::exception &e) {
      TOCR_LOG_ERROR_RL("gRPC stream image error", "error", e.what());
      out.send(make_error_event("INFERENCE_ERROR"));
    }
    ocr::OCRStreamEvent end;
    end.set_event("end");
    end.set_pages(1);
    out.send(end);
    return grpc::Status::OK;
  }

  // ---- PDF ----
  const auto *pdf_data = reinterpret_cast<const uint8_t *>(request->data().data());
  const size_t pdf_len = request->data().size();
  const int dpi = admitted->dpi;
  const pdf::PdfMode req_mode = admitted->mode;
  const int np = admitted->num_pages;

  {
    ocr::OCRStreamEvent meta;
    meta.set_event("meta");
    meta.set_kind("pdf");
    meta.set_pages(np);
    meta.set_dpi(dpi);
    meta.set_mode(std::string(pdf::mode_name(req_mode)));
    out.send(meta);
  }

  pipeline::PdfJobOptions job_opts;
  job_opts.dpi = dpi;
  job_opts.mode = req_mode;
  job_opts.want_layout = opts.want_layout;
  job_opts.want_reading_order = opts.want_reading_order;
  job_opts.want_blocks = opts.want_blocks;
  job_opts.want_tables = opts.want_tables;
  job_opts.want_formulas = opts.want_formulas;
  job_opts.autorotate = opts.want_autorotate;
  // The streaming hooks. These are the whole point of the RPC, and they are the
  // same two PdfJobOptions fields /ocr/stream sets — the pipeline emits a page
  // the moment it is done, out of order, instead of accumulating the document.
  job_opts.on_page_ready = [&out, &opts](int idx, pipeline::PdfPageResult &&pg) {
    ocr::OCRStreamEvent ev;
    ev.set_event("page");
    ev.set_page_index(idx);
    ev.set_width(pg.width);
    ev.set_height(pg.height);
    ev.set_dpi(pg.effective_dpi);
    ev.set_mode(std::string(pdf::mode_name(pg.resolved_mode)));
    ev.set_json_response(
        pipeline::serialize_page_results(pg, opts.want_blocks));
    out.send(ev);
  };
  job_opts.on_page_failed = [&out](int idx) {
    ocr::OCRStreamEvent ev;
    ev.set_event("page_error");
    ev.set_page_index(idx);
    out.send(ev);
  };

  pipeline::PdfJobResult job;
  try {
    // auto_verified resolves inside run_pdf_job — see the note there.
    job = pipeline::run_pdf_job(
        [this](const cv::Mat &img, const InferOptions &o) {
          return from_pipeline_result(
              run_infer(img, o.want_layout, o.want_reading_order, o.want_tables,
                        o.want_formulas, o.routing_override));
        },
        *pdf_renderer_, pdf_data, pdf_len, job_opts, orient_fn_);
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("gRPC stream PDF job error", "error", e.what());
    out.send(make_error_event("INFERENCE_ERROR"));
    return grpc::Status::OK;  // the stream already succeeded; report in-band
  }

  if (job.status != pipeline::PdfJobStatus::Ok) {
    // Once streaming has begun the RPC status is already committed, so a
    // job-level failure travels as an event — the same reason the HTTP route
    // cannot emit a second status line after the 200.
    out.send(make_error_event(pipeline::wire_code(job.status)));
  }

  ocr::OCRStreamEvent end;
  end.set_event("end");
  end.set_pages(job.num_pages);
  end.set_failed(job.page_failures + job.decode_failures);
  out.send(end);
  return grpc::Status::OK;
}

} // namespace turbo_ocr::server
