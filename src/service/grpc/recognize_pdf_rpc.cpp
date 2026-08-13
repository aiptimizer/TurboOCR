// gRPC RecognizePDF: shared run_pdf_job orchestration, per-page fill.
#include "turbo_ocr/service/capability/proto_capability_bridge.h"
#include "turbo_ocr/service/grpc/grpc_service.h"
// The SHARED document builders + per-page hooks, which HTTP /ocr/pdf uses too:
// one markdown assembler and one write_searchable_pdf call site, so the two
// transports cannot return different documents for the same request.
//
// This used to include ../http/pdf/pdf_internal.h across the transport boundary
// — a header whose own banner calls itself HTTP-private — and so pulled Drogon
// into this gRPC translation unit for four functions that touch no transport at
// all. They live in turbo_ocr::pdf::documents now.
#include "turbo_ocr/pdf/pdf_documents.h"

namespace turbo_ocr::server {
namespace {

// EVERYTHING RecognizePDF must agree on before it renders a single page.
//
// Split out for the same reason the HTTP admit_* helpers exist (see
// stream_route.cpp's admit_stream_request and unified_routes.cpp's
// admit_batch_request): every check below produces a plain grpc::Status
// before pipeline::run_pdf_job does any rendering or OCR work. Once that
// call is made, a failure is reported through PdfJobResult's status enum
// instead — a different shape entirely — so keeping the two halves apart is
// what makes the "have we started yet" boundary visible instead of buried in
// a 279-line body.
//
// Returns nullopt having ALREADY produced the error in `out_status`.
struct PdfAdmitted {
  InferOptions opts;
  int dpi = 0;
  pdf::PdfMode mode{};
  bool want_markdown = false;
  bool md_as_pages = false;
  bool want_searchable_pdf = false;
};

std::optional<PdfAdmitted>
admit_pdf_request(grpc::ServerContext *ctx, const ocr::OCRPDFRequest *request,
                  const capability::CapabilityMask &loaded,
                  bool json_bytes_mode, bool have_pdf_renderer,
                  int default_pdf_dpi, pdf::PdfMode default_pdf_mode,
                  int max_pdf_pages, grpc::Status &out_status) {
  if (!have_pdf_renderer) {
    out_status = grpc_error(ctx, grpc::StatusCode::UNIMPLEMENTED,
                            "PDF_NOT_AVAILABLE",
                            "PDF rendering not available on this server");
    return std::nullopt;
  }

  if (request->pdf_data().empty()) {
    out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            "MISSING_PDF", "Empty PDF data");
    return std::nullopt;
  }

  const auto *pdf_data = reinterpret_cast<const uint8_t *>(request->pdf_data().data());
  size_t pdf_len = request->pdf_data().size();

  // The shared gate (see recognize_rpc.cpp), with acts_on = ALL: unlike the
  // image RPCs, the PDF path genuinely applies DocOrientation — it rotates the
  // page before OCR — so autorotate is parsed and availability-gated here
  // (AUTOROTATE_DISABLED) rather than falling through as an unsupported flag.
  // This mirrors parse_pdf_request, which passes CapabilityMask::all() for the
  // same reason.
  InferOptions opts;
  // layout_only comes FROM THE REQUEST. It was hard-coded false here, so
  // OCRPDFRequest.layout_only (proto field 18) — a documented part of the
  // contract — was accepted, ignored, and answered with a full OCR run: the
  // client asked to skip text and got billed for it, with no error to notice.
  // Passing it through means the four text=0 combination rules in
  // options_core.h apply verbatim, exactly as they do for HTTP /ocr/pdf.
  if (auto r = parse_proto_options(*request, request->layout_only(), loaded,
                                   &opts, /*allow_image_only=*/false,
                                   capability::CapabilityMask::all());
      !r.error.empty()) {
    out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            r.error_code.c_str(), r.error);
    return std::nullopt;
  }
  const bool want_blocks = opts.want_blocks;
  if (auto err = grpc_check_structure_backends(
          ctx, opts.requested, loaded, json_bytes_mode,
          want_blocks, /*raw_layout=*/request->layout()); err) {
    out_status = *err;
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
    // An explicit but unrecognized mode is an error, never a silent fall-back.
    // This RPC was the one endpoint of four that fell back: HTTP /ocr/pdf,
    // HTTP /ocr/stream and gRPC RecognizeStream all reject it, so a typo'd mode
    // behaved differently depending on which door the client came through.
    if (!pdf::is_valid_pdf_mode(request->mode())) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER",
                              "mode must be one of ocr, geometric, auto, "
                              "auto_verified");
      return std::nullopt;
    }
    req_mode = pdf::parse_pdf_mode(request->mode(), default_pdf_mode);
  }

  // MAX_PDF_PAGES guard — same env var and limit as HTTP /ocr/pdf
  // (default 2000). Mirror the route's reject_if_too_many_pages: open the
  // doc once just for the page count.
  {
    pdf::PdfDocument probe(pdf_data, pdf_len);
    if (probe.ok() && probe.page_count() > max_pdf_pages) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
          "PDF_TOO_LARGE",
          std::format("PDF has {} pages, maximum is {} "
                      "(set MAX_PDF_PAGES to increase)",
                      probe.page_count(), max_pdf_pages));
      return std::nullopt;
    }
  }

  // ---- Document-output options (proto fields 10-18), the /ocr/pdf parity set.
  // Validated here in the SAME order and with the same messages parse_pdf_request
  // applies, because these are contract rules, not transport rules: a client that
  // moves from HTTP to gRPC must not discover that output=pdf+markdown is legal
  // on one and rejected on the other.
  const bool want_markdown = request->markdown();
  const bool md_as_pages = request->as_pages();
  const std::string output = request->output();
  bool want_searchable_pdf = false;
  if (!output.empty()) {
    if (output == "pdf") want_searchable_pdf = true;
    else if (output != "json") {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER", "output must be json or pdf");
      return std::nullopt;
    }
  }
  if (md_as_pages && !want_markdown) {
    out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            "INVALID_PARAMETER", "as_pages requires markdown");
    return std::nullopt;
  }
  if (request->min_confidence() != 0.0f) {
    if (!want_searchable_pdf) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER",
                              "min_confidence requires output=pdf");
      return std::nullopt;
    }
    // Inclusion test: the client can send NaN bits in the proto float, and
    // `< 0 || > 1` is false for NaN — which then strips the whole text layer
    // silently downstream (keep(): confidence >= NaN is always false).
    if (!(request->min_confidence() >= 0.0f && request->min_confidence() <= 1.0f)) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER",
                              "min_confidence must be between 0 and 1");
      return std::nullopt;
    }
  }
  if (want_searchable_pdf) {
    if (want_markdown) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER",
                              "output=pdf cannot be combined with markdown");
      return std::nullopt;
    }
    if (!opts.want_text) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER",
                              "layout_only cannot be combined with output=pdf (a "
                              "searchable PDF needs the text)");
      return std::nullopt;
    }
  }
  if (want_markdown) {
    if (!loaded.get(capability::CapabilityId::Layout)) {
      const std::string code(
          capability::capability_info(capability::CapabilityId::Layout)
              .error_code);
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, code.c_str(),
                              "markdown requires the layout model (do not start with "
                              "DISABLE_LAYOUT=1)");
      return std::nullopt;
    }
    if (!opts.want_text) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              "INVALID_PARAMETER",
                              "layout_only cannot be combined with markdown "
                              "(markdown needs the text)");
      return std::nullopt;
    }
    // Faithful-export defaults, identical to /ocr/pdf?markdown=1: run the
    // stages the server actually LOADED unless the request disabled them.
    // Field PRESENCE (optional bool) is what makes "disabled" expressible:
    // testing the VALUE treated an explicit tables=false as unset, so a
    // markdown client could never turn structure off while HTTP
    // ?markdown=1&tables=0 could.
    opts.want_layout = true;
    opts.want_reading_order = true;
    if (!request->has_tables())
      opts.want_tables = loaded.get(capability::CapabilityId::Table);
    if (!request->has_formulas())
      opts.want_formulas = loaded.get(capability::CapabilityId::Formula);
    // Keep the REQUESTED mask in step, then re-gate: this mutates the mask
    // AFTER parse_proto_options already checked it, the same hazard
    // request_gate.h's post_parse hook exists for on the HTTP side.
    opts.requested.request(capability::CapabilityId::Layout);
    opts.requested.request(capability::CapabilityId::Table, opts.want_tables);
    opts.requested.request(capability::CapabilityId::Formula,
                           opts.want_formulas);
    if (auto r = check_structure_backends(opts, loaded); !r.error.empty()) {
      out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                              r.error_code.c_str(), r.error);
      return std::nullopt;
    }
  }

  // fields/editable/movable combination rules — the parity gap this admission
  // path used to have: it set detect_page_fields / want_line_styles /
  // want_movable_regions straight from the proto with no check, so gRPC
  // silently accepted `fields+output=pdf`, `movable` without layout, etc. that
  // HTTP rejects. Same transport-free checker, same messages.
  if (auto r = check_pdf_doc_output_combinations(
          request->fields(), request->editable(), request->movable(),
          want_markdown, want_searchable_pdf, opts.want_text, opts.want_layout);
      !r.error.empty()) {
    out_status = grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                            r.error_code.c_str(), r.error);
    return std::nullopt;
  }

  return PdfAdmitted{std::move(opts), dpi, req_mode, want_markdown,
                     md_as_pages, want_searchable_pdf};
}

} // namespace

grpc::Status OCRServiceImpl::RecognizePDF(grpc::ServerContext *ctx,
                          const ocr::OCRPDFRequest *request,
                          ocr::OCRPDFResponse *response) {
  // Admission first: everything above is a plain grpc::Status returned
  // before a single page is rendered. After this point, work has started —
  // a failure is reported through PdfJobResult's status enum (job-level) or
  // guarded_infer's exception mapping (RPC-level), never by falling back
  // into this function's error shape.
  grpc::Status admit_err;
  auto admitted = admit_pdf_request(ctx, request, loaded_,
                                    mode_ == GrpcResponseMode::json_bytes,
                                    pdf_renderer_ != nullptr && render::PdfRenderer::can_render(), default_pdf_dpi_,
                                    default_pdf_mode_, max_pdf_pages_,
                                    admit_err);
  if (!admitted) return admit_err;

  InferOptions &opts = admitted->opts;
  const bool want_blocks = opts.want_blocks;
  const int dpi = admitted->dpi;
  const pdf::PdfMode req_mode = admitted->mode;
  const bool want_markdown = admitted->want_markdown;
  const bool md_as_pages = admitted->md_as_pages;
  const bool want_searchable_pdf = admitted->want_searchable_pdf;

  const auto *pdf_data = reinterpret_cast<const uint8_t *>(request->pdf_data().data());
  size_t pdf_len = request->pdf_data().size();

  // Transport-agnostic orchestrator (H9 + H3): the shared run_pdf_job runs
  // the exact same per-page pipeline the HTTP /ocr/pdf route runs, submitting
  // page work DIRECTLY onto the bounded dispatcher (no per-page std::async /
  // counting_semaphore). Inline page images stay HTTP-only; autorotate
  // (OCRRequest field 17-parity: OCRPDFRequest.autorotate) is honoured through
  // the same job option the HTTP route sets — the availability gate above
  // already rejected it (AUTOROTATE_DISABLED) when this server cannot rotate.
  pipeline::PdfJobOptions job_opts;
  job_opts.dpi = dpi;
  job_opts.mode = req_mode;
  job_opts.autorotate = request->autorotate();
  job_opts.want_layout = opts.want_layout;
  job_opts.want_reading_order = opts.want_reading_order;
  job_opts.want_blocks = want_blocks;
  job_opts.want_tables = opts.want_tables;
  job_opts.want_formulas = opts.want_formulas;
  job_opts.want_text = opts.want_text;
  // Bound the per-page future join with the configured request deadline so a
  // wedged page can't hang the RPC (no-op on the sequential path).
  // The same opt-in hooks /ocr/pdf sets: neither is installed unless asked, so
  // a plain request never pays for per-page markdown assembly or the page-raster
  // morphology the field detector runs.
  if (want_markdown)
    job_opts.render_page_markdown = pdf::documents::make_pdf_page_markdown_renderer();
  if (request->fields())
    job_opts.detect_page_fields = pdf::documents::make_pdf_page_field_detector();
  job_opts.want_line_styles = request->editable();
  job_opts.want_movable_regions = request->movable();

  pipeline::PdfJobResult job;
  // ONE exception mapping for every RPC (guarded_infer, grpc_service.h)
  // instead of a local try/catch: this used to catch only PoolExhaustedError
  // and std::exception, which meant a TimeoutError from a wedged page fell
  // through to the generic INTERNAL/INFERENCE_ERROR mapping instead of
  // DEADLINE_EXCEEDED. `job` is captured by reference because a successful
  // call still has a job.status to switch on below — unlike guarded_infer's
  // other call sites, OK from this call is not the RPC's final answer.
  grpc::Status infer_status = guarded_infer(ctx, "gRPC pdf error", [&] {
    // auto_verified is resolved inside run_pdf_job, so every transport gets
    // the same answer. orient_fn_ powers autorotate exactly as the HTTP route's
    // OrientFunc does; when it is absent, start_grpc_server cleared
    // DocOrientation from loaded_, so autorotate=true never reaches here.
    job = pipeline::run_pdf_job(
        [this](const cv::Mat &img, const InferOptions &o) {
          // from_pipeline_result, NOT a hand-written field list: infer_result.h
          // owns this conversion in both directions, and a transcription here
          // is one forgotten line away from dropping a degradation signal and
          // making a failed stage look like a clean page.
          return from_pipeline_result(
              run_infer(img, o.want_layout, o.want_reading_order, o.want_tables,
                        o.want_formulas, o.routing_override));
        },
        *pdf_renderer_, pdf_data, pdf_len, job_opts, orient_fn_);
  });
  if (!infer_status.ok()) return infer_status;

  // Wire code and message come from the shared pdf_job.h builders — the same
  // ones the HTTP route uses — so they can't drift (this copy had lost the
  // `first_dropped` detail the HTTP Dropped message carries). Only the gRPC
  // StatusCode, which has no HTTP analogue, is selected here.
  if (job.status != pipeline::PdfJobStatus::Ok) {
    const grpc::StatusCode sc =
        job.status == pipeline::PdfJobStatus::Dropped
            ? grpc::StatusCode::RESOURCE_EXHAUSTED
        : job.status == pipeline::PdfJobStatus::TimedOut
            ? grpc::StatusCode::DEADLINE_EXCEEDED
        : (job.status == pipeline::PdfJobStatus::DecodeFailed ||
           job.status == pipeline::PdfJobStatus::PageFailed)
            ? grpc::StatusCode::INTERNAL
            : grpc::StatusCode::INVALID_ARGUMENT;  // RenderFailed, EmptyPdf
    return grpc_error(ctx, sc, pipeline::wire_code(job.status),
                      pipeline::pdf_job_error_message(job));
  }

  // ---- Whole-document forms. Both go through the SHARED builders the HTTP
  // route uses (pdf_internal.h), not a second assembler here, so the bytes a
  // gRPC client receives are the bytes an HTTP client receives.
  if (want_searchable_pdf) {
    pdf::documents::SearchablePdfOptions sopts;
    sopts.min_confidence = request->min_confidence();
    sopts.editable = request->editable();
    sopts.movable = request->movable();
    // Defaults TRUE over HTTP, hence `optional` on the wire: an unset field must
    // mean "on", not "off" (proto3 cannot tell those apart for a plain bool).
    sopts.mark_regions =
        request->has_mark_regions() ? request->mark_regions() : true;
    auto payload = pdf::documents::build_searchable_pdf(
        job.pages, pdf_data, pdf_len, sopts);
    if (payload.bytes.empty())
      return grpc_error(ctx, grpc::StatusCode::INTERNAL, "PDF_WRITE_FAILED",
                        payload.error);
    response->set_document(std::move(payload.bytes));
    // A PDF missing its text layer on some pages is still a byte-valid PDF, so
    // silence here would be indistinguishable from a complete document.
    if (payload.pages_failed > 0)
      response->set_degraded("searchable_pdf:" +
                             std::to_string(payload.pages_failed) +
                             " page(s) could not be stamped");
    return grpc::Status::OK;
  }
  if (want_markdown) {
    auto payload = pdf::documents::build_pdf_markdown(job.pages, md_as_pages);
    response->set_markdown(std::move(payload.body));
    response->set_degraded(std::move(payload.degraded));
    return grpc::Status::OK;
  }

  // Build response
  for (int i = 0; i < job.num_pages; ++i) {
    auto *page = response->add_pages();
    auto &pg = job.pages[static_cast<size_t>(i)];
    page->set_page_number(i + 1);
    page->set_width(pg.width);
    page->set_height(pg.height);
    page->set_dpi(pg.effective_dpi > 0 ? pg.effective_dpi : dpi);
    page->set_mode(std::string(pdf::mode_name(pg.resolved_mode)));
    page->set_text_layer_quality(std::string(pg.text_layer_quality));

    if (mode_ == GrpcResponseMode::json_bytes) {
      // Shared per-page serializer (H7) — identical body to HTTP's per-page
      // JSON, so the two transports can't drift on result/layout shape.
      page->set_json_response(
          pipeline::serialize_page_results(pg, want_blocks));
    } else {
      fill_page_results(page, pg.results);
    }
  }

  return grpc::Status::OK;
}

void OCRServiceImpl::fill_page_results(ocr::OCRPageResult *page,
                       const std::vector<OCRResultItem> &results) {
  page->mutable_results()->Reserve(static_cast<int>(results.size()));
  for (const auto &item : results) {
    auto *result = page->add_results();
    result->set_text(item.text);
    result->set_confidence(item.confidence);
    result->mutable_bounding_box()->Reserve(4);
    for (int k = 0; k < 4; ++k) {
      auto *bbox = result->add_bounding_box();
      bbox->mutable_x()->Reserve(1);
      bbox->mutable_y()->Reserve(1);
      bbox->add_x(static_cast<float>(item.box[k][0]));
      bbox->add_y(static_cast<float>(item.box[k][1]));
    }
  }
}

} // namespace turbo_ocr::server
