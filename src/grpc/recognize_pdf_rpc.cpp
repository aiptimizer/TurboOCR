// gRPC RecognizePDF: shared run_pdf_job orchestration, per-page fill.
#include "turbo_ocr/grpc/grpc_service.h"

namespace turbo_ocr::server {

grpc::Status OCRServiceImpl::RecognizePDF(grpc::ServerContext *ctx,
                          const ocr::OCRPDFRequest *request,
                          ocr::OCRPDFResponse *response) {
  if (!pdf_renderer_)
    return grpc_error(ctx, grpc::StatusCode::UNIMPLEMENTED,
                      "PDF_NOT_AVAILABLE",
                      "PDF rendering not available on this server");

  if (request->pdf_data().empty())
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                      "MISSING_PDF", "Empty PDF data");

  if (auto err = grpc_check_layout_request(ctx, request->layout(),
          /*reading_order=*/request->reading_order() ||
          request->as_blocks() ||
          request->tables() || request->formulas(),
          layout_available_); err)
    return *err;

  const auto *pdf_data = reinterpret_cast<const uint8_t *>(request->pdf_data().data());
  size_t pdf_len = request->pdf_data().size();

  bool want_layout = request->layout();
  const bool want_blocks = request->as_blocks();
  // reading_order is independently requestable (field 8, HTTP-parity with
  // /ocr/pdf?reading_order=1); as_blocks still implies it.
  const bool want_reading_order = request->reading_order() || want_blocks;
  const bool want_tables = request->tables();
  const bool want_formulas = request->formulas();
  if (want_reading_order || want_blocks || want_tables || want_formulas)
    want_layout = true;
  if (auto err = grpc_check_structure_backends(ctx, want_tables, want_formulas,
          table_available_, formula_available_,
          mode_ == GrpcResponseMode::json_bytes,
          request->layout(), request->as_blocks()); err)
    return *err;

  int dpi = request->dpi();
  if (dpi == 0) dpi = default_pdf_dpi_;
  if (dpi < pipeline::kMinPdfDpi || dpi > pipeline::kMaxPdfDpi)
    return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT, "INVALID_DPI",
                      std::format("DPI must be between {} and {}",
                                  pipeline::kMinPdfDpi, pipeline::kMaxPdfDpi));

  pdf::PdfMode req_mode = default_pdf_mode_;
  if (!request->mode().empty())
    req_mode = pdf::parse_pdf_mode(request->mode(), default_pdf_mode_);

  // MAX_PDF_PAGES guard — same env var and limit as HTTP /ocr/pdf
  // (default 2000). Mirror the route's reject_if_too_many_pages: open the
  // doc once just for the page count.
  {
    pdf::PdfDocument probe(pdf_data, pdf_len);
    if (probe.ok() && probe.page_count() > max_pdf_pages_)
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
          "PDF_TOO_LARGE",
          std::format("PDF has {} pages, maximum is {} "
                      "(set MAX_PDF_PAGES to increase)",
                      probe.page_count(), max_pdf_pages_));
  }

  // Transport-agnostic orchestrator (H9 + H3): the shared run_pdf_job runs
  // the exact same per-page pipeline the HTTP /ocr/pdf route runs, submitting
  // page work DIRECTLY onto the bounded dispatcher (no per-page std::async /
  // counting_semaphore). gRPC's PDF surface is text-only: it never requests
  // autorotate or inline page images, so those options stay default-off and
  // the serialised output matches today's byte-for-byte.
  pipeline::PdfJobOptions job_opts;
  job_opts.dpi = dpi;
  job_opts.mode = req_mode;
  job_opts.want_layout = want_layout;
  job_opts.want_reading_order = want_reading_order;
  job_opts.want_blocks = want_blocks;
  job_opts.want_tables = want_tables;
  job_opts.want_formulas = want_formulas;
  // Bound the GPU per-page future join with the configured request deadline so
  // a wedged page can't hang the RPC (no-op on the sequential CPU overload).
  job_opts.request_timeout_ms = request_timeout_ms_;

  pipeline::PdfJobResult job;
  try {
#ifndef USE_CPU_ONLY
    if (dispatcher_) {
      job = pipeline::run_pdf_job(*dispatcher_, *pdf_renderer_, pdf_data,
                                  pdf_len, job_opts);
    } else
#endif
    {
      // CPU build: AutoVerified is GPU-only, alias to Auto (parity with the
      // CPU HTTP route). No orientation backend here (gRPC has no autorotate).
      if (job_opts.mode == pdf::PdfMode::AutoVerified)
        job_opts.mode = pdf::PdfMode::Auto;
      job = pipeline::run_pdf_job(
          [this](const cv::Mat &img, const InferOptions &o) {
            auto r = run_infer(img, o.want_layout, o.want_reading_order,
                               o.want_tables, o.want_formulas,
                               o.routing_override);
            return InferResult{
                .results          = std::move(r.results),
                .layout           = std::move(r.layout),
                .reading_order    = std::move(r.reading_order),
                .tables           = std::move(r.tables),
                .formulas         = std::move(r.formulas),
                .formula_degraded = r.formula_degraded,
                .formula_warning  = std::move(r.formula_warning),
                .table_degraded   = r.table_degraded,
                .table_warning    = std::move(r.table_warning),
                .text_degraded    = r.text_degraded,
                .text_warning     = std::move(r.text_warning),
            };
          },
          *pdf_renderer_, pdf_data, pdf_len, job_opts,
          server::OrientFunc{});
    }
  } catch (const turbo_ocr::PoolExhaustedError &e) {
    return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED,
                      "SERVER_BUSY", e.what());
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("gRPC pdf error", "error", e.what());
    return grpc_error(ctx, grpc::StatusCode::INTERNAL, "INFERENCE_ERROR",
                      "Inference error");
  }

  switch (job.status) {
    case pipeline::PdfJobStatus::Ok: break;
    case pipeline::PdfJobStatus::RenderFailed:
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                        "PDF_RENDER_FAILED", "PDF render failed");
    case pipeline::PdfJobStatus::EmptyPdf:
      return grpc_error(ctx, grpc::StatusCode::INVALID_ARGUMENT,
                        "EMPTY_PDF", "PDF contains no pages");
    case pipeline::PdfJobStatus::Dropped:
      return grpc_error(ctx, grpc::StatusCode::RESOURCE_EXHAUSTED, "SERVER_BUSY",
          std::format("GPU queue full: {} of {} pages could not be processed. "
                      "Retry with backoff.", job.dropped_pages, job.num_pages));
    case pipeline::PdfJobStatus::DecodeFailed:
      return grpc_error(ctx, grpc::StatusCode::INTERNAL, "PAGE_DECODE_FAILED",
          std::format("{} of {} rendered pages could not be decoded; retry",
                      job.decode_failures, job.num_pages));
    case pipeline::PdfJobStatus::PageFailed:
      return grpc_error(ctx, grpc::StatusCode::INTERNAL, "PAGE_FAILED",
          std::format("{} of {} pages failed during OCR; retry",
                      job.page_failures, job.num_pages));
    case pipeline::PdfJobStatus::TimedOut:
      return grpc_error(ctx, grpc::StatusCode::DEADLINE_EXCEEDED,
          "INFERENCE_TIMEOUT", "PDF job exceeded the request deadline");
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
