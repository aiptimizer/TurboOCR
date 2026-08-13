#pragma once

// Transport-agnostic PDF orchestrator (H9).
//
// Both the HTTP /ocr/pdf handler (src/service/http/pdf/pdf_route.cpp) and the gRPC
// RecognizePDF RPC (include/turbo_ocr/service/grpc/grpc_service.h) used to carry a
// near-identical, drifted copy of the per-page PDF pipeline: the text-layer
// pre-pass, the streamed render + OCR fan-out, and the per-page result
// accumulation. This header hosts the single shared implementation; each
// transport calls run_pdf_job() and then serialises the returned per-page
// results into its own envelope (HTTP JSON object vs gRPC proto message).
//
// The orchestrator's PdfJobResult is the contract between the pipeline and the
// transports. It deliberately mirrors the fields the prior code emitted, so
// the serialised output is byte-identical to today's for the same input/mode.
//
// Concurrency: pages are processed by run_streamed_render_cpu's small worker
// pool (TURBO_PDF_PAGE_WORKERS) over the streamed renderer; each page's infer
// takes its own pipeline-pool lease, and the pool's bounded acquire (server
// REQUEST_TIMEOUT_MS) is the backpressure. (An earlier banner here described
// the deleted CUDA dispatcher job — abandoned futures, PdfPageSink shared_ptr
// co-ownership, a per-job deadline field — none of which exists in this
// implementation.)

// Only what the public structs + run_pdf_job declarations need; the heavy
// impl-only headers (<future>, <mutex>, reading_order, pdf_text_layer, …) moved
// to src/pipeline/job/pdf_job.cpp with the bodies.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/core/types.h"
#include "turbo_ocr/analysis/forms/form_field.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pdf/text/font_style.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/text/region_extract.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
// ONLY the two service-boundary aliases (InferFunc / OrientFunc) are needed
// here, and service_fns.h is the leaf that owns them. The server_types.h
// umbrella this used to include pulled Drogon, JsonCpp, the metrics registry
// and the whole validation layer into every TU that touches a PDF page —
// none of which the orchestrator's declarations reference.
#include "turbo_ocr/core/service_fns.h"


namespace turbo_ocr::pipeline {

// Accepted PDF render DPI range — shared by every transport's dpi validation
// (HTTP pdf_routes + gRPC RecognizePDF) so the bound + message can't drift.
constexpr int kMinPdfDpi = 50;
constexpr int kMaxPdfDpi = 600;

// Image-export mode for ?images=... — only "inline" is supported on HTTP; gRPC
// never requests it. Kept here so the orchestrator carries the parity flag.
enum class PdfImageMode { None, Inline };

// One page's resolved output. Shared by every transport. Field-for-field the
// same data the prior HTTP `PdfPageResultBase` / gRPC `PdfPageResult` carried.
struct PdfPageResult {
  std::vector<OCRResultItem>     results;
  std::vector<layout::LayoutBox> layout;
  std::vector<int>               reading_order;
  // Table/formula structure + the per-stage no-silent-failure signals. Previously dropped on
  // /ocr/pdf, so a configured-but-failed table/formula stage was invisible on a PDF page.
  std::vector<router::TableResult>   tables;
  std::vector<router::FormulaResult> formulas;
  bool        formula_degraded = false;
  std::string formula_warning;
  bool        table_degraded = false;
  std::string table_warning;
  bool        text_degraded = false;
  std::string text_warning;
  int width = 0, height = 0, effective_dpi = 0;
  pdf::PdfMode resolved_mode = pdf::PdfMode::Ocr;
  std::string_view text_layer_quality = "absent";
  // Detected page rotation (clockwise, 0/90/180/270) when autorotate=1.
  int orientation_deg = 0;
  // Encoded page-image bytes (set only when image_mode == Inline).
  std::vector<uint8_t> encoded_image;
  // Rendered per-page Markdown (set only when the transport supplied a
  // render_page_markdown hook, i.e. /ocr/pdf?markdown=1).
  std::string markdown;
  // Proposed fillable regions (set only under /ocr/pdf?fields=1). Page pixels,
  // same space as `results`.
  std::vector<forms::FormField> fields;
  // What each recognised line LOOKS like — serif, weight, slant, ink colour.
  // Index-aligned with `results`, and set only when the caller asked for a
  // visible text layer, which is the only thing that needs to know. Measured on
  // the page raster while it is still alive, because it cannot be recovered
  // from the geometry afterwards.
  std::vector<pdf::LineStyle> line_styles;
  // What typeface this page's lines actually look like, decided by rendering
  // candidates and comparing them with the scan. Set alongside line_styles.
  pdf::PageFontMatch font_match;
  // Figures, charts and tables cut out of this page's raster so they can be
  // moved. Set only under ?movable=1, and only where the layout model ran.
  std::vector<pdf::RegionImage> region_images;
  // Printed rules recovered from this page, as shapes rather than pixels.
  std::vector<pdf::RuleShape> rule_shapes;
  // Flat colour blocks — header bars, shaded panels — likewise as shapes.
  std::vector<pdf::BlockShape> block_shapes;
};

// Per-request options for run_pdf_job. Defaults preserve today's behaviour.
struct PdfJobOptions {
  int dpi = 100;
  pdf::PdfMode mode = pdf::PdfMode::Ocr;
  bool want_layout = false;
  bool want_reading_order = false;
  bool want_blocks = false;
  bool want_tables = false;
  bool want_formulas = false;
  // ?text=0 — skip det/rec on every page: geometric pages drop their
  // text-layer text, OCR pages run layout-only (or nothing). Combine with
  // images=inline for a fast pdf->page-images path with no GPU OCR cost.
  bool want_text = true;
  bool autorotate = false;
  // Rendered pages OCR'd concurrently. Each in-flight page takes its own
  // pipeline-pool lease (the pool's bounded acquire is the backpressure), so
  // values beyond the pool size only add idle threads. 0 = resolve from
  // TURBO_PDF_PAGE_WORKERS (default 3); 1 = the old strictly sequential path.
  // Page events still fire exactly once per page, in completion order — the
  // order the streaming transports already document (page_index identifies
  // the page; mixed text/OCR PDFs have always completed out of order).
  int page_workers = 0;
  // Streaming hooks — set by /ocr/stream and gRPC RecognizeStream.
  // on_page_ready receives a COPY of the finished page (emit_page copies
  // deliberately: the non-streaming consumers read the same slot out of the
  // aggregate afterwards, so moving out would blank /ocr/pdf's answer
  // whenever a stream happened to be attached).
  std::function<void(int page_idx, PdfPageResult &&page)> on_page_ready;
  std::function<void(int page_idx)> on_page_failed;
  // Per-page Markdown hook (set by /ocr/pdf?markdown=1). Called on the
  // dispatcher worker after the page is final (post text-layer verification),
  // with the page bitmap still alive, so the pipeline layer stays
  // output-format-agnostic. Must be thread-safe; the returned string is stored
  // in PdfPageResult::markdown.
  std::function<std::string(PdfPageResult &page, const cv::Mat &img)>
      render_page_markdown;
  // Per-page fillable-field hook (set by /ocr/pdf?fields=1). Same contract as
  // render_page_markdown — dispatcher worker, page final, bitmap still alive —
  // because the detectors read the RASTER, not the OCR geometry alone. Null
  // unless the request asked, so fields=1 costs nothing when off. Result is
  // stored in PdfPageResult::fields.
  std::function<std::vector<forms::FormField>(PdfPageResult &page,
                                              const cv::Mat &img)>
      detect_page_fields;
  // ?editable=1 — measure the type each line is set in, at the same point and
  // for the same reason as the two hooks above: the answer is in the raster and
  // nowhere else. A plain flag rather than a hook because the measurement is a
  // pure function of the page and the boxes, with nothing for a transport to
  // decide. Fills PdfPageResult::line_styles.
  bool want_line_styles = false;
  // ?movable=1 — lift each figure, chart, table and seal out of the raster and
  // re-place it as its own object, so a viewer can move one. Costs a JPEG per
  // region, which is why it is opt-in. Needs layout detection to have run.
  bool want_movable_regions = false;
  PdfImageMode image_mode = PdfImageMode::None;
  pdf::EncodeOptions encode_opts{};
};

// NOTE: PdfStreamRenderState used to be declared here. It moved to
// src/pipeline/job/pdf_job_internal.h: it has exactly one producer and one
// consumer, both inside src/pipeline/job/pdf_job.cpp, and the function that
// takes it (detail::run_streamed_render_cpu) is not declared in this header at
// all. Exporting it made every TU that touches a PDF page parse a type it
// cannot use — the opposite of what taking this header off the server_types.h
// umbrella achieved. See the policy three paragraphs down.

// Outcome category for run_pdf_job. The transport maps each to its own error
// surface (HTTP status / gRPC StatusCode) via the shared error_codes table.
enum class PdfJobStatus {
  Ok,
  EmptyPdf,        // no pages -> EMPTY_PDF
  RenderFailed,    // renderer threw -> PDF_RENDER_FAILED
  Dropped,         // GPU queue full mid-stream -> SERVER_BUSY (503 / RES_EXH)
  DecodeFailed,    // rendered PPM unreadable -> PAGE_DECODE_FAILED (500 / INTERNAL)
  PageFailed,      // a page's OCR/inference threw -> PAGE_FAILED (500 / INTERNAL)
  TimedOut,        // job-wide deadline overrun -> INFERENCE_TIMEOUT (504 / DEADLINE_EXCEEDED)
};

struct PdfJobResult {
  PdfJobStatus status = PdfJobStatus::Ok;
  std::vector<PdfPageResult> pages;
  int num_pages = 0;
  int dropped_pages = 0;   // count, for the SERVER_BUSY message
  int first_dropped = -1;  // first dropped page index, for the HTTP message
  int decode_failures = 0;
  int page_failures = 0;   // pages whose inference threw; any > 0 fails the job
};

// The wire error code for a non-Ok job status. This string was written out FOUR
// times (two HTTP paths, two gRPC) — a change had to land in all four or the
// transports disagreed. One spelling here, four call sites keep only their
// envelope.
[[nodiscard]] inline const char *wire_code(PdfJobStatus s) {
  switch (s) {
  case PdfJobStatus::Ok:           return "OK";
  case PdfJobStatus::RenderFailed: return "PDF_RENDER_FAILED";
  case PdfJobStatus::EmptyPdf:     return "EMPTY_PDF";
  case PdfJobStatus::Dropped:      return "SERVER_BUSY";
  case PdfJobStatus::DecodeFailed: return "PAGE_DECODE_FAILED";
  case PdfJobStatus::PageFailed:   return "PAGE_FAILED";
  case PdfJobStatus::TimedOut:     return "INFERENCE_TIMEOUT";
  }
  return "PAGE_FAILED";
}

// The client-facing message for a non-Ok job status, built from the job's own
// counts. Shared by the HTTP and gRPC non-streamed error paths so it cannot
// drift between them — the gRPC copy had already lost the `first_dropped`
// detail the HTTP message carried.
[[nodiscard]] inline std::string pdf_job_error_message(const PdfJobResult &job) {
  switch (job.status) {
  case PdfJobStatus::Ok:           return {};
  case PdfJobStatus::RenderFailed: return "PDF render failed";
  case PdfJobStatus::EmptyPdf:     return "PDF contains no pages";
  case PdfJobStatus::Dropped:
    return std::format("GPU queue full: {} of {} pages could not be processed "
                       "(first dropped page: {}). Retry with backoff.",
                       job.dropped_pages, job.num_pages, job.first_dropped);
  case PdfJobStatus::DecodeFailed:
    return std::format("{} of {} rendered pages could not be decoded; retry",
                       job.decode_failures, job.num_pages);
  case PdfJobStatus::PageFailed:
    return std::format("{} of {} pages failed during OCR; retry",
                       job.page_failures, job.num_pages);
  case PdfJobStatus::TimedOut:
    return "PDF job exceeded the request deadline";
  }
  return "PDF job failed";
}

// ── Public orchestrator API (implementation in src/pipeline/job/pdf_job.cpp) ──
//
// The per-page text-layer pre-pass, streamed render + OCR fan-out, result
// accumulation, and both run_pdf_job overloads live in the .cpp. All the
// helpers they use (fill_from_text_layer_pt, the PdfPageSink, ocr_single_page,
// the streamed-render loops, prepopulate_pages, …) are internal to that TU and
// deliberately not declared here.

// Serialise one finished page to its JSON body (shared by the HTTP /ocr/pdf
// envelope and the gRPC RecognizePDF message so their per-page shape can't
// drift). want_blocks emits the as_blocks grouping.
//
// CONSUMES `pg` on the structure/degraded branch: the pipeline fields
// (results/layout/reading_order/tables/formulas + warnings) are MOVED into
// the serializer there, while the text-only branches leave pg intact. Callers
// must not read those fields afterwards — the non-const reference is a
// consuming pass, not the usual output-parameter convention.
[[nodiscard]] std::string serialize_page_results(PdfPageResult &pg,
                                                 bool want_blocks);

// NOTE: the GPU-only run_pdf_job(PipelineDispatcher&, ...) overload is GONE.
// PipelineDispatcher was typed on the CUDA pipeline, which is why it forced a
// second route family. The InferFunc overload below is backend-neutral and
// already served every vendor; NVIDIA now uses it like everyone else.

// CPU PDF job. Sequential page OCR via the synchronous InferFunc. AutoVerified
// is GPU-only, so the caller aliases it to Auto before invoking.
[[nodiscard]] PdfJobResult run_pdf_job(const server::InferFunc &infer,
                                       render::PdfRenderer &pdf_renderer,
                                       const uint8_t *pdf_data, size_t pdf_len,
                                       const PdfJobOptions &opts,
                                       const server::OrientFunc &orient_fn);

} // namespace turbo_ocr::pipeline
