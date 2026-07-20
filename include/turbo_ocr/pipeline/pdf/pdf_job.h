#pragma once

// Transport-agnostic PDF orchestrator (H9).
//
// Both the HTTP /ocr/pdf handler (src/routes/pdf_routes.cpp) and the gRPC
// RecognizePDF RPC (include/turbo_ocr/grpc/grpc_service.h) used to carry a
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
// Concurrency (H3): page work is submitted DIRECTLY onto the bounded
// PipelineDispatcher queue (the model the HTTP path already used). Backpressure
// is the dispatcher queue depth (PoolExhaustedError -> the caller maps to 503 /
// RESOURCE_EXHAUSTED); there is no per-page std::async / counting_semaphore.
// The GPU job bounds its page-future join against ONE job-wide deadline
// (request_timeout_ms) so a worker wedged on a single page can't hang the whole
// request. A future that overruns is ABANDONED — its task keeps running — so
// every page task co-owns the PdfPageSink and the render StreamHandle by
// shared_ptr (captured by value): the shared state lives until the last task,
// including any abandoned one, drops it, and no task ever touches a dead object.

// Only what the public structs + run_pdf_job declarations need; the heavy
// impl-only headers (<future>, <mutex>, reading_order, pdf_text_layer, …) moved
// to src/pipeline/pdf_job.cpp with the bodies.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/pdf/render/pdf_renderer.h"
#include "turbo_ocr/server/server_types.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pool/pipeline_dispatcher.h"
#endif

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
  // Streaming hooks (see PdfPageSink) — set only by /ocr/stream. on_page_ready
  // receives the finished page MOVED OUT of the sink (the aggregate result then
  // carries an empty slot for it — the streaming route never reads job.pages).
  std::function<void(int page_idx, PdfPageResult &&page)> on_page_ready;
  std::function<void(int page_idx)> on_page_failed;
  // Per-page Markdown hook (set by /ocr/pdf?markdown=1). Called on the
  // dispatcher worker after the page is final (post text-layer verification),
  // with the page bitmap still alive, so the pipeline layer stays
  // output-format-agnostic. Must be thread-safe; the returned string is stored
  // in PdfPageResult::markdown.
  std::function<std::string(PdfPageResult &page, const cv::Mat &img)>
      render_page_markdown;
  PdfImageMode image_mode = PdfImageMode::None;
  pdf::EncodeOptions encode_opts{};
  // Per-request deadline (ms; 0 = unbounded). The GPU job bounds its page-future
  // join against this so a worker wedged mid-page can't hang the whole request.
  long request_timeout_ms = 0;
};

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

// ── Public orchestrator API (implementation in src/pipeline/pdf_job.cpp) ──
//
// The per-page text-layer pre-pass, streamed render + OCR fan-out, result
// accumulation, and both run_pdf_job overloads live in the .cpp. All the
// helpers they use (fill_from_text_layer_pt, the PdfPageSink, ocr_single_page,
// the streamed-render loops, prepopulate_pages, …) are internal to that TU and
// deliberately not declared here.

// Serialise one finished page to its JSON body (shared by the HTTP /ocr/pdf
// envelope and the gRPC RecognizePDF message so their per-page shape can't
// drift). want_blocks emits the as_blocks grouping.
[[nodiscard]] std::string serialize_page_results(PdfPageResult &pg,
                                                 bool want_blocks);

#ifndef USE_CPU_ONLY
// GPU PDF job. Submits page work directly onto the dispatcher (H3). Page tasks
// co-own the sink + StreamHandle by shared_ptr so a task abandoned on a
// deadline overrun stays memory-safe. Backpressure is the dispatcher queue
// depth (PoolExhaustedError -> status=Dropped -> caller 503 / RES_EXHAUSTED).
[[nodiscard]] PdfJobResult run_pdf_job(PipelineDispatcher &dispatcher,
                                       render::PdfRenderer &pdf_renderer,
                                       const uint8_t *pdf_data, size_t pdf_len,
                                       const PdfJobOptions &opts);
#endif

// CPU PDF job. Sequential page OCR via the synchronous InferFunc. AutoVerified
// is GPU-only, so the caller aliases it to Auto before invoking.
[[nodiscard]] PdfJobResult run_pdf_job(const server::InferFunc &infer,
                                       render::PdfRenderer &pdf_renderer,
                                       const uint8_t *pdf_data, size_t pdf_len,
                                       const PdfJobOptions &opts,
                                       const server::OrientFunc &orient_fn);

} // namespace turbo_ocr::pipeline
