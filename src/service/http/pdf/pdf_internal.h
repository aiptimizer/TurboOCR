#pragma once

#include "turbo_ocr/core/capability.h"
#include <functional>
#include <string>
#include <vector>

#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pipeline/job/pdf_job.h"
#include "turbo_ocr/pdf/pdf_documents.h"
#include "turbo_ocr/service/server/server_types.h"

// Internal to src/routes/: PDF request parsing (pdf_request.cpp) and
// response emission (pdf_json.cpp) shared by the /ocr/pdf route TU
// (pdf_route.cpp, GPU + CPU overloads) and /ocr/stream
// (pdf_stream_route.cpp). Public consumers use the registrars declared in
// include/turbo_ocr/service/http/pdf_routes.h.
namespace turbo_ocr::routes::pdfdetail {

using pipeline::PdfImageMode;
using pipeline::PdfJobResult;
using pipeline::PdfPageResult;

// Render-DPI request bounds — the single definition lives in pdf_job.h so
// the gRPC RecognizePDF check uses the same values.
using pipeline::kMinPdfDpi;
using pipeline::kMaxPdfDpi;
// Default render DPI for the CPU route when ?dpi= is absent. The GPU route
// takes its default from ServerConfig (default_dpi); this is the CPU path.
inline constexpr int kCpuDefaultDpi = 100;

// Parse a query-param int safely (std::atoi is UB on overflow). Returns the
// `fallback` for empty/non-numeric/out-of-int-range input, so the caller's
// own range check then rejects it deterministically instead of acting on a
// wrapped/garbage value.
[[nodiscard]] int query_int(const std::string &s, int fallback);

// Everything /ocr/pdf accepts, parsed and validated. The GPU and CPU
// handlers previously each hand-maintained this ~120-line phase and had
// already drifted; the single parser is the drift fix.
struct PdfRequestParams {
  server::InferOptions opts;
  bool want_markdown = false;
  bool md_as_pages = false;
  // ?output=pdf — answer with the source document plus an invisible text
  // layer instead of the JSON envelope.
  bool want_searchable_pdf = false;
  // ?min_confidence= — drop recognised words below this before stamping.
  float min_confidence = 0.0f;
  // ?fields=1 — propose fillable-field rectangles per page ("Prepare Form").
  bool want_fields = false;
  // ?editable=1 (with output=pdf) — draw the recognised words as real type in
  // place of the print, instead of hiding them behind it, so the text can be
  // read and retyped rather than only searched.
  bool want_editable = false;
  // ?movable=1 (with output=pdf) — re-place each figure, chart, table and seal
  // as its own object so a viewer can move one, instead of leaving them as
  // pixels inside the page image.
  bool want_movable = false;
  // ?mark_regions=0 — suppress the outline annotation each figure otherwise
  // gets. On by default so layout=1 keeps meaning what it meant.
  bool want_mark_regions = true;
  // NOTE: there is deliberately NO `bool autorotate` here. It lives in
  // `opts.want_autorotate`, projected from the REQUESTED capability mask by
  // parse_query_options (which can also REJECT with AUTOROTATE_DISABLED, making
  // it the authoritative source). A second copy on this struct, hand-synced in
  // parse_pdf_request, was the one request flag on this type stored twice — and
  // the copy was the one that could go stale. Read `p.opts.want_autorotate`.
  int dpi = 0;
  pdf::PdfMode mode = pdf::PdfMode::Ocr;
  PdfImageMode image_mode = PdfImageMode::None;
  pdf::EncodeOptions encode_opts;
};

// Returns false after answering the request with a 400 through `callback`.
// allow_image_only mirrors parse_query_options: the GPU route accepts
// text=0 (layout-only / image-only responses), the CPU route does not.
[[nodiscard]] bool parse_pdf_request(
    const drogon::HttpRequestPtr &req,
    std::function<void(const drogon::HttpResponsePtr &)> &callback,
    const capability::CapabilityMask &loaded,
    int default_dpi, pdf::PdfMode default_pdf_mode,
    bool allow_image_only, PdfRequestParams &out);

// Parse the page-image capture query params (?images=inline&format=...).
// Returns an error message on invalid values, empty on success.
[[nodiscard]] std::string parse_image_query_params(
    const drogon::HttpRequestPtr &req, PdfImageMode &image_mode,
    pdf::EncodeOptions &encode_opts);

// Extract PDF bytes from a request (raw, base64 JSON, multipart). On failure
// the 400 went through `cb` and false returns; decoded_buf may own decoded
// bytes that pdf_ptr points into.
[[nodiscard]] bool extract_pdf_bytes(
    const drogon::HttpRequestPtr &req, std::string &decoded_buf,
    const char *&pdf_ptr, size_t &pdf_len,
    const std::function<void(const drogon::HttpResponsePtr &)> &cb);

// Pre-parse page-count cap (MAX_PDF_PAGES) — rejects before any render work.
// `out_pages`, when non-null, receives the page count (or -1 when the
// document does not open) so a caller that needs it — /ocr/stream's meta
// event — does not open the PDF a second time or hand-roll the guard.
[[nodiscard]] bool reject_if_too_many_pages(const uint8_t *pdf_data,
                                            size_t pdf_len_local,
                                            int max_pdf_pages,
                                            server::DrogonCallback &cb,
                                            int *out_pages = nullptr);

// Serialize one page into json_str (the /ocr/pdf pages[] element shape;
// also the /ocr/stream "page" event payload). want_fields defaults off so
// /ocr/stream, which never sets the detector hook, is unaffected.
void append_pdf_page_json(std::string &json_str, PdfPageResult &pg, size_t i,
                          int request_dpi, bool want_blocks,
                          PdfImageMode image_mode,
                          const pdf::EncodeOptions &encode_opts,
                          bool want_orientation, bool want_fields = false);

// Serialize the whole-job JSON response ({"pages":[...]}).
[[nodiscard]] std::string emit_pdf_response(
    std::vector<PdfPageResult> &page_results, int request_dpi,
    bool want_blocks, PdfImageMode image_mode,
    const pdf::EncodeOptions &encode_opts, bool want_orientation,
    bool want_fields = false);

// Emit the job-level error response when the job failed; true if handled.
[[nodiscard]] bool emit_job_error(const PdfJobResult &job,
                                  server::DrogonCallback &cb);

// TRANSPORT-NEUTRAL payload for the markdown response: the body bytes plus the
// degradation summary the HTTP emitter puts in X-OCR-Degraded. Split out so gRPC
// can return the identical document instead of growing a second markdown
// assembler — the same rule that governs every other shared policy here.
// The transport-free builders now live in turbo_ocr::pdf::documents
// (include/turbo_ocr/pdf/pdf_documents.h) so the gRPC RPCs can reach them
// without including this HTTP-private header — which is what used to drag
// <drogon/...> into a gRPC translation unit. Re-exported here so the HTTP call
// sites keep their existing spelling.
using pdf::documents::PdfMarkdownPayload;
using pdf::documents::SearchablePdfOptions;
using pdf::documents::SearchablePdfPayload;
using pdf::documents::build_pdf_markdown;
using pdf::documents::build_searchable_pdf;
using pdf::documents::make_pdf_page_field_detector;
using pdf::documents::make_pdf_page_markdown_renderer;

[[nodiscard]] drogon::HttpResponsePtr
emit_pdf_markdown_response(std::vector<PdfPageResult> &pages, bool as_pages);

[[nodiscard]] drogon::HttpResponsePtr
emit_searchable_pdf_response(std::vector<PdfPageResult> &pages,
                             const uint8_t *pdf_data, size_t pdf_len,
                             const SearchablePdfOptions &opts);


} // namespace turbo_ocr::routes::pdfdetail
