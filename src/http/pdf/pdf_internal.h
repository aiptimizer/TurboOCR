#pragma once

#include <functional>
#include <string>
#include <vector>

#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pipeline/pdf/pdf_job.h"
#include "turbo_ocr/server/server_types.h"

// Internal to src/routes/: PDF request parsing (pdf_request.cpp) and
// response emission (pdf_json.cpp) shared by the /ocr/pdf route TU
// (pdf_route.cpp, GPU + CPU overloads) and /ocr/stream
// (pdf_stream_route.cpp). Public consumers use the registrars declared in
// include/turbo_ocr/http/pdf_routes.h.
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
  bool autorotate = false;
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
    bool layout_available, bool table_avail, bool formula_avail,
    bool doc_ori_available, int default_dpi, pdf::PdfMode default_pdf_mode,
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
[[nodiscard]] bool reject_if_too_many_pages(const uint8_t *pdf_data,
                                            size_t pdf_len_local,
                                            int max_pdf_pages,
                                            server::DrogonCallback &cb);

// Serialize one page into json_str (the /ocr/pdf pages[] element shape;
// also the /ocr/stream "page" event payload).
void append_pdf_page_json(std::string &json_str, PdfPageResult &pg, size_t i,
                          int request_dpi, bool want_blocks,
                          PdfImageMode image_mode,
                          const pdf::EncodeOptions &encode_opts,
                          bool want_orientation);

// Serialize the whole-job JSON response ({"pages":[...]}).
[[nodiscard]] std::string emit_pdf_response(
    std::vector<PdfPageResult> &page_results, int request_dpi,
    bool want_blocks, PdfImageMode image_mode,
    const pdf::EncodeOptions &encode_opts, bool want_orientation);

// Emit the job-level error response when the job failed; true if handled.
[[nodiscard]] bool emit_job_error(const PdfJobResult &job,
                                  server::DrogonCallback &cb);

// Whole-document (or as_pages) markdown response.
[[nodiscard]] drogon::HttpResponsePtr
emit_pdf_markdown_response(std::vector<PdfPageResult> &pages, bool as_pages);

// Per-page markdown renderer callback for the PDF job, run on the pipeline
// worker while the page bitmap is still alive.
[[nodiscard]] std::function<std::string(PdfPageResult &, const cv::Mat &)>
make_pdf_page_markdown_renderer();

} // namespace turbo_ocr::routes::pdfdetail
