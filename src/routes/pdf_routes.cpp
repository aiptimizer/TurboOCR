#include "turbo_ocr/routes/pdf_routes.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <format>

#include "turbo_ocr/common/logger.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#endif

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/output/markdown_export.h"
#include "turbo_ocr/pdf/page_image_encoder.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pdf_job.h"
#include "simdutf.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/server_types.h"

using turbo_ocr::pipeline::PdfImageMode;
using turbo_ocr::pipeline::PdfJobOptions;
using turbo_ocr::pipeline::PdfJobResult;
using turbo_ocr::pipeline::PdfJobStatus;
using turbo_ocr::pipeline::PdfPageResult;

namespace turbo_ocr::routes {

namespace {

// Render-DPI request bounds — the single definition lives in pdf_job.h so the
// gRPC RecognizePDF check uses the same values (L5). Aliased here so the
// existing unqualified call sites below keep compiling.
using turbo_ocr::pipeline::kMinPdfDpi;
using turbo_ocr::pipeline::kMaxPdfDpi;
// Default render DPI for the CPU route when ?dpi= is absent. The GPU route
// takes its default from ServerConfig (default_dpi); this is the CPU-only path.
constexpr int kCpuDefaultDpi = 100;

// L3 strict-query-params (opt-in). Default OFF preserves the historical
// lenient behavior (unknown params silently ignored). When
// TURBO_OCR_STRICT_QUERY_PARAMS=1, an unrecognized query parameter is a 400
// INVALID_PARAMETER instead. Read once; cached for the process lifetime.
[[nodiscard]] bool strict_query_params_enabled() noexcept {
  static const bool v = server::env_enabled("TURBO_OCR_STRICT_QUERY_PARAMS");
  return v;
}

// When strict mode is on, reject the request if any query parameter is not in
// `allowed`. Returns true (and invokes `callback`) on rejection. No-op (returns
// false) when strict mode is off, so default behavior is byte-identical.
// NOTE: Drogon's getParameters() merges x-www-form-urlencoded body fields with
// the query string. The PDF route accepts multipart (file fields, which do NOT
// land in getParameters()), JSON, or raw bodies — none populate plain form
// params — so the map here is the query string only. The caller still passes
// the multipart-only check downstream; we only gate query keys.
[[nodiscard]] bool reject_unknown_query_params(
    const drogon::HttpRequestPtr &req,
    std::initializer_list<std::string_view> allowed,
    std::function<void(const drogon::HttpResponsePtr &)> &callback) {
  if (!strict_query_params_enabled()) return false;
  for (const auto &kv : req->getParameters()) {
    const std::string &key = kv.first;
    bool known = false;
    for (auto a : allowed)
      if (a == key) { known = true; break; }
    if (!known) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER",
          std::format("Unknown query parameter '{}' "
                      "(TURBO_OCR_STRICT_QUERY_PARAMS=1)", key)));
      return true;
    }
  }
  return false;
}

// Parse a query-param int safely (std::atoi is UB on overflow). Returns the
// `fallback` for empty/non-numeric/out-of-int-range input, so the caller's
// own range check then rejects it deterministically instead of acting on a
// wrapped/garbage value.
[[nodiscard]] int query_int(const std::string &s, int fallback) {
  if (s.empty()) return fallback;
  errno = 0;
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || errno == ERANGE ||
      v < INT_MIN || v > INT_MAX)
    return fallback;
  return static_cast<int>(v);
}

// Image mode for /ocr/pdf?images=... — only "inline" is supported: the page
// image is embedded as base64 in the JSON response. No server-side cache, no
// GET retrieval endpoint.
PdfImageMode parse_image_mode(const std::string &s) noexcept {
  if (s == "1" || s == "true" || s == "on" || s == "yes" || s == "inline")
    return PdfImageMode::Inline;
  return PdfImageMode::None;
}

// Parse image-capture query params (?images=inline&format=png|jpeg|webp&
// quality=1-100&lossless=0/1&png_compression=0-9&max_side=N). Everything is
// per-request; the only env knob is which JPEG encoder backend runs
// (TURBO_PDF_IMAGE_ENCODER gpu|cpu — same bytes either way). Returns an
// error message on invalid values, empty on success.
std::string parse_image_query_params(const drogon::HttpRequestPtr &req,
                                     PdfImageMode &image_mode,
                                     pdf::EncodeOptions &encode_opts) {
  image_mode = PdfImageMode::None;
  encode_opts = {};

  // Explicit-but-unknown values 400 (same contract as dpi/quality below);
  // only an absent parameter selects the default.
  auto images_str = req->getParameter("images");
  if (!images_str.empty()) {
    if (images_str == "0" || images_str == "false" || images_str == "off" ||
        images_str == "no" || images_str == "none") {
      image_mode = PdfImageMode::None;
    } else {
      image_mode = parse_image_mode(std::string(images_str));
      if (image_mode == PdfImageMode::None)
        return "images must be inline (or 0/none to disable)";
    }
  }

  auto fmt_str = req->getParameter("format");
  if (!fmt_str.empty()) {
    if (!pdf::is_valid_page_image_format(fmt_str.c_str()))
      return "format must be one of png, jpeg, webp";
    encode_opts.format = pdf::parse_page_image_format(fmt_str.c_str());
  }

  // lossless: defaults to true (set in EncodeOptions).
  auto lossless_str = req->getParameter("lossless");
  if (!lossless_str.empty()) {
    if (lossless_str == "0" || lossless_str == "false" || lossless_str == "no" || lossless_str == "off")
      encode_opts.lossless = false;
    else if (lossless_str == "1" || lossless_str == "true" || lossless_str == "yes" || lossless_str == "on")
      encode_opts.lossless = true;
    else
      return "lossless must be 0/1/true/false";
  }

  auto png_comp_str = req->getParameter("png_compression");
  if (!png_comp_str.empty()) {
    int c = query_int(std::string(png_comp_str), -1);
    if (c < 0 || c > 9) return "png_compression must be 0-9";
    encode_opts.png_compression = c;
  }

  auto quality_str = req->getParameter("quality");
  if (!quality_str.empty()) {
    int q = query_int(std::string(quality_str), -1);
    if (q < 1 || q > 100)
      return "quality must be 1-100";
    encode_opts.quality = q;
    // An explicit quality means the client wants lossy. Honor it.
    if (lossless_str.empty()) encode_opts.lossless = false;
  }

  auto max_side_str = req->getParameter("max_side");
  if (!max_side_str.empty()) {
    int ms = query_int(std::string(max_side_str), -1);
    if (ms < 0)
      return "max_side must be >= 0";
    encode_opts.max_side = ms;
  }

  return {};
}

/// Helper: extract PDF bytes from a Drogon request (raw, base64 JSON, multipart).
/// Returns true on success, fills pdf_ptr/pdf_len and may fill decoded_buf.
/// On failure, calls cb with 400 and returns false.
bool extract_pdf_bytes(const drogon::HttpRequestPtr &req,
                       std::string &decoded_buf,
                       const char *&pdf_ptr, size_t &pdf_len,
                       const std::function<void(const drogon::HttpResponsePtr &)> &cb) {
  auto ct = req->getHeader("Content-Type");
  if (ct.find("multipart/form-data") != std::string::npos) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
      cb(server::error_response(drogon::k400BadRequest, "INVALID_MULTIPART", "Failed to parse multipart body"));
      return false;
    }
    for (auto &file : parser.getFiles()) {
      const auto &name = file.getItemName();
      if (name == "file" || name == "pdf") {
        decoded_buf.assign(file.fileData(), file.fileLength());
        break;
      }
    }
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_FILE",
          "Multipart request must contain a 'file' or 'pdf' form field"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else if (ct.find("application/json") != std::string::npos) {
    auto json = req->getJsonObject();
    if (!json || !json->isMember("pdf")) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_PDF",
          R"(JSON body must contain {"pdf": "<base64>"})"));
      return false;
    }
    auto b64 = (*json)["pdf"].asString();
    decoded_buf = turbo_ocr::base64_decode(b64);
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64 PDF"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else {
    if (req->body().empty()) {
      cb(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
      return false;
    }
    pdf_ptr = req->body().data();
    pdf_len = req->body().size();
  }
  return true;
}

// Inline page-count guard: emits PDF_TOO_LARGE if the document exceeds the
// configured limit (cfg.max_pdf_pages — honors --max-pdf-pages AND
// MAX_PDF_PAGES, matching gRPC). Returns true on guard-trip (caller aborts).
bool reject_if_too_many_pages(const uint8_t *pdf_data, size_t pdf_len_local,
                               int limit, server::DrogonCallback &cb) {
  pdf::PdfDocument check_doc(pdf_data, pdf_len_local);
  if (!check_doc.ok()) return false;
  int np = check_doc.page_count();
  if (np > limit) {
    cb(server::error_response(drogon::k400BadRequest, "PDF_TOO_LARGE",
        std::format("PDF has {} pages, maximum is {} (set MAX_PDF_PAGES to increase)",
                    np, limit)));
    return true;
  }
  return false;
}

// Append ONE page object (without the surrounding {pages:[...]} envelope) to
// `json_str`. Shared by the batch emitter below and the /ocr/stream NDJSON
// page events so the per-page shape cannot drift between the two.
void append_pdf_page_json(std::string &json_str, PdfPageResult &pg, size_t i,
                          int request_dpi, bool want_blocks,
                          PdfImageMode image_mode,
                          const pdf::EncodeOptions &encode_opts,
                          bool want_orientation) {
  int page_dpi = pg.effective_dpi > 0 ? pg.effective_dpi : request_dpi;
  json_str += "{\"page\":";
  json_str += std::to_string(i + 1);
  json_str += ",\"page_index\":";
  json_str += std::to_string(i);
  json_str += ",\"dpi\":";
  json_str += std::to_string(page_dpi);
  json_str += ",\"width\":";
  json_str += std::to_string(pg.width);
  json_str += ",\"height\":";
  json_str += std::to_string(pg.height);
  json_str += ',';
  // serialize_page_results returns a full "{...}" object; splice its interior
  // (drop the braces) into this page object. An empty "{}" would splice
  // nothing and leave a trailing comma — invalid JSON. The serializer always
  // emits at least "results":[], so this is defensive, not currently reached.
  auto page_json = pipeline::serialize_page_results(pg, want_blocks);
  if (page_json.size() > 2)
    json_str.append(page_json.data() + 1, page_json.size() - 2);
  else
    json_str += "\"results\":[]";
  json_str += ",\"mode\":\"";
  json_str += pdf::mode_name(pg.resolved_mode);
  json_str += "\",\"text_layer_quality\":\"";
  json_str += pg.text_layer_quality;
  json_str += '"';

  // Detected page rotation (the image + boxes were de-rotated upright by it).
  if (want_orientation) {
    json_str += ",\"orientation_deg\":";
    json_str += std::to_string(pg.orientation_deg);
  }

  // Inline page image: base64 of the encoded bytes (simdutf, SIMD path).
  if (image_mode == PdfImageMode::Inline && !pg.encoded_image.empty()) {
    const auto &raw = pg.encoded_image;
    size_t b64_len = ((raw.size() + 2) / 3) * 4;
    std::string b64(b64_len, '\0');
    simdutf::binary_to_base64(
        reinterpret_cast<const char *>(raw.data()), raw.size(),
        b64.data());
    json_str += ",\"image_b64\":\"";
    json_str += b64;
    json_str += "\",\"image_content_type\":\"";
    json_str += pdf::page_image_content_type(encode_opts.format);
    json_str += '"';
  }

  json_str += '}';
}

// Build the final {pages: [...]} JSON envelope. The per-page result body comes
// from the shared serializer (turbo_ocr::pipeline::serialize_page_results) so
// the result/layout/reading_order/blocks shape can't drift from the gRPC path.
// The per-result + per-page byte estimate keeps dense pages from reallocating
// and tiny pages from over-allocating.
std::string emit_pdf_response(std::vector<PdfPageResult> &page_results,
                              int request_dpi, bool want_blocks,
                              PdfImageMode image_mode,
                              const pdf::EncodeOptions &encode_opts,
                              bool want_orientation) {
  size_t n_pages = page_results.size();
  size_t total_results = 0;
  size_t total_image_bytes = 0;
  for (size_t i = 0; i < n_pages; ++i) {
    total_results += page_results[i].results.size() + page_results[i].layout.size();
    total_image_bytes += page_results[i].encoded_image.size();
  }
  std::string json_str;
  json_str.reserve(total_results * 256 + n_pages * 256 + 64 +
                   (total_image_bytes * 4) / 3 + n_pages * 48);
  json_str += "{\"pages\":[";
  for (size_t i = 0; i < n_pages; ++i) {
    if (i > 0) json_str += ',';
    append_pdf_page_json(json_str, page_results[i], i, request_dpi, want_blocks,
                         image_mode, encode_opts, want_orientation);
  }
  json_str += "]}";
  return json_str;
}

// Map a PdfJobResult terminal status to the HTTP error response, matching the
// codes/statuses the route emitted before the orchestrator extraction. Returns
// true (and invokes cb) when the job did NOT succeed.
bool emit_job_error(const PdfJobResult &job, server::DrogonCallback &cb) {
  switch (job.status) {
    case PdfJobStatus::Ok:
      return false;
    case PdfJobStatus::RenderFailed:
      cb(server::error_response(drogon::k400BadRequest, "PDF_RENDER_FAILED",
          "PDF render failed"));
      return true;
    case PdfJobStatus::EmptyPdf:
      cb(server::error_response(drogon::k400BadRequest, "EMPTY_PDF",
          "PDF contains no pages"));
      return true;
    case PdfJobStatus::Dropped:
      cb(server::error_response(drogon::k503ServiceUnavailable, "SERVER_BUSY",
          std::format("GPU queue full: {} of {} pages could not be processed "
                      "(first dropped page: {}). Retry with backoff.",
                      job.dropped_pages, job.num_pages, job.first_dropped)));
      return true;
    case PdfJobStatus::DecodeFailed:
      // We rendered these PPMs ourselves — failing to read them back is a
      // server-side fault (tmpfs pressure, truncated write), not client input.
      cb(server::error_response(drogon::k500InternalServerError,
          "PAGE_DECODE_FAILED",
          std::format("{} of {} rendered pages could not be decoded; retry",
                      job.decode_failures, job.num_pages)));
      return true;
    case PdfJobStatus::PageFailed:
      // A page's OCR/inference threw. Fail the whole request rather than return
      // a 200 with silently-empty pages — partial OCR must never look complete.
      cb(server::error_response(drogon::k500InternalServerError,
          "PAGE_FAILED",
          std::format("{} of {} pages failed during OCR; retry",
                      job.page_failures, job.num_pages)));
      return true;
    case PdfJobStatus::TimedOut:
      // The job-wide deadline (scaled by page count) was exceeded — a 504, not an
      // inference failure. Distinct from PAGE_FAILED so clients retry / raise the
      // deadline rather than treating it as bad input.
      cb(server::error_response(drogon::k504GatewayTimeout, "INFERENCE_TIMEOUT",
          "PDF job exceeded the request deadline"));
      return true;
  }
  return false;
}

// /ocr/pdf?markdown=1 response. Default: one text/markdown document, pages
// prefixed with `<!-- page N -->` (invisible when rendered, splittable by
// chunkers). ?as_pages=1: JSON array of per-page markdown for programmatic
// consumers. The markdown body intentionally drops failed/garbage regions, so
// per-stage degradation is surfaced in the X-OCR-Degraded header (with page
// numbers) and per-page flags in the as_pages shape — never silently.
[[nodiscard]] drogon::HttpResponsePtr
emit_pdf_markdown_response(std::vector<PdfPageResult> &pages, bool as_pages) {
  std::string dt, dtab, df;
  for (size_t i = 0; i < pages.size(); ++i) {
    auto mark = [&](std::string &s) {
      if (!s.empty()) s += ",";
      s += std::to_string(i + 1);
    };
    if (pages[i].text_degraded) mark(dt);
    if (pages[i].table_degraded) mark(dtab);
    if (pages[i].formula_degraded) mark(df);
  }
  std::string degraded;
  auto add = [&](const char *stage, const std::string &plist) {
    if (plist.empty()) return;
    if (!degraded.empty()) degraded += "; ";
    degraded += stage;
    degraded += "(p";
    degraded += plist;
    degraded += ")";
  };
  add("text", dt);
  add("table", dtab);
  add("formula", df);

  drogon::HttpResponsePtr resp;
  if (as_pages) {
    std::string body = "{\"pages\":[";
    for (size_t i = 0; i < pages.size(); ++i) {
      if (i) body += ",";
      body += "{\"page_index\":" + std::to_string(i) + ",\"markdown\":\"";
      turbo_ocr::detail::append_escaped_string(body, pages[i].markdown);
      body += "\"";
      if (pages[i].text_degraded) body += ",\"text_degraded\":true";
      if (pages[i].table_degraded) body += ",\"table_degraded\":true";
      if (pages[i].formula_degraded) body += ",\"formula_degraded\":true";
      body += "}";
    }
    body += "]}";
    resp = server::json_response(std::move(body));
  } else {
    size_t total = 0;
    for (const auto &pg : pages) total += pg.markdown.size() + 24;
    std::string body;
    body.reserve(total);
    for (size_t i = 0; i < pages.size(); ++i) {
      body += "<!-- page ";
      body += std::to_string(i + 1);
      body += " -->\n\n";
      body += pages[i].markdown;
      if (i + 1 < pages.size()) body += "\n\n";
    }
    resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k200OK);
    resp->setBody(std::move(body));
    resp->setContentTypeString("text/markdown; charset=utf-8");
  }
  if (!degraded.empty()) resp->addHeader("X-OCR-Degraded", degraded);
  return resp;
}

// Per-page Markdown renderer shared by the GPU and CPU /ocr/pdf handlers (set as
// PdfJobOptions::render_page_markdown). Moves the finished page's fields into an
// OcrPipelineResult, renders self-contained Markdown (figures embedded as data:
// URIs), and moves them back so the JSON envelope is unaffected. Runs on the
// pipeline worker while the page bitmap is still alive.
[[nodiscard]] std::function<std::string(PdfPageResult &, const cv::Mat &)>
make_pdf_page_markdown_renderer() {
  return [](PdfPageResult &pg, const cv::Mat &img) -> std::string {
    turbo_ocr::assign_layout_ids(pg.results, pg.layout);
    pipeline::OcrPipelineResult res;
    res.results = std::move(pg.results);
    res.layout = std::move(pg.layout);
    res.reading_order = std::move(pg.reading_order);
    res.tables = std::move(pg.tables);
    res.formulas = std::move(pg.formulas);
    std::string md = output::render_markdown_with_assets(
        res, img, /*base_dir=*/".", /*embed_images=*/true);
    pg.results = std::move(res.results);
    pg.layout = std::move(res.layout);
    pg.reading_order = std::move(res.reading_order);
    pg.tables = std::move(res.tables);
    pg.formulas = std::move(res.formulas);
    return md;
  };
}

// Everything /ocr/pdf accepts, parsed and validated. The GPU and CPU
// handlers previously each hand-maintained this ~120-line phase and had
// already drifted (the CPU copy lost the want_text handling); the single
// parser is the drift fix.
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
    bool allow_image_only, PdfRequestParams &out) {
  auto fail = [&](const char *code, const std::string &msg) {
    callback(server::error_response(drogon::k400BadRequest, code, msg));
    return false;
  };

  if (auto r = server::parse_query_options(req, layout_available, &out.opts,
                                           allow_image_only);
      !r.error.empty())
    return fail(r.error_code.c_str(), r.error);

  // ?markdown=1: run the /ocr/markdown pipeline per page and return the
  // assembled Markdown document instead of the JSON envelope.
  if (auto err = server::parse_bool_query(req, "markdown", &out.want_markdown);
      !err.empty())
    return fail("INVALID_PARAMETER", err);
  if (auto err = server::parse_bool_query(req, "as_pages", &out.md_as_pages);
      !err.empty())
    return fail("INVALID_PARAMETER", err);
  if (out.md_as_pages && !out.want_markdown)
    return fail("INVALID_PARAMETER", "as_pages=1 requires markdown=1");
  if (out.want_markdown) {
    if (!layout_available)
      return fail("LAYOUT_DISABLED",
                  "markdown=1 requires the layout model (do not start with "
                  "DISABLE_LAYOUT=1)");
    if (!out.opts.want_text)
      return fail("INVALID_PARAMETER",
                  "text=0 cannot be combined with markdown=1 (markdown needs "
                  "the text)");
    out.opts.want_layout = true;
    out.opts.want_reading_order = true;
    // Faithful-export defaults (mirror /ocr/markdown): stages the server
    // actually loaded run unless the query explicitly disabled them, so a
    // text-only server produces honest text markdown rather than silently
    // dropping table/formula sections. Safe in every mode — geometric pages
    // recognize tables/formulas on the rendered image while KEEPING the exact
    // text layer (run_layout_and_structure), so structure never replaces
    // born-digital text.
    if (req->getParameter("tables").empty()) out.opts.want_tables = table_avail;
    if (req->getParameter("formulas").empty())
      out.opts.want_formulas = formula_avail;
  }

  if (reject_unknown_query_params(
          req, {"layout", "reading_order", "as_blocks", "tables", "formulas",
                "text", "dpi", "mode", "markdown", "as_pages",
                "images", "format", "lossless", "png_compression", "quality",
                "max_side", "autorotate"}, callback))
    return false;
  if (auto r = server::check_structure_backends(out.opts, table_avail,
                                                formula_avail);
      !r.error.empty())
    return fail(r.error_code.c_str(), r.error);

  auto dpi_str = req->getParameter("dpi");
  // Absent -> default; present-but-garbage/overflow -> -1 -> rejected below
  // (don't silently fall back to default on a bad explicit value).
  out.dpi = dpi_str.empty() ? default_dpi : query_int(std::string(dpi_str), -1);
  if (out.dpi < kMinPdfDpi || out.dpi > kMaxPdfDpi)
    return fail("INVALID_DPI", std::format("DPI must be between {} and {}",
                                           kMinPdfDpi, kMaxPdfDpi));

  out.mode = default_pdf_mode;
  auto mode_str = req->getParameter("mode");
  if (!mode_str.empty()) {
    if (!pdf::is_valid_pdf_mode(mode_str))
      return fail("INVALID_PARAMETER",
                  "mode must be one of ocr, geometric, auto, auto_verified");
    out.mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);
  }

  // Page-image export params (?images=inline&format=...&quality=...)
  if (auto err = parse_image_query_params(req, out.image_mode, out.encode_opts);
      !err.empty())
    return fail("INVALID_PARAMETER", err);
  if (out.want_markdown && out.image_mode != PdfImageMode::None)
    return fail("INVALID_PARAMETER",
                "images= page-image export is not available with markdown=1 "
                "(figure crops are already embedded in the markdown)");

  // text=0 on /ocr/pdf must produce SOMETHING: a layout-only run or inline
  // page images (the fast pdf->images path). Bare text=0 returns empty pages.
  if (allow_image_only && !out.opts.want_text && !out.opts.want_layout &&
      out.image_mode != PdfImageMode::Inline)
    return fail("INVALID_PARAMETER",
                "text=0 without layout=1 or images=inline would return empty "
                "pages; add layout=1 (layout-only) and/or images=inline "
                "(page images)");

  // autorotate=1: de-rotate each OCR'd page upright using the doc-orientation
  // model. Rejected when the model isn't loaded (parity with LAYOUT_DISABLED).
  if (auto err = server::parse_bool_query(req, "autorotate", &out.autorotate);
      !err.empty())
    return fail("INVALID_PARAMETER", err);
  if (out.autorotate && !doc_ori_available)
    return fail("AUTOROTATE_DISABLED",
                "autorotate=1 requires the doc-orientation model "
                "(models/doc_ori.onnx); it was not found at startup");

  return true;
}

} // namespace

#ifndef USE_CPU_ONLY
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        bool table_available,
                        bool formula_available,
                        int default_dpi,
                        int max_pdf_pages,
                        bool doc_ori_available) {

  // Availability from the warmed pipeline (single source of truth), threaded
  // from main(); not re-derived from config.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;
  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &dispatcher, &pdf_renderer, default_pdf_mode, layout_available,
       table_avail, formula_avail,
       default_dpi, max_pdf_pages, doc_ori_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    // Extract PDF bytes (lightweight, on event loop)
    auto pdf_buf = std::make_shared<std::string>();
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, *pdf_buf, pdf_ptr, pdf_len, callback))
      return;

    PdfRequestParams p;
    if (!parse_pdf_request(req, callback, layout_available, table_avail,
                           formula_avail, doc_ori_available, default_dpi,
                           default_pdf_mode, /*allow_image_only=*/true, p))
      return;

    const bool layout_enabled = p.opts.want_layout;
    const bool want_reading_order = p.opts.want_reading_order;
    const bool want_blocks = p.opts.want_blocks;
    const bool want_tables = p.opts.want_tables;
    const bool want_formulas = p.opts.want_formulas;
    const bool want_text = p.opts.want_text;
    const bool want_markdown = p.want_markdown;
    const bool md_as_pages = p.md_as_pages;
    const bool autorotate = p.autorotate;
    const int dpi = p.dpi;
    const pdf::PdfMode req_mode = p.mode;
    const PdfImageMode image_mode = p.image_mode;
    const pdf::EncodeOptions encode_opts = p.encode_opts;

    // For raw body case, pdf_ptr points into req->body() — copy into pdf_buf
    if (pdf_buf->empty())
      pdf_buf->assign(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &dispatcher, &pdf_renderer, // req dropped: unused, kept the full request (raw body) resident alongside pdf_buf
         layout_enabled, want_reading_order, want_blocks, want_tables, want_formulas,
         want_text, want_markdown, md_as_pages,
         dpi, req_mode, image_mode,
         encode_opts, max_pdf_pages, autorotate](server::DrogonCallback &cb) {
     // Wrap the whole body: post-render work (emit_pdf_response's multi-GB
     // reserve under images=inline) can throw bad_alloc, which the WorkPool
     // worker would otherwise swallow — leaving the client hung with no
     // response. run_with_error_handling turns it into 500.
     server::run_with_error_handling(cb, "/ocr/pdf", [&] {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, max_pdf_pages, cb)) return;

      PdfJobOptions job_opts;
      job_opts.dpi = dpi;
      job_opts.mode = req_mode;
      job_opts.want_layout = layout_enabled;
      job_opts.want_reading_order = want_reading_order;
      job_opts.want_blocks = want_blocks;
      job_opts.want_tables = want_tables;
      job_opts.want_formulas = want_formulas;
      job_opts.want_text = want_text;
      job_opts.autorotate = autorotate;
      job_opts.image_mode = image_mode;
      job_opts.encode_opts = encode_opts;
      // Bound the per-page future join with the configured request deadline so a
      // wedged page can't hang the whole PDF request (same value the dispatcher
      // applies to single-image submits).
      job_opts.request_timeout_ms = dispatcher.request_timeout_ms();

      if (want_markdown)
        job_opts.render_page_markdown = make_pdf_page_markdown_renderer();

      auto job = pipeline::run_pdf_job(dispatcher, pdf_renderer, pdf_data,
                                       pdf_len_local, job_opts);
      if (emit_job_error(job, cb)) return;

      if (want_markdown) {
        cb(emit_pdf_markdown_response(job.pages, md_as_pages));
        return;
      }

      cb(server::json_response(emit_pdf_response(job.pages, dpi, want_blocks,
                                                  image_mode, encode_opts,
                                                  autorotate)));
     });
    });
  }, {drogon::Post});
}

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
    if (auto r = server::parse_query_options(req, layout_available, &opts,
                                             /*allow_image_only=*/true);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    if (reject_unknown_query_params(
            req, {"layout", "reading_order", "as_blocks", "tables", "formulas", "text",
                  "dpi", "mode",
                  "images", "format", "lossless", "png_compression", "quality",
                  "max_side", "autorotate"}, callback))
      return;
    if (auto r = server::check_structure_backends(opts, table_avail, formula_avail);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }

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
#endif // !USE_CPU_ONLY

// --- CPU overload: sequential page OCR via InferFunc ---
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available,
                        bool table_available,
                        bool formula_available,
                        int max_pdf_pages,
                        server::OrientFunc orient_fn) {
  const bool doc_ori_available = static_cast<bool>(orient_fn);
  // Availability passed in (CPU: env-derived = what actually loaded), not
  // routing-derived — the CPU pipeline loads table/formula from env, not routing.
  const bool table_avail   = table_available;
  const bool formula_avail = formula_available;

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &infer, &pdf_renderer, default_pdf_mode, layout_available,
       table_avail, formula_avail,
       max_pdf_pages, orient_fn, doc_ori_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    std::string decoded_buf;
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, decoded_buf, pdf_ptr, pdf_len, callback))
      return;

    PdfRequestParams p;
    if (!parse_pdf_request(req, callback, layout_available, table_avail,
                           formula_avail, doc_ori_available, kCpuDefaultDpi,
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
    const bool autorotate = p.autorotate;
    const int dpi = p.dpi;
    const pdf::PdfMode req_mode = p.mode;
    const PdfImageMode image_mode = p.image_mode;
    const pdf::EncodeOptions encode_opts = p.encode_opts;

    auto pdf_buf = std::make_shared<std::string>(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &infer, &pdf_renderer, want_layout,
         want_reading_order, want_blocks, want_tables, want_formulas, want_text,
         dpi, want_markdown, md_as_pages,
         req_mode, image_mode, encode_opts, max_pdf_pages,
         autorotate, orient_fn](server::DrogonCallback &cb) {
     // See GPU route: wrap the body so a post-render bad_alloc returns 500
     // instead of being swallowed by the WorkPool (client hang).
     server::run_with_error_handling(cb, "/ocr/pdf", [&] {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, max_pdf_pages, cb)) return;

      // CPU server runs sequentially; the GPU AutoVerified path cross-checks
      // every OCR detection against the text layer in parallel. Doing the same
      // on CPU would require an extra pdfium text_in_rect call per detection
      // per page, doubling latency on a single-thread pipeline. Honest
      // behavior: alias auto_verified to auto on CPU and emit the actually-
      // resolved per-page mode in the response, so clients who set
      // auto_verified get auto's text-layer fast-path without us claiming
      // verification we didn't perform.
      pdf::PdfMode mode = req_mode;
      if (mode == pdf::PdfMode::AutoVerified) mode = pdf::PdfMode::Auto;

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

      auto job = pipeline::run_pdf_job(infer, pdf_renderer, pdf_data,
                                       pdf_len_local, job_opts, orient_fn);
      if (emit_job_error(job, cb)) return;

      if (want_markdown) {
        cb(emit_pdf_markdown_response(job.pages, md_as_pages));
        return;
      }

      cb(server::json_response(emit_pdf_response(job.pages, dpi, want_blocks,
                                                  image_mode, encode_opts,
                                                  autorotate)));
     });
    });
  }, {drogon::Post});
}

} // namespace turbo_ocr::routes
