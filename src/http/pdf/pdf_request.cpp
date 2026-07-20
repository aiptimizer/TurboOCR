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

namespace turbo_ocr::routes::pdfdetail {

int query_int(const std::string &s, int fallback) {
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

namespace {
PdfImageMode parse_image_mode(const std::string &s) noexcept {
  if (s == "1" || s == "true" || s == "on" || s == "yes" || s == "inline")
    return PdfImageMode::Inline;
  return PdfImageMode::None;
}


/// Helper: extract PDF bytes from a Drogon request (raw, base64 JSON, multipart).
/// Returns true on success, fills pdf_ptr/pdf_len and may fill decoded_buf.
/// On failure, calls cb with 400 and returns false.
} // namespace

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


// Build the final {pages: [...]} JSON envelope. The per-page result body comes
// from the shared serializer (turbo_ocr::pipeline::serialize_page_results) so
// the result/layout/reading_order/blocks shape can't drift from the gRPC path.

// Returns false after answering the request with a 400 through `callback`.
// allow_image_only mirrors parse_query_options: the GPU route accepts
// text=0 (layout-only / image-only responses), the CPU route does not.
bool parse_pdf_request(
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

  {
    // Parameter classification + PDF routing policy (request_validation.h).
    // The spec's default ocr_options=true keeps the OCR param names in the
    // allowed set; validate_params only CLASSIFIES — parse_query_options
    // already ran above and is not re-run here.
    server::EndpointSpec spec;
    spec.pdf_options = true;
    spec.routing_unsupported_reason = server::kRoutingUnsupportedPdf;
    std::vector<std::string> ignored;
    if (auto e = server::validate_params(server::query_only_params(req), spec,
                                         /*valid_route_table=*/{},
                                         /*valid_route_formula=*/{},
                                         server::strict_query_params_enabled(),
                                         &out.opts.routing_override, &ignored);
        !e.ok())
      return fail(e.code.c_str(), e.message);
    if (!ignored.empty()) {
      // Same deprecation surface as the request gate (v4 rejects these).
      std::string csv;
      for (const auto &n : ignored) {
        if (!csv.empty()) csv += ',';
        csv += n;
      }
      req->addHeader("X-Ignored-Params", csv);  // relayed by the middleware
      TOCR_LOG_WARN_RL("Ignoring unsupported query parameter(s) — deprecated "
                       "tolerance, v4 rejects with 400",
                       "params", csv, "path", req->path());
    }
  }
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

} // namespace turbo_ocr::routes::pdfdetail
