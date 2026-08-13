#include "turbo_ocr/service/http/pdf_routes.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <format>

#include "turbo_ocr/base/log/logger.h"


#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pipeline/job/pdf_job.h"
#include "simdutf.h"
#include "turbo_ocr/base/env_utils.h"
#include "turbo_ocr/service/validation/request_gate.h"
#include "turbo_ocr/service/server/server_types.h"

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
      cb(server::error_response(server::ErrorCode::kInvalidMultipart,
                                "Failed to parse multipart body"));
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
      cb(server::error_response(server::ErrorCode::kMissingFile,
                                "Multipart request must contain a 'file' or 'pdf' form field"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else if (ct.find("application/json") != std::string::npos) {
    auto json = req->getJsonObject();
    if (!json || !json->isMember("pdf")) {
      cb(server::error_response(server::ErrorCode::kMissingPdf,
                                R"(JSON body must contain {"pdf": "<base64>"})"));
      return false;
    }
    auto b64 = (*json)["pdf"].asString();
    decoded_buf = turbo_ocr::base64_decode(b64);
    if (decoded_buf.empty()) {
      cb(server::error_response(server::ErrorCode::kBase64DecodeFailed,
                                "Failed to decode base64 PDF"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else {
    if (req->body().empty()) {
      cb(server::error_response(server::ErrorCode::kEmptyBody, "Empty body"));
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
                               int limit, server::DrogonCallback &cb,
                               int *out_pages) {
  pdf::PdfDocument check_doc(pdf_data, pdf_len_local);
  if (!check_doc.ok()) {
    if (out_pages) *out_pages = -1;
    return false;
  }
  int np = check_doc.page_count();
  if (out_pages) *out_pages = np;
  if (np > limit) {
    cb(server::error_response(server::ErrorCode::kPdfTooLarge,
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
    const capability::CapabilityMask &loaded, int default_dpi, pdf::PdfMode default_pdf_mode,
    bool allow_image_only, PdfRequestParams &out) {
  auto fail = [&](const char *code, const std::string &msg) {
    callback(server::error_response(code, msg));
    return false;
  };

  // THE SHARED GATE — same call every other inference endpoint makes.
  //
  // This route used to hand-sequence the gate's steps (parse_query_options
  // here, validate_params ~100 lines below, check_structure_backends after
  // that), because its ?markdown=1 defaults MUTATE the parsed options and so
  // need a second capability check that the one-shot gate had no seam for. That
  // seam is now `post_parse`, so the sequencing lives in request_gate.h once
  // instead of being reproduced — and drifting — here.
  //
  // The hook below carries everything that must run between the parse and the
  // classification. Errors are returned, not emitted: the gate owns the 400.
  const auto post_parse =
      [&](server::InferOptions *opts) -> server::ValidationError {
    const auto bad = [](const std::string &msg) {
      return server::ValidationError{msg, "INVALID_PARAMETER"};
    };

    // ?markdown=1: run the /ocr/markdown pipeline per page and return the
    // assembled Markdown document instead of the JSON envelope.
    if (auto err = server::parse_bool_query(req, "markdown", &out.want_markdown);
        !err.empty())
      return bad(err);
    if (auto err = server::parse_bool_query(req, "as_pages", &out.md_as_pages);
        !err.empty())
      return bad(err);
    if (out.md_as_pages && !out.want_markdown)
      return bad("as_pages=1 requires markdown=1");

    // ?output=pdf: stamp the recognised words back onto the source document as
    // an invisible text layer and return that, rather than geometry the caller
    // would have to render itself.
    if (auto output = req->getParameter("output"); !output.empty()) {
      if (output == "pdf") out.want_searchable_pdf = true;
      else if (output != "json")
        return bad("output must be json or pdf");
    }
    if (auto floor = req->getParameter("min_confidence"); !floor.empty()) {
      if (!out.want_searchable_pdf)
        return bad("min_confidence requires output=pdf");
      // The string must outlive the *end read below — strtod's end pointer
      // points INTO this buffer, so a temporary here is a dangling read.
      const std::string floor_s(floor);
      char *end = nullptr;
      const double v = std::strtod(floor_s.c_str(), &end);
      // Inclusion test, not exclusion: `v < 0 || v > 1` is FALSE for NaN, so
      // strtod("nan") sailed through and every keep() check (r.confidence >=
      // NaN) then returned false — a 200 with a byte-valid PDF whose text layer
      // is silently empty, indistinguishable from a blank scan.
      if (end == floor_s.c_str() || *end != '\0' || !(v >= 0.0 && v <= 1.0))
        return bad("min_confidence must be between 0 and 1");
      out.min_confidence = static_cast<float>(v);
    }
    if (out.want_searchable_pdf) {
      if (out.want_markdown)
        return bad("output=pdf cannot be combined with markdown=1");
      if (!opts->want_text)
        return bad("text=0 cannot be combined with output=pdf (a searchable "
                   "PDF needs the text)");
    }
    if (out.want_markdown) {
      if (!loaded.get(capability::CapabilityId::Layout)) {
        // Code from capability_table.def, which owns it — not a literal a
        // rename would leave stale.
        return server::ValidationError{
            "markdown=1 requires the layout model (do not start with "
            "DISABLE_LAYOUT=1)",
            std::string(
                capability::capability_info(capability::CapabilityId::Layout)
                    .error_code)};
      }
      if (!opts->want_text)
        return bad("text=0 cannot be combined with markdown=1 (markdown needs "
                   "the text)");
      opts->want_layout = true;
      opts->want_reading_order = true;
      // Faithful-export defaults (mirror /ocr/markdown): stages the server
      // actually loaded run unless the query explicitly disabled them, so a
      // text-only server produces honest text markdown rather than silently
      // dropping table/formula sections. Safe in every mode — geometric pages
      // recognize tables/formulas on the rendered image while KEEPING the exact
      // text layer (run_layout_and_structure), so structure never replaces
      // born-digital text.
      //
      // THIS is the mutation that made the shared gate unusable here: it adds
      // to opts->requested AFTER the parse already gated it, so the gate must
      // re-check. request_gate.h now does exactly that whenever a hook ran.
      if (req->getParameter("tables").empty())
        opts->want_tables = loaded.get(capability::CapabilityId::Table);
      if (req->getParameter("formulas").empty())
        opts->want_formulas = loaded.get(capability::CapabilityId::Formula);
      // Keep the mask the gate reads in step with the bools just set — they are
      // the same request, and check_structure_backends reads the mask.
      // request(), not set(), on the REQUESTED axis: set() is literal and would
      // drop the capability's dependencies (capability.h:105-127). Table and
      // Formula both TURBO_CAPABILITY_IMPLIES(Layout), so this only happened to
      // be correct because Layout is requested on the line above — an ordering
      // accident, not a construction. request(id, false) is identical to
      // set(id, false), so there is no reason for the exception. This was the
      // only set() on a requested mask in the tree.
      opts->requested.request(capability::CapabilityId::Layout);
      opts->requested.request(capability::CapabilityId::Table,
                              opts->want_tables);
      opts->requested.request(capability::CapabilityId::Formula,
                              opts->want_formulas);
    }
    return {};
  };

  // PDF endpoints act on EVERY capability, autorotate included — they rotate
  // the page before OCR — so acts_on is all(), and autorotate is parsed and
  // availability-gated (AUTOROTATE_DISABLED) by the shared parser rather than
  // by a route-local copy that could drift from the mask the gate enforces.
  server::EndpointSpec spec;
  spec.acts_on = capability::CapabilityMask::all();
  spec.pdf_options = true;
  spec.pdf_doc_params = true; // output/min_confidence/fields/editable/movable/mark_regions
  spec.routing_unsupported_reason = server::kRoutingUnsupportedPdf;
  if (!server::validate_request(req, spec, loaded, /*valid_route_table=*/{},
                                /*valid_route_formula=*/{}, &out.opts, callback,
                                allow_image_only, /*body=*/nullptr, post_parse))
    return false; // the gate already answered with the 400

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

  // fields=1: propose fillable-field rectangles from page geometry (rules,
  // closed boxes, checkboxes, label+gap, empty table cells). Strictly opt-in —
  // the detectors binarise and morphologically open the whole page raster, so
  // the default request must not pay for them. Needs the text to label the
  // fields and to know a box is empty, so text=0 is refused rather than
  // silently returning unlabelled geometry.
  // fields=1: propose fillable-field rectangles from page geometry. editable=1:
  // rewrite the words as real type. movable=1: lift figures into objects. All
  // opt-in (each runs extra per-page work the default must not pay for) and all
  // constrained in how they combine with text/markdown/output — those
  // combination rules are CONTRACT, not transport, so they live in one
  // transport-free checker (check_pdf_doc_output_combinations) that gRPC's
  // admit path calls with the same messages. Read the three flags, then check.
  if (auto err = server::parse_bool_query(req, "fields", &out.want_fields);
      !err.empty())
    return fail("INVALID_PARAMETER", err);
  if (auto err = server::parse_bool_query(req, "editable", &out.want_editable);
      !err.empty())
    return fail("INVALID_PARAMETER", err);
  if (auto err = server::parse_bool_query(req, "movable", &out.want_movable);
      !err.empty())
    return fail("INVALID_PARAMETER", err);
  if (auto r = server::check_pdf_doc_output_combinations(
          out.want_fields, out.want_editable, out.want_movable, out.want_markdown,
          out.want_searchable_pdf, out.opts.want_text, out.opts.want_layout);
      !r.error.empty())
    return fail(r.error_code.c_str(), r.error);

  // Only parsed when actually PRESENT. parse_bool_query clears its output
  // before it looks, so calling it unconditionally would turn every default-on
  // flag off the moment the caller left it out — which is exactly what happened
  // here: region outlines stopped being drawn for anyone who never asked about
  // them.
  if (!req->getParameter("mark_regions").empty()) {
    if (auto err =
            server::parse_bool_query(req, "mark_regions", &out.want_mark_regions);
        !err.empty())
      return fail("INVALID_PARAMETER", err);
  }

  // autorotate=1: de-rotate each OCR'd page upright using the doc-orientation
  // model. Parsed AND availability-gated by the shared parse_query_options
  // above (this endpoint's acts_on includes DocOrientation), so by this point
  // the flag is either honoured or the request was already rejected with
  // AUTOROTATE_DISABLED.

  return true;
}

} // namespace turbo_ocr::routes::pdfdetail
