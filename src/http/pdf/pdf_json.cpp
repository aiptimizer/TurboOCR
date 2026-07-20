#include "turbo_ocr/common/encoding.h"
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
    std::string b64 = base64_encode(raw.data(), raw.size());
    json_str += ",\"image_b64\":\"";
    json_str += b64;
    json_str += "\",\"image_content_type\":\"";
    json_str += pdf::page_image_content_type(encode_opts.format);
    json_str += '"';
  }

  json_str += '}';
}

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
drogon::HttpResponsePtr
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
std::function<std::string(PdfPageResult &, const cv::Mat &)>
make_pdf_page_markdown_renderer() {
  return [](PdfPageResult &pg, const cv::Mat &img) -> std::string {
    turbo_ocr::assign_layout_ids(pg.results, pg.layout);
    pipeline::OcrPipelineResult res;
    res.results = std::move(pg.results);
    res.layout = std::move(pg.layout);
    res.reading_order = std::move(pg.reading_order);
    res.tables = std::move(pg.tables);
    res.formulas = std::move(pg.formulas);
    std::string md = markdown::render_markdown_with_assets(
        res, img, /*base_dir=*/".", /*embed_images=*/true);
    pg.results = std::move(res.results);
    pg.layout = std::move(res.layout);
    pg.reading_order = std::move(res.reading_order);
    pg.tables = std::move(res.tables);
    pg.formulas = std::move(res.formulas);
    return md;
  };
}

} // namespace turbo_ocr::routes::pdfdetail
