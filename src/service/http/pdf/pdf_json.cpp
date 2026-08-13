#include "turbo_ocr/base/encoding.h"
#include "turbo_ocr/service/http/pdf_routes.h"

#include <cerrno>
#include <climits>
#include <cstdlib>

#include "turbo_ocr/base/log/logger.h"


#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/serialization/serialization.h"
#include "turbo_ocr/image/image_config.h"
#include "turbo_ocr/document/markdown_export.h"
#include "turbo_ocr/image/page_image_encoder.h"
#include "turbo_ocr/pdf/text/pdf_searchable.h"
#include "turbo_ocr/analysis/forms/field_detector.h"
#include "turbo_ocr/analysis/forms/field_model.h"
#include "turbo_ocr/analysis/forms/field_serialization.h"
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
#include "turbo_ocr/pdf/pdf_documents.h"

namespace turbo_ocr::routes::pdfdetail {

// Append ONE page object (without the surrounding {pages:[...]} envelope) to
// `json_str`. Shared by the batch emitter below and the /ocr/stream NDJSON
// page events so the per-page shape cannot drift between the two.
void append_pdf_page_json(std::string &json_str, PdfPageResult &pg, size_t i,
                          int request_dpi, bool want_blocks,
                          PdfImageMode image_mode,
                          const pdf::EncodeOptions &encode_opts,
                          bool want_orientation, bool want_fields) {
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

  // ?fields=1 — proposed fillable regions. Emitted whenever the detector RAN,
  // including as an empty array: "we looked and found none" and "we did not
  // look" must not be the same response, or a plain page is indistinguishable
  // from a broken detector. Absent entirely when fields=0.
  if (want_fields) forms::append_fields_array(json_str, pg.fields);

  // Inline page image: base64 of the encoded bytes (simdutf, SIMD path).
  // Same "looked-and-found-nothing must differ from didn't-look" rule as the
  // fields array above: when images=inline was requested but the encode failed
  // (page_image_encoder returns {} — e.g. WebP's 16383-px side limit sits just
  // under our 16384 per-side cap, so a legal max-side page silently produced no
  // image), emit an explicit image_error rather than omit the key, which would
  // be byte-identical to "images were never requested".
  if (image_mode == PdfImageMode::Inline) {
    if (!pg.encoded_image.empty()) {
      const auto &raw = pg.encoded_image;
      std::string b64 = base64_encode(raw.data(), raw.size());
      json_str += ",\"image_b64\":\"";
      json_str += b64;
      json_str += "\",\"image_content_type\":\"";
      json_str += pdf::page_image_content_type(encode_opts.format);
      json_str += '"';
    } else {
      json_str += ",\"image_error\":\"ENCODE_FAILED\"";
    }
  }

  json_str += '}';
}

// The per-result + per-page byte estimate keeps dense pages from reallocating
// and tiny pages from over-allocating.
std::string emit_pdf_response(std::vector<PdfPageResult> &page_results,
                              int request_dpi, bool want_blocks,
                              PdfImageMode image_mode,
                              const pdf::EncodeOptions &encode_opts,
                              bool want_orientation, bool want_fields) {
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
                         image_mode, encode_opts, want_orientation, want_fields);
  }
  json_str += "]}";
  return json_str;
}

// Map a PdfJobResult terminal status to the HTTP error response. Wire code,
// HTTP status, and message all come from the shared pdf_job.h builders (the
// same ones the gRPC and streaming paths use), so the four transports cannot
// drift. error_response(string_view) resolves the HTTP status from the wire
// code via error_codes.h — the same status each ErrorCode carried before.
// Returns true (and invokes cb) when the job did NOT succeed.
bool emit_job_error(const PdfJobResult &job, server::DrogonCallback &cb) {
  if (job.status == PdfJobStatus::Ok) return false;
  cb(server::error_response(pipeline::wire_code(job.status),
                            pipeline::pdf_job_error_message(job)));
  return true;
}


// HTTP wrapper over build_pdf_markdown. Owns only the response shaping — the
// document itself comes from the shared builder so gRPC serves identical bytes.
drogon::HttpResponsePtr
emit_pdf_markdown_response(std::vector<PdfPageResult> &pages, bool as_pages) {
  auto p = build_pdf_markdown(pages, as_pages);
  drogon::HttpResponsePtr resp;
  if (p.is_json) {
    resp = server::json_response(std::move(p.body));
  } else {
    resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k200OK);
    resp->setBody(std::move(p.body));
    resp->setContentTypeString("text/markdown; charset=utf-8");
  }
  if (!p.degraded.empty()) resp->addHeader("X-OCR-Degraded", p.degraded);
  return resp;
}


// HTTP wrapper over build_searchable_pdf — response shaping only.
drogon::HttpResponsePtr
emit_searchable_pdf_response(std::vector<PdfPageResult> &pages,
                             const uint8_t *pdf_data, size_t pdf_len,
                             const SearchablePdfOptions &sopts) {
  auto p = build_searchable_pdf(pages, pdf_data, pdf_len, sopts);
  if (p.bytes.empty())
    return server::error_response(server::ErrorCode::kPdfWriteFailed, p.error);
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(drogon::k200OK);
  // no-silent-failure: a PDF whose text layer is missing on some pages is
  // byte-for-byte a valid PDF, so a 200 with no signal is indistinguishable from
  // a complete one. Same mechanism /ocr/markdown uses for a degraded stage.
  if (p.pages_failed > 0)
    resp->addHeader("X-OCR-Degraded",
                    "searchable_pdf:" + std::to_string(p.pages_failed) +
                        " page(s) could not be stamped");
  resp->setBody(std::move(p.bytes));
  resp->setContentTypeString("application/pdf");
  resp->addHeader("Content-Disposition", "attachment; filename=\"searchable.pdf\"");
  if (p.dropped_words > 0)
    resp->addHeader("X-OCR-Dropped-Words", std::to_string(p.dropped_words));
  return resp;
}


} // namespace turbo_ocr::routes::pdfdetail
