#pragma once

// pdf_documents.h — what to DO with a finished PDF job, independent of transport.
//
// Assembling a Markdown document from a PdfJobResult, stamping the recognised
// words back onto the source PDF, and the two per-page hooks the job runs while
// the page raster is still alive. None of it knows HTTP from gRPC.
//
// WHY THIS HEADER EXISTS: these four were declared in
// src/service/http/pdf/pdf_internal.h, a header whose own banner calls itself
// internal to the HTTP routes — and src/service/grpc/recognize_pdf_rpc.cpp
// included it with a relative path to reach them, which pulled <drogon/...> into
// a gRPC translation unit for four functions that touch no transport at all.
// The declarations that DO touch Drogon (the emit_* response wrappers, and
// PdfRequestParams) stayed behind.
//
// It lives under pdf/ because that is pdf/'s stated rule — the PDF medium, both
// directions, searchable-PDF out.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/analysis/forms/form_field.h"
#include "turbo_ocr/pipeline/job/pdf_job.h"

namespace turbo_ocr::pdf::documents {

using pipeline::PdfPageResult;

// Markdown for a whole PDF job, plus the degradation summary the HTTP emitter
// puts in X-OCR-Degraded and the gRPC response carries in its own field.
struct PdfMarkdownPayload {
  std::string body;      // markdown document, or the {"pages":[...]} JSON
  std::string degraded;  // empty when no stage degraded
  bool is_json = false;  // true when as_pages produced the JSON envelope
};
[[nodiscard]] PdfMarkdownPayload
build_pdf_markdown(std::vector<PdfPageResult> &pages, bool as_pages);

// The four "what to do with the document" knobs travel as ONE named value, not
// as `float, bool, bool, bool`: three same-typed bools in a row is a call site
// where transposing any two compiles clean and silently changes the OUTPUT
// DOCUMENT (editable and movable are wholly different products, and
// mark_regions defaults TRUE while the other two default false, so a mis-ordered
// call also flips a default).
struct SearchablePdfOptions {
  float min_confidence = 0.0f;  // drop recognised words below this
  bool editable = false;        // draw words as real type in place of the print
  bool movable = false;         // re-place each figure as its own object
  bool mark_regions = true;     // outline annotation on each figure
};

// `bytes` empty means the write failed and `error` says why.
struct SearchablePdfPayload {
  std::string bytes;
  std::string error;         // non-empty only when bytes is empty
  int pages_failed = 0;      // >0 => degraded (some pages could not be stamped)
  int dropped_words = 0;
};
[[nodiscard]] SearchablePdfPayload
build_searchable_pdf(std::vector<PdfPageResult> &pages, const uint8_t *pdf_data,
                     size_t pdf_len, const SearchablePdfOptions &opts);

// Per-page markdown renderer for PdfJobOptions::render_page_markdown, run on the
// pipeline worker while the page bitmap is still alive.
[[nodiscard]] std::function<std::string(PdfPageResult &, const cv::Mat &)>
make_pdf_page_markdown_renderer();

// ?fields=1 — per-page fillable-field detector for
// PdfJobOptions::detect_page_fields. Same worker, same reason: it reads the
// page raster.
[[nodiscard]] std::function<std::vector<forms::FormField>(PdfPageResult &,
                                                          const cv::Mat &)>
make_pdf_page_field_detector();

} // namespace turbo_ocr::pdf::documents
