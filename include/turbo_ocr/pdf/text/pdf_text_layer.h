#pragma once

// In-process PDFium text-layer extraction for /ocr/pdf's extraction modes.
// Walks the CHAR FLOW (FPDFText_GetUnicode/GetCharBox, splitting on
// PDFium's generated \r\n) — NOT FPDFText_CountRects/GetRect: rects are
// per same-font/style RUN, so a mid-line font change fragments one visual
// line and nested run rects duplicate text (see pdf_text_extract.cpp).
//
// Coordinate convention: PDF points (1/72 inch), **top-left origin**
// (matching turbo-ocr's convention for all other endpoints). PDFium's
// native coordinate system is bottom-left; we flip y at the boundary.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "turbo_ocr/core/types.h"

namespace turbo_ocr::pdf {

// One VISUAL LINE from the char-flow walk (full line, not a same-font run):
// its text plus the AABB of its glyph boxes, transformed to visual space.
struct PdfTextLine {
  std::string text;   // utf-8
  float x0_pt = 0.0f; // top-left origin, PDF points
  float y0_pt = 0.0f;
  float x1_pt = 0.0f;
  float y1_pt = 0.0f;
};

// Per-page snapshot returned by PdfDocument. The lines vector is pre-grouped
// by PDFium — callers should emit one OCRResultItem per line directly.
struct PdfPageText {
  std::vector<PdfTextLine> lines;
  float page_width_pt  = 0.0f;
  float page_height_pt = 0.0f;
  int   rotation_deg   = 0;  // 0, 90, 180, or 270
  int   char_count     = 0;  // FPDFText_CountChars total
  int   fffd_count     = 0;  // U+FFFD replacement chars in the page text
  int   nonprint_count = 0;  // control chars excluding tab/newline
};

// RAII wrapper around a loaded PDF document. Holds one FPDF_DOCUMENT plus
// a lazy cache of FPDF_PAGE / FPDF_TEXTPAGE handles so repeated lookups
// against the same page are cheap.
class PdfDocument {
public:
  PdfDocument(const uint8_t *data, size_t len);
  ~PdfDocument() noexcept;

  PdfDocument(const PdfDocument &) = delete;
  PdfDocument &operator=(const PdfDocument &) = delete;
  PdfDocument(PdfDocument &&) noexcept;
  PdfDocument &operator=(PdfDocument &&) noexcept;

  [[nodiscard]] bool ok() const noexcept { return doc_ != nullptr; }
  [[nodiscard]] int  page_count() const noexcept;

  // Extract the full per-page text snapshot. Returns an empty PdfPageText
  // (lines empty, char_count=0) if the page has no content-stream text.
  [[nodiscard]] PdfPageText extract_page(int page_index) const;

  // Get the utf-8 text inside a top-left-origin PDF-point rectangle on
  // `page_index`. Used by mode=auto_verified to look up native text for
  // each detection box. Returns empty string if the page has no text layer
  // or the rect contains nothing.
  [[nodiscard]] std::string text_in_rect_pt(int page_index,
                                            float x0_pt, float y0_pt,
                                            float x1_pt, float y1_pt) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  void *doc_ = nullptr; // FPDF_DOCUMENT
};

// Initialize PDFium once per process. Thread-safe and idempotent.
void ensure_pdfium_initialized();

struct SanityVerdict {
  bool accept = false;
  const char *reason = "";
};

// Decide whether a native-text string recovered from a detection box can
// be trusted as the "real" text for that region. Rejects on:
//   - empty string
//   - too many U+FFFD / non-printable
//   - implausible char count for the box width
// (Page rotation is NOT its concern — this function never sees it; the
// per-page gate lives in pdf_job_pages.cpp text_layer_quality_for.)
[[nodiscard]] SanityVerdict passes_sanity_check(
    const std::string &text,
    float box_width_pt, float box_height_pt);

// mode=auto_verified core: for every OCR result, look up the native text
// inside its detection box (rendered-pixel space at `dpi`) and replace the
// OCR text with it when the sanity check accepts — the native layer then
// becomes the trusted source for that region (source="pdf", confidence 1).
// Shared by the HTTP and gRPC PDF routes (single-page and batched paths).
void verify_results_with_text_layer(std::vector<OCRResultItem> &results,
                                    const PdfDocument &doc, int page_index,
                                    int dpi);

} // namespace turbo_ocr::pdf
