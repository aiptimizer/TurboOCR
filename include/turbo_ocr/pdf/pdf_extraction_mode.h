#pragma once

// PDF extraction mode for the /ocr/pdf endpoint. Four settable modes, with
// `Ocr` as the default. Chosen at request time via ?mode=<name> or at server
// start via the ENABLE_PDF_MODE env var.
//
// Orthogonal to ENABLE_LAYOUT: any mode can run with layout on or off.
//
// The pipeline per page:
//
//   Ocr            — render → det → cls → rec   (today's behavior)
//   Geometric      — PDFium text layer only, no rasterization (unless
//                    layout is enabled, in which case we rasterize anyway
//                    for the layout stage and still use PDFium for text)
//   Auto           — if FPDFText_CountChars > N → Geometric, else Ocr.
//                    Per-page decision.
//   AutoVerified   — render → det → for each det box, look up native text
//                    via PDFium and accept if sanity-check passes, else
//                    rec that box. Detection is the ground truth.
//

#include <string>
#include <string_view>

namespace turbo_ocr::pdf {

enum class PdfMode {
  Ocr = 0,
  Geometric,
  Auto,
  AutoVerified,
};

constexpr std::string_view mode_name(PdfMode m) noexcept {
  switch (m) {
    case PdfMode::Ocr:          return "ocr";
    case PdfMode::Geometric:    return "geometric";
    case PdfMode::Auto:         return "auto";
    case PdfMode::AutoVerified: return "auto_verified";
  }
  return "ocr";
}

// Parse a string from a query param or env var into a mode.
// Unknown / empty → returns `fallback`.
constexpr PdfMode parse_pdf_mode(std::string_view s,
                                  PdfMode fallback = PdfMode::Ocr) noexcept {
  if (s == "ocr")           return PdfMode::Ocr;
  if (s == "geometric")     return PdfMode::Geometric;
  if (s == "auto")          return PdfMode::Auto;
  if (s == "auto_verified") return PdfMode::AutoVerified;
  return fallback;
}

// Strict variant for request parsing: an explicit unknown value is the
// caller's error and must 400, never silently coerce to a default (that is
// the same contract bad dpi/quality values already get).
constexpr bool is_valid_pdf_mode(std::string_view s) noexcept {
  return s == "ocr" || s == "geometric" || s == "auto" || s == "auto_verified";
}

// Per-page resolution result for Auto / AutoVerified (where the effective
// mode may differ from the requested mode). Emitted in the response so
// clients can see which path ran on each page.
struct ResolvedPageMode {
  PdfMode mode = PdfMode::Ocr;
  // "trusted" | "rejected" | "absent" — only meaningful for Geometric /
  // Auto / AutoVerified. For Ocr this is always "absent".
  std::string_view text_layer_quality = "absent";
};

// Whether the given mode always needs a rasterized page (independent of
// ENABLE_LAYOUT). Geometric and Auto (on native pages) can skip rendering.
constexpr bool mode_always_rasterizes(PdfMode m) noexcept {
  return m == PdfMode::Ocr || m == PdfMode::AutoVerified;
}

} // namespace turbo_ocr::pdf
