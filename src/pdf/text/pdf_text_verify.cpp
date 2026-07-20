// Text-layer trust checks: the sanity heuristics for a native-text box and
// the auto_verified per-item replacement pass. Pure consumers of the public
// PdfDocument API — no direct PDFium calls, so no pdfium_lock here.

#include "turbo_ocr/pdf/pdf_text_layer.h"

#include <algorithm>
#include <string>
#include <vector>

#include "turbo_ocr/common/geometry/box.h"

namespace turbo_ocr::pdf {

SanityVerdict passes_sanity_check(const std::string &text,
                                  float box_width_pt,
                                  float box_height_pt) {
  if (text.empty())
    return {false, "no native text in box"};

  int n = 0, fffd = 0, nonprint = 0;
  for (size_t i = 0; i < text.size(); ) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    uint32_t cp = 0;
    int step = 1;
    if (c < 0x80) { cp = c; step = 1; }
    else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
      cp = (c & 0x1F) << 6 | (static_cast<unsigned char>(text[i+1]) & 0x3F);
      step = 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
      cp = (c & 0x0F) << 12
         | (static_cast<unsigned char>(text[i+1]) & 0x3F) << 6
         | (static_cast<unsigned char>(text[i+2]) & 0x3F);
      step = 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
      cp = (c & 0x07) << 18
         | (static_cast<unsigned char>(text[i+1]) & 0x3F) << 12
         | (static_cast<unsigned char>(text[i+2]) & 0x3F) << 6
         | (static_cast<unsigned char>(text[i+3]) & 0x3F);
      step = 4;
    }
    if (cp == 0xFFFD) ++fffd;
    else if (cp < 0x20 && cp != '\t' && cp != '\n' && cp != '\r') ++nonprint;
    ++n;
    i += step;
  }
  if (n == 0) return {false, "empty after decode"};
  if (fffd * 20 > n) return {false, "too many U+FFFD replacement chars"};
  if (nonprint * 10 > n) return {false, "too many non-printable chars"};

  if (box_width_pt > 0 && box_height_pt > 0) {
    float min_expected = box_width_pt / 30.0f;
    float max_expected = box_width_pt / 2.0f;
    if (static_cast<float>(n) < min_expected * 0.5f ||
        static_cast<float>(n) > max_expected * 2.0f)
      return {false, "char count implausible for box width"};
  }

  return {true, "trusted"};
}

void verify_results_with_text_layer(std::vector<OCRResultItem> &results,
                                    const PdfDocument &doc, int page_index,
                                    int dpi) {
  // dpi is caller-supplied; a zero/negative value would make px_to_pt inf/nan
  // and silently blank the text-layer coords for every item (mis-verifying the
  // whole page). Callers validate DPI, but this guard keeps the invariant local.
  if (dpi <= 0) return;
  const float px_to_pt = 72.0f / static_cast<float>(dpi);
  for (auto &item : results) {
    auto [ix0, iy0, ix1, iy1] = turbo_ocr::aabb(item.box);
    float x0 = ix0 * px_to_pt, y0 = iy0 * px_to_pt;
    float x1 = ix1 * px_to_pt, y1 = iy1 * px_to_pt;
    std::string native = doc.text_in_rect_pt(page_index, x0, y0, x1, y1);
    auto verdict = passes_sanity_check(native, x1 - x0, y1 - y0);
    if (verdict.accept) {
      item.text = std::move(native);
      item.source = "pdf";
      item.confidence = 1.0f;
    }
  }
}

} // namespace turbo_ocr::pdf
