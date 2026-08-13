// Searchable-PDF writer: text round-trip and box placement.
//
// The geometry check is a true round trip — stamp a known pixel box, reload the
// produced PDF, and map the extracted char rect back to device pixels of the
// same raster. That catches a wrong rotation case, which reading the matrix
// alone does not.

#include <cmath>
#include <string>
#include <vector>

#include <fpdf_annot.h>
#include <fpdf_edit.h>
#include <fpdf_save.h>
#include <fpdf_text.h>
#include <fpdfview.h>

#include "catch_amalgamated.hpp"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pdf/text/pdf_searchable.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::pdf::SearchablePage;
using turbo_ocr::pdf::SearchableStats;
using turbo_ocr::pdf::write_searchable_pdf;

namespace {

struct Writer : FPDF_FILEWRITE {
  std::string buf;
};

int write_block(FPDF_FILEWRITE *w, const void *data, unsigned long size) {
  static_cast<Writer *>(w)->buf.append(static_cast<const char *>(data), size);
  return 1;
}

// A blank page with the given /Rotate, as PDF bytes.
std::string blank_pdf(int rotate_quarter) {
  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc = FPDF_CreateNewDocument();
  FPDF_PAGE page = FPDFPage_New(doc, 0, 612, 792);
  FPDFPage_SetRotation(page, rotate_quarter);
  FPDFPage_GenerateContent(page);
  FPDF_ClosePage(page);
  Writer w;
  w.version = 1;
  w.WriteBlock = write_block;
  FPDF_SaveAsCopy(doc, &w, 0);
  FPDF_CloseDocument(doc);
  return w.buf;
}

std::string utf8_of(const unsigned short *b, int n) {
  std::string o;
  for (int i = 0; i < n; ++i) {
    unsigned cp = b[i];
    if (!cp) continue;
    if (cp < 0x80) {
      o += static_cast<char>(cp);
    } else if (cp < 0x800) {
      o += static_cast<char>(0xC0 | (cp >> 6));
      o += static_cast<char>(0x80 | (cp & 63));
    } else {
      o += static_cast<char>(0xE0 | (cp >> 12));
      o += static_cast<char>(0x80 | ((cp >> 6) & 63));
      o += static_cast<char>(0x80 | (cp & 63));
    }
  }
  return o;
}

OCRResultItem word_at(const char *text, int x0, int y0, int x1, int y1) {
  OCRResultItem item;
  item.text = text;
  item.confidence = 0.99f;
  item.box = Box{{{{{x0, y0}}, {{x1, y0}}, {{x1, y1}}, {{x0, y1}}}}};
  return item;
}

} // namespace

TEST_CASE("searchable pdf places words where they were detected", "[pdf]") {
  constexpr int kDpi = 150;
  constexpr int kX0 = 120, kY0 = 200, kX1 = 520, kY1 = 250;
  const char *kWord = "Rotation";

  for (int quarter = 0; quarter < 4; ++quarter) {
    for (int orientation : {0, 90}) {
      CAPTURE(quarter * 90, orientation);
      const std::string src = blank_pdf(quarter);

      FPDF_DOCUMENT probe =
          FPDF_LoadMemDocument(src.data(), static_cast<int>(src.size()), nullptr);
      REQUIRE(probe != nullptr);
      FPDF_PAGE probe_page = FPDF_LoadPage(probe, 0);
      float vis_w = FPDF_GetPageWidthF(probe_page);
      float vis_h = FPDF_GetPageHeightF(probe_page);
      FPDF_ClosePage(probe_page);
      FPDF_CloseDocument(probe);
      if (orientation % 180) std::swap(vis_w, vis_h);

      const int raster_w = static_cast<int>(std::lround(vis_w * kDpi / 72.0));
      const int raster_h = static_cast<int>(std::lround(vis_h * kDpi / 72.0));

      std::vector<OCRResultItem> results{word_at(kWord, kX0, kY0, kX1, kY1)};
      SearchablePage page;
      page.page_index = 0;
      page.raster_w = raster_w;
      page.raster_h = raster_h;
      page.orientation_deg = orientation;
      page.results = &results;

      SearchableStats stats;
      std::string err;
      const std::string out = write_searchable_pdf(
          reinterpret_cast<const uint8_t *>(src.data()), src.size(), {page},
          0.0f, &stats, err);
      REQUIRE(err.empty());
      REQUIRE_FALSE(out.empty());
      REQUIRE(stats.words == 1);

      FPDF_DOCUMENT doc =
          FPDF_LoadMemDocument(out.data(), static_cast<int>(out.size()), nullptr);
      REQUIRE(doc != nullptr);
      FPDF_PAGE page_handle = FPDF_LoadPage(doc, 0);
      FPDF_TEXTPAGE text = FPDFText_LoadPage(page_handle);

      const int chars = FPDFText_CountChars(text);
      std::vector<unsigned short> buf(chars + 1);
      FPDFText_GetText(text, 0, chars, buf.data());
      CHECK(utf8_of(buf.data(), chars) == kWord);

      REQUIRE(FPDFText_CountRects(text, 0, chars) >= 1);
      double left, top, right, bottom;
      FPDFText_GetRect(text, 0, &left, &top, &right, &bottom);
      int dx0, dy0, dx1, dy1;
      FPDF_PageToDevice(page_handle, 0, 0, raster_w, raster_h, 0, left, top, &dx0, &dy0);
      FPDF_PageToDevice(page_handle, 0, 0, raster_w, raster_h, 0, right, bottom, &dx1, &dy1);
      if (dx0 > dx1) std::swap(dx0, dx1);
      if (dy0 > dy1) std::swap(dy0, dy1);

      // 3 px at 150 dpi is under 1.5 pt — tight enough that a wrong rotation,
      // flip or origin offset fails, loose enough for float rounding.
      CHECK(std::abs(dx0 - kX0) <= 3);
      CHECK(std::abs(dx1 - kX1) <= 3);
      CHECK(std::abs(dy0 - kY0) <= 3);
      CHECK(std::abs(dy1 - kY1) <= 3);

      FPDFText_ClosePage(text);
      FPDF_ClosePage(page_handle);
      FPDF_CloseDocument(doc);
    }
  }
}

TEST_CASE("searchable pdf carries every script through ToUnicode", "[pdf]") {
  const std::string src = blank_pdf(0);
  std::vector<OCRResultItem> results{
      word_at("Grüße", 100, 100, 300, 140),
      word_at("你好世界", 100, 200, 300, 240),
      word_at("Ελληνικά", 100, 300, 300, 340),
  };
  // PDFium reorders RTL runs into visual order when extracting, so this one is
  // checked character by character rather than as a substring.
  const std::string rtl = "مرحبا";
  results.push_back(word_at(rtl.c_str(), 100, 400, 300, 440));
  SearchablePage page;
  page.raster_w = 1275;
  page.raster_h = 1650;
  page.results = &results;

  SearchableStats stats;
  std::string err;
  const std::string out =
      write_searchable_pdf(reinterpret_cast<const uint8_t *>(src.data()),
                           src.size(), {page}, 0.0f, &stats, err);
  REQUIRE(err.empty());
  REQUIRE(stats.words == 4);

  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(out.data(), static_cast<int>(out.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page_handle = FPDF_LoadPage(doc, 0);
  FPDF_TEXTPAGE text = FPDFText_LoadPage(page_handle);
  const int chars = FPDFText_CountChars(text);
  std::vector<unsigned short> buf(chars + 1);
  FPDFText_GetText(text, 0, chars, buf.data());
  const std::string got = utf8_of(buf.data(), chars);

  for (size_t i = 0; i + 1 < results.size(); ++i)
    CHECK(got.find(results[i].text) != std::string::npos);
  for (size_t i = 0; i < rtl.size();) {
    const size_t n = (static_cast<unsigned char>(rtl[i]) & 0xE0) == 0xC0 ? 2 : 3;
    CHECK(got.find(rtl.substr(i, n)) != std::string::npos);
    i += n;
  }

  FPDFText_ClosePage(text);
  FPDF_ClosePage(page_handle);
  FPDF_CloseDocument(doc);
}

TEST_CASE("searchable pdf skips words already in the document's text layer", "[pdf]") {
  const std::string src = blank_pdf(0);
  std::vector<OCRResultItem> results{word_at("native", 100, 100, 300, 140),
                                     word_at("recognised", 100, 200, 300, 240)};
  results[0].source = "pdf";

  SearchablePage page;
  page.raster_w = 1275;
  page.raster_h = 1650;
  page.results = &results;

  SearchableStats stats;
  std::string err;
  const std::string out =
      write_searchable_pdf(reinterpret_cast<const uint8_t *>(src.data()),
                           src.size(), {page}, 0.0f, &stats, err);
  REQUIRE(err.empty());
  CHECK(stats.words == 1);

  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(out.data(), static_cast<int>(out.size()), nullptr);
  FPDF_PAGE page_handle = FPDF_LoadPage(doc, 0);
  FPDF_TEXTPAGE text = FPDFText_LoadPage(page_handle);
  const int chars = FPDFText_CountChars(text);
  std::vector<unsigned short> buf(chars + 1);
  FPDFText_GetText(text, 0, chars, buf.data());
  const std::string got = utf8_of(buf.data(), chars);
  CHECK(got.find("recognised") != std::string::npos);
  CHECK(got.find("native") == std::string::npos);

  FPDFText_ClosePage(text);
  FPDF_ClosePage(page_handle);
  FPDF_CloseDocument(doc);
}

TEST_CASE("searchable pdf honours the confidence floor", "[pdf]") {
  const std::string src = blank_pdf(0);
  std::vector<OCRResultItem> results{word_at("certain", 100, 100, 300, 140),
                                     word_at("doubtful", 100, 200, 300, 240)};
  results[1].confidence = 0.20f;

  SearchablePage page;
  page.raster_w = 1275;
  page.raster_h = 1650;
  page.results = &results;

  SearchableStats stats;
  std::string err;
  const std::string out =
      write_searchable_pdf(reinterpret_cast<const uint8_t *>(src.data()),
                           src.size(), {page}, 0.5f, &stats, err);
  REQUIRE(err.empty());
  CHECK(stats.words == 1);
  CHECK_FALSE(out.empty());
}

TEST_CASE("searchable pdf returns the original when there is nothing to stamp", "[pdf]") {
  const std::string src = blank_pdf(0);
  std::vector<OCRResultItem> results;
  SearchablePage page;
  page.raster_w = 1275;
  page.raster_h = 1650;
  page.results = &results;

  SearchableStats stats;
  std::string err;
  const std::string out =
      write_searchable_pdf(reinterpret_cast<const uint8_t *>(src.data()),
                           src.size(), {page}, 0.0f, &stats, err);
  CHECK(err.empty());
  CHECK(out == src);
  CHECK(stats.words == 0);
  // `pages` is pages STAMPED, and nothing was stamped — the original bytes came
  // straight back. Reporting the submitted page count here made an
  // all-native-text document log "pages=1 words=0", which reads exactly like
  // "stamped one page, found no words".
  CHECK(stats.pages == 0);
  CHECK(stats.pages_failed == 0);
}

TEST_CASE("searchable pdf rejects input it cannot open", "[pdf]") {
  const std::string junk = "not a pdf at all";
  std::vector<OCRResultItem> results{word_at("x", 10, 10, 50, 30)};
  SearchablePage page;
  page.raster_w = 100;
  page.raster_h = 100;
  page.results = &results;

  std::string err;
  const std::string out =
      write_searchable_pdf(reinterpret_cast<const uint8_t *>(junk.data()),
                           junk.size(), {page}, 0.0f, nullptr, err);
  CHECK(out.empty());
  CHECK_FALSE(err.empty());
}

TEST_CASE("searchable pdf marks figure regions so they can be selected", "[pdf]") {
  const std::string src = blank_pdf(0);
  std::vector<OCRResultItem> results{word_at("caption", 100, 900, 400, 940)};

  turbo_ocr::layout::LayoutBox figure;
  figure.class_id = turbo_ocr::layout::kImageClassId; // "image"
  figure.score = 0.95f;
  figure.box = Box{{{{{200, 200}}, {{900, 200}}, {{900, 800}}, {{200, 800}}}}};
  turbo_ocr::layout::LayoutBox paragraph;
  paragraph.class_id = 22; // "text" — selectable through the text layer already
  paragraph.box = Box{{{{{100, 900}}, {{400, 900}}, {{400, 940}}, {{100, 940}}}}};
  std::vector<turbo_ocr::layout::LayoutBox> layout{figure, paragraph};

  SearchablePage page;
  page.raster_w = 1275;
  page.raster_h = 1650;
  page.results = &results;
  page.layout = &layout;

  SearchableStats stats;
  std::string err;
  const std::string out =
      write_searchable_pdf(reinterpret_cast<const uint8_t *>(src.data()),
                           src.size(), {page}, 0.0f, &stats, err);
  REQUIRE(err.empty());
  CHECK(stats.regions == 1); // the figure, not the paragraph

  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(out.data(), static_cast<int>(out.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page_handle = FPDF_LoadPage(doc, 0);
  REQUIRE(FPDFPage_GetAnnotCount(page_handle) == 1);

  FPDF_ANNOTATION annot = FPDFPage_GetAnnot(page_handle, 0);
  REQUIRE(annot != nullptr);
  FS_RECTF rect{};
  REQUIRE(FPDFAnnot_GetRect(annot, &rect));

  // The region must cover the detected pixels, mapped to points at 150 dpi.
  const float scale = 612.0f / 1275.0f;
  CHECK(std::abs(rect.left - 200 * scale) <= 2.0f);
  CHECK(std::abs(rect.right - 900 * scale) <= 2.0f);
  CHECK(std::abs(rect.top - (792.0f - 200 * scale)) <= 2.0f);
  CHECK(std::abs(rect.bottom - (792.0f - 800 * scale)) <= 2.0f);

  unsigned short label[32] = {};
  FPDFAnnot_GetStringValue(annot, "Contents", label, sizeof(label));
  CHECK(utf8_of(label, 5) == "image");

  FPDFPage_CloseAnnot(annot);
  FPDF_ClosePage(page_handle);
  FPDF_CloseDocument(doc);
}
