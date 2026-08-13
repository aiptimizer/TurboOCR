// Visible ("editable") text layer: the words are drawn as real type where the
// print was, instead of hidden behind it.
//
// The round trip here is the point. A page of text is built and rasterised —
// that raster stands in for the scan, and is the only thing the estimator is
// allowed to look at — then written back with mode=Visible, then rendered
// again. Asserting on the SECOND raster is what proves the type landed where
// the print was rather than merely that some object was added to the file.

#include <cmath>
#include <string>
#include <vector>

#include <fpdf_edit.h>
#include <fpdf_save.h>
#include <fpdf_text.h>
#include <fpdfview.h>

#include <opencv2/imgproc.hpp>

#include "catch_amalgamated.hpp"
#include "turbo_ocr/pdf/text/font_style.h"
#include "turbo_ocr/pdf/text/pdf_searchable.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"


using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::pdf::LineStyle;
using turbo_ocr::pdf::SearchablePage;
using turbo_ocr::pdf::SearchableStats;
using turbo_ocr::pdf::TextLayerMode;
using turbo_ocr::pdf::write_searchable_pdf;
using turbo_ocr::pdf::measure_page_line_styles;

namespace {

constexpr float kPageW = 480;
constexpr float kPageH = 120;
constexpr float kScale = 2.0F;  // ~144 dpi, where real scans sit

struct Writer : FPDF_FILEWRITE {
  std::string buf;
};

int write_block(FPDF_FILEWRITE *w, const void *data, unsigned long size) {
  static_cast<Writer *>(w)->buf.append(static_cast<const char *>(data), size);
  return 1;
}

std::vector<unsigned short> utf16(const std::string &s) {
  // The samples are ASCII apart from the deliberately non-Latin-1 case, which
  // arrives already as codepoints below 0xFFFF.
  std::vector<unsigned short> out;
  for (size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    unsigned cp = c;
    int n = 1;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1FU; n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0FU; n = 3; }
    for (int k = 1; k < n && i + k < s.size(); ++k)
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3FU);
    i += n;
    out.push_back(static_cast<unsigned short>(cp));
  }
  out.push_back(0);
  return out;
}

// A one-page PDF holding `text` in `face`. Stands in for the source scan.
std::string page_with_text(const char *face, const std::string &text,
                           float pt = 16.0F) {
  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc = FPDF_CreateNewDocument();
  FPDF_PAGE page = FPDFPage_New(doc, 0, kPageW, kPageH);
  FPDF_FONT font = FPDFText_LoadStandardFont(doc, face);
  FPDF_PAGEOBJECT obj = FPDFPageObj_CreateTextObj(doc, font, pt);
  const auto wide = utf16(text);
  FPDFText_SetText(obj, reinterpret_cast<FPDF_WIDESTRING>(wide.data()));
  FPDFPageObj_Transform(obj, 1, 0, 0, 1, 40, 50);
  FPDFPage_InsertObject(page, obj);
  FPDFPage_GenerateContent(page);
  FPDFFont_Close(font);
  FPDF_ClosePage(page);

  Writer w;
  w.version = 1;
  w.WriteBlock = write_block;
  FPDF_SaveAsCopy(doc, &w, 0);
  FPDF_CloseDocument(doc);
  return w.buf;
}

// Rasterises page 0 of `pdf` at kScale. This is the "scan".
cv::Mat rasterise(const std::string &pdf) {
  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(pdf.data(), static_cast<int>(pdf.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  REQUIRE(page != nullptr);
  const int w = static_cast<int>(FPDF_GetPageWidthF(page) * kScale);
  const int h = static_cast<int>(FPDF_GetPageHeightF(page) * kScale);
  FPDF_BITMAP bmp = FPDFBitmap_Create(w, h, 0);
  FPDFBitmap_FillRect(bmp, 0, 0, w, h, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bmp, page, 0, 0, w, h, 0, 0);
  const cv::Mat view(h, w, CV_8UC4, FPDFBitmap_GetBuffer(bmp),
                     static_cast<size_t>(FPDFBitmap_GetStride(bmp)));
  cv::Mat out;
  cv::cvtColor(view, out, cv::COLOR_BGRA2BGR);
  FPDFBitmap_Destroy(bmp);
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
  return out;
}

cv::Rect ink_box(const cv::Mat &bgr) {
  cv::Mat gray;
  cv::Mat mask;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
  return cv::boundingRect(mask);
}

Box box_of(const cv::Rect &r) {
  return Box{{{{r.x, r.y},
               {r.x + r.width, r.y},
               {r.x + r.width, r.y + r.height},
               {r.x, r.y + r.height}}}};
}

// Wraps a raster as the sole content of a one-page PDF — a scan, in other
// words. Building the input this way rather than reusing the text page it was
// rendered from matters: a text page would arrive at the writer already
// carrying a text layer, so every assertion about what the writer added would
// really be reading what was there beforehand.
std::string image_pdf(const cv::Mat &bgr) {
  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc = FPDF_CreateNewDocument();
  FPDF_PAGE page = FPDFPage_New(doc, 0, kPageW, kPageH);

  FPDF_BITMAP bmp = FPDFBitmap_Create(bgr.cols, bgr.rows, 0);
  REQUIRE(bmp != nullptr);
  auto *dst = static_cast<uint8_t *>(FPDFBitmap_GetBuffer(bmp));
  const int stride = FPDFBitmap_GetStride(bmp);
  for (int y = 0; y < bgr.rows; ++y) {
    const auto *src = bgr.ptr<cv::Vec3b>(y);
    uint8_t *row = dst + static_cast<size_t>(y) * static_cast<size_t>(stride);
    for (int x = 0; x < bgr.cols; ++x) {
      row[4 * x + 0] = src[x][0];
      row[4 * x + 1] = src[x][1];
      row[4 * x + 2] = src[x][2];
      row[4 * x + 3] = 255;
    }
  }

  FPDF_PAGEOBJECT img = FPDFPageObj_NewImageObj(doc);
  REQUIRE(FPDFImageObj_SetBitmap(&page, 1, img, bmp));
  // An image object's own space is the unit square, so this scales it to fill
  // the page.
  FPDFPageObj_Transform(img, kPageW, 0, 0, kPageH, 0, 0);
  FPDFPage_InsertObject(page, img);
  FPDFPage_GenerateContent(page);

  FPDFBitmap_Destroy(bmp);
  FPDF_ClosePage(page);
  Writer w;
  w.version = 1;
  w.WriteBlock = write_block;
  FPDF_SaveAsCopy(doc, &w, 0);
  FPDF_CloseDocument(doc);
  return w.buf;
}

struct Scan {
  std::string pdf;
  cv::Mat raster;
  std::vector<OCRResultItem> results;
  std::vector<LineStyle> styles;
  cv::Rect ink;
};

// Build the whole input the writer sees: a scanned page, its raster, one
// recognised line covering the ink, and the styles measured off that raster.
Scan scan_of(const char *face, const std::string &text) {
  Scan s;
  s.raster = rasterise(page_with_text(face, text));
  s.pdf = image_pdf(s.raster);
  s.ink = ink_box(s.raster);
  REQUIRE(s.ink.width > 20);
  REQUIRE(s.ink.height > 6);

  OCRResultItem item;
  item.text = text;
  item.confidence = 0.99F;
  item.box = box_of(s.ink);
  s.results.push_back(item);
  s.styles = measure_page_line_styles(s.results, s.raster);
  return s;
}

SearchablePage page_of(const Scan &s, bool with_styles = true) {
  SearchablePage p;
  p.page_index = 0;
  p.raster_w = s.raster.cols;
  p.raster_h = s.raster.rows;
  p.results = &s.results;
  if (with_styles) p.styles = &s.styles;
  return p;
}

std::string write(const Scan &s, const SearchablePage &p, TextLayerMode mode,
                  SearchableStats &stats) {
  std::string err;
  const std::string out = write_searchable_pdf(
      reinterpret_cast<const uint8_t *>(s.pdf.data()), s.pdf.size(), {p}, 0.0F,
      &stats, err, mode);
  INFO("writer error: " << err);
  REQUIRE(err.empty());
  REQUIRE_FALSE(out.empty());
  return out;
}

std::string extracted_text(const std::string &pdf) {
  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(pdf.data(), static_cast<int>(pdf.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
  const int n = FPDFText_CountChars(tp);
  std::vector<unsigned short> buf(static_cast<size_t>(n) + 1, 0);
  FPDFText_GetText(tp, 0, n, buf.data());
  std::string out;
  for (int i = 0; i < n; ++i) {
    const unsigned cp = buf[static_cast<size_t>(i)];
    if (cp && cp < 0x80) out += static_cast<char>(cp);
  }
  FPDFText_ClosePage(tp);
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
  return out;
}

// How the text objects on page 0 are rendered. Invisible mode must leave every
// one at mode 3; visible mode must leave at least one filling.
int count_render_mode(const std::string &pdf, FPDF_TEXT_RENDERMODE want) {
  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(pdf.data(), static_cast<int>(pdf.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  int found = 0;
  for (int i = 0; i < FPDFPage_CountObjects(page); ++i) {
    FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, i);
    if (FPDFPageObj_GetType(o) != FPDF_PAGEOBJ_TEXT) continue;
    if (FPDFTextObj_GetTextRenderMode(o) == want) ++found;
  }
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
  return found;
}

} // namespace

TEST_CASE("visible mode draws real type where the print was", "[pdf][editable]") {
  const std::string kText = "Invoice total due on receipt";
  const Scan s = scan_of("Helvetica", kText);
  REQUIRE(s.styles.size() == 1);
  REQUIRE(s.styles[0].measured);

  SearchableStats stats;
  const SearchablePage p = page_of(s);
  const std::string out = write(s, p, TextLayerMode::Visible, stats);

  CHECK(stats.visible == 1);
  CHECK(stats.uncovered == 0);

  // Still searchable — visible type must not cost what the invisible layer gave.
  CHECK(extracted_text(out) == kText);

  // Really drawn, not hidden.
  CHECK(count_render_mode(out, FPDF_TEXTRENDERMODE_FILL) >= 1);
  CHECK(count_render_mode(out, FPDF_TEXTRENDERMODE_INVISIBLE) == 0);

  // And drawn where the original was. Re-rendering and comparing ink boxes is
  // the assertion that matters: it fails if the matrix, the rotation or the
  // baseline is wrong, none of which reading the file back would catch.
  const cv::Rect before = s.ink;
  const cv::Rect after = ink_box(rasterise(out));
  INFO("before " << before << " after " << after);
  const auto tol = static_cast<int>(std::lround(before.height * 0.5));
  CHECK(std::abs(after.x - before.x) <= tol);
  CHECK(std::abs(after.y - before.y) <= tol);
  CHECK(std::abs(after.width - before.width) <= tol);
  CHECK(std::abs(after.height - before.height) <= tol);
}

TEST_CASE("the covered print does not show through", "[pdf][editable]") {
  // Cover the original, draw the new type in white, and what is left must be
  // blank paper. If the print were still underneath, it would show here.
  const Scan s = scan_of("Helvetica", "Ghosting check");
  Scan white = s;
  white.styles[0].ink = cv::Vec3b(255, 255, 255);

  SearchableStats stats;
  const SearchablePage p = page_of(white);
  const std::string out = write(white, p, TextLayerMode::Visible, stats);
  REQUIRE(stats.visible == 1);

  const cv::Rect ink = ink_box(rasterise(out));
  INFO("residual ink " << ink);
  // boundingRect of an empty mask is 0x0.
  CHECK(ink.area() == 0);
}

TEST_CASE("invisible stays the default and stays invisible", "[pdf][editable]") {
  const std::string kText = "Unchanged behaviour";
  const Scan s = scan_of("Helvetica", kText);
  SearchableStats stats;
  const SearchablePage p = page_of(s);

  // No mode argument at all: the existing call sites must not change meaning.
  std::string err;
  const std::string out = write_searchable_pdf(
      reinterpret_cast<const uint8_t *>(s.pdf.data()), s.pdf.size(), {p}, 0.0F,
      &stats, err);
  REQUIRE(err.empty());
  CHECK(stats.visible == 0);
  CHECK(count_render_mode(out, FPDF_TEXTRENDERMODE_FILL) == 0);
  CHECK(count_render_mode(out, FPDF_TEXTRENDERMODE_INVISIBLE) == 1);
  CHECK(extracted_text(out) == kText);
}

TEST_CASE("visible mode without styles falls back to invisible", "[pdf][editable]") {
  // A caller that asks for visible type but never measured any cannot be given
  // it. Falling back silently to a searchable document is right; guessing a
  // font and a paper colour is not.
  const Scan s = scan_of("Helvetica", "No styles supplied");
  SearchableStats stats;
  const SearchablePage p = page_of(s, /*with_styles=*/false);
  const std::string out = write(s, p, TextLayerMode::Visible, stats);

  CHECK(stats.visible == 0);
  CHECK(stats.words == 1);
  CHECK(count_render_mode(out, FPDF_TEXTRENDERMODE_INVISIBLE) == 1);
}

TEST_CASE("text the standard fonts cannot spell is left alone", "[pdf][editable]") {
  // Greek has no place in the standard-14 encodings. Drawing it would put the
  // wrong glyphs on the page AND cover the right ones, which is strictly worse
  // than leaving the scan to speak for itself.
  Scan s = scan_of("Helvetica", "Total");
  s.results[0].text = "\xCE\xA9\xCE\xBC\xCE\xB5\xCE\xB3\xCE\xB1";  // Ωμεγα

  SearchableStats stats;
  const SearchablePage p = page_of(s);
  const std::string out = write(s, p, TextLayerMode::Visible, stats);

  CHECK(stats.visible == 0);
  CHECK(stats.uncovered == 1);
  CHECK(count_render_mode(out, FPDF_TEXTRENDERMODE_INVISIBLE) == 1);

  // The page still looks exactly as it did: nothing was covered.
  const cv::Rect after = ink_box(rasterise(out));
  CHECK(after.area() > 0);
}

TEST_CASE("a line on patterned ground is not covered", "[pdf][editable]") {
  // Covering here would erase the rules, the shading or the logo the line sits
  // on, which is page content the OCR never claimed to have read.
  Scan s = scan_of("Helvetica", "Ruled cell");
  s.styles[0].flat_paper = false;

  SearchableStats stats;
  const SearchablePage p = page_of(s);
  const std::string out = write(s, p, TextLayerMode::Visible, stats);

  CHECK(stats.visible == 0);
  CHECK(stats.uncovered == 1);
  CHECK(count_render_mode(out, FPDF_TEXTRENDERMODE_INVISIBLE) == 1);
}

TEST_CASE("the ink colour is taken off the page", "[pdf][editable]") {
  // A form printed in navy stays navy. Rendering everything in black would be
  // a visible, avoidable change to the document.
  Scan s = scan_of("Helvetica", "Coloured heading");
  s.styles[0].ink = cv::Vec3b(180, 40, 40);  // BGR: a strong blue

  SearchableStats stats;
  const SearchablePage p = page_of(s);
  const std::string out = write(s, p, TextLayerMode::Visible, stats);
  REQUIRE(stats.visible == 1);

  const cv::Mat rendered = rasterise(out);
  const cv::Rect ink = ink_box(rendered);
  REQUIRE(ink.area() > 0);
  // Average the darkest pixels: on this page they are the glyphs.
  cv::Mat crop = rendered(ink);
  cv::Mat gray;
  cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
  cv::Mat mask;
  cv::threshold(gray, mask, 200, 255, cv::THRESH_BINARY_INV);
  const cv::Scalar mean = cv::mean(crop, mask);
  INFO("mean glyph colour B=" << mean[0] << " G=" << mean[1] << " R=" << mean[2]);
  CHECK(mean[0] > mean[2] + 40);  // decidedly more blue than red
}

TEST_CASE("a serif document comes back in a serif face", "[pdf][editable]") {
  // End to end: the type on the page is measured, voted on, and the matching
  // standard-14 face is what gets embedded.
  for (const auto &[face, want] : std::vector<std::pair<const char *, const char *>>{
           {"Helvetica", "Helvetica"}, {"Times-Roman", "Times"}}) {
    const Scan s = scan_of(face, "The quick brown fox jumps over it");
    SearchableStats stats;
    const SearchablePage p = page_of(s);
    const std::string out = write(s, p, TextLayerMode::Visible, stats);
    REQUIRE(stats.visible == 1);

    FPDF_DOCUMENT doc =
        FPDF_LoadMemDocument(out.data(), static_cast<int>(out.size()), nullptr);
    REQUIRE(doc != nullptr);
    FPDF_PAGE page = FPDF_LoadPage(doc, 0);
    std::string used;
    for (int i = 0; i < FPDFPage_CountObjects(page); ++i) {
      FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, i);
      if (FPDFPageObj_GetType(o) != FPDF_PAGEOBJ_TEXT) continue;
      if (FPDFTextObj_GetTextRenderMode(o) != FPDF_TEXTRENDERMODE_FILL) continue;
      FPDF_FONT f = FPDFTextObj_GetFont(o);
      char name[128] = {0};
      if (FPDFFont_GetBaseFontName(f, name, sizeof(name)) > 0) used = name;
    }
    FPDF_ClosePage(page);
    FPDF_CloseDocument(doc);
    INFO("source face " << face << " embedded as " << used);
    CHECK(used.find(want) != std::string::npos);
  }
}
