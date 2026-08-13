// Typeface identification by rendering candidates and comparing them with the
// page.
//
// The samples are produced by PDFium and handed back to the matcher, which is
// circular for the standard-14 and openly so — these cases check the mechanism,
// not the accuracy. Accuracy is measured against the fonts actually installed
// on the machine by the hidden [fonteval] / [fontmatch] / [fontid] cases in
// test_font_style.cpp, which is where the honest numbers come from: 96.7%
// family over 90 verified text faces, and 100% family / 70% exact face when the
// catalogue holds all 97 of them.

#include <cstdio>
#include <string>
#include <vector>

#include <fpdf_edit.h>
#include <fpdf_save.h>
#include <fpdfview.h>

#include <opencv2/imgproc.hpp>

#include "catch_amalgamated.hpp"
#include "turbo_ocr/pdf/text/font_match.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"

using turbo_ocr::pdf::FamilyPrior;
using turbo_ocr::pdf::FontCandidate;
using turbo_ocr::pdf::FontFamily;
using turbo_ocr::pdf::FontSample;
using turbo_ocr::pdf::load_font_bytes;
using turbo_ocr::pdf::match_font;
using turbo_ocr::pdf::shape_agreement;
using turbo_ocr::pdf::standard_font_catalogue;

namespace {

constexpr const char *kSpecimen = "The quality of mercy is not strained";

std::vector<unsigned short> utf16(const std::string &s) {
  std::vector<unsigned short> out(s.size() + 1, 0);
  for (size_t i = 0; i < s.size(); ++i)
    out[i] = static_cast<unsigned char>(s[i]);
  return out;
}

// A page set in `face` at scan-like resolution, cropped to the ink — the shape
// of thing the matcher meets in production.
cv::Mat specimen_crop(const char *face, const std::string &text = kSpecimen) {
  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc = FPDF_CreateNewDocument();
  if (!doc) return {};
  const double pw = 700;
  const double ph = 90;
  FPDF_PAGE page = FPDFPage_New(doc, 0, pw, ph);
  FPDF_FONT font = FPDFText_LoadStandardFont(doc, face);
  cv::Mat out;
  if (page && font) {
    FPDF_PAGEOBJECT obj = FPDFPageObj_CreateTextObj(doc, font, 20);
    const auto wide = utf16(text);
    if (obj && FPDFText_SetText(obj, reinterpret_cast<FPDF_WIDESTRING>(wide.data()))) {
      FPDFPageObj_Transform(obj, 1, 0, 0, 1, 24, 34);
      FPDFPage_InsertObject(page, obj);
      FPDFPage_GenerateContent(page);
      const int bw = static_cast<int>(pw * 2);
      const int bh = static_cast<int>(ph * 2);
      FPDF_BITMAP bmp = FPDFBitmap_Create(bw, bh, 0);
      if (bmp) {
        FPDFBitmap_FillRect(bmp, 0, 0, bw, bh, 0xFFFFFFFF);
        FPDF_RenderPageBitmap(bmp, page, 0, 0, bw, bh, 0, 0);
        const cv::Mat view(bh, bw, CV_8UC4, FPDFBitmap_GetBuffer(bmp),
                           static_cast<size_t>(FPDFBitmap_GetStride(bmp)));
        cv::Mat bgr;
        cv::cvtColor(view, bgr, cv::COLOR_BGRA2BGR);
        cv::Mat gray;
        cv::Mat mask;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
        const cv::Rect ink = cv::boundingRect(mask);
        if (ink.width > 20 && ink.height > 6) out = bgr(ink).clone();
        FPDFBitmap_Destroy(bmp);
      }
    } else if (obj) {
      FPDFPageObj_Destroy(obj);
    }
  }
  if (font) FPDFFont_Close(font);
  if (page) FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
  return out;
}

const FontCandidate &named(const char *std14) {
  for (const auto &c : standard_font_catalogue())
    if (c.standard14 == std14) return c;
  FAIL("no such candidate: " << std14);
  return standard_font_catalogue()[0];
}

} // namespace

TEST_CASE("a face resembles itself more than it resembles another", "[fontmatch]") {
  const cv::Mat serif = specimen_crop("Times-Roman");
  REQUIRE_FALSE(serif.empty());

  const float same = shape_agreement(serif, kSpecimen, named("Times-Roman"));
  const float other = shape_agreement(serif, kSpecimen, named("Helvetica"));
  INFO("Times against Times " << same << ", against Helvetica " << other);
  CHECK(same > other);
  // Setting the very same words in the very same face measures 0.84, not 1.0:
  // the masks are dilated and re-thresholded before overlap, which costs a
  // little even on an identical pair. What matters is the distance to the next
  // family, which sits at 0.51 — a gap wide enough to decide on.
  CHECK(same > 0.80F);
  CHECK(same - other > 0.25F);
}

TEST_CASE("the matcher picks the face the page was set in", "[fontmatch]") {
  for (const char *face : {"Times-Roman", "Helvetica", "Courier"}) {
    const cv::Mat crop = specimen_crop(face);
    REQUIRE_FALSE(crop.empty());
    const std::vector<FontSample> samples{{crop, kSpecimen}};
    const auto m = match_font(samples, standard_font_catalogue());
    REQUIRE(m.index >= 0);
    const auto &won = standard_font_catalogue()[static_cast<size_t>(m.index)];
    INFO("set in " << face << ", matched " << won.standard14 << " at " << m.score);
    CHECK(won.standard14 == std::string(face));
  }
}

TEST_CASE("the family prior breaks ties without overruling the shapes",
          "[fontmatch]") {
  const cv::Mat serif = specimen_crop("Times-Roman");
  REQUIRE_FALSE(serif.empty());
  const std::vector<FontSample> samples{{serif, kSpecimen}};

  // A wrong prior, at the strength actually shipped, must not turn a clear
  // shape win into a loss. If it can, the prior is doing the deciding.
  FamilyPrior wrong;
  wrong.family = FontFamily::Sans;
  wrong.strength = 0.25F;
  const auto m = match_font(samples, standard_font_catalogue(), wrong);
  REQUIRE(m.index >= 0);
  CHECK(standard_font_catalogue()[static_cast<size_t>(m.index)].family ==
        FontFamily::Serif);
}

TEST_CASE("nothing to go on is reported, not guessed", "[fontmatch]") {
  CHECK(match_font({}, standard_font_catalogue()).index < 0);

  const cv::Mat crop = specimen_crop("Helvetica");
  REQUIRE_FALSE(crop.empty());
  // Text the standard-14 encodings cannot spell: there is no rendering to
  // compare against, so the sample is skipped rather than scored at zero.
  const std::vector<FontSample> greek{{crop, "\xCE\xA9\xCE\xBC\xCE\xB5"}};
  CHECK(match_font(greek, standard_font_catalogue()).index < 0);

  // A blank crop carries no ink to compare.
  const cv::Mat blank(40, 200, CV_8UC3, cv::Scalar(255, 255, 255));
  const std::vector<FontSample> empty{{blank, kSpecimen}};
  CHECK(match_font(empty, standard_font_catalogue()).index < 0);

  CHECK(shape_agreement(cv::Mat(), kSpecimen, named("Helvetica")) == 0.0F);
  CHECK(shape_agreement(crop, "", named("Helvetica")) == 0.0F);
}

TEST_CASE("the catalogue covers every style of every standard family",
          "[fontmatch]") {
  const auto &cat = standard_font_catalogue();
  CHECK(cat.size() == 12);
  int sans = 0;
  int serif = 0;
  int mono = 0;
  for (const auto &c : cat) {
    CHECK_FALSE(c.standard14.empty());
    // Standard-14 needs no font file, which is the whole reason the shipped
    // catalogue is made of it: nothing is embedded, so nothing is licensed.
    CHECK(c.file.empty());
    CHECK(c.embeddable);
    if (c.family == FontFamily::Sans) ++sans;
    else if (c.family == FontFamily::Serif) ++serif;
    else ++mono;
  }
  CHECK(sans == 4);
  CHECK(serif == 4);
  CHECK(mono == 4);
}

TEST_CASE("a plain font file is passed through unchanged", "[fontmatch]") {
  // Any .ttf on the machine will do; skip where there is none.
  const char *candidates[] = {
      "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
      "/System/Library/Fonts/Supplemental/Arial.ttf",
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"};
  for (const char *path : candidates) {
    const auto bytes = load_font_bytes(path, 0);
    if (bytes.empty()) continue;
    CHECK(bytes.size() > 1000);
    // A single-face file starts with the sfnt version and is handed over as-is.
    const bool sfnt = (bytes[0] == 0x00 && bytes[1] == 0x01) ||
                      (bytes[0] == 'O' && bytes[1] == 'T');
    CHECK(sfnt);
    return;
  }
  WARN("no plain font file found to test with");
}

TEST_CASE("a face is lifted out of a collection as a standalone font",
          "[fontmatch]") {
  // macOS ships most of its text faces inside .ttc collections. Without this
  // extraction the matcher cannot see any of them.
  const char *collections[] = {"/System/Library/Fonts/Supplemental/Baskerville.ttc",
                               "/System/Library/Fonts/Helvetica.ttc",
                               "/System/Library/Fonts/Supplemental/Bodoni 72.ttc"};
  for (const char *path : collections) {
    const auto bytes = load_font_bytes(path, 0);
    if (bytes.empty()) continue;
    REQUIRE(bytes.size() > 1000);
    // The result must be a single-face file, not the collection it came from.
    CHECK(bytes[0] == 0x00);
    CHECK(bytes[1] == 0x01);
    CHECK(bytes[2] == 0x00);
    CHECK(bytes[3] == 0x00);

    // Its table directory has to be self-consistent, or PDFium rejects it.
    const int num = (bytes[4] << 8) | bytes[5];
    CHECK(num > 4);
    for (int i = 0; i < num; ++i) {
      const size_t rec = 12 + static_cast<size_t>(i) * 16;
      REQUIRE(rec + 16 <= bytes.size());
      const size_t off = (static_cast<size_t>(bytes[rec + 8]) << 24) |
                         (static_cast<size_t>(bytes[rec + 9]) << 16) |
                         (static_cast<size_t>(bytes[rec + 10]) << 8) | bytes[rec + 11];
      const size_t len = (static_cast<size_t>(bytes[rec + 12]) << 24) |
                         (static_cast<size_t>(bytes[rec + 13]) << 16) |
                         (static_cast<size_t>(bytes[rec + 14]) << 8) | bytes[rec + 15];
      CHECK(off + len <= bytes.size());
    }

    // Out-of-range faces are refused rather than returning something arbitrary.
    CHECK(load_font_bytes(path, 999).empty());
    return;
  }
  WARN("no font collection found to test with");
}

TEST_CASE("a missing font file is refused", "[fontmatch]") {
  CHECK(load_font_bytes("/nonexistent/font.ttf", 0).empty());
  CHECK(load_font_bytes("", 0).empty());
}
