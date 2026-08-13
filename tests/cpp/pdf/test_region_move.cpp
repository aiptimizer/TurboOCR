// Figures lifted out of the page raster so they can actually be moved.
//
// The test that matters is not "an object was added" — it is what happens when
// that object is MOVED. A page marked up with annotations passes any check that
// only counts objects, and then drags an empty outline across a picture that
// never budges. So each case here moves the region and looks at what is left
// behind: the hole has to be clean, and the figure has to be somewhere else.

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <fpdf_edit.h>
#include <fpdf_save.h>
#include <fpdfview.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "catch_amalgamated.hpp"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pdf/text/pdf_searchable.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"
#include "turbo_ocr/pdf/text/region_extract.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::pdf::extract_movable_regions;
using turbo_ocr::pdf::RegionImage;
using turbo_ocr::pdf::RuleShape;
using turbo_ocr::pdf::extract_rules;
using turbo_ocr::pdf::SearchablePage;
using turbo_ocr::pdf::SearchableStats;
using turbo_ocr::pdf::write_searchable_pdf;

namespace {
// The layout header exposes id -> name; this is the inverse, which only the
// tests need.
int class_id_named(std::string_view want) {
  for (int i = 0; i < 64; ++i)
    if (turbo_ocr::layout::label_name(i) == want) return i;
  FAIL("no layout class named " << want);
  return 0;
}
} // namespace

namespace {

constexpr int kPageW = 480;
constexpr int kPageH = 360;
// Where the "figure" sits on the mock scan.
const cv::Rect kFigure(60, 40, 200, 120);

struct Writer : FPDF_FILEWRITE {
  std::string buf;
};

int write_block(FPDF_FILEWRITE *w, const void *data, unsigned long size) {
  static_cast<Writer *>(w)->buf.append(static_cast<const char *>(data), size);
  return 1;
}

// A page with one unmistakable coloured figure on plain paper, plus a line of
// dark text well away from it.
cv::Mat mock_scan() {
  cv::Mat page(kPageH, kPageW, CV_8UC3, cv::Scalar(250, 248, 246));
  cv::rectangle(page, kFigure, cv::Scalar(40, 40, 220), cv::FILLED);
  cv::circle(page, {kFigure.x + 60, kFigure.y + 60}, 30, cv::Scalar(30, 200, 30),
             cv::FILLED);
  cv::putText(page, "caption below", {60, 250}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(20, 20, 20), 2);
  return page;
}

std::vector<turbo_ocr::layout::LayoutBox> figure_layout() {
  turbo_ocr::layout::LayoutBox b;
  // "image" is one of the movable classes.
  b.class_id = class_id_named("image");
  b.box = Box{{{{kFigure.x, kFigure.y},
                {kFigure.x + kFigure.width, kFigure.y},
                {kFigure.x + kFigure.width, kFigure.y + kFigure.height},
                {kFigure.x, kFigure.y + kFigure.height}}}};
  b.score = 0.95F;
  return {b};
}

// A one-page PDF whose only content is `bgr` — a scan.
std::string scan_pdf(const cv::Mat &bgr) {
  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc = FPDF_CreateNewDocument();
  FPDF_PAGE page = FPDFPage_New(doc, 0, kPageW, kPageH);
  FPDF_BITMAP bmp = FPDFBitmap_Create(bgr.cols, bgr.rows, 0);
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
  FPDFImageObj_SetBitmap(&page, 1, img, bmp);
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

cv::Mat rasterise(const std::string &pdf) {
  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(pdf.data(), static_cast<int>(pdf.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  REQUIRE(page != nullptr);
  FPDF_BITMAP bmp = FPDFBitmap_Create(kPageW, kPageH, 0);
  FPDFBitmap_FillRect(bmp, 0, 0, kPageW, kPageH, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bmp, page, 0, 0, kPageW, kPageH, 0, 0);
  const cv::Mat view(kPageH, kPageW, CV_8UC4, FPDFBitmap_GetBuffer(bmp),
                     static_cast<size_t>(FPDFBitmap_GetStride(bmp)));
  cv::Mat out;
  cv::cvtColor(view, out, cv::COLOR_BGRA2BGR);
  FPDFBitmap_Destroy(bmp);
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
  return out;
}

// One recognised word, so a page can be made to have text work on it.
OCRResultItem word_at(const char *text, int x0, int y0, int x1, int y1) {
  OCRResultItem item;
  item.text = text;
  item.confidence = 0.99f;
  item.box = Box{{{{{x0, y0}}, {{x1, y0}}, {{x1, y1}}, {{x0, y1}}}}};
  return item;
}

// Writes the scan out with its figure lifted into its own object.
std::string with_movable_figure(const cv::Mat &scan, SearchableStats &stats,
                                std::vector<RegionImage> &regions) {
  const std::string src = scan_pdf(scan);
  const auto layout = figure_layout();
  regions = extract_movable_regions(scan, layout);
  REQUIRE(regions.size() == 1);

  std::vector<OCRResultItem> results;
  SearchablePage p;
  p.page_index = 0;
  p.raster_w = scan.cols;
  p.raster_h = scan.rows;
  p.results = &results;
  p.regions = &regions;

  std::string err;
  const std::string out = write_searchable_pdf(
      reinterpret_cast<const uint8_t *>(src.data()), src.size(), {p}, 0.0F,
      &stats, err, turbo_ocr::pdf::TextLayerMode::Invisible,
      /*movable_regions=*/true);
  INFO("writer error: " << err);
  REQUIRE(err.empty());
  REQUIRE_FALSE(out.empty());
  return out;
}

// Shifts every image object on page 0 by (dx, dy) — what a viewer does when
// someone drags a figure.
std::string move_images(const std::string &pdf, float dx, float dy,
                        int &moved) {
  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(pdf.data(), static_cast<int>(pdf.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  moved = 0;
  const int n = FPDFPage_CountObjects(page);
  for (int i = 0; i < n; ++i) {
    FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, i);
    if (FPDFPageObj_GetType(o) != FPDF_PAGEOBJ_IMAGE) continue;
    FS_MATRIX m{};
    if (!FPDFPageObj_GetMatrix(o, &m)) continue;
    // The page's own background image spans the sheet; leave it where it is and
    // move only the lifted region, which is what a person would drag.
    if (m.a > static_cast<float>(kPageW) * 0.9F) continue;
    m.e += dx;
    m.f += dy;
    FPDFPageObj_SetMatrix(o, &m);
    ++moved;
  }
  FPDFPage_GenerateContent(page);
  Writer w;
  w.version = 1;
  w.WriteBlock = write_block;
  FPDF_SaveAsCopy(doc, &w, 0);
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
  return w.buf;
}

// How much of `r` is the figure's strong blue/green, as a fraction.
double figure_ink(const cv::Mat &page, const cv::Rect &r) {
  const cv::Rect safe = r & cv::Rect(0, 0, page.cols, page.rows);
  if (safe.width <= 0 || safe.height <= 0) return 0.0;
  const cv::Mat crop = page(safe);
  int hits = 0;
  for (int y = 0; y < crop.rows; ++y) {
    const auto *row = crop.ptr<cv::Vec3b>(y);
    for (int x = 0; x < crop.cols; ++x) {
      // The figure is drawn Scalar(40,40,220) and Scalar(30,200,30) — in BGR
      // that is RED and green.
      const int b = row[x][0], g = row[x][1], rr = row[x][2];
      const bool red = rr > 150 && b < 110 && g < 110;
      const bool green = g > 150 && rr < 110 && b < 110;
      if (red || green) ++hits;
    }
  }
  return static_cast<double>(hits) / (crop.rows * crop.cols);
}

} // namespace

// Writes before/after pictures of a figure being dragged, for looking at.
//   TOCR_MOVDEMO=/tmp/out turbo_ocr_tests "[movdemo]"
TEST_CASE("picture of a figure being moved", "[.movdemo]") {
  const char *dir = std::getenv("TOCR_MOVDEMO");
  if (dir == nullptr) {
    WARN("set TOCR_MOVDEMO to a directory");
    return;
  }
  const cv::Mat scan = mock_scan();
  SearchableStats stats;
  std::vector<RegionImage> regions;
  const std::string lifted = with_movable_figure(scan, stats, regions);
  int moved = 0;
  const cv::Mat after = rasterise(move_images(lifted, 220.0F, -80.0F, moved));
  cv::imwrite(std::string(dir) + "/move_before.png", scan);
  cv::imwrite(std::string(dir) + "/move_after.png", after);
  printf("wrote before/after to %s (moved %d object)\n", dir, moved);
}

TEST_CASE("a figure is lifted out as its own object", "[pdf][movable]") {
  const cv::Mat scan = mock_scan();
  SearchableStats stats;
  std::vector<RegionImage> regions;
  const std::string out = with_movable_figure(scan, stats, regions);

  CHECK(stats.movable == 1);
  CHECK(regions[0].label == "image");
  CHECK(regions[0].w == kFigure.width);
  CHECK(regions[0].h == kFigure.height);
  // The paper colour is read off the page, so a cream form does not gain a
  // white rectangle.
  CHECK(regions[0].paper[0] > 200);
  CHECK(regions[0].paper[2] > 200);
}

TEST_CASE("a figure that cannot be re-embedded is left alone, not erased",
          "[pdf][movable]") {
  // REGRESSION. The paper "hole" that makes the lift work also DESTROYS the
  // original pixels, so it must never go down before the replacement image is
  // known good. It used to: the patch was inserted first and both bail-outs
  // (FPDFImageObj_NewImageObj null, and the LoadJpegFileInline/decode ladder
  // failing) `continue`d with the patch already on the page's object list. One
  // OCR'd word elsewhere on the page then made FPDFPage_GenerateContent commit
  // it, and the figure came back as a flat rectangle of paper with nothing
  // drawn in its place — silently, with stats.movable == 0.
  const cv::Mat scan = mock_scan();
  const std::string src = scan_pdf(scan);
  const auto layout = figure_layout();
  std::vector<RegionImage> regions = extract_movable_regions(scan, layout);
  REQUIRE(regions.size() == 1);
  // Keep the geometry and the paper colour; poison ONLY the payload, so both
  // LoadJpegFileInline and the cv::imdecode fallback refuse it.
  regions[0].bytes.assign(64, static_cast<uint8_t>(0x5A));

  // One real word, so `any` is true for the page and GenerateContent runs —
  // that is what turns an orphan patch into a committed erasure.
  std::vector<OCRResultItem> results{word_at("caption", 60, 235, 200, 260)};
  SearchablePage p;
  p.page_index = 0;
  p.raster_w = scan.cols;
  p.raster_h = scan.rows;
  p.results = &results;
  p.regions = &regions;

  SearchableStats stats;
  std::string err;
  const std::string out = write_searchable_pdf(
      reinterpret_cast<const uint8_t *>(src.data()), src.size(), {p}, 0.0F,
      &stats, err, turbo_ocr::pdf::TextLayerMode::Invisible,
      /*movable_regions=*/true);
  INFO("writer error: " << err);
  REQUIRE(err.empty());
  REQUIRE_FALSE(out.empty());
  CHECK(stats.movable == 0);   // nothing was lifted
  CHECK(stats.words == 1);     // ...but the page WAS rewritten

  const cv::Mat after = rasterise(out);
  const double before_ink = figure_ink(scan, kFigure);
  const double after_ink = figure_ink(after, kFigure);
  INFO("figure ink before " << before_ink << " after " << after_ink);
  CHECK(before_ink > 0.8);
  CHECK(after_ink > 0.8);      // the figure is still there
}

TEST_CASE("the page still looks the same before anything is moved",
          "[pdf][movable]") {
  // Lifting the figure out must be invisible until someone acts on it.
  const cv::Mat scan = mock_scan();
  SearchableStats stats;
  std::vector<RegionImage> regions;
  const cv::Mat after = rasterise(with_movable_figure(scan, stats, regions));

  const double before_ink = figure_ink(scan, kFigure);
  const double after_ink = figure_ink(after, kFigure);
  INFO("figure ink before " << before_ink << " after " << after_ink);
  CHECK(before_ink > 0.8);
  CHECK(after_ink > 0.8);
}

TEST_CASE("moving the figure moves the FIGURE, and leaves clean paper",
          "[pdf][movable]") {
  // This is the whole point. Drag the object and the picture goes with it;
  // where it was is blank page, not the original showing through.
  const cv::Mat scan = mock_scan();
  SearchableStats stats;
  std::vector<RegionImage> regions;
  const std::string lifted = with_movable_figure(scan, stats, regions);

  int moved = 0;
  // Far enough that the destination does not overlap the origin — the figure
  // is 200 wide, so a 150 shift would leave a quarter of it on top of its own
  // old position and the "clean paper" check would be measuring the figure.
  const cv::Mat after = rasterise(move_images(lifted, 220.0F, -80.0F, moved));
  REQUIRE(moved == 1);

  const double left_behind = figure_ink(after, kFigure);
  INFO("ink left where the figure was: " << left_behind);
  CHECK(left_behind < 0.05);

  // And it arrived. PDF y runs up the page, so a +60 pt drop in device terms.
  const cv::Rect dest(kFigure.x + 220, kFigure.y + 80, kFigure.width,
                      kFigure.height);
  const double arrived = figure_ink(after, dest);
  INFO("ink at the destination: " << arrived);
  CHECK(arrived > 0.7);
}

TEST_CASE("text near a lifted figure is not disturbed", "[pdf][movable]") {
  const cv::Mat scan = mock_scan();
  SearchableStats stats;
  std::vector<RegionImage> regions;
  const cv::Mat after = rasterise(with_movable_figure(scan, stats, regions));

  // The caption sits well below the figure and must survive untouched.
  const cv::Rect caption(50, 225, 260, 40);
  cv::Mat a;
  cv::Mat b;
  cv::cvtColor(scan(caption), a, cv::COLOR_BGR2GRAY);
  cv::cvtColor(after(caption), b, cv::COLOR_BGR2GRAY);
  cv::Mat diff;
  cv::absdiff(a, b, diff);
  const double changed =
      static_cast<double>(cv::countNonZero(diff > 60)) / diff.total();
  INFO("caption pixels changed: " << changed);
  CHECK(changed < 0.02);
}

TEST_CASE("regions too small or too large to be worth moving are left alone",
          "[pdf][movable]") {
  const cv::Mat scan = mock_scan();

  turbo_ocr::layout::LayoutBox tiny;
  tiny.class_id = class_id_named("image");
  tiny.box = Box{{{{10, 10}, {24, 10}, {24, 24}, {10, 24}}}};
  CHECK(extract_movable_regions(scan, {tiny}).empty());

  turbo_ocr::layout::LayoutBox whole;
  whole.class_id = class_id_named("image");
  whole.box = Box{{{{0, 0}, {kPageW, 0}, {kPageW, kPageH}, {0, kPageH}}}};
  CHECK(extract_movable_regions(scan, {whole}).empty());
}

TEST_CASE("text regions are never lifted", "[pdf][movable]") {
  // Words are handled by the text layer. Lifting them would put a picture of
  // the words on top of the words.
  const cv::Mat scan = mock_scan();
  turbo_ocr::layout::LayoutBox para;
  para.class_id = class_id_named("text");
  para.box = Box{{{{kFigure.x, kFigure.y},
                   {kFigure.x + kFigure.width, kFigure.y},
                   {kFigure.x + kFigure.width, kFigure.y + kFigure.height},
                   {kFigure.x, kFigure.y + kFigure.height}}}};
  CHECK(extract_movable_regions(scan, {para}).empty());
}

TEST_CASE("asking for movable regions without any is harmless",
          "[pdf][movable]") {
  const cv::Mat scan = mock_scan();
  const std::string src = scan_pdf(scan);
  std::vector<OCRResultItem> results;
  std::vector<RegionImage> none;
  SearchablePage p;
  p.page_index = 0;
  p.raster_w = scan.cols;
  p.raster_h = scan.rows;
  p.results = &results;
  p.regions = &none;

  SearchableStats stats;
  std::string err;
  const std::string out = write_searchable_pdf(
      reinterpret_cast<const uint8_t *>(src.data()), src.size(), {p}, 0.0F,
      &stats, err, turbo_ocr::pdf::TextLayerMode::Invisible, true);
  REQUIRE(err.empty());
  CHECK(stats.movable == 0);
  CHECK_FALSE(out.empty());
}

// ── printed rules ─────────────────────────────────────────────────────────

namespace {

// A form-like page: a ruled table and an underline, on plain paper.
cv::Mat ruled_page() {
  cv::Mat page(kPageH, kPageW, CV_8UC3, cv::Scalar(252, 250, 248));
  for (int y : {60, 100, 140, 180})
    cv::line(page, {40, y}, {440, y}, cv::Scalar(30, 30, 30), 2);
  for (int x : {40, 200, 440})
    cv::line(page, {x, 60}, {x, 180}, cv::Scalar(30, 30, 30), 2);
  cv::line(page, {40, 260}, {300, 260}, cv::Scalar(30, 30, 30), 2);
  cv::putText(page, "Name", {50, 250}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(20, 20, 20), 1);
  return page;
}

std::string with_rules(const cv::Mat &scan, SearchableStats &stats,
                       std::vector<RuleShape> &rules) {
  const std::string src = scan_pdf(scan);
  rules = extract_rules(scan);
  REQUIRE_FALSE(rules.empty());

  std::vector<OCRResultItem> results;
  SearchablePage p;
  p.page_index = 0;
  p.raster_w = scan.cols;
  p.raster_h = scan.rows;
  p.results = &results;
  p.rules = &rules;

  std::string err;
  const std::string out = write_searchable_pdf(
      reinterpret_cast<const uint8_t *>(src.data()), src.size(), {p}, 0.0F,
      &stats, err, turbo_ocr::pdf::TextLayerMode::Invisible, true);
  INFO("writer error: " << err);
  REQUIRE(err.empty());
  return out;
}

int path_objects(const std::string &pdf) {
  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(pdf.data(), static_cast<int>(pdf.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  int n = 0;
  for (int i = 0; i < FPDFPage_CountObjects(page); ++i)
    if (FPDFPageObj_GetType(FPDFPage_GetObject(page, i)) == FPDF_PAGEOBJ_PATH) ++n;
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
  return n;
}

} // namespace

TEST_CASE("printed rules are found as shapes, not pixels", "[pdf][movable]") {
  const auto rules = extract_rules(ruled_page());
  INFO("rules found: " << rules.size());
  // Four horizontals and three verticals were drawn; detection may merge or
  // split a couple, so this asks for most of them rather than exactly them.
  CHECK(rules.size() >= 6);
  const auto horizontals =
      std::ranges::count_if(rules, [](const RuleShape &r) { return r.horizontal; });
  CHECK(horizontals >= 3);
  // Ink and paper are read off the page, not assumed.
  for (const RuleShape &r : rules) {
    CHECK(r.ink[0] < 120);
    CHECK(r.paper[0] > 200);
  }
}

TEST_CASE("a letter stroke is not mistaken for a rule", "[pdf][movable]") {
  // Text alone must produce nothing: a rule is defined by running unbroken
  // across a real fraction of the page, which no glyph does.
  cv::Mat words(kPageH, kPageW, CV_8UC3, cv::Scalar(252, 250, 248));
  cv::putText(words, "lllllll HHHHH IIIII", {40, 120}, cv::FONT_HERSHEY_SIMPLEX,
              1.2, cv::Scalar(20, 20, 20), 3);
  CHECK(extract_rules(words).empty());
}

TEST_CASE("each rule becomes a selectable object", "[pdf][movable]") {
  const cv::Mat scan = ruled_page();
  SearchableStats stats;
  std::vector<RuleShape> rules;
  const std::string out = with_rules(scan, stats, rules);

  CHECK(stats.rules == static_cast<int>(rules.size()));
  // One patch and one shape per rule — both are paths, so both are selectable
  // and the patch can be removed if someone wants the scan back.
  CHECK(path_objects(out) >= static_cast<int>(rules.size()) * 2);
}

TEST_CASE("the ruled page still looks ruled", "[pdf][movable]") {
  // Replacing ink with shapes must be invisible until someone acts on it.
  const cv::Mat scan = ruled_page();
  SearchableStats stats;
  std::vector<RuleShape> rules;
  const cv::Mat after = rasterise(with_rules(scan, stats, rules));

  cv::Mat a;
  cv::Mat b;
  cv::cvtColor(scan, a, cv::COLOR_BGR2GRAY);
  cv::cvtColor(after, b, cv::COLOR_BGR2GRAY);
  const double dark_before =
      static_cast<double>(cv::countNonZero(a < 128)) / a.total();
  const double dark_after =
      static_cast<double>(cv::countNonZero(b < 128)) / b.total();
  INFO("dark before " << dark_before << " after " << dark_after);
  // The rules are still there, in about the same quantity of ink.
  CHECK(dark_after > dark_before * 0.6);
  CHECK(dark_after < dark_before * 1.8);
}

TEST_CASE("moving a rule leaves clean paper", "[pdf][movable]") {
  const cv::Mat scan = ruled_page();
  SearchableStats stats;
  std::vector<RuleShape> rules;
  const std::string out = with_rules(scan, stats, rules);

  // Delete every rule SHAPE (keeping the patches) and the page should come
  // back blank where the rules were — proof the ink really was replaced rather
  // than merely covered over.
  FPDF_DOCUMENT doc =
      FPDF_LoadMemDocument(out.data(), static_cast<int>(out.size()), nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  std::vector<FPDF_PAGEOBJECT> dark;
  for (int i = 0; i < FPDFPage_CountObjects(page); ++i) {
    FPDF_PAGEOBJECT o = FPDFPage_GetObject(page, i);
    if (FPDFPageObj_GetType(o) != FPDF_PAGEOBJ_PATH) continue;
    unsigned int r = 0, g = 0, b = 0, a = 0;
    if (!FPDFPageObj_GetFillColor(o, &r, &g, &b, &a)) continue;
    if (r < 120 && g < 120 && b < 120) dark.push_back(o);
  }
  INFO("dark rule shapes: " << dark.size());
  CHECK_FALSE(dark.empty());
  for (FPDF_PAGEOBJECT o : dark) FPDFPage_RemoveObject(page, o);
  FPDFPage_GenerateContent(page);

  Writer w;
  w.version = 1;
  w.WriteBlock = write_block;
  FPDF_SaveAsCopy(doc, &w, 0);
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);

  const cv::Mat stripped = rasterise(w.buf);
  cv::Mat gray;
  cv::cvtColor(stripped, gray, cv::COLOR_BGR2GRAY);
  // Only the caption should be left.
  const double remaining =
      static_cast<double>(cv::countNonZero(gray < 128)) / gray.total();
  INFO("ink left after removing the rules: " << remaining);
  CHECK(remaining < 0.01);
}
