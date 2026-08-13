// Font-style estimation, calibrated against type rendered by PDFium itself.
//
// Every sample here is generated at run time: a one-page PDF is built in a
// chosen face, rasterised, and handed back to the estimator as if it had come
// off a scanner. That matters more than it sounds. The thresholds in
// font_style.cpp are the whole feature — get them wrong and every document
// comes out in the wrong typeface — and a checked-in raster fixture would let
// them drift out from under the estimator silently. Regenerating the ground
// truth on every run means these numbers cannot go stale without the suite
// saying so.
//
// The FACE, though, is pinned. The specimens used to be rendered from the PDF
// standard-14 names (Helvetica, Courier, …), which PDFium resolves through
// per-OS font substitution — so "Helvetica-Bold" came out bold on macOS/Linux
// but with a non-bold substitute on Windows, and "Courier" rendered a face
// measure_line_style could not read there. Same test, different pixels per OS,
// two spurious Windows failures. The specimens now render from bundled DejaVu
// faces the test ships (tests/fixtures/fonts/, via TURBO_TEST_FONT_DIR), so the
// raster is identical on every platform. Nothing rasterised is checked in — the
// live-regeneration guarantee above is untouched; only the input font is now
// deterministic instead of whatever the host happened to substitute.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <fpdf_edit.h>
#include <fpdf_text.h>
#include <fpdfview.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "catch_amalgamated.hpp"
#include "turbo_ocr/pdf/text/font_match.h"
#include "turbo_ocr/pdf/text/font_style.h"
#include "turbo_ocr/pdf/text/pdf_text_layer.h"

using turbo_ocr::Box;
using turbo_ocr::pdf::FontFamily;
using turbo_ocr::pdf::LineStyle;
using turbo_ocr::pdf::measure_line_style;
using turbo_ocr::pdf::resolve_document_fonts;

namespace {

// A pangram-ish sample: ascenders, descenders, round and straight stems, and
// enough word gaps for the pitch measurement to have something to work with.
constexpr const char *kSpecimen = "Handgloves quick brown fox jumps 12345";

std::vector<unsigned short> utf16(const std::string &s) {
  std::vector<unsigned short> out(s.size() + 1, 0);
  for (size_t i = 0; i < s.size(); ++i)
    out[i] = static_cast<unsigned char>(s[i]);
  return out;
}

struct Sample {
  cv::Mat image;  // BGR, white page with one line of black text
  Box box{};      // tight around the ink, as a detector would report it
  bool ok = false;
};

// Resolve a specimen face to a font handle. The standard-14 names the tests ask
// for are mapped to the bundled DejaVu faces (see the file header for why), read
// as bytes and embedded with FPDFText_LoadFont — the same path production's
// font matcher uses (load_font_bytes + FPDFText_LoadFont). If the fixture dir is
// not defined at compile time, or a name is unmapped, or the file is missing, it
// falls back to the standard-14 name so the suite still runs (just non-hermetic,
// as it did before).
FPDF_FONT test_font(FPDF_DOCUMENT doc, const char *font_name) {
#ifdef TURBO_TEST_FONT_DIR
  static const std::map<std::string, std::string> kFace = {
      {"Helvetica",             "DejaVuSans.ttf"},
      {"Helvetica-Bold",        "DejaVuSans-Bold.ttf"},
      {"Helvetica-Oblique",     "DejaVuSans-Oblique.ttf"},
      {"Helvetica-BoldOblique", "DejaVuSans-BoldOblique.ttf"},
      {"Times-Roman",           "DejaVuSerif.ttf"},
      {"Times-Bold",            "DejaVuSerif-Bold.ttf"},
      {"Times-Italic",          "DejaVuSerif-Italic.ttf"},
      {"Times-BoldItalic",      "DejaVuSerif-BoldItalic.ttf"},
      {"Courier",               "DejaVuSansMono.ttf"},
      {"Courier-Bold",          "DejaVuSansMono-Bold.ttf"},
      {"Courier-BoldOblique",   "DejaVuSansMono-BoldOblique.ttf"},
  };
  if (auto it = kFace.find(font_name); it != kFace.end()) {
    const std::string path = std::string(TURBO_TEST_FONT_DIR) + "/" + it->second;
    const std::vector<uint8_t> bytes = turbo_ocr::pdf::load_font_bytes(path, 0);
    if (!bytes.empty())
      if (FPDF_FONT f = FPDFText_LoadFont(
              doc, bytes.data(), static_cast<uint32_t>(bytes.size()),
              FPDF_FONT_TRUETYPE, /*cid=*/1))
        return f;
  }
#endif
  return FPDFText_LoadStandardFont(doc, font_name);
}

// Renders `text` in `font_name` at `pt`, then rasterises the page at `scale`
// times PDF units — scale 2 lands near 144 dpi, which is where real scans sit.
Sample render_text(const char *font_name, float pt, const std::string &text,
                   float scale) {
  Sample s;
  turbo_ocr::pdf::ensure_pdfium_initialized();

  FPDF_DOCUMENT doc = FPDF_CreateNewDocument();
  if (!doc) return s;
  const double pw = 640;
  const double ph = 100;
  FPDF_PAGE page = FPDFPage_New(doc, 0, pw, ph);
  FPDF_FONT font = test_font(doc, font_name);
  if (!page || !font) {
    FPDF_CloseDocument(doc);
    return s;
  }

  FPDF_PAGEOBJECT obj = FPDFPageObj_CreateTextObj(doc, font, pt);
  const auto wide = utf16(text);
  if (!obj || !FPDFText_SetText(obj, reinterpret_cast<FPDF_WIDESTRING>(wide.data()))) {
    if (obj) FPDFPageObj_Destroy(obj);
    FPDFFont_Close(font);
    FPDF_ClosePage(page);
    FPDF_CloseDocument(doc);
    return s;
  }
  FPDFPageObj_Transform(obj, 1, 0, 0, 1, 24, 36);
  FPDFPage_InsertObject(page, obj);
  FPDFPage_GenerateContent(page);

  const int bw = static_cast<int>(pw * scale);
  const int bh = static_cast<int>(ph * scale);
  FPDF_BITMAP bmp = FPDFBitmap_Create(bw, bh, 0);
  if (bmp) {
    FPDFBitmap_FillRect(bmp, 0, 0, bw, bh, 0xFFFFFFFF);
    FPDF_RenderPageBitmap(bmp, page, 0, 0, bw, bh, 0, 0);
    const cv::Mat bgrx(bh, bw, CV_8UC4, FPDFBitmap_GetBuffer(bmp),
                       static_cast<size_t>(FPDFBitmap_GetStride(bmp)));
    // cvtColor allocates, so the result outlives the PDFium buffer.
    cv::cvtColor(bgrx, s.image, cv::COLOR_BGRA2BGR);
    FPDFBitmap_Destroy(bmp);
  }
  FPDFFont_Close(font);
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);

  if (s.image.empty()) return s;

  // The box a detector would hand us: tight around the ink.
  cv::Mat gray;
  cv::Mat mask;
  cv::cvtColor(s.image, gray, cv::COLOR_BGR2GRAY);
  cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
  const cv::Rect r = cv::boundingRect(mask);
  if (r.width < 8 || r.height < 6) return s;
  s.box = Box{{{{r.x, r.y},
                {r.x + r.width, r.y},
                {r.x + r.width, r.y + r.height},
                {r.x, r.y + r.height}}}};
  s.ok = true;
  return s;
}

LineStyle measure(const char *font_name, float pt = 13.0F, float scale = 2.0F) {
  const Sample s = render_text(font_name, pt, kSpecimen, scale);
  REQUIRE(s.ok);
  const LineStyle st = measure_line_style(
      s.image, s.box, static_cast<int>(std::string(kSpecimen).size()));
  REQUIRE(st.measured);
  return st;
}

// Lines whose per-character advance differs sharply in a proportional face and
// not at all in a monospaced one — which is the whole of the pitch test.
constexpr const char *kWidthProbes[] = {
    "Illinois filliniti jilllif", "WAXWORK QUOMODO MMWWMM",
    "Handgloves quick brown fox", "mmmmwwww iiiillll ttttffff",
    "sample text for measurement"};

std::vector<LineStyle> pitch_document(const char *font_name) {
  std::vector<LineStyle> out;
  for (const char *probe : kWidthProbes) {
    const Sample s = render_text(font_name, 14.0F, probe, 2.0F);
    REQUIRE(s.ok);
    const LineStyle st = measure_line_style(
        s.image, s.box, static_cast<int>(std::string(probe).size()));
    REQUIRE(st.measured);
    out.push_back(st);
  }
  return out;
}

// A whole document set in one face, as resolve_document_fonts sees it.
std::vector<LineStyle> document_of(const char *font_name, int lines,
                                   float pt = 13.0F) {
  std::vector<LineStyle> out;
  out.reserve(static_cast<size_t>(lines));
  for (int i = 0; i < lines; ++i) {
    // Vary the size a little so the vote is not being handed identical copies
    // of one measurement, which would prove nothing about its stability.
    const Sample s = render_text(font_name, pt + static_cast<float>(i % 3),
                                 kSpecimen, 2.0F);
    REQUIRE(s.ok);
    out.push_back(measure_line_style(s.image, s.box));
  }
  return out;
}

} // namespace

// Hidden by the leading dot: run it with `turbo_ocr_tests [fontcal]` when a
// threshold needs revisiting. It prints every face at every size the estimator
// is expected to cope with, which is how the constants in font_style.cpp were
// picked in the first place.
TEST_CASE("calibration sweep", "[.fontcal]") {
  const char *faces[] = {"Helvetica",   "Helvetica-Bold", "Helvetica-Oblique",
                         "Times-Roman", "Times-Bold",     "Times-Italic",
                         "Courier",     "Courier-Bold"};
  printf("\n%-20s %5s %5s %7s %7s %7s %7s %6s\n", "face", "pt", "scale", "serif",
         "weight", "advance", "slant", "xh");
  const auto chars = static_cast<int>(std::string(kSpecimen).size());
  for (const char *face : faces) {
    // 14 and 15 are here because document_of() renders 13/14/15 — the sweep
    // used to skip exactly the sizes the tests exercise, so a threshold could
    // (and did) sit right on a quantisation step that the table never showed.
    for (float pt : {9.0F, 13.0F, 14.0F, 15.0F, 20.0F}) {
      for (float scale : {1.5F, 2.0F, 3.0F}) {
        const Sample s = render_text(face, pt, kSpecimen, scale);
        if (!s.ok) continue;
        const LineStyle st = measure_line_style(s.image, s.box, chars);
        if (!st.measured) {
          printf("%-20s %5.0f %5.1f   (unmeasured)\n", face, pt, scale);
          continue;
        }
        printf("%-20s %5.0f %5.1f %7.3f %7.3f %7.3f %7.1f %6.1f\n", face, pt,
               scale, st.serif, st.weight, st.advance_ratio, st.slant_deg,
               st.x_height_px);
      }
    }
  }

  // Real forms are not pangrams. Their lines are "Name:" and "Telefon:", at
  // whatever resolution the caller asked for, and a measurement that only holds
  // on a long specimen does not hold on a form.
  printf("\nshort form labels\n%-14s %-14s %5s %7s %7s %6s\n", "face", "text",
         "scale", "serif", "weight", "xh");
  for (const char *face : {"Helvetica", "Times-Roman"}) {
    for (const char *label : {"Name:", "Vorname:", "Geburtsdatum:", "PLZ / Ort:"}) {
      for (float scale : {1.4F, 2.0F, 4.0F}) {
        const Sample s = render_text(face, 11.0F, label, scale);
        if (!s.ok) continue;
        const LineStyle st = measure_line_style(
            s.image, s.box, static_cast<int>(std::string(label).size()));
        if (!st.measured) {
          printf("%-14s %-14s %5.1f  (unmeasured)\n", face, label, scale);
          continue;
        }
        printf("%-14s %-14s %5.1f %7.3f %7.3f %6.1f\n", face, label, scale,
               st.serif, st.weight, st.x_height_px);
      }
    }
  }

  printf("\nper-document advance spread (mono when small)\n");
  for (const char *face : {"Helvetica", "Times-Roman", "Courier"}) {
    printf("%-14s", face);
    for (const char *probe : kWidthProbes) {
      const Sample s = render_text(face, 14.0F, probe, 2.0F);
      if (!s.ok) continue;
      const LineStyle st = measure_line_style(
          s.image, s.box, static_cast<int>(std::string(probe).size()));
      printf(" %6.3f", st.advance_ratio);
    }
    printf("\n");
  }
}

// Measures a REAL document rather than a rendered specimen. PDFium's own text
// rectangles stand in for the detector's boxes, so this reads the same lines
// the OCR would, at whatever resolution is asked for — which is how the
// low-resolution failure was found: synthetic specimens are long, and a real
// form's lines are three words each.
//
//   STYLE_SCAN_PDF=path STYLE_SCAN_DPI=100 turbo_ocr_tests "[stylescan]"
TEST_CASE("style scan of a real document", "[.stylescan]") {
  const char *path = std::getenv("STYLE_SCAN_PDF");
  if (!path) path = ".ocr-demo/form.pdf";
  const char *dpi_env = std::getenv("STYLE_SCAN_DPI");
  const double dpi = dpi_env ? std::atof(dpi_env) : 100.0;

  turbo_ocr::pdf::ensure_pdfium_initialized();
  FPDF_DOCUMENT doc = FPDF_LoadDocument(path, nullptr);
  REQUIRE(doc != nullptr);
  FPDF_PAGE page = FPDF_LoadPage(doc, 0);
  REQUIRE(page != nullptr);

  const double scale = dpi / 72.0;
  const int bw = static_cast<int>(FPDF_GetPageWidthF(page) * scale);
  const int bh = static_cast<int>(FPDF_GetPageHeightF(page) * scale);
  FPDF_BITMAP bmp = FPDFBitmap_Create(bw, bh, 0);
  FPDFBitmap_FillRect(bmp, 0, 0, bw, bh, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bmp, page, 0, 0, bw, bh, 0, 0);
  const cv::Mat view(bh, bw, CV_8UC4, FPDFBitmap_GetBuffer(bmp),
                     static_cast<size_t>(FPDFBitmap_GetStride(bmp)));
  cv::Mat raster;
  cv::cvtColor(view, raster, cv::COLOR_BGRA2BGR);
  FPDFBitmap_Destroy(bmp);

  FPDF_TEXTPAGE tp = FPDFText_LoadPage(page);
  const int rects = FPDFText_CountRects(tp, 0, -1);
  printf("\n%s @ %.0f dpi — %d line(s)\n", path, dpi, rects);
  printf("%-26s %7s %7s %7s %6s %6s\n", "text", "serif", "weight", "slant",
         "xh", "flat");
  std::vector<LineStyle> doc_styles;
  const auto page_h = static_cast<double>(FPDF_GetPageHeightF(page));
  for (int i = 0; i < rects; ++i) {
    double l = 0;
    double t = 0;
    double r = 0;
    double b = 0;
    FPDFText_GetRect(tp, i, &l, &t, &r, &b);
    // PDFium's text space is y-up; the raster is y-down.
    const int x0 = static_cast<int>(l * scale);
    const int x1 = static_cast<int>(r * scale);
    const int y0 = static_cast<int>((page_h - t) * scale);
    const int y1 = static_cast<int>((page_h - b) * scale);
    if (x1 - x0 < 4 || y1 - y0 < 4) continue;
    const Box box{{{{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}}}};

    std::vector<unsigned short> buf(64, 0);
    const int n = FPDFText_GetBoundedText(tp, l, t, r, b, buf.data(), 60);
    std::string txt;
    for (int k = 0; k < n && k < 60; ++k)
      if (buf[static_cast<size_t>(k)] > 31 && buf[static_cast<size_t>(k)] < 127)
        txt += static_cast<char>(buf[static_cast<size_t>(k)]);

    const LineStyle st = measure_line_style(raster, box, n > 0 ? n : 0);
    doc_styles.push_back(st);
    if (!st.measured) {
      printf("%-26s   (unmeasured)\n", txt.c_str());
      continue;
    }
    printf("%-26s %7.3f %7.3f %7.1f %6.1f %6d\n", txt.c_str(), st.serif,
           st.weight, st.slant_deg, st.x_height_px, st.flat_paper ? 1 : 0);
  }
  const auto fonts = resolve_document_fonts(doc_styles);
  if (!fonts.empty())
    printf("=> document resolves to %s\n", fonts[0].standard_name());

  FPDFText_ClosePage(tp);  // NOLINT
  FPDF_ClosePage(page);
  FPDF_CloseDocument(doc);
}

// Evaluates the family decision against every text face installed on the
// machine, not against the handful the estimator's own thresholds were fitted
// to. Rendering the standard-14 with PDFium and measuring them back proves
// nothing on its own — it is the same three designs going out and coming in.
//
//   python3 tools/bench/formbench/font_ground_truth.py > /tmp/fonts.tsv
//   python3 tools/bench/formbench/render_font_pages.py /tmp/fonts.tsv /tmp/fontpages
//   FONT_EVAL=/tmp/fontpages/manifest.tsv turbo_ocr_tests "[fonteval]"
// Must stay in step with LINES in tools/bench/formbench/render_font_pages.py.

// The line quadrilaterals the renderer emitted next to the page, when it left
// any. On a skewed page a detector returns rotated quads; recovering lines by
// horizontal row projection would merge them, and the evaluation would then be
// measuring the harness rather than the estimator.
inline std::vector<Box> load_boxes(const std::string &png) {
  std::vector<Box> out;
  std::ifstream in(png + ".boxes");
  if (!in) return out;
  std::string ln;
  while (std::getline(in, ln)) {
    int v[8];
    std::istringstream ss(ln);
    bool ok = true;
    for (int &x : v)
      if (!(ss >> x)) ok = false;
    if (!ok) continue;
    out.push_back(Box{{{{v[0], v[1]}, {v[2], v[3]}, {v[4], v[5]}, {v[6], v[7]}}}});
  }
  return out;
}

static const std::vector<std::string> kEvalLines = {
    "Handgloves quick brown fox",
    "Invoice total due on receipt",
    "PLZ / Ort: Hamburg 20095",
    "The quality of mercy is not strained",
    "Name Vorname Strasse Telefon",
    "Reference 4711-A jumps over it"};

TEST_CASE("family accuracy over installed fonts", "[.fonteval]") {
  const char *manifest = std::getenv("FONT_EVAL");
  if (!manifest) manifest = "/tmp/fontpages/manifest.tsv";
  std::ifstream in(manifest);
  if (!in) {
    WARN("no manifest at " << manifest << " — see the header comment");
    return;
  }

  struct Row {
    std::string png;
    std::string truth;
    std::string face;
  };
  std::vector<Row> rows;
  std::string line;
  while (std::getline(in, line)) {
    const size_t a = line.find('\t');
    if (a == std::string::npos) continue;
    const size_t b = line.find('\t', a + 1);
    if (b == std::string::npos) continue;
    rows.push_back({line.substr(0, a), line.substr(a + 1, b - a - 1),
                    line.substr(b + 1)});
  }
  REQUIRE_FALSE(rows.empty());

  std::map<std::string, std::map<std::string, int>> confusion;
  std::vector<std::string> wrong;
  int unmeasured = 0;
  for (const Row &row : rows) {
    const cv::Mat page = cv::imread(row.png, cv::IMREAD_COLOR);
    if (page.empty()) continue;

    // Find the rendered lines the way a detector would: bands of rows carrying
    // ink, separated by blank ones.
    cv::Mat gray;
    cv::Mat mask;
    cv::cvtColor(page, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 200, 255, cv::THRESH_BINARY_INV);
    cv::Mat rowsum;
    cv::reduce(mask, rowsum, 1, cv::REDUCE_SUM, CV_32S);
    const auto *rs = rowsum.ptr<int>(0);

    std::vector<LineStyle> styles;
    const std::vector<Box> given = load_boxes(row.png);
    if (!given.empty()) {
      for (size_t i = 0; i < given.size(); ++i)
        styles.push_back(measure_line_style(
            page, given[i],
            i < kEvalLines.size() ? static_cast<int>(kEvalLines[i].size()) : 0));
    }
    int y = given.empty() ? 0 : page.rows;
    while (y < page.rows) {
      if (rs[y] <= 0) {
        ++y;
        continue;
      }
      int end = y;
      while (end + 1 < page.rows && rs[end + 1] > 0) ++end;
      const cv::Mat band = mask.rowRange(y, end + 1);
      cv::Mat colsum;
      cv::reduce(band, colsum, 0, cv::REDUCE_SUM, CV_32S);
      const auto *cs = colsum.ptr<int>(0);
      int x0 = -1;
      int x1 = -1;
      for (int x = 0; x < page.cols; ++x) {
        if (cs[x] <= 0) continue;
        if (x0 < 0) x0 = x;
        x1 = x;
      }
      if (x0 >= 0 && x1 - x0 > 40 && end - y >= 8) {
        const Box box{{{{x0, y}, {x1, y}, {x1, end}, {x0, end}}}};
        // The real pipeline knows how many characters the line holds, and the
        // pitch test is worthless without it — passing 0 here silently disabled
        // monospace detection and made the estimator look worse than it is.
        const size_t idx = styles.size();
        const int chars =
            idx < kEvalLines.size() ? static_cast<int>(kEvalLines[idx].size()) : 0;
        styles.push_back(measure_line_style(page, box, chars));
      }
      y = end + 1;
    }
    if (styles.empty()) {
      ++unmeasured;
      continue;
    }

    const auto fonts = resolve_document_fonts(styles);
    const char *got = "sans";
    if (fonts[0].family == FontFamily::Serif) got = "serif";
    else if (fonts[0].family == FontFamily::Mono) got = "mono";
    confusion[row.truth][got]++;
    if (row.truth != got) wrong.push_back(row.face + " (" + row.truth + " read as " + got + ")");
  }

  int total = 0;
  int right = 0;
  printf("\ntruth \\ predicted   sans  serif   mono\n");
  for (const auto &[truth, preds] : confusion) {
    printf("%-16s", truth.c_str());
    for (const char *p : {"sans", "serif", "mono"}) {
      const auto it = preds.find(p);
      const int n = it == preds.end() ? 0 : it->second;
      printf("%6d", n);
      total += n;
      if (truth == p) right += n;
    }
    printf("\n");
  }
  printf("\noverall %d/%d = %.1f%%   (%d pages unmeasurable)\n", right, total,
         total ? 100.0 * right / total : 0.0, unmeasured);
  printf("\nmisread:\n");
  for (const std::string &w : wrong) printf("  %s\n", w.c_str());
}

// The same corpus, answered by SHAPE MATCHING instead of measured features:
// each candidate face sets the very words the page shows and the renders are
// compared with it. Run alongside [fonteval] to see the two side by side.
TEST_CASE("family accuracy by shape matching", "[.fontmatch]") {
  using turbo_ocr::pdf::FontSample;
  using turbo_ocr::pdf::match_font;
  using turbo_ocr::pdf::standard_font_catalogue;

  // Must stay in step with LINES in tools/bench/formbench/render_font_pages.py.
  static const std::vector<std::string> kPageLines = {
      "Handgloves quick brown fox",
      "Invoice total due on receipt",
      "PLZ / Ort: Hamburg 20095",
      "The quality of mercy is not strained",
      "Name Vorname Strasse Telefon",
      "Reference 4711-A jumps over it"};

  const char *manifest = std::getenv("FONT_EVAL");
  if (!manifest) manifest = "/tmp/fontpages/manifest.tsv";
  std::ifstream in(manifest);
  if (!in) {
    WARN("no manifest at " << manifest);
    return;
  }
  struct Row { std::string png, truth, face; };
  std::vector<Row> rows;
  std::string line;
  while (std::getline(in, line)) {
    const size_t a = line.find('\t');
    if (a == std::string::npos) continue;
    const size_t b = line.find('\t', a + 1);
    if (b == std::string::npos) continue;
    rows.push_back({line.substr(0, a), line.substr(a + 1, b - a - 1), line.substr(b + 1)});
  }
  REQUIRE_FALSE(rows.empty());

  const auto &catalogue = standard_font_catalogue();
  std::map<std::string, std::map<std::string, int>> confusion;
  std::vector<std::string> wrong;
  for (const Row &row : rows) {
    const cv::Mat page = cv::imread(row.png, cv::IMREAD_COLOR);
    if (page.empty()) continue;

    cv::Mat gray, mask;
    cv::cvtColor(page, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 200, 255, cv::THRESH_BINARY_INV);
    cv::Mat rowsum;
    cv::reduce(mask, rowsum, 1, cv::REDUCE_SUM, CV_32S);
    const auto *rs = rowsum.ptr<int>(0);

    std::vector<FontSample> samples;
    std::vector<Box> boxes = load_boxes(row.png);
    if (!boxes.empty()) {
      for (size_t i = 0; i < boxes.size() && i < kPageLines.size(); ++i) {
        cv::Mat crop;
        if (turbo_ocr::pdf::rectify_line(page, boxes[i], crop))
          samples.push_back({std::move(crop), kPageLines[i]});
      }
    }
    int y = samples.empty() ? 0 : page.rows;
    while (y < page.rows && samples.size() < kPageLines.size()) {
      if (rs[y] <= 0) { ++y; continue; }
      int end = y;
      while (end + 1 < page.rows && rs[end + 1] > 0) ++end;
      cv::Mat band = mask.rowRange(y, end + 1);
      cv::Mat colsum;
      cv::reduce(band, colsum, 0, cv::REDUCE_SUM, CV_32S);
      const auto *cs = colsum.ptr<int>(0);
      int x0 = -1, x1 = -1;
      for (int x = 0; x < page.cols; ++x) {
        if (cs[x] <= 0) continue;
        if (x0 < 0) x0 = x;
        x1 = x;
      }
      if (x0 >= 0 && x1 - x0 > 40 && end - y >= 8) {
        boxes.push_back(Box{{{{x0, y}, {x1, y}, {x1, end}, {x0, end}}}});
        samples.push_back({page(cv::Rect(x0, y, x1 - x0 + 1, end - y + 1)).clone(),
                           kPageLines[samples.size()]});
      }
      y = end + 1;
    }
    if (samples.empty()) continue;

    const char *dbg = std::getenv("FONT_MATCH_DEBUG");
    if (dbg && row.face.find(dbg) != std::string::npos) {
      printf("\n--- %s (truth %s) ---\n", row.face.c_str(), row.truth.c_str());
      for (size_t c = 0; c < catalogue.size(); ++c) {
        float tot = 0;
        int n = 0;
        for (const auto &smp : samples) {
          const float sc = turbo_ocr::pdf::shape_agreement(smp.crop, smp.text, catalogue[c]);
          if (sc > 0) { tot += sc; ++n; }
        }
        printf("  %-24s %.4f\n", catalogue[c].name.c_str(), n ? tot / n : 0.0F);
      }
    }
    // The feature estimator's family, offered to the matcher as a prior.
    // Weight comes from the environment so the sweep needs no rebuild.
    turbo_ocr::pdf::FamilyPrior prior;
    const char *pw = std::getenv("FONT_MATCH_PRIOR");
    if (pw) {
      std::vector<LineStyle> styles;
      for (size_t i = 0; i < samples.size(); ++i)
        styles.push_back(measure_line_style(
            page, boxes[i], static_cast<int>(samples[i].text.size())));
      const auto ff = resolve_document_fonts(styles);
      if (!ff.empty()) {
        prior.family = ff[0].family;
        prior.strength = static_cast<float>(std::atof(pw));
      }
    }
    const auto m = match_font(samples, catalogue, prior);
    const char *got = "sans";
    if (m.index >= 0) {
      const auto f = catalogue[static_cast<size_t>(m.index)].family;
      if (f == FontFamily::Serif) got = "serif";
      else if (f == FontFamily::Mono) got = "mono";
    }
    confusion[row.truth][got]++;
    if (row.truth != got)
      wrong.push_back(row.face + " (" + row.truth + " read as " + got + ", score " +
                      std::to_string(m.score).substr(0, 4) + ")");
  }

  int total = 0, right = 0;
  printf("\nSHAPE MATCHING\ntruth \\ predicted   sans  serif   mono\n");
  for (const auto &[truth, preds] : confusion) {
    printf("%-16s", truth.c_str());
    for (const char *p : {"sans", "serif", "mono"}) {
      const auto it = preds.find(p);
      const int n = it == preds.end() ? 0 : it->second;
      printf("%6d", n);
      total += n;
      if (truth == p) right += n;
    }
    printf("\n");
  }
  printf("\noverall %d/%d = %.1f%%\n\nmisread:\n", right, total,
         total ? 100.0 * right / total : 0.0);
  for (const std::string &w : wrong) printf("  %s\n", w.c_str());
}

// The question the family tests cannot answer: given a page, can it name the
// FACE? The catalogue here is every text font installed on the machine, so the
// matcher is choosing among ~90 real designs rather than three.
//
//   FONT_EVAL=/tmp/fp2/manifest.tsv FONT_LIST=/tmp/fonts.tsv \
//     turbo_ocr_tests "[fontid]"
TEST_CASE("names the actual face from a large catalogue", "[.fontid]") {
  using turbo_ocr::pdf::FontCandidate;
  using turbo_ocr::pdf::FontSample;
  using turbo_ocr::pdf::match_font;

  const char *list = std::getenv("FONT_LIST");
  if (!list) list = "/tmp/fonts.tsv";
  const char *manifest = std::getenv("FONT_EVAL");
  if (!manifest) manifest = "/tmp/fp2/manifest.tsv";

  std::vector<FontCandidate> catalogue;
  {
    std::ifstream in(list);
    if (!in) { WARN("no font list at " << list); return; }
    std::string ln;
    while (std::getline(in, ln)) {
      if (ln.empty() || ln[0] == '#') continue;
      std::vector<std::string> f;
      size_t start = 0;
      for (size_t i = 0; i <= ln.size(); ++i)
        if (i == ln.size() || ln[i] == '\t') { f.push_back(ln.substr(start, i - start)); start = i + 1; }
      if (f.size() < 4) continue;
      FontCandidate c;
      c.file = f[0];
      c.face = std::atoi(f[1].c_str());
      c.name = f[3];
      c.family = f[2] == "serif" ? FontFamily::Serif
               : f[2] == "mono"  ? FontFamily::Mono : FontFamily::Sans;
      catalogue.push_back(std::move(c));
    }
  }
  REQUIRE(catalogue.size() > 20);

  std::ifstream in(manifest);
  if (!in) { WARN("no manifest"); return; }
  struct Row { std::string png, truth, face; };
  std::vector<Row> rows;
  std::string line;
  while (std::getline(in, line)) {
    const size_t a = line.find('\t');
    if (a == std::string::npos) continue;
    const size_t b = line.find('\t', a + 1);
    if (b == std::string::npos) continue;
    rows.push_back({line.substr(0, a), line.substr(a + 1, b - a - 1), line.substr(b + 1)});
  }

  int exact = 0, family_ok = 0, total = 0;
  std::vector<std::string> misses;
  for (const Row &row : rows) {
    const cv::Mat page = cv::imread(row.png, cv::IMREAD_COLOR);
    if (page.empty()) continue;
    cv::Mat gray, mask;
    cv::cvtColor(page, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 200, 255, cv::THRESH_BINARY_INV);
    cv::Mat rowsum;
    cv::reduce(mask, rowsum, 1, cv::REDUCE_SUM, CV_32S);
    const auto *rs = rowsum.ptr<int>(0);
    std::vector<FontSample> samples;
    int y = 0;
    while (y < page.rows && samples.size() < kEvalLines.size()) {
      if (rs[y] <= 0) { ++y; continue; }
      int end = y;
      while (end + 1 < page.rows && rs[end + 1] > 0) ++end;
      cv::Mat band = mask.rowRange(y, end + 1);
      cv::Mat colsum;
      cv::reduce(band, colsum, 0, cv::REDUCE_SUM, CV_32S);
      const auto *cs = colsum.ptr<int>(0);
      int x0 = -1, x1 = -1;
      for (int x = 0; x < page.cols; ++x) {
        if (cs[x] <= 0) continue;
        if (x0 < 0) x0 = x;
        x1 = x;
      }
      if (x0 >= 0 && x1 - x0 > 40 && end - y >= 8)
        samples.push_back({page(cv::Rect(x0, y, x1 - x0 + 1, end - y + 1)).clone(),
                           kEvalLines[samples.size()]});
      y = end + 1;
    }
    if (samples.empty()) continue;
    // Only three lines, to keep ~90 candidates x N pages tractable.
    if (samples.size() > 3) samples.resize(3);

    const auto m = match_font(samples, catalogue);
    ++total;
    if (m.index < 0) { misses.push_back(row.face + " -> (none)"); continue; }
    const auto &won = catalogue[static_cast<size_t>(m.index)];
    if (won.name == row.face) ++exact;
    else misses.push_back(row.face + " -> " + won.name + " (" +
                          std::to_string(m.score).substr(0, 4) + ")");
    const char *gf = won.family == FontFamily::Serif ? "serif"
                   : won.family == FontFamily::Mono ? "mono" : "sans";
    if (row.truth == gf) ++family_ok;
  }
  printf("\nFONT IDENTIFICATION over %zu candidates\n", catalogue.size());
  printf("  exact face : %d/%d = %.1f%%\n", exact, total, total ? 100.0 * exact / total : 0.0);
  printf("  family     : %d/%d = %.1f%%\n", family_ok, total, total ? 100.0 * family_ok / total : 0.0);
  printf("\nnot exact:\n");
  for (size_t i = 0; i < misses.size() && i < 40; ++i) printf("  %s\n", misses[i].c_str());
}

TEST_CASE("serif score separates Times from Helvetica", "[fontstyle]") {
  const LineStyle sans = measure("Helvetica");
  const LineStyle serif = measure("Times-Roman");

  INFO("Helvetica serif=" << sans.serif << " Times serif=" << serif.serif);
  CHECK(serif.serif > sans.serif);

  // The threshold has to land strictly between the two, or the family vote is
  // decided by which face happens to be commoner rather than by the evidence.
  const std::vector<LineStyle> sans_doc{sans};
  const std::vector<LineStyle> serif_doc{serif};
  CHECK(resolve_document_fonts(sans_doc)[0].family == FontFamily::Sans);
  CHECK(resolve_document_fonts(serif_doc)[0].family == FontFamily::Serif);
}

TEST_CASE("bold measures heavier than regular", "[fontstyle]") {
  const LineStyle regular = measure("Helvetica");
  const LineStyle bold = measure("Helvetica-Bold");
  INFO("regular weight=" << regular.weight << " bold weight=" << bold.weight);
  CHECK(bold.weight > regular.weight * 1.2F);
}

TEST_CASE("stroke weight is scale free", "[fontstyle]") {
  // weight is stroke over x-height, so the same face read at two resolutions
  // must give the same number — that is what lets one absolute bold threshold
  // hold at every dpi.
  const LineStyle small = measure("Helvetica", 13.0F, 1.5F);
  const LineStyle large = measure("Helvetica", 13.0F, 3.0F);
  INFO("weight at 1.5x=" << small.weight << " at 3x=" << large.weight);
  CHECK(std::abs(small.weight - large.weight) < 0.05F);
}

TEST_CASE("oblique measures a right lean, upright does not", "[fontstyle]") {
  const LineStyle upright = measure("Helvetica");
  const LineStyle oblique = measure("Helvetica-Oblique");
  INFO("upright slant=" << upright.slant_deg
                        << " oblique slant=" << oblique.slant_deg);
  CHECK(std::abs(upright.slant_deg) <= 2.0F);
  CHECK(oblique.slant_deg > 6.0F);
}

TEST_CASE("Courier reads as monospaced and Helvetica does not", "[fontstyle]") {
  // The evidence for pitch is how little the advance moves as the CONTENT
  // changes, so this needs a document of differing lines, not one line.
  const auto mono = pitch_document("Courier");
  const auto prop = pitch_document("Helvetica");
  INFO("Courier advances "
       << mono[0].advance_ratio << " " << mono[1].advance_ratio << " "
       << mono[2].advance_ratio << " | Helvetica " << prop[0].advance_ratio
       << " " << prop[1].advance_ratio << " " << prop[2].advance_ratio);

  CHECK(resolve_document_fonts(mono)[0].family == FontFamily::Mono);
  CHECK(resolve_document_fonts(prop)[0].family != FontFamily::Mono);
}

TEST_CASE("pitch is not judged without the recognised text", "[fontstyle]") {
  // char_count 0 means "not known"; it must leave the advance unset rather than
  // inventing a ratio that would then vote in the family decision.
  const Sample s = render_text("Courier", 14.0F, kSpecimen, 2.0F);
  REQUIRE(s.ok);
  const LineStyle st = measure_line_style(s.image, s.box);
  REQUIRE(st.measured);
  CHECK(st.advance_ratio == 0.0F);
}

TEST_CASE("a document in one face resolves to one font throughout", "[fontstyle]") {
  // The user-visible promise: lines that look alike get the SAME font, so a
  // page does not flicker between typefaces as it is read down.
  for (const char *face : {"Helvetica", "Times-Roman"}) {
    const auto doc = document_of(face, 6);
    const auto fonts = resolve_document_fonts(doc);
    REQUIRE(fonts.size() == doc.size());
    INFO("face " << face);
    for (const auto &f : fonts) {
      CHECK(f.family == fonts[0].family);
      CHECK(f.bold == fonts[0].bold);
      CHECK(f.italic == fonts[0].italic);
    }
    CHECK_FALSE(fonts[0].bold);
    CHECK_FALSE(fonts[0].italic);
    CHECK(fonts[0].family ==
          (std::string(face) == "Helvetica" ? FontFamily::Sans : FontFamily::Serif));
  }
}

TEST_CASE("bold headings stand out against a regular body", "[fontstyle]") {
  std::vector<LineStyle> doc = document_of("Helvetica", 5);
  const Sample heading = render_text("Helvetica-Bold", 13.0F, kSpecimen, 2.0F);
  REQUIRE(heading.ok);
  doc.push_back(measure_line_style(heading.image, heading.box));

  const auto fonts = resolve_document_fonts(doc);
  REQUIRE(fonts.size() == doc.size());
  for (size_t i = 0; i + 1 < fonts.size(); ++i) {
    INFO("body line " << i << " weight " << doc[i].weight);
    CHECK_FALSE(fonts[i].bold);
  }
  INFO("heading weight " << doc.back().weight);
  CHECK(fonts.back().bold);
}

TEST_CASE("a document that is bold throughout still reads as bold", "[fontstyle]") {
  // The relative rule cannot see this — the median IS the bold weight — so it
  // is the absolute arm of the test being exercised here.
  const auto doc = document_of("Helvetica-Bold", 5);
  const auto fonts = resolve_document_fonts(doc);
  for (const auto &f : fonts) CHECK(f.bold);
}

TEST_CASE("page skew is not mistaken for italic", "[fontstyle]") {
  // Every line leaning the same way is a crooked scan; one line leaning
  // against its neighbours is italic. Rotating the whole document must not
  // turn it italic.
  std::vector<LineStyle> doc;
  for (int i = 0; i < 5; ++i) {
    Sample s = render_text("Helvetica", 13.0F, kSpecimen, 2.0F);
    REQUIRE(s.ok);
    // Shear the page, then keep the box on the ink as a detector would.
    const cv::Matx23f shear(1.0F, -0.10F, 0.10F * static_cast<float>(s.image.rows),
                            0.0F, 1.0F, 0.0F);
    cv::Mat sheared;
    cv::warpAffine(s.image, sheared, shear, s.image.size(), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::Mat gray;
    cv::Mat mask;
    cv::cvtColor(sheared, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask, 250, 255, cv::THRESH_BINARY_INV);
    const cv::Rect r = cv::boundingRect(mask);
    const Box b{{{{r.x, r.y},
                  {r.x + r.width, r.y},
                  {r.x + r.width, r.y + r.height},
                  {r.x, r.y + r.height}}}};
    const LineStyle st = measure_line_style(sheared, b);
    REQUIRE(st.measured);
    doc.push_back(st);
  }
  const auto fonts = resolve_document_fonts(doc);
  INFO("sheared slant " << doc[0].slant_deg);
  for (const auto &f : fonts) CHECK_FALSE(f.italic);
}

TEST_CASE("ink and paper colours come back off the page", "[fontstyle]") {
  const LineStyle s = measure("Helvetica");
  INFO("ink " << int(s.ink[0]) << "," << int(s.ink[1]) << "," << int(s.ink[2])
              << " paper " << int(s.paper[0]) << "," << int(s.paper[1]) << ","
              << int(s.paper[2]));
  CHECK(s.ink[0] < 120);
  CHECK(s.ink[1] < 120);
  CHECK(s.ink[2] < 120);
  CHECK(s.paper[0] > 200);
  CHECK(s.paper[1] > 200);
  CHECK(s.paper[2] > 200);
  // Plain black on plain white is the clearest case there is; if this is not
  // flat, nothing will be, and the writer will never dare cover anything.
  CHECK(s.flat_paper);
}

TEST_CASE("a busy background is not reported flat", "[fontstyle]") {
  Sample s = render_text("Helvetica", 13.0F, kSpecimen, 2.0F);
  REQUIRE(s.ok);
  // Rule the page like a form: covering this area would erase the rules.
  for (int y = 0; y < s.image.rows; y += 4)
    cv::line(s.image, {0, y}, {s.image.cols, y}, cv::Scalar(40, 40, 40), 1);
  const LineStyle st = measure_line_style(s.image, s.box);
  if (st.measured) CHECK_FALSE(st.flat_paper);
}

TEST_CASE("degenerate input is refused rather than guessed", "[fontstyle]") {
  const Box b{{{{0, 0}, {30, 0}, {30, 20}, {0, 20}}}};
  CHECK_FALSE(measure_line_style(cv::Mat(), b).measured);

  // Blank paper: no ink to measure.
  const cv::Mat blank(60, 200, CV_8UC3, cv::Scalar(255, 255, 255));
  CHECK_FALSE(measure_line_style(blank, b).measured);

  // Too few rows for a projection profile to have a plateau at all.
  const Sample s = render_text("Helvetica", 13.0F, kSpecimen, 2.0F);
  REQUIRE(s.ok);
  const Box tiny{{{{10, 10}, {14, 10}, {14, 13}, {10, 13}}}};
  CHECK_FALSE(measure_line_style(s.image, tiny).measured);
}

TEST_CASE("unmeasured lines still get the document's font", "[fontstyle]") {
  // A line the estimator refused is far better served by the majority answer
  // than by a hardcoded default that ignores the other 400 lines on the page.
  std::vector<LineStyle> doc = document_of("Times-Roman", 4);
  doc.emplace_back();  // measured == false
  const auto fonts = resolve_document_fonts(doc);
  REQUIRE(fonts.size() == doc.size());
  CHECK(fonts.back().family == FontFamily::Serif);
}

TEST_CASE("standard-14 names cover every combination", "[fontstyle]") {
  using turbo_ocr::pdf::FontChoice;
  CHECK(std::string(FontChoice{FontFamily::Sans, false, false}.standard_name()) ==
        "Helvetica");
  CHECK(std::string(FontChoice{FontFamily::Sans, true, true}.standard_name()) ==
        "Helvetica-BoldOblique");
  CHECK(std::string(FontChoice{FontFamily::Serif, false, false}.standard_name()) ==
        "Times-Roman");
  CHECK(std::string(FontChoice{FontFamily::Serif, true, true}.standard_name()) ==
        "Times-BoldItalic");
  CHECK(std::string(FontChoice{FontFamily::Mono, false, false}.standard_name()) ==
        "Courier");
  CHECK(std::string(FontChoice{FontFamily::Mono, true, true}.standard_name()) ==
        "Courier-BoldOblique");
}
