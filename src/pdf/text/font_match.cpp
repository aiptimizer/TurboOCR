#include "turbo_ocr/pdf/text/font_match.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <cmath>
#include <map>
#include <mutex>

#include <fpdf_edit.h>
#include <fpdfview.h>
#include <opencv2/imgproc.hpp>

#include "pdf_text_internal.h"

namespace turbo_ocr::pdf {
namespace {

// Both images are scaled to this ink height before they are compared, so the
// comparison is about shape and proportion rather than size. Tall enough to
// keep a serif from vanishing, small enough that a document's worth of
// comparisons stays cheap.
constexpr int kCanonHeight = 40;
// Guard against a pathological aspect ratio eating memory.
constexpr int kMaxCanonWidth = 4000;

// How much the measured-feature family is allowed to lift a candidate of the
// same family. Swept over 90 installed text faces: 0 gives 92.2% family
// accuracy and 0.20 to 0.50 all give 96.7%, so the middle of that plateau is
// taken. Push it to 0.8 and the prior starts overruling clear shape wins, and
// accuracy falls back to 95.6%.
constexpr float kMeasuredPriorStrength = 0.25F;

// Below this the winner is not a match, it is merely the least bad. Measured:
// a face rendering the words as .notdef boxes scores ~0.20, a real but
// imperfect match 0.49 upward, and setting the same words in the same face
// 0.84. The floor sits above the first and well below the second.
constexpr float kRejectBelow = 0.35F;

// The size the candidate is rendered at before being scaled down. Large enough
// that its own rasterisation is not the thing being measured.
constexpr float kRenderSize = 64.0F;

std::vector<unsigned short> to_utf16(const std::string &s) {
  std::vector<unsigned short> out;
  for (size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    unsigned cp = c;
    int n = 1;
    if ((c & 0xE0) == 0xC0) { cp = c & 0x1FU; n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0FU; n = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07U; n = 4; }
    for (int k = 1; k < n && i + static_cast<size_t>(k) < s.size(); ++k)
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + static_cast<size_t>(k)]) & 0x3FU);
    i += static_cast<size_t>(n);
    if (cp == 0 || cp > 0xFFFF) continue;
    out.push_back(static_cast<unsigned short>(cp));
  }
  out.push_back(0);
  return out;
}

// The standard-14 encodings are Latin-1. Anything else cannot be set in them,
// so there is nothing to compare and the sample is skipped rather than scored.
bool spellable(const std::string &s) {
  for (size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      if (c < 0x20) return false;
      ++i;
      continue;
    }
    if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
      const unsigned cp = ((c & 0x1FU) << 6) |
                          (static_cast<unsigned char>(s[i + 1]) & 0x3FU);
      if (cp < 0xA0 || cp > 0xFF) return false;
      i += 2;
      continue;
    }
    return false;
  }
  return true;
}

// Ink mask of an image, scaled so the ink is kCanonHeight tall and the aspect
// ratio is kept. Keeping the aspect is what lets a condensed face lose to a
// normal one: the widths then genuinely disagree, and the overlap falls.
bool canonical_ink(const cv::Mat &bgr, cv::Mat &out) {
  if (bgr.empty()) return false;
  cv::Mat gray;
  if (bgr.channels() == 1) gray = bgr;
  else if (bgr.channels() == 4) cv::cvtColor(bgr, gray, cv::COLOR_BGRA2GRAY);
  else cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

  cv::Mat mask;
  cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  if (cv::countNonZero(mask) * 2 > static_cast<int>(mask.total()))
    cv::bitwise_not(mask, mask);

  const cv::Rect ink = cv::boundingRect(mask);
  if (ink.width < 4 || ink.height < 4) return false;

  const double scale = static_cast<double>(kCanonHeight) / ink.height;
  const int w = static_cast<int>(std::lround(ink.width * scale));
  if (w < 4 || w > kMaxCanonWidth) return false;
  // AREA when shrinking, LINEAR when growing. INTER_AREA degenerates to
  // nearest-neighbour on upscale, and the scan is nearly always upscaled here
  // (12-25 px of ink to 40) while the render is downscaled — so one side of the
  // comparison was getting blocky nearest-neighbour and the other proper area
  // averaging. That is the exact asymmetry this method exists to avoid.
  cv::resize(mask(ink), out, cv::Size(w, kCanonHeight), 0, 0,
             ink.height >= kCanonHeight ? cv::INTER_AREA : cv::INTER_LINEAR);
  // INTER_AREA leaves greys at the edges; the comparison wants ink or not.
  cv::threshold(out, out, 127, 255, cv::THRESH_BINARY);
  return true;
}

// Agreement between two ink masks of the same words.
//
// The candidate is stretched onto the scan's exact box before comparison, and
// that is the load-bearing step. Comparing at natural width instead — letting a
// wider face simply overhang — sounds more informative and is much worse in
// practice: two faces put their glyphs on different advances, so by the middle
// of a line the two images are a whole letter out of step and the overlap
// collapses into noise. It measured 80% that way against 91% for the crude
// feature test it was meant to replace. Normalising the width cancels the
// accumulated drift and leaves the letterforms themselves to decide.
//
// Both are then softened before overlap is taken. A scan's strokes are fatter
// than a clean render's, and its glyphs never land exactly where the render
// puts them; without some tolerance the metric rewards weight over shape.
float mask_iou(const cv::Mat &a, const cv::Mat &b) {
  cv::Mat fitted;
  if (b.cols == a.cols) fitted = b;
  else
    cv::resize(b, fitted, cv::Size(a.cols, kCanonHeight), 0, 0,
               b.cols >= a.cols ? cv::INTER_AREA : cv::INTER_LINEAR);
  cv::threshold(fitted, fitted, 127, 255, cv::THRESH_BINARY);

  const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
  cv::Mat sa;
  cv::Mat sb;
  cv::dilate(a, sa, k);
  cv::dilate(fitted, sb, k);

  cv::Mat inter;
  cv::Mat uni;
  cv::bitwise_and(sa, sb, inter);
  cv::bitwise_or(sa, sb, uni);
  const double u = cv::countNonZero(uni);
  if (u <= 0.0) return 0.0F;
  const auto overlap = static_cast<float>(cv::countNonZero(inter) / u);

  // Width still carries information — a condensed face really is narrower — so
  // it is kept, as a gentle penalty rather than as the whole signal.
  const float wa = static_cast<float>(a.cols);
  const float wb = static_cast<float>(b.cols);
  const float ratio = std::min(wa, wb) / std::max(wa, wb);
  return overlap * (0.75F + 0.25F * ratio);
}

uint32_t be32(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

uint16_t be16(const uint8_t *p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

void put32(std::vector<uint8_t> &v, size_t at, uint32_t x) {
  v[at] = static_cast<uint8_t>(x >> 24);
  v[at + 1] = static_cast<uint8_t>(x >> 16);
  v[at + 2] = static_cast<uint8_t>(x >> 8);
  v[at + 3] = static_cast<uint8_t>(x);
}

void put16(std::vector<uint8_t> &v, size_t at, uint16_t x) {
  v[at] = static_cast<uint8_t>(x >> 8);
  v[at + 1] = static_cast<uint8_t>(x);
}

// A PDFium document kept open for the life of a matching session, so a hundred
// renders do not open a hundred documents.
class Renderer {
public:
  Renderer() {
    ensure_pdfium_initialized();
    doc_ = FPDF_CreateNewDocument();
  }
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;
  Renderer(Renderer &&) = delete;
  Renderer &operator=(Renderer &&) = delete;
  ~Renderer() {
    for (auto &[name, font] : fonts_)
      if (font) FPDFFont_Close(font);
    if (doc_) FPDF_CloseDocument(doc_);
  }

  [[nodiscard]] bool ok() const { return doc_ != nullptr; }

  // Sets `text` in `candidate` and returns its ink, canonicalised.
  bool render(const FontCandidate &candidate, const std::string &text,
              cv::Mat &out) {
    if (!doc_) return false;
    FPDF_FONT font = font_for(candidate);
    if (!font) return false;

    // Room for the longest plausible line at kRenderSize, with margin so a
    // descender or an overhanging italic cannot be clipped.
    const double pw = 40.0 + kRenderSize * 0.75 * static_cast<double>(text.size());
    const double ph = kRenderSize * 2.5;
    FPDF_PAGE page = FPDFPage_New(doc_, 0, pw, ph);
    if (!page) return false;

    FPDF_PAGEOBJECT obj = FPDFPageObj_CreateTextObj(doc_, font, kRenderSize);
    bool ok = false;
    if (obj) {
      const auto wide = to_utf16(text);
      if (FPDFText_SetText(obj, reinterpret_cast<FPDF_WIDESTRING>(wide.data()))) {
        FPDFPageObj_Transform(obj, 1, 0, 0, 1, 20, kRenderSize);
        FPDFPage_InsertObject(page, obj);
        FPDFPage_GenerateContent(page);
        ok = true;
      } else {
        FPDFPageObj_Destroy(obj);
      }
    }

    if (ok) {
      const int bw = static_cast<int>(pw);
      const int bh = static_cast<int>(ph);
      FPDF_BITMAP bmp = FPDFBitmap_Create(bw, bh, 0);
      if (bmp) {
        FPDFBitmap_FillRect(bmp, 0, 0, bw, bh, 0xFFFFFFFF);
        FPDF_RenderPageBitmap(bmp, page, 0, 0, bw, bh, 0, 0);
        const cv::Mat view(bh, bw, CV_8UC4, FPDFBitmap_GetBuffer(bmp),
                           static_cast<size_t>(FPDFBitmap_GetStride(bmp)));
        ok = canonical_ink(view, out);
        FPDFBitmap_Destroy(bmp);
      } else {
        ok = false;
      }
    }
    // Close the handle BEFORE dropping the page from the document. Deleting a
    // page that is still loaded leaves PDFium holding a page that no longer
    // exists, and the renders that follow on the same document come back
    // wrong — which showed up as the matcher scoring correctly when each call
    // built its own document and incorrectly when a document was shared across
    // a page's worth of comparisons, the exact case that matters.
    FPDF_ClosePage(page);
    FPDFPage_Delete(doc_, 0);
    return ok;
  }

private:
  FPDF_FONT font_for(const FontCandidate &c) {
    const std::string key =
        c.standard14.empty() ? c.file + "#" + std::to_string(c.face) : c.standard14;
    auto [it, inserted] = fonts_.try_emplace(key, nullptr);
    if (!inserted) return it->second;
    if (!c.standard14.empty()) {
      it->second = FPDFText_LoadStandardFont(doc_, c.standard14.c_str());
    } else {
      const std::vector<uint8_t> bytes = load_font_bytes(c.file, c.face);
      if (!bytes.empty())
        it->second = FPDFText_LoadFont(doc_, bytes.data(),
                                       static_cast<uint32_t>(bytes.size()),
                                       FPDF_FONT_TRUETYPE, /*cid=*/1);
    }
    return it->second;
  }

  FPDF_DOCUMENT doc_ = nullptr;
  std::map<std::string, FPDF_FONT> fonts_;
};

} // namespace

std::vector<uint8_t> load_font_bytes(const std::string &path, int face) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  const std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  if (raw.size() < 12) return {};

  // A plain .ttf or .otf is already what PDFium wants.
  if (std::memcmp(raw.data(), "ttcf", 4) != 0) return raw;

  // A collection shares its table DATA between faces and gives each its own
  // table directory. PDFium takes one font, not a collection, so the requested
  // face is rebuilt as a standalone file: its directory, then copies of just
  // the tables it names. Without this, most of the type actually installed on
  // a Mac — Baskerville, Palatino, Georgia's siblings, the whole Hiragino
  // family — is invisible to the matcher, because Apple ships it in .ttc.
  const uint32_t count = be32(raw.data() + 8);
  if (face < 0 || static_cast<uint32_t>(face) >= count) return {};
  const size_t off_at = 12 + static_cast<size_t>(face) * 4;
  if (off_at + 4 > raw.size()) return {};
  // WIDEN BEFORE ADDING. `dir` is a 32-bit offset read straight out of the
  // file, so `dir + 12` is computed in 32 bits and wraps: a directory offset of
  // 0xFFFFFFFF gives 11, sails past a bounds check against a 64-bit size, and
  // the read below lands about four gigabytes past the buffer.
  const auto dir = static_cast<size_t>(be32(raw.data() + off_at));
  if (dir + 12 > raw.size()) return {};

  const uint16_t num = be16(raw.data() + dir + 4);
  if (num == 0 || num > 512) return {};
  if (dir + 12 + static_cast<size_t>(num) * 16 > raw.size()) return {};

  struct Entry {
    uint8_t tag[4];
    uint32_t checksum;
    uint32_t offset;
    uint32_t length;
  };
  std::vector<Entry> entries;
  entries.reserve(num);
  size_t body = 0;
  for (uint16_t i = 0; i < num; ++i) {
    const uint8_t *rec = raw.data() + dir + 12 + static_cast<size_t>(i) * 16;
    Entry e{};
    std::memcpy(e.tag, rec, 4);
    e.checksum = be32(rec + 4);
    e.offset = be32(rec + 8);
    e.length = be32(rec + 12);
    if (static_cast<size_t>(e.offset) + e.length > raw.size()) return {};
    entries.push_back(e);
    body += (static_cast<size_t>(e.length) + 3U) & ~static_cast<size_t>(3U);
    // Tables in a collection are SHARED between faces, so a damaged or hostile
    // directory can legitimately-looking name one 20 MB table five hundred
    // times. One face cannot honestly exceed the file it came from.
    if (body > raw.size()) return {};
  }

  const size_t header = 12 + static_cast<size_t>(num) * 16;
  std::vector<uint8_t> out(header + body, 0);
  // sfnt version, then the binary-search hints the format asks for. Readers
  // vary in how much they trust these, so they are written correctly rather
  // than zeroed.
  put32(out, 0, 0x00010000U);
  put16(out, 4, num);
  uint16_t pow2 = 1;
  uint16_t sel = 0;
  while (static_cast<uint16_t>(pow2 * 2) <= num) {
    pow2 = static_cast<uint16_t>(pow2 * 2);
    ++sel;
  }
  put16(out, 6, static_cast<uint16_t>(pow2 * 16));
  put16(out, 8, sel);
  put16(out, 10, static_cast<uint16_t>((num - pow2) * 16));

  size_t cursor = header;
  for (size_t i = 0; i < entries.size(); ++i) {
    const Entry &e = entries[i];
    const size_t rec = 12 + i * 16;
    std::memcpy(out.data() + rec, e.tag, 4);
    put32(out, rec + 4, e.checksum);
    put32(out, rec + 8, static_cast<uint32_t>(cursor));
    put32(out, rec + 12, e.length);
    std::memcpy(out.data() + cursor, raw.data() + e.offset, e.length);
    cursor += (e.length + 3U) & ~3U;
  }
  return out;
}

const std::vector<FontCandidate> &standard_font_catalogue() {
  static const std::vector<FontCandidate> kCatalogue = [] {
    std::vector<FontCandidate> out;
    struct Entry {
      const char *name;
      const char *std14;
      FontFamily family;
      bool bold;
      bool italic;
    };
    static constexpr Entry kEntries[] = {
        {"Helvetica", "Helvetica", FontFamily::Sans, false, false},
        {"Helvetica Bold", "Helvetica-Bold", FontFamily::Sans, true, false},
        {"Helvetica Oblique", "Helvetica-Oblique", FontFamily::Sans, false, true},
        {"Helvetica Bold Oblique", "Helvetica-BoldOblique", FontFamily::Sans, true, true},
        {"Times", "Times-Roman", FontFamily::Serif, false, false},
        {"Times Bold", "Times-Bold", FontFamily::Serif, true, false},
        {"Times Italic", "Times-Italic", FontFamily::Serif, false, true},
        {"Times Bold Italic", "Times-BoldItalic", FontFamily::Serif, true, true},
        {"Courier", "Courier", FontFamily::Mono, false, false},
        {"Courier Bold", "Courier-Bold", FontFamily::Mono, true, false},
        {"Courier Oblique", "Courier-Oblique", FontFamily::Mono, false, true},
        {"Courier Bold Oblique", "Courier-BoldOblique", FontFamily::Mono, true, true},
    };
    for (const Entry &e : kEntries) {
      FontCandidate c;
      c.name = e.name;
      c.standard14 = e.std14;
      c.family = e.family;
      c.bold = e.bold;
      c.italic = e.italic;
      // Standard-14 is embeddable in the sense that matters: it needs no
      // embedding at all, because every reader already has it.
      c.embeddable = true;
      out.push_back(std::move(c));
    }
    return out;
  }();
  return kCatalogue;
}

PageFontMatch match_page_family(const std::vector<OCRResultItem> &results,
                                const cv::Mat &page,
                                const std::vector<LineStyle> &styles) {
  PageFontMatch out;
  if (page.empty() || results.empty() || styles.size() != results.size())
    return out;

  // At most this many lines per page. Beyond it the answer stops moving and
  // the renders stop being free.
  constexpr size_t kMaxSamples = 4;
  // Shorter than this and the line is a label, not evidence.
  constexpr size_t kMinChars = 10;

  std::vector<size_t> order;
  for (size_t i = 0; i < results.size(); ++i) {
    if (!styles[i].measured) continue;
    if (results[i].text.size() < kMinChars) continue;
    order.push_back(i);
  }
  // Longest first: more glyphs, more evidence.
  std::ranges::sort(order, [&](size_t a, size_t b) {
    return results[a].text.size() > results[b].text.size();
  });
  if (order.size() > kMaxSamples) order.resize(kMaxSamples);
  if (order.empty()) return out;

  std::vector<FontSample> samples;
  for (size_t i : order) {
    // Rectified, not merely cropped. The crop is normalised by its ink height
    // before comparison, and on a page that went through the feeder crooked an
    // axis-aligned box is materially taller than the line inside it.
    cv::Mat crop;
    if (!rectify_line(page, results[i].box, crop)) continue;
    if (crop.cols < 20 || crop.rows < 8) continue;
    samples.push_back({std::move(crop), results[i].text});
  }
  if (samples.empty()) return out;

  // The measured family, offered as a prior. Alone the two methods fail in
  // almost disjoint places; together they beat either.
  FamilyPrior prior;
  const auto measured = resolve_document_fonts(styles);
  if (!measured.empty()) {
    prior.family = measured[0].family;
    prior.strength = kMeasuredPriorStrength;
  }

  const auto m = match_font(samples, standard_font_catalogue(), prior);
  if (m.index < 0) return out;
  out.family = standard_font_catalogue()[static_cast<size_t>(m.index)].family;
  out.score = m.score;
  return out;
}

float shape_agreement(const cv::Mat &crop, const std::string &text,
                      const FontCandidate &candidate) {
  if (crop.empty() || text.empty() || !spellable(text)) return 0.0F;
  cv::Mat scanned;
  if (!canonical_ink(crop, scanned)) return 0.0F;

  std::lock_guard<std::mutex> guard(detail::pdfium_lock());
  Renderer renderer;
  if (!renderer.ok()) return 0.0F;
  cv::Mat drawn;
  if (!renderer.render(candidate, text, drawn)) return 0.0F;
  return mask_iou(scanned, drawn);
}

FontMatch match_font(const std::vector<FontSample> &samples,
                     const std::vector<FontCandidate> &catalogue,
                     FamilyPrior prior) {
  FontMatch result;
  if (samples.empty() || catalogue.empty()) return result;

  // Canonicalise the scanned lines once; every candidate is compared against
  // the same masks.
  std::vector<cv::Mat> scanned;
  std::vector<const FontSample *> usable;
  for (const FontSample &s : samples) {
    if (s.text.empty() || !spellable(s.text)) continue;
    cv::Mat mask;
    if (!canonical_ink(s.crop, mask)) continue;
    scanned.push_back(mask);
    usable.push_back(&s);
  }
  if (scanned.empty()) return result;

  std::lock_guard<std::mutex> guard(detail::pdfium_lock());
  Renderer renderer;
  if (!renderer.ok()) return result;

  std::vector<float> totals(catalogue.size(), 0.0F);
  std::vector<int> counts(catalogue.size(), 0);
  cv::Mat drawn;
  for (size_t c = 0; c < catalogue.size(); ++c) {
    for (size_t i = 0; i < scanned.size(); ++i) {
      if (!renderer.render(catalogue[c], usable[i]->text, drawn)) continue;
      totals[c] += mask_iou(scanned[i], drawn);
      ++counts[c];
    }
  }

  float best = -1.0F;
  float second = -1.0F;
  float best_raw = 0.0F;
  for (size_t c = 0; c < catalogue.size(); ++c) {
    // A candidate that managed only one of four samples is not comparable with
    // one that managed all four; averaging over different denominators lets a
    // single lucky line beat four consistent ones.
    if (counts[c] != static_cast<int>(scanned.size())) continue;
    const float raw = totals[c] / static_cast<float>(counts[c]);
    // Multiplicative, so the prior scales what the shapes already said instead
    // of adding a constant that would matter most where the evidence is
    // weakest.
    float ranked = raw;
    if (prior.strength > 0.0F && catalogue[c].family == prior.family)
      ranked *= 1.0F + prior.strength;
    if (ranked > best) {
      second = best;
      best = ranked;
      best_raw = raw;
      result.index = static_cast<int>(c);
    } else if (ranked > second) {
      second = ranked;
    }
  }
  if (result.index < 0) return result;

  // Nothing resembled the page. Say so instead of returning the least bad of a
  // dozen wrong answers — that ability is the one real advantage this has over
  // measuring features, which can only ever fall back to a default. A font
  // rendering the specimen as .notdef boxes scores about 0.20 here, a genuine
  // but imperfect match 0.49 and up, an exact one 0.84.
  if (best_raw < kRejectBelow) {
    result.index = -1;
    return result;
  }

  // The UN-primed agreement is reported, because the prior already moved the
  // ranking and letting it move the reported confidence too would count it
  // twice — most visibly in the document-wide vote, which weights each page by
  // this number.
  result.score = best_raw;
  result.margin = second >= 0.0F ? best - second : best;
  return result;
}

} // namespace turbo_ocr::pdf
