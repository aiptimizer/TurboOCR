#include "turbo_ocr/pdf/text/font_style.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/base/statistics.h"

namespace turbo_ocr::pdf {

using turbo_ocr::stats::median_of;
namespace {

// Below this a crop has too few glyph rows for the projection profile to show
// the step between the x-height band and the ascenders, which every other
// measurement is taken relative to.
constexpr int kMinCropHeight = 8;
constexpr int kMinCropWidth = 6;
constexpr int kMinInkPixels = 24;
constexpr int kMinXHeight = 3;

// Rows carrying at least this share of the busiest row form the x-height band.
// Between the baseline and the x-height nearly every glyph contributes ink;
// above and below only ascenders and descenders do, so the profile steps there.
constexpr float kXHeightBandShare = 0.5f;
// Ascenders are sparse, so finding the top of the tallest glyph needs a much
// lower bar than the band itself.
constexpr float kAscentShare = 0.12f;

// A horizontal ink run wider than this share of the line is a rule, an
// underline or a filled cell, not a stroke of a letter.
constexpr float kMaxRunShare = 0.4f;
// The same idea used as a safety test rather than a measurement filter: a run
// this long means a printed rule crosses the line, and covering it would erase
// the rule. Lower than kMaxRunShare because here a false alarm costs only a
// line left as scan, while a miss destroys page content.
constexpr float kRuleRunShare = 0.25f;

// Slant search. Past 20 degrees a shear stops being italic and starts being a
// skewed scan we should not be silently correcting for.
constexpr int kMaxSlantDeg = 20;

// Background is "flat" when its spread is under this many grey levels. Chosen
// to accept the mottling of a real scan and reject a rule, a shaded cell or a
// photograph — anything where covering the area would destroy page content.
constexpr double kFlatPaperStdDev = 22.0;

// ── decision thresholds ───────────────────────────────────────────────────
//
// Calibrated against text rendered by PDFium itself in the standard-14 faces
// and measured back, over three point sizes and three resolutions each — see
// the "calibration sweep" case in tests/cpp/pdf/test_font_style.cpp, which
// regenerates every sample on the spot, so these numbers cannot quietly go
// stale if the estimator changes. The measured values behind each:
//
//   foot spread   Helvetica 1.00 flat across every size | Times 1.33-2.50
//                 | Courier 1.43-3.50, correctly, because Courier is a slab
//                 serif. Italic faces scatter, since a leaning stem is not a
//                 full-height column — which the document vote absorbs.
//   weight        Helvetica 0.19-0.21 | Helvetica-Bold 0.28-0.31
//                 | Times 0.21-0.25 | Times-Bold 0.31-0.36
//   slant         upright 0 | oblique and italic 5-16
constexpr float kSerifThreshold = 1.25F;
// Bold, as a multiple of the document's own median stroke weight...
constexpr float kBoldRelative = 1.28F;
// ...or outright, as a fraction of x-height. Scale-free, because the weight is
// already divided by x-height, so this one constant holds at every dpi.
//
// 0.258 is the MIDPOINT of the measured gap, not a round number: across the
// [.fontcal] sweep (trustworthy lines only, x-height >= kMinTrustworthyXHeight)
// regular faces top out at 0.250 (Times-Roman, Times-Italic, Helvetica-Oblique)
// and bold faces bottom out at 0.267 (Helvetica-Bold at 14pt). It was 0.27,
// which sat ABOVE that floor, so genuinely bold 14pt text read as regular and
// "a document that is bold throughout" failed on two of its five lines. The
// sweep did not show it because it stepped 9/13/20pt and skipped 14.
//
// The arm is a floor, not a separator, and cannot be made into one: `weight` is
// stem pixels over x-height pixels, so it lands on coarse steps (4/15 = 0.267,
// 3/12 = 0.250) and the classes genuinely OVERLAP — Courier-Bold at 15pt/3.0
// measures 0.250, exactly a regular Times-Roman. No threshold separates those
// two. Ordinary documents are decided by the relative arm; this one only has to
// catch the case where the median is itself bold.
constexpr float kBoldAbsolute = 0.258F;
constexpr float kItalicRelativeDeg = 5.0F;
constexpr float kItalicAbsoluteDeg = 8.0F;
// Monospace: how little the per-character advance may move across a document
// before the face is one. Measured over five probe lines of deliberately
// different width per character: Courier came back at 0.004, Times at 0.031 and
// Helvetica at 0.11. The threshold sits far under Times rather than midway,
// because the two errors are not equal — a Courier document set in Times only
// looks wrong, while a Times document set in Courier is unreadable — so this
// asks for near-perfect agreement before it will believe a document is
// monospaced, and otherwise leaves it to the shape test.
//
// The one thing that moves the advance in a genuinely monospaced document is a
// line of ALL CAPS: the x-height it is divided by is really the cap height, so
// that line reads narrow. Taking a median absolute deviation rather than a mean
// is what keeps such a line from deciding the answer.
constexpr float kMonoSpread = 0.012F;
constexpr size_t kMinMonoLines = 5;
// Below this x-height in pixels a line is too coarsely sampled to judge on its
// own: one pixel of stroke is a tenth of the measurement. Such lines take the
// document's answer instead of voting a noisy one of their own.
constexpr float kMinTrustworthyXHeight = 8.0F;

// Straightens the line into an axis-aligned crop, with a margin of surrounding
// page kept around it.
//
// The corners arrive clockwise from top-left, so bl->br is the baseline:
// mapping them onto a rectangle removes the page skew and the detector's own
// rotation in one warp, which every measurement below assumes has happened.
// The margin is what the paper colour and flatness are read from — the box
// hugs the ink, so without it there is no unprinted page to sample.
bool upright_crop(const cv::Mat &page, const Box &box, cv::Mat &out,
                  cv::Rect &inner) {
  const cv::Point2f tl(box[0][0], box[0][1]);
  const cv::Point2f tr(box[1][0], box[1][1]);
  const cv::Point2f br(box[2][0], box[2][1]);
  const cv::Point2f bl(box[3][0], box[3][1]);

  const auto w = static_cast<float>((cv::norm(br - bl) + cv::norm(tr - tl)) * 0.5);
  const auto h = static_cast<float>((cv::norm(bl - tl) + cv::norm(br - tr)) * 0.5);
  const int cw = static_cast<int>(std::lround(w));
  const int ch = static_cast<int>(std::lround(h));
  if (cw < kMinCropWidth || ch < kMinCropHeight) return false;
  if (cw > 20000 || ch > 20000) return false;

  const int pad = std::max(2, static_cast<int>(std::lround(h * 0.35F)));
  inner = cv::Rect(pad, pad, cw, ch);

  const cv::Point2f src[3] = {tl, tr, bl};
  const cv::Point2f dst[3] = {
      {static_cast<float>(pad), static_cast<float>(pad)},
      {static_cast<float>(pad + cw), static_cast<float>(pad)},
      {static_cast<float>(pad), static_cast<float>(pad + ch)}};
  const cv::Mat m = cv::getAffineTransform(src, dst);
  // REPLICATE, not a constant: a black border would be read as ink and would
  // poison both the paper colour and the flatness test.
  cv::warpAffine(page, out, m, cv::Size(cw + 2 * pad, ch + 2 * pad),
                 cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  return true;
}

// Per-channel median over the pixels `mask` selects. Median rather than mean so
// one dark speck or one bright blowout cannot move the answer.
cv::Vec3b masked_median_colour(const cv::Mat &bgr, const cv::Mat &mask,
                               cv::Vec3b fallback) {
  std::vector<float> ch[3];
  const int n = cv::countNonZero(mask);
  if (n < 4) return fallback;
  for (auto &c : ch) c.reserve(static_cast<size_t>(n));
  for (int y = 0; y < bgr.rows; ++y) {
    const auto *mrow = mask.ptr<uint8_t>(y);
    const auto *prow = bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < bgr.cols; ++x) {
      if (!mrow[x]) continue;
      for (int c = 0; c < 3; ++c) ch[c].push_back(prow[x][c]);
    }
  }
  cv::Vec3b out;
  for (int c = 0; c < 3; ++c)
    out[c] = static_cast<uint8_t>(std::lround(median_of(ch[c])));
  return out;
}

// Degrees the stems lean from vertical, by shear search.
//
// Sliding each row sideways in proportion to its height above the baseline and
// asking which shift packs the ink into the narrowest columns: upright stems
// only stack into tall narrow peaks when the shear that produced them has been
// undone, so the angle with the most concentrated column profile IS the slant.
float measure_slant(const cv::Mat &band) {
  if (band.rows < 3 || band.cols < 4) return 0.0F;

  std::vector<cv::Point> ink;
  cv::findNonZero(band, ink);
  if (ink.size() < kMinInkPixels) return 0.0F;

  const auto ref_y = static_cast<float>(band.rows - 1);
  const float span = std::tan(static_cast<float>(kMaxSlantDeg) * CV_PI / 180.0F) *
                     static_cast<float>(band.rows);
  const int lo = -static_cast<int>(std::ceil(span)) - 1;
  const int width = band.cols + 2 * (static_cast<int>(std::ceil(span)) + 1);

  std::vector<float> energy(static_cast<size_t>(2 * kMaxSlantDeg + 1), 0.0F);
  std::vector<int> hist;
  for (int deg = -kMaxSlantDeg; deg <= kMaxSlantDeg; ++deg) {
    const float t = std::tan(static_cast<float>(deg) * CV_PI / 180.0F);
    hist.assign(static_cast<size_t>(width), 0);
    for (const cv::Point &p : ink) {
      const float shifted =
          static_cast<float>(p.x) - t * (ref_y - static_cast<float>(p.y));
      const int b = static_cast<int>(std::lround(shifted)) - lo;
      if (b >= 0 && b < width) ++hist[static_cast<size_t>(b)];
    }
    // Sum of squares: maximal when the same ink is packed into fewer columns.
    float e = 0.0F;
    for (int c : hist) e += static_cast<float>(c) * static_cast<float>(c);
    energy[static_cast<size_t>(deg + kMaxSlantDeg)] = e;
  }

  const float best = *std::max_element(energy.begin(), energy.end());
  if (best <= 0.0F) return 0.0F;
  // Upright text at a small x-height ties across several angles, because one
  // degree of shear moves a 10-pixel stem by less than a pixel. Taking the
  // first maximum then reports whichever end of the sweep was scanned first,
  // which reads as a consistent lean on text that has none — so among the
  // angles that are effectively tied, the most upright one wins.
  float answer = 0.0F;
  float best_abs = 1e9F;
  for (int deg = -kMaxSlantDeg; deg <= kMaxSlantDeg; ++deg) {
    if (energy[static_cast<size_t>(deg + kMaxSlantDeg)] < best * 0.995F) continue;
    const auto mag = std::abs(static_cast<float>(deg));
    if (mag < best_abs) {
      best_abs = mag;
      answer = static_cast<float>(deg);
    }
  }
  return answer;
}

// How much wider a vertical stem is at its foot than at its waist, or 0 when
// the line offers no evidence either way.
//
// This is the definition of a serif, measured directly. Isolate the columns
// that carry ink down the full height of the band — those are the stems of
// h, n, l, d, i, u — then compare the width of each stem where it meets the
// baseline against its width halfway up. A sans stem is the same width all the
// way down, so the ratio sits at 1; a serif stem lands on a foot two to three
// times its own width, so the ratio climbs well past it.
//
// Measuring stems rather than the whole line is what makes this survive scan
// resolution. Total ink near the baseline is no use — every face piles ink
// there, because round letters flatten as they meet the line — and at the 10-20
// pixel x-heights a real scan gives, that swamps the serifs entirely.
//
// The trap, and the reason for the symmetry test below: L, E and Z end in a
// horizontal ARM at the baseline, in every typeface there is. To a naive width
// comparison an arm is indistinguishable from a serif foot, so a sans line
// reading "PLZ / Ort:" measured as strongly serifed — 3.0 against Helvetica's
// usual 1.0 — while a long pangram did not, because there the plain stems of
// h, n and m outvoted the arms. Short form labels are exactly the case that has
// no such majority to fall back on. A serif foot brackets its stem on BOTH
// sides; an arm goes one way only, and that is what separates them.
float measure_foot_spread(const cv::Mat &band) {
  const int h = band.rows;
  if (h < 5 || band.cols < 8) return 0.0F;

  cv::Mat colsum;
  cv::reduce(band, colsum, 0, cv::REDUCE_SUM, CV_32S);
  const auto *col = colsum.ptr<int>(0);
  const int full = 255 * h;

  // The ink run through (y, x), as [lo, hi].
  const auto run_at = [&](int y, int x, int &lo, int &hi) {
    lo = hi = -1;
    if (y < 0 || y >= h) return false;
    const auto *row = band.ptr<uint8_t>(y);
    if (!row[x]) return false;
    lo = hi = x;
    while (lo > 0 && row[lo - 1]) --lo;
    while (hi + 1 < band.cols && row[hi + 1]) ++hi;
    return true;
  };

  const int mid_y = h / 2;
  const int foot_y = h - 1;
  std::vector<float> ratios;
  int x = 0;
  while (x < band.cols) {
    if (col[x] < static_cast<int>(0.85 * full)) {
      ++x;
      continue;
    }
    int end = x;
    while (end + 1 < band.cols && col[end + 1] >= static_cast<int>(0.85 * full)) ++end;
    const int centre = (x + end) / 2;
    x = end + 1;

    int wlo = 0;
    int whi = 0;
    int flo = 0;
    int fhi = 0;
    if (!run_at(mid_y, centre, wlo, whi)) continue;
    if (!run_at(foot_y, centre, flo, fhi)) continue;
    const int waist = whi - wlo + 1;
    const int foot = fhi - flo + 1;
    // A stem already joined to something at its waist — the crossbar of an 'e',
    // the shoulder of an 'n' — is not measuring a stem width at all.
    if (waist <= 0 || waist > 4 * (end - centre + 2)) continue;
    if (foot <= waist) {
      ratios.push_back(1.0F);
      continue;
    }
    // Symmetry: how far the foot reaches either side of the stem it stands on.
    const int left = wlo - flo;
    const int right = fhi - whi;
    // Float division, deliberately. Written as `max / 4` in integers this is
    // dead code at the resolutions that matter: a foot extends one to three
    // pixels at a 10-16 px x-height, `max / 4` truncates to zero, and nothing
    // is ever rejected — so the L/E/Z arm it exists to catch was only caught
    // above about 300 dpi.
    const auto lo = static_cast<float>(std::min(left, right));
    const auto hi = static_cast<float>(std::max(left, right));
    if (hi > 0.0F && lo < 0.25F * hi) continue;  // an arm, not a foot
    ratios.push_back(static_cast<float>(foot) / static_cast<float>(waist));
  }
  // Fewer than three usable stems is not evidence of a sans face, it is no
  // evidence at all — and saying "sans" anyway would let every short line on a
  // serif document vote against it.
  if (ratios.size() < 3) return 0.0F;
  return median_of(ratios);
}

// Coefficient of variation of a sample, or a large number when it has too
// little to say. Used to ask how much the per-character advance moves across a
// document, which is what separates a monospaced face from a proportional one.
float spread_ratio(std::vector<float> &v) {
  if (v.size() < 3) return 1.0F;
  const float med = median_of(v);
  if (med <= 0.0F) return 1.0F;
  std::vector<float> dev;
  dev.reserve(v.size());
  for (float x : v) dev.push_back(std::abs(x - med));
  // Median absolute deviation, not standard deviation: one short line, or one
  // line the recogniser mangled, must not decide the answer for the document.
  return median_of(dev) / med;
}

} // namespace

const char *FontChoice::standard_name() const noexcept {
  switch (family) {
    case FontFamily::Serif:
      if (bold && italic) return "Times-BoldItalic";
      if (bold) return "Times-Bold";
      if (italic) return "Times-Italic";
      return "Times-Roman";
    case FontFamily::Mono:
      if (bold && italic) return "Courier-BoldOblique";
      if (bold) return "Courier-Bold";
      if (italic) return "Courier-Oblique";
      return "Courier";
    case FontFamily::Sans:
    default:
      if (bold && italic) return "Helvetica-BoldOblique";
      if (bold) return "Helvetica-Bold";
      if (italic) return "Helvetica-Oblique";
      return "Helvetica";
  }
}

bool rectify_line(const cv::Mat &page, const Box &box, cv::Mat &out) {
  if (page.empty()) return false;
  cv::Rect inner;
  cv::Mat padded;
  if (!upright_crop(page, box, padded, inner)) return false;
  out = padded(inner).clone();
  return !out.empty();
}

LineStyle measure_line_style(const cv::Mat &page, const Box &box, int char_count) {
  LineStyle out;
  if (page.empty()) return out;

  cv::Mat crop;
  cv::Rect inner;
  if (!upright_crop(page, box, crop, inner)) return out;

  cv::Mat bgr;
  if (crop.channels() == 1) cv::cvtColor(crop, bgr, cv::COLOR_GRAY2BGR);
  else if (crop.channels() == 4) cv::cvtColor(crop, bgr, cv::COLOR_BGRA2BGR);
  else bgr = crop;

  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

  // Threshold on the LINE only, then apply that threshold to the whole padded
  // crop. Letting the margin into Otsu would bias it towards the background,
  // which on a sparse line is most of the pixels.
  cv::Mat line_gray = gray(inner);
  cv::Mat line_mask;
  const double thr = cv::threshold(line_gray, line_mask, 0, 255,
                                   cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

  double ink_share = static_cast<double>(cv::countNonZero(line_mask)) /
                     static_cast<double>(line_mask.total());
  // Light text on a dark ground: Otsu still splits the two populations, it just
  // labels the wrong one as ink. Text never covers most of its own line, so the
  // majority class is the paper whichever way round the page is printed.
  const bool inverted = ink_share > 0.55;
  cv::Mat full_mask;
  cv::threshold(gray, full_mask, thr, 255,
                inverted ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV);
  if (inverted) {
    cv::bitwise_not(line_mask, line_mask);
    ink_share = 1.0 - ink_share;
  }

  if (cv::countNonZero(line_mask) < kMinInkPixels) return out;

  // ── vertical extents ────────────────────────────────────────────────────
  cv::Mat rowsum;
  cv::reduce(line_mask, rowsum, 1, cv::REDUCE_SUM, CV_32S);
  const auto *rs = rowsum.ptr<int>(0);
  const int rows = line_mask.rows;
  int peak = 0;
  for (int y = 0; y < rows; ++y) peak = std::max(peak, rs[y]);
  if (peak <= 0) return out;

  const auto band_floor = static_cast<int>(kXHeightBandShare * static_cast<float>(peak));
  const auto ascent_floor = static_cast<int>(kAscentShare * static_cast<float>(peak));
  int x_top = -1;
  int x_bot = -1;
  int a_top = -1;
  for (int y = 0; y < rows; ++y) {
    if (rs[y] >= band_floor) {
      if (x_top < 0) x_top = y;
      x_bot = y;
    }
    if (a_top < 0 && rs[y] >= ascent_floor) a_top = y;
  }
  if (x_top < 0 || x_bot - x_top + 1 < kMinXHeight) return out;

  // The inked extent, which is what replacement type has to be fitted to.
  // A low bar rather than zero, so a speck of scanner noise in the margin of
  // the box does not stretch it.
  const auto ink_floor = static_cast<int>(0.02F * static_cast<float>(peak));
  int ink_top = -1;
  int ink_bot = -1;
  for (int y = 0; y < rows; ++y) {
    if (rs[y] <= ink_floor) continue;
    if (ink_top < 0) ink_top = y;
    ink_bot = y;
  }
  cv::Mat colsum;
  cv::reduce(line_mask, colsum, 0, cv::REDUCE_SUM, CV_32S);
  const auto *cs = colsum.ptr<int>(0);
  int ink_left = -1;
  int ink_right = -1;
  for (int x = 0; x < line_mask.cols; ++x) {
    if (cs[x] <= 0) continue;
    if (ink_left < 0) ink_left = x;
    ink_right = x;
  }
  if (ink_top >= 0 && ink_left >= 0) {
    out.ink_x = ink_left;
    out.ink_y = ink_top;
    out.ink_w = ink_right - ink_left + 1;
    out.ink_h = ink_bot - ink_top + 1;
  }

  const auto x_height = static_cast<float>(x_bot - x_top + 1);
  // The baseline sits just under the last dense row: that row is where the
  // stems stop, and it is where replacement text has to sit to line up.
  const auto baseline = static_cast<float>(x_bot + 1);
  out.x_height_px = x_height;
  out.baseline_px = baseline;
  out.ascent_px = baseline - static_cast<float>(a_top < 0 ? x_top : a_top);

  // ── stroke width, from horizontal runs across the x-height band ─────────
  const cv::Mat band = line_mask(cv::Rect(0, x_top, line_mask.cols, x_bot - x_top + 1));
  const auto max_run = static_cast<int>(kMaxRunShare * static_cast<float>(band.cols));
  std::vector<float> runs;
  for (int y = 0; y < band.rows; ++y) {
    const auto *row = band.ptr<uint8_t>(y);
    int run = 0;
    for (int x = 0; x < band.cols; ++x) {
      if (row[x]) {
        ++run;
        continue;
      }
      if (run > 0 && run <= max_run) runs.push_back(static_cast<float>(run));
      run = 0;
    }
    if (run > 0 && run <= max_run) runs.push_back(static_cast<float>(run));
  }
  if (runs.size() < 8) return out;

  const float stroke = median_of(runs);
  if (stroke <= 0.0F) return out;
  out.weight = stroke / x_height;

  out.serif = measure_foot_spread(band);
  out.slant_deg = measure_slant(band);
  if (char_count > 0)
    out.advance_ratio =
        static_cast<float>(line_mask.cols) / static_cast<float>(char_count) / x_height;

  // ── colour ──────────────────────────────────────────────────────────────
  //
  // Erode the ink before sampling it and dilate it before sampling the paper,
  // so the antialiased rim between them — which is neither colour — lands in
  // neither sample.
  cv::Mat ink_core;
  cv::Mat ink_halo;
  const cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
  cv::erode(full_mask, ink_core, k);
  cv::dilate(full_mask, ink_halo, k, {-1, -1}, 2);
  if (cv::countNonZero(ink_core) < 4) ink_core = full_mask;

  cv::Mat paper_mask;
  cv::bitwise_not(ink_halo, paper_mask);
  out.ink = masked_median_colour(bgr, ink_core, cv::Vec3b(0, 0, 0));
  out.paper = masked_median_colour(bgr, paper_mask, cv::Vec3b(255, 255, 255));

  // FLATNESS. Measured over the margin around the line, with nothing masked
  // out — which is the whole point, and the opposite of what this did before.
  //
  // Taking the spread over "everything that is not ink" looks right and is
  // exactly backwards: a printed rule, a cell border, a shaded panel are all
  // dark, so all of them land in the ink mask, get dilated, and are removed
  // from the sample. The test then reports beautifully flat paper for precisely
  // the lines that must never be painted over, and form underlines were
  // destroyed. The margin ring carries the structures the glyphs do not.
  cv::Mat ring = cv::Mat::zeros(gray.size(), CV_8UC1);
  ring.setTo(255);
  ring(inner).setTo(0);
  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(gray, mean, stddev, ring);
  const bool ring_flat =
      cv::countNonZero(ring) >= 16 && stddev[0] < kFlatPaperStdDev;

  // A rule running UNDER the words sits inside the box, where it cannot be told
  // from ink by threshold alone — but it gives itself away by length. No letter
  // stroke runs for half a line.
  bool ruled = false;
  for (int y = 0; y < line_mask.rows && !ruled; ++y) {
    const auto *row = line_mask.ptr<uint8_t>(y);
    int run_len = 0;
    for (int x = 0; x < line_mask.cols; ++x) {
      run_len = row[x] ? run_len + 1 : 0;
      if (run_len > static_cast<int>(kRuleRunShare * static_cast<float>(line_mask.cols))) {
        ruled = true;
        break;
      }
    }
  }

  out.flat_paper = ring_flat && !ruled;

  out.measured = true;
  return out;
}

std::vector<LineStyle>
measure_page_line_styles(const std::vector<OCRResultItem> &results,
                         const cv::Mat &page) {
  std::vector<LineStyle> out;
  if (page.empty()) return out;
  out.reserve(results.size());
  for (const auto &item : results) {
    // Character count, not byte count: the advance-per-character measurement is
    // meaningless if a line of Japanese counts three times its glyphs.
    int chars = 0;
    for (unsigned char c : item.text)
      if ((c & 0xC0) != 0x80) ++chars;
    out.push_back(measure_line_style(page, item.box, chars));
  }
  return out;
}

std::vector<FontChoice>
resolve_document_fonts(const std::vector<LineStyle> &lines,
                       const FontFamily *family_override) {
  std::vector<FontChoice> out(lines.size());

  std::vector<float> weights;
  std::vector<float> slants;
  std::vector<float> advances;
  double serif_num = 0;
  double evidence = 0;
  for (const LineStyle &l : lines) {
    if (!l.measured) continue;
    weights.push_back(l.weight);
    slants.push_back(l.slant_deg);
    if (l.advance_ratio > 0.0F) advances.push_back(l.advance_ratio);
    // Only lines that HAVE an opinion on shape vote on it. A line of three
    // words may hold no measurable stem at all, and counting its silence as a
    // vote for sans would let a form of short labels outvote the paragraphs.
    if (l.serif > 0.0F) {
      // A taller line was measured on more pixels, so it gets more of a say.
      const double w = l.x_height_px;
      serif_num += w * l.serif;
      evidence += w;
    }
  }
  if (weights.empty()) return out;

  // No line anywhere had a measurable stem. Sans is both the commoner answer
  // and the cheaper mistake: serif type set in Helvetica reads plainly, while
  // sans set in Times acquires feet the original never had.
  const auto doc_serif =
      evidence > 0.0 ? static_cast<float>(serif_num / evidence) : 0.0F;
  const float med_weight = median_of(weights);
  const float med_slant = median_of(slants);
  const float advance_spread = spread_ratio(advances);

  // ONE family for the whole document. Type is a property of the document, not
  // of the line, and a page that flickers between Times and Helvetica line by
  // line looks far worse than one that picks the wrong family consistently.
  //
  // Pitch is tested before shape, because Courier is a slab serif and would
  // otherwise be answered "serif" — true of its shape, but the wrong answer:
  // setting a monospaced document in Times throws away the column alignment
  // that was the reason for using it.
  FontFamily family = FontFamily::Sans;
  if (family_override != nullptr) {
    family = *family_override;
  } else if (advances.size() >= kMinMonoLines && advance_spread <= kMonoSpread) {
    family = FontFamily::Mono;
  } else if (doc_serif >= kSerifThreshold) {
    family = FontFamily::Serif;
  }

  // What the document does on the whole, for the lines too small to judge on
  // their own. A short line inside a bold heading block belongs with the block.
  int bold_votes = 0;
  int italic_votes = 0;
  int voters = 0;
  const auto judge_bold = [&](const LineStyle &l) {
    // Relative OR absolute, and it needs both arms. The relative arm survives a
    // binarisation that fattens every stroke on the page; the absolute arm is
    // what still finds bold in a document that is bold throughout, where the
    // median is itself bold and nothing stands out against it.
    return l.weight > med_weight * kBoldRelative || l.weight > kBoldAbsolute;
  };
  const auto judge_italic = [&](const LineStyle &l) {
    // Subtracting the document's own median slant is what separates italic from
    // a scan that simply went through the feeder crooked: a whole page leaning
    // 3 degrees is skew, one line leaning 11 against its neighbours is italic.
    return (l.slant_deg - med_slant) > kItalicRelativeDeg ||
           l.slant_deg > kItalicAbsoluteDeg;
  };
  for (const LineStyle &l : lines) {
    if (!l.measured || l.x_height_px < kMinTrustworthyXHeight) continue;
    ++voters;
    bold_votes += judge_bold(l) ? 1 : 0;
    italic_votes += judge_italic(l) ? 1 : 0;
  }
  const bool doc_bold = voters > 0 && bold_votes * 2 > voters;
  const bool doc_italic = voters > 0 && italic_votes * 2 > voters;

  for (size_t i = 0; i < lines.size(); ++i) {
    const LineStyle &l = lines[i];
    out[i].family = family;
    if (!l.measured || l.x_height_px < kMinTrustworthyXHeight) {
      out[i].bold = doc_bold;
      out[i].italic = doc_italic;
      continue;
    }
    out[i].bold = judge_bold(l);
    out[i].italic = judge_italic(l);
  }
  return out;
}

} // namespace turbo_ocr::pdf
