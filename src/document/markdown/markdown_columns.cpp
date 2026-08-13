#include "markdown_internal.h"

#include <algorithm>
#include <climits>
#include <optional>
#include <vector>

namespace turbo_ocr::markdown::mddetail {
namespace {

// ── column-aware body re-ordering (Markdown view only) ───────────────────
//
// PaddleX's xycut_enhanced weaves the columns of a multi-column page by
// horizontal band (L,R,L,R…) rather than finishing the left column first, so a
// reading_order-faithful render of a two-column page zig-zags. This pass
// detects a CLEAR column structure purely from the body block bboxes and
// re-emits column-major. It NEVER touches the JSON / scorer reading_order —
// only the order body cells are walked for Markdown emission.
//
// Detection (all must hold, else fall back to reading_order):
//  • >= kColMinBodyBlocks body cells.
//  • A vertical "gutter": a contiguous interior x-band wide enough
//    (>= kColGutterMinW) over which the summed height of straddling blocks is
//    <= kColGutterCov of the content height — i.e. a real whitespace corridor.
//    Full-width blocks crossing the gutter are tolerated up to that coverage,
//    so single-column pages (whose full-width text straddles every interior x)
//    never register a gutter. Multiple gutters ⇒ N columns.
//  • Each column holds >= kColMinColBlocks AND >= kColMinColShare of the
//    non-spanning blocks (no thin equation-number "column").
//  • Adjacent columns vertically overlap by >= kColMinVOverlap (side-by-side,
//    not stacked sections).
// A block is a full-width break (emitted between column groups at its vertical
// position) only when it genuinely spans both sides of a gutter — a wide
// paragraph that merely pokes a few px past the gutter stays in its column.
constexpr double kColGutterCov      = 0.18; // max straddle-coverage in a gutter
constexpr double kColEdgeFrac       = 0.12; // ignore page-margin "gutters"
constexpr double kColGutterMinWFrac = 0.015;
constexpr int    kColGutterMinWAbs  = 8;
constexpr int    kColMinColBlocks   = 2;
constexpr double kColMinColShare    = 0.15;
constexpr double kColMinVOverlap    = 0.25;
constexpr double kColSpanPen        = 0.18; // min penetration into BOTH sides

} // namespace

// `rects` are the axis-aligned bboxes of the body cells in their current
// (reading_order) emission order. Returns a permutation of [0,rects.size())
// in column-major reading order, or nullopt when no clear column structure is
// found (caller keeps the original order).
[[nodiscard]] std::optional<std::vector<int>>
column_major_order(const std::vector<std::array<int, 4>> &rects) {
  const int n = static_cast<int>(rects.size());
  if (n < kColMinBodyBlocks) return std::nullopt;

  int cL = INT_MAX, cR = INT_MIN, cT = INT_MAX, cB = INT_MIN;
  for (const auto &r : rects) {
    cL = std::min(cL, r[0]); cR = std::max(cR, r[2]);
    cT = std::min(cT, r[1]); cB = std::max(cB, r[3]);
  }
  const double cW = cR - cL, cH = cB - cT;
  if (cW <= 0 || cH <= 0) return std::nullopt;

  // Summed height of blocks strictly straddling x, normalised by content
  // height. A gutter is where this is near zero (overestimate vs the true
  // union of y-intervals is conservative — it can only suppress a split).
  auto cov = [&](double x) -> double {
    double s = 0;
    for (const auto &r : rects)
      if (r[0] < x && x < r[2]) s += (r[3] - r[1]);
    return s / cH;
  };

  const double edge = kColEdgeFrac * cW;
  const double lo = cL + edge, hi = cR - edge;
  if (hi <= lo) return std::nullopt;
  const int step = std::max(1, static_cast<int>(cW / 500));
  const double min_w = std::max(static_cast<double>(kColGutterMinWAbs),
                                kColGutterMinWFrac * cW);

  std::vector<double> boundaries;
  bool in_band = false; double band_lo = 0, band_hi = 0;
  auto close_band = [&] {
    if (in_band && band_hi - band_lo >= min_w)
      boundaries.push_back((band_lo + band_hi) * 0.5);
    in_band = false;
  };
  for (int xi = static_cast<int>(lo); xi <= static_cast<int>(hi); xi += step) {
    if (cov(static_cast<double>(xi)) <= kColGutterCov) {
      if (!in_band) { in_band = true; band_lo = xi; }
      band_hi = xi;
    } else {
      close_band();
    }
  }
  close_band();
  if (boundaries.empty()) return std::nullopt;
  std::sort(boundaries.begin(), boundaries.end());

  const int ncols = static_cast<int>(boundaries.size()) + 1;
  std::vector<double> edges;
  edges.push_back(cL);
  for (double b : boundaries) edges.push_back(b);
  edges.push_back(cR);
  auto col_of = [&](double xc) {
    int c = 0;
    for (double b : boundaries) if (xc >= b) ++c;
    return c;
  };

  std::vector<std::vector<int>> cols(ncols);
  std::vector<int> spanning;
  for (int i = 0; i < n; ++i) {
    const auto &r = rects[i];
    int crossed = -1, ncross = 0;
    for (int k = 0; k < static_cast<int>(boundaries.size()); ++k)
      if (r[0] < boundaries[k] && boundaries[k] < r[2]) { crossed = k; ++ncross; }
    bool is_span = false;
    if (ncross >= 2) {
      is_span = true;
    } else if (ncross == 1) {
      const double bd = boundaries[crossed];
      const double wL = bd - edges[crossed], wR = edges[crossed + 2] - bd;
      const double pen = std::min(bd - r[0], static_cast<double>(r[2]) - bd);
      if (pen >= kColSpanPen * std::min(wL, wR)) is_span = true;
    }
    if (is_span) spanning.push_back(i);
    else cols[col_of((r[0] + r[2]) * 0.5)].push_back(i);
  }

  int nonspan = 0;
  for (const auto &c : cols) nonspan += static_cast<int>(c.size());
  if (nonspan == 0) return std::nullopt;
  for (const auto &c : cols) {
    if (static_cast<int>(c.size()) < kColMinColBlocks) return std::nullopt;
    if (static_cast<double>(c.size()) < kColMinColShare * nonspan)
      return std::nullopt;
  }

  // Adjacent columns must run side by side (vertical overlap), else they are
  // stacked sections that only look like columns.
  std::vector<std::array<int, 2>> yext(ncols, {INT_MAX, INT_MIN});
  for (int c = 0; c < ncols; ++c)
    for (int i : cols[c]) {
      yext[c][0] = std::min(yext[c][0], rects[i][1]);
      yext[c][1] = std::max(yext[c][1], rects[i][3]);
    }
  for (int a = 0; a + 1 < ncols; ++a) {
    const int ov = std::min(yext[a][1], yext[a + 1][1]) -
                   std::max(yext[a][0], yext[a + 1][0]);
    const int sm = std::min(yext[a][1] - yext[a][0],
                            yext[a + 1][1] - yext[a + 1][0]);
    if (sm <= 0 || ov < kColMinVOverlap * sm) return std::nullopt;
  }

  auto yc = [&](int i) { return (rects[i][1] + rects[i][3]) * 0.5; };
  std::sort(spanning.begin(), spanning.end(),
            [&](int a, int b) { return yc(a) < yc(b); });
  auto band_of = [&](int i) {
    int b = 0;
    for (int s : spanning) if (yc(s) < yc(i)) ++b;
    return b;
  };

  const int nb = static_cast<int>(spanning.size());
  std::vector<int> out;
  out.reserve(n);
  for (int band = 0; band <= nb; ++band) {
    for (int c = 0; c < ncols; ++c) {
      std::vector<int> mem;
      for (int i : cols[c]) if (band_of(i) == band) mem.push_back(i);
      std::sort(mem.begin(), mem.end(), [&](int a, int b) {
        if (rects[a][1] != rects[b][1]) return rects[a][1] < rects[b][1];
        return rects[a][0] < rects[b][0];
      });
      for (int i : mem) out.push_back(i);
    }
    if (band < nb) out.push_back(spanning[band]);
  }
  return out;
}

} // namespace turbo_ocr::markdown::mddetail
