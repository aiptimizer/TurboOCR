#include "turbo_ocr/output/markdown_export.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/router/router_types.h"

namespace turbo_ocr::output {
namespace {

// ── small string utils ──────────────────────────────────────────────────

[[nodiscard]] std::string trim(std::string_view s) {
  size_t a = 0, b = s.size();
  while (a < b && static_cast<unsigned char>(s[a]) <= ' ') ++a;
  while (b > a && static_cast<unsigned char>(s[b - 1]) <= ' ') --b;
  return std::string(s.substr(a, b - a));
}

[[nodiscard]] std::string to_lower_ascii(std::string_view s) {
  std::string o(s);
  for (char &c : o)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return o;
}

// Count Unicode codepoints (UTF-8 lead bytes) so the orphan-length gate is
// script-agnostic — a single CJK glyph counts as one, not three.
[[nodiscard]] int codepoint_count(std::string_view s) {
  int n = 0;
  for (unsigned char c : s)
    if ((c & 0xC0) != 0x80) ++n; // not a continuation byte
  return n;
}

// Join the visible text of an ordered set of OCR lines with single spaces,
// collapsing internal whitespace runs so a wrapped paragraph reads as one line.
[[nodiscard]] std::string join_lines(const std::vector<std::string> &lines) {
  std::string out;
  for (const auto &raw : lines) {
    std::string t = trim(raw);
    if (t.empty()) continue;
    if (!out.empty()) out += ' ';
    out += t;
  }
  return out;
}

// ── table HTML hygiene ───────────────────────────────────────────────────
//
// Recognizers emit `<html><body><table …>…</table></body></html>`. The
// document wrapper is invalid inside Markdown and makes most viewers show raw
// text or drop the table. Keep ONLY the bare `<table>…</table>` (colspan /
// rowspan preserved). Returns "" when there is no table to embed.
[[nodiscard]] std::string strip_table_wrapper(const std::string &html) {
  const std::string lo = to_lower_ascii(html);
  const size_t a = lo.find("<table");
  const size_t b = lo.rfind("</table>");
  if (a == std::string::npos || b == std::string::npos || b < a) return {};
  return trim(std::string_view(html).substr(a, (b + 8) - a)); // 8 = "</table>"
}

// ── LaTeX render-safety ──────────────────────────────────────────────────
//
// We cannot run KaTeX in-process, so reject the structural failures that make
// it throw: unbalanced groups, unbalanced \begin/\end, and a \left|\right not
// immediately followed by a valid delimiter (the real-world breaker
// `\left\mathrm…`). Conservative: a flagged formula falls back to code, never
// to a $$ that errors. `\{` / `\}` are literal braces and do NOT open groups.

[[nodiscard]] bool is_delimiter_at(const std::string &s, size_t k) {
  while (k < s.size() && s[k] == ' ') ++k;
  if (k >= s.size()) return false;
  const char c = s[k];
  if (c == '(' || c == ')' || c == '[' || c == ']' || c == '|' || c == '.' ||
      c == '/' || c == '<' || c == '>')
    return true;
  if (c == '\\') {
    size_t j = k + 1;
    if (j < s.size()) {
      const char d = s[j];
      if (d == '{' || d == '}' || d == '|' || d == '\\') return true; // \{ \} \| or \\ esc
      std::string cmd;
      while (j < s.size() && std::isalpha(static_cast<unsigned char>(s[j])))
        cmd += s[j++];
      static const std::unordered_set<std::string> kDelims = {
          "lbrace",  "rbrace",  "langle",  "rangle",   "lvert",
          "rvert",   "lVert",   "rVert",   "vert",     "Vert",
          "lfloor",  "rfloor",  "lceil",   "rceil",    "uparrow",
          "downarrow", "Uparrow", "Downarrow", "updownarrow", "backslash"};
      return kDelims.count(cmd) > 0;
    }
  }
  return false;
}

// Commands that, like \left/\right, REQUIRE a delimiter argument — a missing
// or non-delimiter argument throws in KaTeX (e.g. `\big -`).
[[nodiscard]] bool needs_delimiter(const std::string &cmd) {
  static const std::unordered_set<std::string> kSized = {
      "big",  "Big",  "bigg",  "Bigg",  "bigl",  "Bigl",  "biggl", "Biggl",
      "bigr", "Bigr", "biggr", "Biggr", "bigm",  "Bigm",  "biggm", "Biggm"};
  return cmd == "left" || cmd == "right" || kSized.count(cmd) > 0;
}

// Skip one sub/superscript argument starting at k: a {…} group, a \command,
// or a single atom. Returns the index just past it.
[[nodiscard]] size_t skip_script_arg(const std::string &s, size_t k) {
  if (k >= s.size()) return k;
  if (s[k] == '{') {
    int d = 0;
    for (size_t i = k; i < s.size(); ++i) {
      if (s[i] == '{') ++d;
      else if (s[i] == '}' && --d == 0) return i + 1;
    }
    return s.size();
  }
  if (s[k] == '\\') {
    size_t j = k + 1;
    if (j < s.size() && std::isalpha(static_cast<unsigned char>(s[j]))) {
      while (j < s.size() && std::isalpha(static_cast<unsigned char>(s[j]))) ++j;
      return j;
    }
    return std::min(s.size(), k + 2); // \<char>
  }
  return k + 1; // single atom
}

// A base carrying two same-type scripts (`x^a^b`, `\beta_{}_{f}`) throws
// "Double sub/superscript" in KaTeX. Detect it without disturbing the
// brace/delimiter accounting of the main pass.
[[nodiscard]] bool has_double_script(const std::string &s) {
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '\\') { // step over an escaped op or command so \_ / \^ aren't ops
      size_t j = i + 1;
      if (j < s.size() && std::isalpha(static_cast<unsigned char>(s[j]))) {
        while (j < s.size() && std::isalpha(static_cast<unsigned char>(s[j]))) ++j;
        i = j - 1;
      } else {
        i = j;
      }
      continue;
    }
    if (c == '_' || c == '^') {
      size_t k = i + 1;
      while (k < s.size() && s[k] == ' ') ++k;
      k = skip_script_arg(s, k);
      while (k < s.size() && s[k] == ' ') ++k;
      if (k < s.size() && s[k] == c) return true;
    }
  }
  return false;
}

// Conservative structural KaTeX safety check. Catches the OCR-garble classes
// that actually throw: unbalanced groups / \begin-\end / \left-\right, a
// sizing/fence command not followed by a delimiter, a dangling backslash, an
// undefined `\<punct>` escape (e.g. `\?`), an accent escape with no argument,
// and an empty / doubled sub-superscript. A flagged formula falls back to code
// rather than to a $$ that errors. Errs toward fallback, never toward breakage.
[[nodiscard]] bool latex_is_render_safe(const std::string &s) {
  if (trim(s).empty()) return false;
  if (has_double_script(s)) return false;
  int brace = 0, leftright = 0, beginend = 0;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (c == '\\') {
      size_t j = i + 1;
      if (j >= s.size()) return false; // dangling backslash at end
      const char nx = s[j];
      if (!std::isalpha(static_cast<unsigned char>(nx))) {
        if (nx == '\\' || nx == ' ') { i = j; continue; } // \\ / control space
        static const std::string kOkPunct = "{}$&#_%~|,;:!.";
        if (kOkPunct.find(nx) != std::string::npos) { i = j; continue; }
        static const std::string kAccent = "=\"'^`";
        if (kAccent.find(nx) != std::string::npos) {
          size_t k = j + 1;
          while (k < s.size() && s[k] == ' ') ++k;
          if (k >= s.size()) return false; // accent with no argument (`\=` at end)
          const char a = s[k];
          if (a == '}' || a == ']' || a == ')' || a == '_' || a == '^')
            return false;
          i = j; continue;
        }
        return false; // undefined escape such as `\?`
      }
      std::string cmd;
      while (j < s.size() && std::isalpha(static_cast<unsigned char>(s[j])))
        cmd += s[j++];
      if (needs_delimiter(cmd)) {
        if (!is_delimiter_at(s, j)) return false;
        if (cmd == "left") ++leftright;
        else if (cmd == "right") { if (--leftright < 0) return false; }
      } else if (cmd == "begin") {
        ++beginend;
      } else if (cmd == "end") {
        if (--beginend < 0) return false;
      }
      i = (j > i) ? j - 1 : i;
      continue;
    }
    if (c == '{') {
      ++brace;
    } else if (c == '}') {
      if (--brace < 0) return false;
    } else if (c == '_' || c == '^') {
      size_t k = i + 1;
      while (k < s.size() && s[k] == ' ') ++k;
      if (k >= s.size()) return false; // trailing script operator
      const char a = s[k];
      if (a == '_' || a == '^' || a == '}' || a == ']' || a == ')' || a == '&')
        return false; // empty / doubled script
    }
  }
  return brace == 0 && leftright == 0 && beginend == 0;
}

[[nodiscard]] std::string inline_code(const std::string &s) {
  // A `…` span can't contain a backtick; widen the fence if it does.
  std::string fence = "`";
  if (s.find('`') != std::string::npos) fence = "``";
  return fence + " " + s + " " + fence;
}

// ── runaway / mode-collapse garbage detector (Markdown view only) ─────────
//
// PP-FormulaNet-S degenerates on Chinese-text-in-formula crops and emits
// syntactically valid but semantically dead LaTeX: a short unit
// (`\mathsf { W } _ { \mathsf { a } }`, `\mathrm { ( ) }`, a bare `\ `)
// repeated dozens of times, or a 1500+ char runaway. latex_is_render_safe()
// passes these — the braces balance — so they would render as a wall of
// garbage that dominates the page. We mirror formula::formula_is_mode_collapsed
// but on the LaTeX string (the serializer has no token ids). Scoped to the
// Markdown view ONLY: the JSON/scorer path keeps every formula.
//
// Tuned on the OmniDocBench result set (1679 formulas). Legitimate matrices /
// \begin{aligned} blocks peak at a 5–8-token window recurring ~14× and a
// back-to-back identical run of ~3 (worst legit OCR ~7); genuine collapse sits
// at window ≥27 and run ≥14. The gates (window ≥20, run ≥8, len ≥1500) leave a
// margin on both sides and flag 13/1679 — every one garbage — while keeping
// every real matrix / aligned system.

[[nodiscard]] std::vector<std::string_view> ws_tokens(const std::string &s) {
  std::vector<std::string_view> t;
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    while (i < n && static_cast<unsigned char>(s[i]) <= ' ') ++i;
    const size_t a = i;
    while (i < n && static_cast<unsigned char>(s[i]) > ' ') ++i;
    if (i > a) t.emplace_back(s.data() + a, i - a);
  }
  return t;
}

// Highest occurrence count of any identical w-token window (overlapping).
[[nodiscard]] int max_window_count(const std::vector<std::string_view> &t, int w) {
  const int n = static_cast<int>(t.size());
  if (n < w) return 0;
  std::unordered_map<std::string, int> freq;
  int best = 0;
  for (int i = 0; i + w <= n; ++i) {
    std::string key;
    for (int k = 0; k < w; ++k) { key.append(t[i + k]); key.push_back('\x1f'); }
    best = std::max(best, ++freq[key]);
  }
  return best;
}

// Longest run of a unit repeated back-to-back, over periods 1..6 (stride =
// period). Catches `\# \# \# …` and `\ \ \ …` style collapse that a window
// count understates because the unit is the same single token.
[[nodiscard]] int max_consecutive_run(const std::vector<std::string_view> &t) {
  const int n = static_cast<int>(t.size());
  int best = 1;
  for (int w = 1; w <= 6; ++w) {
    for (int i = 0; i + w <= n;) {
      int reps = 1, j = i + w;
      while (j + w <= n &&
             std::equal(t.begin() + i, t.begin() + i + w, t.begin() + j))
        { ++reps; j += w; }
      best = std::max(best, reps);
      i = (reps > 1) ? j : i + 1;
    }
  }
  return best;
}

[[nodiscard]] bool latex_is_mode_collapsed(const std::string &latex) {
  if (latex.size() >= 1500) return true;          // absolute runaway
  const auto t = ws_tokens(latex);
  if (t.size() < 24) return false;                 // short formulas always kept
  for (int w = 5; w <= 8; ++w)
    if (max_window_count(t, w) >= 20) return true;  // long identical-unit repeat
  return max_consecutive_run(t) >= 8;              // back-to-back collapse
}

// Strip surrounding parens/whitespace from a detected formula number so
// \tag{N} renders "(N)" rather than "((N))".
[[nodiscard]] std::string clean_tag(std::string s) {
  s = trim(s);
  if (s.size() >= 2 && s.front() == '(' && s.back() == ')')
    s = trim(s.substr(1, s.size() - 2));
  return s;
}

// ── base64 (data-URI embed path only) ────────────────────────────────────
[[nodiscard]] std::string base64_encode(const unsigned char *p, size_t n) {
  static const char *T =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  size_t i = 0;
  for (; i + 3 <= n; i += 3) {
    uint32_t v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
    out += T[(v >> 18) & 63];
    out += T[(v >> 12) & 63];
    out += T[(v >> 6) & 63];
    out += T[v & 63];
  }
  if (i < n) {
    uint32_t v = p[i] << 16;
    if (i + 1 < n) v |= p[i + 1] << 8;
    out += T[(v >> 18) & 63];
    out += T[(v >> 12) & 63];
    out += (i + 1 < n) ? T[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

// Class-id constants (pinned by the static_asserts in layout_types.h).
constexpr int kClassDisplayFormula = 5;
constexpr int kClassFormulaNumber  = 11;
constexpr int kClassInlineFormula  = 15;

[[nodiscard]] bool is_image_label(std::string_view l) {
  return l == "image" || l == "chart" || l == "header_image" ||
         l == "footer_image" || l == "seal";
}

// Text-bearing classes the orphan-length gate applies to (titles / formulas /
// tables / images are exempt — handled by their own branches).
[[nodiscard]] bool length_gated(std::string_view l) {
  return l == "text" || l == "content" || l == "abstract" ||
         l == "aside_text" || l == "footnote" || l == "vision_footnote" ||
         l == "vertical_text" || l == "reference" || l == "reference_content" ||
         l == "number" || l == "SupplementaryRegion";
}

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
constexpr int    kColMinBodyBlocks  = 4;
constexpr double kColSpanPen        = 0.18; // min penetration into BOTH sides

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

// Escape the characters that terminate a Markdown image-link caption or
// destination — an OCR'd ']' or ')' in a figure caption must not truncate
// the link and leak the rest as raw text.
std::string escape_md_link_text(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    // A newline would terminate the ![...] link; captions are single-line by
    // construction but OCR text is untrusted, so fold to spaces.
    if (c == '\n' || c == '\r') {
      if (!out.empty() && out.back() != ' ') out.push_back(' ');
      continue;
    }
    if (c == '[' || c == ']' || c == '(' || c == ')' || c == '\\')
      out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

} // namespace

const std::unordered_set<std::string> &default_ignore_labels() {
  static const std::unordered_set<std::string> kIgnore = {
      "header", "header_image", "footer", "footer_image", "number", "seal"};
  return kIgnore;
}

std::string render_markdown(const pipeline::OcrPipelineResult &res,
                            const MarkdownOptions &opts,
                            std::vector<MarkdownAsset> *assets_out,
                            const ImageSrcResolver &resolver) {
  const auto &layout = res.layout;
  const auto &results = res.results;
  const size_t nL = layout.size();
  const size_t nR = results.size();

  if (nL == 0) {
    // No layout: emit text lines in their native order as one paragraph each.
    std::vector<std::string> parts;
    for (const auto &it : results) {
      std::string t = trim(it.text);
      if (codepoint_count(t) >= opts.min_text_codepoints) parts.push_back(t);
    }
    std::string md;
    for (size_t i = 0; i < parts.size(); ++i) {
      if (i) md += "\n\n";
      md += parts[i];
    }
    return md.empty() ? md : md + "\n";
  }

  // Axis-aligned rect per layout cell.
  std::vector<std::array<int, 4>> lr(nL);
  for (size_t i = 0; i < nL; ++i) lr[i] = turbo_ocr::aabb(layout[i].box);

  auto li_of = [&](const OCRResultItem &it) -> int {
    if (it.layout_id >= 0 && static_cast<size_t>(it.layout_id) < nL)
      return it.layout_id;
    float cx = 0, cy = 0;
    for (int k = 0; k < 4; ++k) { cx += it.box[k][0]; cy += it.box[k][1]; }
    cx *= 0.25f; cy *= 0.25f;
    for (size_t j = 0; j < nL; ++j)
      if (cx >= lr[j][0] && cx <= lr[j][2] && cy >= lr[j][1] && cy <= lr[j][3])
        return static_cast<int>(j);
    return -1;
  };

  // Rank every result: reading_order first (text XY-cut), then any leftover by
  // geometry, so each result has a total order even if reading_order is partial.
  std::vector<int> rank(nR, -1);
  int next_rank = 0;
  for (int ri : res.reading_order)
    if (ri >= 0 && static_cast<size_t>(ri) < nR && rank[ri] < 0)
      rank[ri] = next_rank++;
  std::vector<int> leftover;
  for (size_t ri = 0; ri < nR; ++ri)
    if (rank[ri] < 0) leftover.push_back(static_cast<int>(ri));
  std::sort(leftover.begin(), leftover.end(), [&](int a, int b) {
    auto ra = turbo_ocr::aabb(results[a].box), rb = turbo_ocr::aabb(results[b].box);
    return ra[1] != rb[1] ? ra[1] < rb[1] : ra[0] < rb[0];
  });
  for (int ri : leftover) rank[ri] = next_rank++;

  // Group results by region (members carry result indices, kept in rank order).
  std::vector<std::vector<int>> members(nL);
  for (size_t ri = 0; ri < nR; ++ri) {
    int li = li_of(results[ri]);
    if (li >= 0) members[li].push_back(static_cast<int>(ri));
  }
  for (auto &m : members)
    std::sort(m.begin(), m.end(), [&](int a, int b) { return rank[a] < rank[b]; });

  // Per-region structure payloads, keyed by layout index (== their layout_id).
  std::unordered_map<int, const router::TableResult *> table_by_li;
  std::unordered_map<int, const router::FormulaResult *> formula_by_li;
  for (const auto &t : res.tables)
    if (t.layout_id >= 0 && static_cast<size_t>(t.layout_id) < nL)
      table_by_li[t.layout_id] = &t;
  for (const auto &f : res.formulas)
    if (f.layout_id >= 0 && static_cast<size_t>(f.layout_id) < nL)
      formula_by_li[f.layout_id] = &f;

  // Fold formula_number regions into the nearest display_formula on the same
  // vertical band; consumed numbers are not emitted as stray paragraphs.
  std::vector<char> consumed(nL, 0);
  std::unordered_map<int, std::string> tag_of; // display-formula li -> tag text
  if (opts.fold_formula_numbers) {
    for (size_t ni = 0; ni < nL; ++ni) {
      if (layout[ni].class_id != kClassFormulaNumber) continue;
      const int ncy = (lr[ni][1] + lr[ni][3]) / 2;
      int best = -1, best_dx = INT_MAX;
      for (size_t di = 0; di < nL; ++di) {
        if (layout[di].class_id != kClassDisplayFormula) continue;
        if (ncy < lr[di][1] || ncy > lr[di][3]) continue; // band overlap
        const int dx = std::abs(lr[ni][0] - lr[di][2]);    // number right of formula
        if (dx < best_dx) { best_dx = dx; best = static_cast<int>(di); }
      }
      if (best < 0) continue;
      std::string num;
      for (int ri : members[ni]) num += (num.empty() ? "" : " ") + trim(results[ri].text);
      num = clean_tag(num);
      if (!num.empty()) { tag_of[best] = num; consumed[ni] = 1; }
    }
  }

  // Center of a region, for placing structure-only cells (no OCR line) inline.
  auto center_rank = [&](size_t li) -> double {
    if (!members[li].empty()) return rank[members[li].front()];
    if (nR == 0) return 1e9 + lr[li][1]; // pure-geometric page
    const double cx = (lr[li][0] + lr[li][2]) * 0.5;
    const double cy = (lr[li][1] + lr[li][3]) * 0.5;
    int best = -1; double best_d = 1e30;
    for (size_t ri = 0; ri < nR; ++ri) {
      auto rb = turbo_ocr::aabb(results[ri].box);
      const double rx = (rb[0] + rb[2]) * 0.5, ry = (rb[1] + rb[3]) * 0.5;
      const double d = (rx - cx) * (rx - cx) + (ry - cy) * (ry - cy);
      if (d < best_d) { best_d = d; best = static_cast<int>(ri); }
    }
    return best >= 0 ? rank[best] + 0.5 : 1e9 + lr[li][1];
  };

  // Emit order: class-priority bucket (header/body/footer), then inline rank.
  std::vector<int> order(nL);
  for (size_t i = 0; i < nL; ++i) order[i] = static_cast<int>(i);
  std::vector<double> key(nL);
  for (size_t i = 0; i < nL; ++i) key[i] = center_rank(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const int ba = layout::reading_priority_bucket(layout[a].class_id);
    const int bb = layout::reading_priority_bucket(layout[b].class_id);
    if (ba != bb) return ba < bb;
    if (key[a] != key[b]) return key[a] < key[b];
    if (lr[a][1] != lr[b][1]) return lr[a][1] < lr[b][1];
    return lr[a][0] < lr[b][0];
  });

  // Column-aware re-ordering of the BODY bucket (Markdown view only). `order`
  // is bucket-major (TOP, then BODY, then BOTTOM are each contiguous), so the
  // body cells form one contiguous run we can reorder in place without
  // disturbing header/footer placement. A clean multi-column page is emitted
  // column-major; anything ambiguous keeps the reading_order sequence.
  if (opts.column_aware_order) {
    size_t blo = 0;
    while (blo < order.size() &&
           layout::reading_priority_bucket(layout[order[blo]].class_id) != 1)
      ++blo;
    size_t bhi = blo;
    while (bhi < order.size() &&
           layout::reading_priority_bucket(layout[order[bhi]].class_id) == 1)
      ++bhi;
    if (bhi - blo >= static_cast<size_t>(kColMinBodyBlocks)) {
      std::vector<std::array<int, 4>> rects;
      rects.reserve(bhi - blo);
      for (size_t i = blo; i < bhi; ++i) rects.push_back(lr[order[i]]);
      if (auto perm = column_major_order(rects)) {
        std::vector<int> reordered;
        reordered.reserve(perm->size());
        for (int p : *perm) reordered.push_back(order[blo + p]);
        std::copy(reordered.begin(), reordered.end(), order.begin() + blo);
      }
    }
  }

  auto gather = [&](int li) -> std::string {
    std::vector<std::string> lines;
    for (int ri : members[li]) lines.push_back(results[ri].text);
    return join_lines(lines);
  };

  std::vector<std::string> parts;
  bool refs_open = false;

  for (int li : order) {
    if (consumed[li]) continue;
    const auto &cell = layout[li];
    const std::string label(layout::label_name(cell.class_id));
    if (opts.ignore_labels.count(label)) continue;

    const int cls = cell.class_id;

    // Structure-first dispatch (these own their payload, not gathered text).
    if (label == "table") {
      // Recognized HTML when the table backend ran; otherwise (backend off, e.g.
      // geometric PDF mode) fall back to the region's raw text so the table's
      // content is not silently dropped — it just lacks grid structure.
      if (auto it = table_by_li.find(li); it != table_by_li.end()) {
        std::string html = strip_table_wrapper(it->second->html);
        if (!html.empty()) { parts.push_back(std::move(html)); continue; }
      }
      std::string txt = gather(li);
      if (!txt.empty()) parts.push_back(std::move(txt));
      continue;
    }
    if (cls == kClassDisplayFormula) {
      // Only content from the formula recognizer is LaTeX. When no recognized
      // result exists (formula backend off, e.g. geometric PDF mode, or the
      // region failed to recognize), gather(li) returns the region's raw
      // OCR / PDF-text-layer characters — NOT LaTeX. Wrapping those in $$ makes
      // KaTeX/MathJax render broken math, so emit them as a plain paragraph.
      if (auto f = formula_by_li.find(li); f != formula_by_li.end()) {
        std::string latex = trim(f->second->latex);
        if (latex.empty()) latex = gather(li);  // recognizer ran but empty: keep old fallback
        if (latex.empty()) continue;
        if (opts.drop_collapsed_formulas && latex_is_mode_collapsed(latex)) {
          if (!opts.collapsed_formula_note.empty())
            parts.push_back(opts.collapsed_formula_note);
          continue;
        }
        if (auto t = tag_of.find(li); t != tag_of.end())
          latex += " \\tag{" + t->second + "}";
        if (!opts.safe_formula_fallback || latex_is_render_safe(latex))
          parts.push_back("$$\n" + latex + "\n$$");
        else
          parts.push_back("```latex\n" + latex + "\n```");
      } else {
        std::string txt = gather(li);
        if (!txt.empty()) parts.push_back(std::move(txt));
      }
      continue;
    }
    if (cls == kClassInlineFormula) {
      if (auto f = formula_by_li.find(li); f != formula_by_li.end()) {
        std::string latex = trim(f->second->latex);
        if (latex.empty()) latex = gather(li);
        if (latex.empty()) continue;
        if (opts.drop_collapsed_formulas && latex_is_mode_collapsed(latex)) {
          if (!opts.collapsed_formula_note.empty())
            parts.push_back(opts.collapsed_formula_note);
          continue;
        }
        if (!opts.safe_formula_fallback || latex_is_render_safe(latex))
          parts.push_back("$" + latex + "$");
        else
          parts.push_back(inline_code(latex));
      } else {
        // Unrecognized inline formula: emit its text inline as plain prose
        // rather than as broken $…$ math.
        std::string txt = gather(li);
        if (!txt.empty()) parts.push_back(std::move(txt));
      }
      continue;
    }
    if (cls == kClassFormulaNumber) {
      // A formula_number is either folded into its display formula above (and
      // skipped via `consumed`) or orphaned because no display formula matched
      // / its host was dropped as garbage. Never emit a bare "(n)" paragraph —
      // standalone equation numbers are out-of-order reading-order noise.
      continue;
    }
    if (is_image_label(label)) {
      const int bid = cell.id >= 0 ? cell.id : li;
      MarkdownAsset a;
      a.layout_index = li;
      a.block_id = bid;
      a.kind = label;
      a.box = cell.box;
      a.rel_path = (opts.assets_dir.empty() ? std::string{}
                                            : opts.assets_dir + "/") +
                   "block" + std::to_string(bid) + ".png";
      const std::string caption = gather(li);
      const std::string src = resolver ? resolver(a) : a.rel_path;
      parts.push_back("![" + escape_md_link_text(caption) + "](" + src + ")");
      if (assets_out) assets_out->push_back(std::move(a));
      continue;
    }

    // Text-bearing regions.
    std::string text = gather(li);
    if (text.empty()) continue;
    if (length_gated(label) && codepoint_count(text) < opts.min_text_codepoints)
      continue;

    if (label == "doc_title") {
      parts.push_back("# " + text);
    } else if (label == "paragraph_title") {
      parts.push_back("## " + text);
    } else if (label == "figure_title") {
      parts.push_back("### " + text);
    } else if (label == "abstract") {
      parts.push_back("**Abstract** " + text);
    } else if (label == "algorithm") {
      parts.push_back("```\n" + text + "\n```");
    } else if (label == "reference" || label == "reference_content") {
      if (!refs_open) { parts.push_back("### References"); refs_open = true; }
      parts.push_back("- " + text);
    } else {
      // text, vertical_text, content, aside_text, footnote, vision_footnote,
      // SupplementaryRegion, and any unknown label → plain paragraph.
      parts.push_back(text);
    }
  }

  std::string md;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) md += "\n\n";
    md += parts[i];
  }
  return md.empty() ? md : md + "\n";
}

// ── OpenCV-backed helpers ────────────────────────────────────────────────

static cv::Mat crop_region(const cv::Mat &page, const turbo_ocr::Box &box) {
  if (page.empty()) return {};
  auto r = turbo_ocr::clamped_crop_rect(box, page.cols, page.rows);
  return page(cv::Rect(r[0], r[1], r[2], r[3])).clone();
}

int write_asset_crops(const cv::Mat &page,
                      const std::vector<MarkdownAsset> &assets,
                      const std::string &base_dir) {
  int n = 0;
  for (const auto &a : assets) {
    cv::Mat crop = crop_region(page, a.box);
    if (crop.empty()) continue;
    std::filesystem::path out = std::filesystem::path(base_dir) / a.rel_path;
    std::error_code ec;
    std::filesystem::create_directories(out.parent_path(), ec);
    if (cv::imwrite(out.string(), crop)) ++n;
  }
  return n;
}

std::string crop_to_png_data_uri(const cv::Mat &page,
                                 const turbo_ocr::Box &box) {
  cv::Mat crop = crop_region(page, box);
  if (crop.empty()) return {};
  std::vector<unsigned char> buf;
  if (!cv::imencode(".png", crop, buf) || buf.empty()) return {};
  return "data:image/png;base64," + base64_encode(buf.data(), buf.size());
}

std::string render_markdown_with_assets(const pipeline::OcrPipelineResult &res,
                                        const cv::Mat &page,
                                        const std::string &base_dir,
                                        bool embed_images,
                                        const MarkdownOptions &opts) {
  if (embed_images) {
    ImageSrcResolver resolver = [&](const MarkdownAsset &a) {
      std::string uri = crop_to_png_data_uri(page, a.box);
      return uri.empty() ? a.rel_path : uri;
    };
    return render_markdown(res, opts, nullptr, resolver);
  }
  std::vector<MarkdownAsset> assets;
  std::string md = render_markdown(res, opts, &assets);
  write_asset_crops(page, assets, base_dir);
  return md;
}

} // namespace turbo_ocr::output
