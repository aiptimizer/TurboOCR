#include "markdown_internal.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace turbo_ocr::markdown::mddetail {
namespace {

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

} // namespace

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

[[nodiscard]] bool latex_is_mode_collapsed(const std::string &latex) {
  if (latex.size() >= 1500) return true;          // absolute runaway
  const auto t = ws_tokens(latex);
  if (t.size() < 24) return false;                 // short formulas always kept
  for (int w = 5; w <= 8; ++w)
    if (max_window_count(t, w) >= 20) return true;  // long identical-unit repeat
  return max_consecutive_run(t) >= 8;              // back-to-back collapse
}

} // namespace turbo_ocr::markdown::mddetail
