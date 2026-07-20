#include "turbo_ocr/table/html_reconstruct.h"
#include "turbo_ocr/markdown/markdown_export.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "markdown_internal.h"
#include "turbo_ocr/common/geometry/box.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/router/router_types.h"

namespace turbo_ocr::markdown {
namespace {

using mddetail::column_major_order;
using mddetail::inline_code;
using mddetail::kColMinBodyBlocks;
using mddetail::latex_is_mode_collapsed;
using mddetail::latex_is_render_safe;
using mddetail::trim;

// ── small string utils ──────────────────────────────────────────────────

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
  // Single choke point for every backend's table HTML entering the Markdown
  // document — sanitize here so no <script>/on*= can survive into rendered
  // output regardless of which recognizer produced it (defense in depth;
  // SLANeXt already escapes cell text, the VLM passthrough is also sanitized
  // at source). Bare `<table>…</table>`, colspan/rowspan kept.
  const std::string inner =
      trim(std::string_view(html).substr(a, (b + 8) - a)); // 8 = "</table>"
  return turbo_ocr::table::sanitize_table_html(inner);
}

// Strip surrounding parens/whitespace from a detected formula number so
// \tag{N} renders "(N)" rather than "((N))".
[[nodiscard]] std::string clean_tag(std::string s) {
  s = trim(s);
  if (s.size() >= 2 && s.front() == '(' && s.back() == ')')
    s = trim(s.substr(1, s.size() - 2));
  return s;
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
    for (size_t j = 0; j < nL; ++j)
      if (turbo_ocr::centroid_in_aabb(it.box, lr[j]))
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

} // namespace turbo_ocr::markdown
