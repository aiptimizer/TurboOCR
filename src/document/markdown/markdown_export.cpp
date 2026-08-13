#include "turbo_ocr/analysis/table/html_reconstruct.h"
#include "turbo_ocr/document/markdown_export.h"

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
#include <utility>
#include <vector>

#include "markdown_internal.h"
#include "turbo_ocr/base/geometry/box.h"
#include "turbo_ocr/core/types.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/core/router_types.h"

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

// Single definition of the document's block separator: blocks are joined by a
// blank line and a non-empty document ends with exactly one newline. Both the
// layout-less fast path and the region walk finish here so they can never
// drift apart.
[[nodiscard]] std::string join_blocks(const std::vector<std::string> &parts) {
  std::string md;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) md += "\n\n";
    md += parts[i];
  }
  return md.empty() ? md : md + "\n";
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
//
// The result is spliced straight into the LaTeX stream, and it comes from OCR of
// a scanned equation number — so it is rejected outright if it contains any
// character that is syntax in that stream. An equation number legitimately never
// does, and letting one through was a real hazard: a mis-OCR'd "1}" unbalances
// the braces and demotes the whole (otherwise perfectly safe) equation to a code
// fence, while a bare '$' terminates the $$…$$ block and corrupts the rest of
// the document (latex_is_render_safe does not inspect bare '$').
[[nodiscard]] std::string clean_tag(std::string s) {
  s = trim(s);
  if (s.size() >= 2 && s.front() == '(' && s.back() == ')')
    s = trim(s.substr(1, s.size() - 2));
  if (s.find_first_of("{}$\\%&#^_~") != std::string::npos) return {};
  return s;
}

// Class-id constants, PINNED HERE against the shared label table. The previous
// comment claimed layout_types.h's static_asserts covered these: they do not —
// those live in the same file as the array they check, so renumbering the
// classes means editing the array and its asserts together while these three
// literals drift silently. A shift would route display formulas into emit_text
// (raw LaTeX as prose) and formula_numbers into the paragraph path this file
// explicitly refuses. Zero runtime cost; a mis-render becomes a compile error.
constexpr int kClassDisplayFormula = 5;
constexpr int kClassFormulaNumber  = 11;
constexpr int kClassInlineFormula  = 15;
static_assert(layout::kLayoutLabels[kClassDisplayFormula] == "display_formula");
static_assert(layout::kLayoutLabels[kClassFormulaNumber] == "formula_number");
static_assert(layout::kLayoutLabels[kClassInlineFormula] == "inline_formula");

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

// ── phase data ───────────────────────────────────────────────────────────
//
// render_markdown is a five-phase pipeline over one page:
//   1. index   — rect / rank / region-membership assignment  (build_page_index)
//   2. payload — layout_id → table / formula result maps     (collect_payloads)
//   3. fold    — formula_number regions → \tag{…}            (fold_formula_numbers)
//   4. order   — priority buckets, then column-aware reorder (bucket_order /
//                                                             apply_column_order)
//   5. emit    — one Markdown block per surviving region     (emit_region)
// Each phase reads only what the structs below hand it, so a change to (say)
// ordering cannot silently perturb the emission policy.

// Geometry + ordering facts about the page, computed once in phase 1.
struct PageIndex {
  std::vector<std::array<int, 4>> lr;      // axis-aligned rect per layout cell
  std::vector<int>                rank;    // total order over results
  std::vector<std::vector<int>>   members; // layout li → result indices, ranked
};

// Per-region structure payloads, keyed by layout index (== their layout_id).
struct StructurePayloads {
  std::unordered_map<int, const router::TableResult *>   table_by_li;
  std::unordered_map<int, const router::FormulaResult *> formula_by_li;
};

// Result of the formula-number fold: which layout cells were absorbed, and the
// tag text each display formula inherited.
struct FormulaTags {
  std::vector<char>                    consumed; // layout li → folded away
  std::unordered_map<int, std::string> tag_of;   // display-formula li → tag
};

// Everything the emission phase may read. Bundled so the per-region emitters
// take one const ref instead of eight parallel parameters.
struct EmitContext {
  const pipeline::OcrPipelineResult &res;
  const MarkdownOptions             &opts;
  const PageIndex                   &index;
  const StructurePayloads           &payloads;
  const FormulaTags                 &tags;
  const ImageSrcResolver            &resolver;
  std::vector<MarkdownAsset>        *assets_out;
};

// The only state the emitters mutate.
struct EmitState {
  std::vector<std::string> parts;
  bool refs_open = false; // "### References" heading already emitted
};

// ── phase 1: region / rank assignment ────────────────────────────────────

[[nodiscard]] PageIndex build_page_index(const pipeline::OcrPipelineResult &res) {
  const auto &layout = res.layout;
  const auto &results = res.results;
  const size_t nL = layout.size();
  const size_t nR = results.size();

  PageIndex ix;
  ix.lr.resize(nL);
  for (size_t i = 0; i < nL; ++i) ix.lr[i] = turbo_ocr::aabb(layout[i].box);

  auto li_of = [&](const OCRResultItem &it) -> int {
    if (it.layout_id >= 0 && static_cast<size_t>(it.layout_id) < nL)
      return it.layout_id;
    for (size_t j = 0; j < nL; ++j)
      if (turbo_ocr::centroid_in_aabb(it.box, ix.lr[j]))
        return static_cast<int>(j);
    return -1;
  };

  // Rank every result: reading_order first (text XY-cut), then any leftover by
  // geometry, so each result has a total order even if reading_order is partial.
  ix.rank.assign(nR, -1);
  int next_rank = 0;
  for (int ri : res.reading_order)
    if (ri >= 0 && static_cast<size_t>(ri) < nR && ix.rank[ri] < 0)
      ix.rank[ri] = next_rank++;
  std::vector<int> leftover;
  for (size_t ri = 0; ri < nR; ++ri)
    if (ix.rank[ri] < 0) leftover.push_back(static_cast<int>(ri));
  std::sort(leftover.begin(), leftover.end(), [&](int a, int b) {
    auto ra = turbo_ocr::aabb(results[a].box), rb = turbo_ocr::aabb(results[b].box);
    return ra[1] != rb[1] ? ra[1] < rb[1] : ra[0] < rb[0];
  });
  for (int ri : leftover) ix.rank[ri] = next_rank++;

  // Group results by region (members carry result indices, kept in rank order).
  ix.members.assign(nL, {});
  for (size_t ri = 0; ri < nR; ++ri) {
    int li = li_of(results[ri]);
    if (li >= 0) ix.members[li].push_back(static_cast<int>(ri));
  }
  for (auto &m : ix.members)
    std::sort(m.begin(), m.end(),
              [&](int a, int b) { return ix.rank[a] < ix.rank[b]; });
  return ix;
}

// ── phase 2: structure payload lookup ────────────────────────────────────

[[nodiscard]] StructurePayloads
collect_payloads(const pipeline::OcrPipelineResult &res) {
  const size_t nL = res.layout.size();
  StructurePayloads p;
  for (const auto &t : res.tables)
    if (t.layout_id >= 0 && static_cast<size_t>(t.layout_id) < nL)
      p.table_by_li[t.layout_id] = &t;
  for (const auto &f : res.formulas)
    if (f.layout_id >= 0 && static_cast<size_t>(f.layout_id) < nL)
      p.formula_by_li[f.layout_id] = &f;
  return p;
}

// ── phase 3: formula-number folding ──────────────────────────────────────

// Fold formula_number regions into the nearest display_formula on the same
// vertical band; consumed numbers are not emitted as stray paragraphs.
[[nodiscard]] FormulaTags
fold_formula_numbers(const pipeline::OcrPipelineResult &res,
                     const MarkdownOptions &opts, const PageIndex &ix) {
  const auto &layout = res.layout;
  const auto &lr = ix.lr;
  const size_t nL = layout.size();

  FormulaTags ft;
  ft.consumed.assign(nL, 0);
  if (!opts.fold_formula_numbers) return ft;

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
    for (int ri : ix.members[ni])
      num += (num.empty() ? "" : " ") + trim(res.results[ri].text);
    num = clean_tag(num);
    if (!num.empty()) { ft.tag_of[best] = num; ft.consumed[ni] = 1; }
  }
  return ft;
}

// ── phase 4: emission order ──────────────────────────────────────────────

// Center of a region, for placing structure-only cells (no OCR line) inline.
[[nodiscard]] double center_rank(const pipeline::OcrPipelineResult &res,
                                 const PageIndex &ix, size_t li) {
  const auto &lr = ix.lr;
  const size_t nR = res.results.size();
  if (!ix.members[li].empty()) return ix.rank[ix.members[li].front()];
  if (nR == 0) return 1e9 + lr[li][1]; // pure-geometric page
  const double cx = (lr[li][0] + lr[li][2]) * 0.5;
  const double cy = (lr[li][1] + lr[li][3]) * 0.5;
  int best = -1; double best_d = 1e30;
  for (size_t ri = 0; ri < nR; ++ri) {
    auto rb = turbo_ocr::aabb(res.results[ri].box);
    const double rx = (rb[0] + rb[2]) * 0.5, ry = (rb[1] + rb[3]) * 0.5;
    const double d = (rx - cx) * (rx - cx) + (ry - cy) * (ry - cy);
    if (d < best_d) { best_d = d; best = static_cast<int>(ri); }
  }
  return best >= 0 ? ix.rank[best] + 0.5 : 1e9 + lr[li][1];
}

// Emit order: class-priority bucket (header/body/footer), then inline rank.
[[nodiscard]] std::vector<int> bucket_order(const pipeline::OcrPipelineResult &res,
                                            const PageIndex &ix) {
  const auto &layout = res.layout;
  const auto &lr = ix.lr;
  const size_t nL = layout.size();

  std::vector<int> order(nL);
  for (size_t i = 0; i < nL; ++i) order[i] = static_cast<int>(i);
  std::vector<double> key(nL);
  for (size_t i = 0; i < nL; ++i) key[i] = center_rank(res, ix, i);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    const int ba = layout::reading_priority_bucket(layout[a].class_id);
    const int bb = layout::reading_priority_bucket(layout[b].class_id);
    if (ba != bb) return ba < bb;
    if (key[a] != key[b]) return key[a] < key[b];
    if (lr[a][1] != lr[b][1]) return lr[a][1] < lr[b][1];
    return lr[a][0] < lr[b][0];
  });
  return order;
}

// Column-aware re-ordering of the BODY bucket (Markdown view only). `order`
// is bucket-major (TOP, then BODY, then BOTTOM are each contiguous), so the
// body cells form one contiguous run we can reorder in place without
// disturbing header/footer placement. A clean multi-column page is emitted
// column-major; anything ambiguous keeps the reading_order sequence.
void apply_column_order(const pipeline::OcrPipelineResult &res,
                        const PageIndex &ix, std::vector<int> &order) {
  const auto &layout = res.layout;
  size_t blo = 0;
  while (blo < order.size() &&
         layout::reading_priority_bucket(layout[order[blo]].class_id) != 1)
    ++blo;
  size_t bhi = blo;
  while (bhi < order.size() &&
         layout::reading_priority_bucket(layout[order[bhi]].class_id) == 1)
    ++bhi;
  if (bhi - blo < static_cast<size_t>(kColMinBodyBlocks)) return;

  std::vector<std::array<int, 4>> rects;
  rects.reserve(bhi - blo);
  for (size_t i = blo; i < bhi; ++i) rects.push_back(ix.lr[order[i]]);
  auto perm = column_major_order(rects);
  if (!perm) return;
  std::vector<int> reordered;
  reordered.reserve(perm->size());
  for (int p : *perm) reordered.push_back(order[blo + p]);
  std::copy(reordered.begin(), reordered.end(), order.begin() + blo);
}

// ── phase 5: per-region emission ─────────────────────────────────────────

// Visible text of a region: its member OCR lines in rank order, space-joined.
[[nodiscard]] std::string gather(const EmitContext &ctx, int li) {
  std::vector<std::string> lines;
  for (int ri : ctx.index.members[li]) lines.push_back(ctx.res.results[ri].text);
  return join_lines(lines);
}

// The fallback shared by every structure branch whose recognizer produced
// nothing: emit the region's text as escaped prose, or nothing at all when it
// is empty. Escaping here matters as much as in emit_text — these branches
// carry the same untrusted OCR / PDF-text-layer bytes, just from regions the
// table/formula recognizers declined.
void push_text_fallback(EmitState &st, const std::string &text) {
  if (!text.empty()) st.parts.push_back(mddetail::escape_md_text(text));
}

enum class FormulaKind { Display, Inline };

[[nodiscard]] bool renders_safely(const MarkdownOptions &opts,
                                  const std::string &latex) {
  return !opts.safe_formula_fallback || latex_is_render_safe(latex);
}

// Recognized HTML when the table backend ran; otherwise (backend off, e.g.
// geometric PDF mode) fall back to the region's raw text so the table's
// content is not silently dropped — it just lacks grid structure.
void emit_table(const EmitContext &ctx, int li, EmitState &st) {
  const auto &m = ctx.payloads.table_by_li;
  if (auto it = m.find(li); it != m.end()) {
    std::string html = strip_table_wrapper(it->second->html);
    if (!html.empty()) { st.parts.push_back(std::move(html)); return; }
  }
  push_text_fallback(st, gather(ctx, li));
}

// Display and inline formulas share ONE policy — recognizer lookup, raw-text
// fallback, mode-collapse drop, render-safety fallback — and differ only in
// the delimiters (and in the \tag{…} a display formula may have inherited from
// a folded formula_number). Keeping the policy in one place is deliberate: the
// two branches drifted apart historically, and only the delimiters are a real
// difference of kind.
void emit_formula(const EmitContext &ctx, int li, FormulaKind kind,
                  EmitState &st) {
  const auto &opts = ctx.opts;
  const auto &m = ctx.payloads.formula_by_li;
  auto f = m.find(li);
  if (f == m.end()) {
    // Only content from the formula recognizer is LaTeX. With no recognized
    // result (formula backend off, e.g. geometric PDF mode, or the region
    // failed to recognize), gather(li) returns the region's raw OCR /
    // PDF-text-layer characters — NOT LaTeX. Wrapping those in $$…$$ (display)
    // or $…$ (inline) makes KaTeX/MathJax render broken math, so BOTH kinds
    // emit them as plain prose.
    push_text_fallback(st, gather(ctx, li));
    return;
  }
  std::string latex = trim(f->second->latex);
  if (latex.empty()) latex = gather(ctx, li); // recognizer ran but empty: keep old fallback
  if (latex.empty()) return;
  if (opts.drop_collapsed_formulas && latex_is_mode_collapsed(latex)) {
    if (!opts.collapsed_formula_note.empty())
      st.parts.push_back(opts.collapsed_formula_note);
    return;
  }
  if (kind == FormulaKind::Display) {
    // A BAD TAG COSTS ONLY THE TAG. Safety is decided on the UNTAGGED latex
    // first; the tag is then appended only if the tagged string is also safe.
    // Appending unconditionally (as this did) let one mis-OCR'd equation number
    // demote an otherwise perfectly renderable equation to a fenced listing.
    // clean_tag already refuses anything containing LaTeX syntax, so reaching
    // the else here means the equation itself was the problem.
    const bool safe = renders_safely(opts, latex);
    if (auto t = ctx.tags.tag_of.find(li); t != ctx.tags.tag_of.end()) {
      const std::string tagged = latex + " \\tag{" + t->second + "}";
      if (!safe || renders_safely(opts, tagged)) latex = tagged;
    }
    st.parts.push_back(renders_safely(opts, latex)
                           ? "$$\n" + latex + "\n$$"
                           // Fence sized past any embedded backtick run — the
                           // unsafe-latex fallback is exactly the case where
                           // the content is untrusted (see fenced_block).
                           : mddetail::fenced_block(latex, "latex"));
  } else {
    st.parts.push_back(renders_safely(opts, latex) ? "$" + latex + "$"
                                                   : inline_code(latex));
  }
}

void emit_image(const EmitContext &ctx, int li, const std::string &label,
                EmitState &st) {
  const auto &cell = ctx.res.layout[li];
  const int bid = cell.id >= 0 ? cell.id : li;
  MarkdownAsset a;
  a.layout_index = li;
  a.block_id = bid;
  a.kind = label;
  a.box = cell.box;
  a.rel_path = (ctx.opts.assets_dir.empty() ? std::string{}
                                            : ctx.opts.assets_dir + "/") +
               "block" + std::to_string(bid) + ".png";
  const std::string caption = gather(ctx, li);
  const std::string src = ctx.resolver ? ctx.resolver(a) : a.rel_path;
  st.parts.push_back("![" + escape_md_link_text(caption) + "](" + src + ")");
  if (ctx.assets_out) ctx.assets_out->push_back(std::move(a));
}

void emit_text(const EmitContext &ctx, int li, const std::string &label,
               EmitState &st) {
  std::string raw = gather(ctx, li);
  if (raw.empty()) return;
  if (length_gated(label) &&
      codepoint_count(raw) < ctx.opts.min_text_codepoints)
    return;

  // UNTRUSTED CONTENT GATE. This function emits the majority of a document's
  // text. It was gated first, but it was NOT the only ungated path: the
  // structure fallbacks (push_text_fallback, reached from emit_table and
  // emit_formula whenever those backends are off) and the no-layout path
  // (render_lines_only) carry the same untrusted bytes and were escaped
  // afterwards. Every text-bearing emit path now funnels through
  // escape_md_text — keep it that way when adding one. Captions go through
  // escape_md_link_text (emit_image), table HTML through sanitize_table_html.
  // In mode=auto the PDF text layer arrives byte-exact, so an unescaped
  // `<img onerror=...>` here was stored XSS across all six
  // markdown-producing call sites. The algorithm branch
  // keeps the RAW text — it is fenced, and the fence is sized past any
  // embedded backtick run so the payload cannot break out of it.
  if (label == "algorithm") {
    st.parts.push_back(mddetail::fenced_block(raw, ""));
    return;
  }
  const std::string text = mddetail::escape_md_text(raw);

  if (label == "doc_title") {
    st.parts.push_back("# " + text);
  } else if (label == "paragraph_title") {
    st.parts.push_back("## " + text);
  } else if (label == "figure_title") {
    st.parts.push_back("### " + text);
  } else if (label == "abstract") {
    st.parts.push_back("**Abstract** " + text);
  } else if (label == "reference" || label == "reference_content") {
    if (!st.refs_open) { st.parts.push_back("### References"); st.refs_open = true; }
    st.parts.push_back("- " + text);
  } else {
    // text, vertical_text, content, aside_text, footnote, vision_footnote,
    // SupplementaryRegion, and any unknown label → plain paragraph.
    st.parts.push_back(text);
  }
}

// Structure-first dispatch: table / formula / image regions own their payload
// and are NOT rendered from gathered text; everything else is text-bearing.
void emit_region(const EmitContext &ctx, int li, EmitState &st) {
  if (ctx.tags.consumed[li]) return;
  const auto &cell = ctx.res.layout[li];
  const std::string label(layout::label_name(cell.class_id));
  if (ctx.opts.ignore_labels.count(label)) return;

  const int cls = cell.class_id;
  if (label == "table") return emit_table(ctx, li, st);
  if (cls == kClassDisplayFormula)
    return emit_formula(ctx, li, FormulaKind::Display, st);
  if (cls == kClassInlineFormula)
    return emit_formula(ctx, li, FormulaKind::Inline, st);
  if (cls == kClassFormulaNumber) {
    // A formula_number is either folded into its display formula (and skipped
    // via `consumed`) or orphaned because no display formula matched / its
    // host was dropped as garbage. Never emit a bare "(n)" paragraph —
    // standalone equation numbers are out-of-order reading-order noise.
    return;
  }
  if (is_image_label(label)) return emit_image(ctx, li, label, st);
  emit_text(ctx, li, label, st);
}

// No layout: emit text lines in their native order as one paragraph each.
// Escaped like every other prose path — this is the DEFAULT branch for plain
// OCR (no layout model), so it carries the same untrusted bytes as emit_text.
[[nodiscard]] std::string render_lines_only(const pipeline::OcrPipelineResult &res,
                                            const MarkdownOptions &opts) {
  std::vector<std::string> parts;
  for (const auto &it : res.results) {
    std::string t = trim(it.text);
    if (codepoint_count(t) >= opts.min_text_codepoints)
      parts.push_back(mddetail::escape_md_text(t));
  }
  return join_blocks(parts);
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
  if (res.layout.empty()) return render_lines_only(res, opts);

  const PageIndex index = build_page_index(res);
  const StructurePayloads payloads = collect_payloads(res);
  const FormulaTags tags = fold_formula_numbers(res, opts, index);

  std::vector<int> order = bucket_order(res, index);
  if (opts.column_aware_order) apply_column_order(res, index, order);

  const EmitContext ctx{res, opts, index, payloads, tags, resolver, assets_out};
  EmitState st;
  for (int li : order) emit_region(ctx, li, st);
  return join_blocks(st.parts);
}

} // namespace turbo_ocr::markdown
