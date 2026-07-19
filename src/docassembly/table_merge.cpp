#include "turbo_ocr/docassembly/table_merge.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <span>
#include <string_view>

namespace turbo_ocr::docassembly {

namespace {

struct Cell {
  int colspan = 1;
  int rowspan = 1;
  std::string text;  // inner text, tags stripped (owned; outlives parse views)
};
using Row = std::vector<Cell>;

// Case-insensitive match of `lit` (already lower-case) at position `i` of `s`.
bool ieq_at(std::string_view s, std::size_t i, std::string_view lit) {
  if (i + lit.size() > s.size()) return false;
  for (std::size_t k = 0; k < lit.size(); ++k)
    if (std::tolower(static_cast<unsigned char>(s[i + k])) != lit[k]) return false;
  return true;
}

// Case-insensitive find/rfind of a lower-case '<'-prefixed tag literal. The
// open-tag scans already match case-insensitively via ieq_at; close tags must
// too, or an upper-case `</TR>` from a VLM would end the row scan early and
// silently drop every remaining row.
std::size_t ifind(std::string_view s, std::string_view lit, std::size_t from) {
  if (lit.size() > s.size()) return std::string_view::npos;
  for (std::size_t i = from; i + lit.size() <= s.size(); ++i)
    if (s[i] == '<' && ieq_at(s, i, lit)) return i;
  return std::string_view::npos;
}
std::size_t irfind(std::string_view s, std::string_view lit) {
  if (lit.size() > s.size()) return std::string_view::npos;
  for (std::size_t i = s.size() - lit.size() + 1; i-- > 0;)
    if (s[i] == '<' && ieq_at(s, i, lit)) return i;
  return std::string_view::npos;
}

// Sane ceiling for colspan/rowspan. A degenerate model emission like
// colspan="9999999999" would otherwise overflow the int accumulate (UB) and
// then drive a multi-gigabyte blocked_until.resize — a one-cell OOM.
constexpr int kMaxSpan = 512;

int attr_int(std::string_view tag, std::string_view name, int def) {
  // Match `name` only as a whole attribute (preceded by '<', space, or a
  // quote), so `colspan` doesn't substring-match inside `data-colspanx` and
  // silently return the default span, skewing the merge column math.
  std::size_t p = std::string_view::npos;
  for (std::size_t i = tag.find(name); i != std::string_view::npos;
       i = tag.find(name, i + 1)) {
    if (i == 0 || tag[i - 1] == '<' || tag[i - 1] == ' ' ||
        tag[i - 1] == '\t' || tag[i - 1] == '"' || tag[i - 1] == '\'') {
      p = i;
      break;
    }
  }
  if (p == std::string_view::npos) return def;
  p += name.size();
  while (p < tag.size() &&
         (tag[p] == ' ' || tag[p] == '=' || tag[p] == '"' || tag[p] == '\''))
    ++p;
  int val = 0;
  bool any = false;
  while (p < tag.size() && std::isdigit(static_cast<unsigned char>(tag[p]))) {
    if (val < kMaxSpan)  // overflow-safe: stop growing once past the cap
      val = val * 10 + (tag[p] - '0');
    ++p;
    any = true;
  }
  if (!any) return def;
  // Clamp: 0 would occupy no grid column (skewing the merge column math) and
  // HTML treats non-positive spans as 1 anyway.
  return std::clamp(val, 1, kMaxSpan);
}

std::string strip_tags(std::string_view html) {
  std::string out;
  out.reserve(html.size());
  bool in_tag = false;
  for (char c : html) {
    if (c == '<') in_tag = true;
    else if (c == '>') in_tag = false;
    else if (!in_tag) out.push_back(c);
  }
  return out;
}

// Non-owning "<tr>...</tr>" spans (top-level rows). Views alias `html`, which
// outlives every use here and in perform_table_merge / parse_rows.
std::vector<std::string_view> extract_tr_raw(std::string_view html) {
  std::vector<std::string_view> rows;
  std::size_t i = 0;
  while (i < html.size()) {
    std::size_t lt = html.find('<', i);
    if (lt == std::string_view::npos) break;
    i = lt;
    if (ieq_at(html, i, "<tr")) {
      std::size_t close = ifind(html, "</tr>", i);
      if (close == std::string_view::npos) break;
      std::size_t end = close + 5;
      rows.push_back(html.substr(i, end - i));
      i = end;
    } else {
      i = lt + 1;
    }
  }
  return rows;
}

Row extract_cells(std::string_view tr_html) {
  Row cells;
  std::size_t i = 0;
  while (i < tr_html.size()) {
    std::size_t lt = tr_html.find('<', i);
    if (lt == std::string_view::npos) break;
    i = lt;
    bool td = ieq_at(tr_html, i, "<td");
    bool th = ieq_at(tr_html, i, "<th");
    if (td || th) {
      std::size_t tag_end = tr_html.find('>', i);
      if (tag_end == std::string_view::npos) break;
      std::string_view open_tag = tr_html.substr(i, tag_end - i);
      std::string_view close_lit = td ? "</td>" : "</th>";
      std::size_t close = ifind(tr_html, close_lit, tag_end);
      std::size_t inner_end = (close == std::string_view::npos) ? tr_html.size() : close;
      std::string_view inner = tr_html.substr(tag_end + 1, inner_end - tag_end - 1);
      Cell c;
      c.colspan = attr_int(open_tag, "colspan", 1);
      c.rowspan = attr_int(open_tag, "rowspan", 1);
      c.text = strip_tags(inner);
      cells.push_back(std::move(c));
      i = (close == std::string_view::npos) ? tr_html.size() : close + 5;
    } else {
      i = lt + 1;
    }
  }
  return cells;
}

std::vector<Row> parse_rows(std::string_view html) {
  auto raw = extract_tr_raw(html);
  std::vector<Row> rows;
  rows.reserve(raw.size());
  for (std::string_view tr : raw) rows.push_back(extract_cells(tr));
  return rows;
}

// Grid width accounting for colspan/rowspan. `blocked_until[c]` is the first row
// index at which column c is free again — a single vector, no per-cell sets.
//
// blocked_until grows to the running SUM of colspans, so many cells each near
// the per-cell kMaxSpan cap would otherwise amplify into a multi-GB resize on
// adversarial model HTML (the per-cell clamp bounds one span, not the total).
// Cap the grid width — a real table never approaches it — mirroring the OTSL
// path's kMaxTableCells guard; past it, stop widening and return the cap.
int calc_total_columns(const std::vector<Row>& rows) {
  constexpr int kMaxGridColumns = 4096;
  int max_cols = 0;
  std::vector<int> blocked_until;
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const int row = static_cast<int>(r);
    int col = 0;
    for (const Cell& cell : rows[r]) {
      while (col < static_cast<int>(blocked_until.size()) && blocked_until[col] > row)
        ++col;
      const int end = std::min(col + cell.colspan, kMaxGridColumns);
      if (end > static_cast<int>(blocked_until.size())) blocked_until.resize(end, 0);
      const int free_row = row + cell.rowspan;
      for (int c = col; c < end; ++c) blocked_until[c] = free_row;
      col = end;
      max_cols = std::max(max_cols, col);
      if (max_cols >= kMaxGridColumns) return kMaxGridColumns;
    }
  }
  return max_cols;
}

int calc_row_columns(const Row& row) {
  int s = 0;
  for (const Cell& c : row) s += c.colspan;
  return s;
}

int calc_visual_columns(const Row& row) { return static_cast<int>(row.size()); }

std::string norm_text(const std::string& s) {
  std::string fh = full_to_half(s);
  std::string out;
  out.reserve(fh.size());
  for (char c : fh)
    if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
  return out;
}

// Count leading rows that are byte-identical (after normalization) across both
// tables — the repeated header of a continued table.
int count_header_rows(const std::vector<Row>& r1, const std::vector<Row>& r2,
                      int max_header_rows = 5) {
  const int min_rows = std::min({static_cast<int>(r1.size()),
                                 static_cast<int>(r2.size()), max_header_rows});
  int header_rows = 0;
  for (int i = 0; i < min_rows; ++i) {
    if (r1[i].size() != r2[i].size()) break;
    bool match = true;
    for (std::size_t k = 0; k < r1[i].size(); ++k) {
      if (r1[i][k].colspan != r2[i][k].colspan ||
          norm_text(r1[i][k].text) != norm_text(r2[i][k].text)) {
        match = false;
        break;
      }
    }
    if (!match) break;
    ++header_rows;
  }
  return header_rows;
}

bool check_rows_match(const std::vector<Row>& r1, const std::vector<Row>& r2) {
  if (r1.empty() || r2.empty()) return false;
  const int header_count = count_header_rows(r1, r2);
  if (static_cast<int>(r2.size()) <= header_count) return false;
  const Row& last_row = r1.back();
  const Row& first_data_row = r2[header_count];
  return calc_row_columns(last_row) == calc_row_columns(first_data_row) ||
         calc_visual_columns(last_row) == calc_visual_columns(first_data_row);
}

bool label_in(std::string_view label, std::span<const std::string_view> set) {
  return std::find(set.begin(), set.end(), label) != set.end();
}

bool is_skippable(const ParsingBlock& block,
                  std::span<const std::string_view> allowed_labels) {
  if (label_in(block.label, allowed_labels)) return true;
  // PaddleX inspects text/title attrs; our block carries text in `content`.
  static constexpr std::string_view kw[] = {"continue", "continued", "cont'd",
                                             "\xe7\xbb\xad" /*续*/};
  std::string lower;
  lower.reserve(block.content.size());
  for (char c : block.content)
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  for (std::string_view k : kw)
    if (lower.find(k) != std::string::npos) return true;
  return false;
}

int box_width(const ParsingBlock& b) { return b.bbox[2] - b.bbox[0]; }

bool can_merge_tables(const std::vector<ParsingBlock>& prev_page, std::size_t prev_idx,
                      const std::vector<ParsingBlock>& curr_page, std::size_t curr_idx) {
  const ParsingBlock& prev = prev_page[prev_idx];
  const ParsingBlock& curr = curr_page[curr_idx];
  const int pw = box_width(prev), cw = box_width(curr);
  if (pw == 0 || cw == 0) return false;
  if (static_cast<double>(std::abs(cw - pw)) / std::min(cw, pw) >= 0.1) return false;

  static constexpr std::string_view follow_ok[] = {
      "footer", "vision_footnote", "number", "footnote", "footer_image", "seal"};
  for (std::size_t i = prev_idx + 1; i < prev_page.size(); ++i)
    if (!label_in(prev_page[i].label, follow_ok)) return false;

  static constexpr std::string_view before_ok[] = {"header", "header_image",
                                                   "number", "seal"};
  for (std::size_t i = 0; i < curr_idx; ++i)
    if (!is_skippable(curr_page[i], before_ok)) return false;

  if (prev.content.empty() || curr.content.empty()) return false;
  const auto rows_prev = parse_rows(prev.content);
  const auto rows_curr = parse_rows(curr.content);
  const bool tables_match =
      calc_total_columns(rows_prev) == calc_total_columns(rows_curr);
  return tables_match || check_rows_match(rows_prev, rows_curr);
}

std::string perform_table_merge(const std::string& prev_html,
                                const std::string& curr_html) {
  const auto rows_prev = parse_rows(prev_html);
  const auto rows_curr = parse_rows(curr_html);
  const int header_count = count_header_rows(rows_prev, rows_curr);
  const auto curr_raw = extract_tr_raw(curr_html);

  std::size_t appended_len = 0;
  for (std::size_t i = static_cast<std::size_t>(header_count); i < curr_raw.size(); ++i)
    appended_len += curr_raw[i].size();
  if (appended_len == 0) return prev_html;

  std::string appended;
  appended.reserve(appended_len);
  for (std::size_t i = static_cast<std::size_t>(header_count); i < curr_raw.size(); ++i)
    appended.append(curr_raw[i]);

  std::size_t insert_at = irfind(prev_html, "</tr>");
  insert_at = (insert_at != std::string::npos) ? insert_at + 5
                                               : ifind(prev_html, "</table>", 0);
  if (insert_at == std::string::npos) return prev_html + appended;

  std::string out;
  out.reserve(prev_html.size() + appended.size());
  out.append(prev_html, 0, insert_at);
  out.append(appended);
  out.append(prev_html, insert_at, std::string::npos);
  return out;
}

} // namespace

std::string full_to_half(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  std::size_t i = 0;
  while (i < text.size()) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if ((c >> 4) == 0xE && i + 2 < text.size()) {
      std::uint32_t cp = ((c & 0x0Fu) << 12) |
                         ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 6) |
                         (static_cast<unsigned char>(text[i + 2]) & 0x3Fu);
      if (cp >= 0xFF01 && cp <= 0xFF5E) {
        out.push_back(static_cast<char>(cp - 0xFEE0));
        i += 3;
        continue;
      }
      out.append(text, i, 3);
      i += 3;
    } else {
      out.push_back(text[i]);
      ++i;
    }
  }
  return out;
}

void merge_tables_across_pages(std::vector<std::vector<ParsingBlock>>& pages) {
  // Assign flattened-order ids (restructure_pages does this before merge).
  int gid = 0;
  std::vector<ParsingBlock*> all_blocks;
  for (auto& page : pages)
    for (auto& b : page) {
      b.global_block_id = gid;
      b.global_group_id = gid;
      all_blocks.push_back(&b);
      ++gid;
    }

  for (int i = static_cast<int>(pages.size()) - 1; i > 0; --i) {
    auto& page_curr = pages[i];
    auto& page_prev = pages[i - 1];

    int curr_idx = -1;
    for (std::size_t j = 0; j < page_curr.size(); ++j)
      if (page_curr[j].label == "table") { curr_idx = static_cast<int>(j); break; }
    int prev_idx = -1;
    for (int j = static_cast<int>(page_prev.size()) - 1; j >= 0; --j)
      if (page_prev[j].label == "table") { prev_idx = j; break; }

    if (curr_idx < 0 || prev_idx < 0) continue;
    if (!can_merge_tables(page_prev, prev_idx, page_curr, curr_idx)) continue;

    ParsingBlock& prev_block = page_prev[prev_idx];
    ParsingBlock& curr_block = page_curr[curr_idx];
    prev_block.content = perform_table_merge(prev_block.content, curr_block.content);
    curr_block.content.clear();
    curr_block.global_group_id = prev_block.global_block_id;
  }

  // Resolve chains: a block folded into another inherits that host's group.
  // Hosts always precede their followers in id order, so a single hop over the
  // already-resolved host fully compresses the chain.
  for (auto& page : pages)
    for (auto& b : page)
      if (b.global_block_id != b.global_group_id)
        b.global_group_id = all_blocks[b.global_group_id]->global_group_id;
}

} // namespace turbo_ocr::docassembly
