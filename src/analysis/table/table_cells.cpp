#include "turbo_ocr/analysis/table/table_cells.h"

#include <algorithm>
#include <cstddef>
#include <string_view>

#include "turbo_ocr/analysis/table/html_reconstruct.h"  // td_fragment_text

namespace turbo_ocr::table {

namespace {

// The bundled dict tops out at colspan="25" / rowspan="20", so a span beyond
// this cannot come from a well-formed stream; clamping keeps a corrupt token
// from sizing the occupancy map.
constexpr int kMaxSpan = 64;

// Parse the integer out of a span attribute token — the dict emits them
// verbatim as ` colspan="3"`, so the digits follow the first quote. Returns 1
// for anything unparseable, which is the no-span default.
int parse_span(std::string_view tok) noexcept {
  const std::size_t q = tok.find('"');
  if (q == std::string_view::npos) return 1;
  int v = 0;
  for (std::size_t i = q + 1; i < tok.size() && tok[i] >= '0' && tok[i] <= '9'; ++i)
    v = v * 10 + (tok[i] - '0');
  return std::clamp(v, 1, kMaxSpan);
}

// Join the fragments matched into one cell, trimmed and single-space separated —
// the plain-text twin of the <td> body reconstruct_html emits.
std::string join_fragments(const std::vector<std::size_t>& idx,
                           const std::vector<std::string>& ocr_texts) {
  std::string out;
  for (const std::size_t j : idx) {
    if (j >= ocr_texts.size()) continue;
    const std::string_view frag = td_fragment_text(ocr_texts[j]);
    if (frag.empty()) continue;
    if (!out.empty() && out.back() != ' ') out.push_back(' ');
    out += frag;
  }
  return out;
}

} // namespace

std::vector<router::TableCell> build_table_cells(
    const std::vector<std::string>& structure,
    const std::vector<std::array<int, 8>>& quads,
    const std::vector<MatchedCell>& matched,
    const std::vector<std::string>& ocr_texts) {
  std::vector<router::TableCell> out;
  out.reserve(quads.size());
  for (std::size_t i = 0; i < quads.size(); ++i) {
    const auto& q = quads[i];
    router::TableCell c;
    c.box.pts = {{{q[0], q[1]}, {q[2], q[3]}, {q[4], q[5]}, {q[6], q[7]}}};
    if (i < matched.size()) c.text = join_fragments(matched[i].ocr_indices, ocr_texts);
    out.push_back(std::move(c));
  }

  // HTML table grid walk. free_at[c] is the first row index at which column c is
  // free again, so a rowspan from an earlier row keeps later rows off it — the
  // same occupancy rule a browser applies, which is what makes row/col derived
  // rather than assumed.
  std::vector<int> free_at;
  int         cur_row = -1;   // no <tr> seen yet
  std::size_t cursor  = 0;    // next candidate column in the current row

  auto place = [&](std::size_t slot, int rowspan, int colspan) {
    if (cur_row < 0 || slot >= out.size()) return;
    // The WHOLE span must be free, not just its first column. Claiming
    // [c0, c0+colspan) after checking only c0 stamped over the free_at entry
    // of any interior column still occupied by an earlier row's live rowspan
    // — two cells then claimed one grid position (a browser shifts the new
    // cell RIGHT past the blocker instead, which is the occupancy rule the
    // comment above promises). Scan restarts past the blocking column.
    std::size_t c0 = cursor;
    for (;;) {
      while (c0 < free_at.size() && free_at[c0] > cur_row) ++c0;
      std::size_t k = c0;
      const std::size_t span_end = c0 + static_cast<std::size_t>(colspan);
      while (k < span_end && (k >= free_at.size() || free_at[k] <= cur_row))
        ++k;
      if (k == span_end) break; // whole span free
      c0 = k + 1;               // blocked at k — resume past it
    }
    const std::size_t end = c0 + static_cast<std::size_t>(colspan);
    if (free_at.size() < end) free_at.resize(end, 0);
    for (std::size_t k = c0; k < end; ++k) free_at[k] = cur_row + rowspan;
    cursor = end;
    router::TableCell& c = out[slot];
    c.row     = cur_row;
    c.col     = static_cast<int>(c0);
    c.rowspan = rowspan;
    c.colspan = colspan;
  };

  // Slot counting must mirror decode_structure, which pushes one cell per
  // is_td_token() token: `<td` (attributed cell, spans follow) and `<td></td>`
  // (fused empty cell). `<td>` is erased from the dict by CharDict::build, but
  // is counted here too so a custom dict that kept it stays aligned.
  std::size_t next_slot   = 0;
  bool        open_td     = false;  // inside `<td` … `>`
  std::size_t open_slot   = 0;
  int         open_rspan  = 1;
  int         open_cspan  = 1;

  for (const std::string& tok : structure) {
    if (tok == "<tr>") {
      ++cur_row;
      cursor  = 0;
      open_td = false;  // an unterminated <td open tag does not cross rows
      continue;
    }
    if (tok == "<td></td>" || tok == "<td>") {
      place(next_slot++, 1, 1);
      continue;
    }
    if (tok == "<td") {
      open_td    = true;
      open_slot  = next_slot++;
      open_rspan = 1;
      open_cspan = 1;
      continue;
    }
    if (!open_td) continue;
    if (tok.starts_with(" colspan=")) {
      open_cspan = parse_span(tok);
    } else if (tok.starts_with(" rowspan=")) {
      open_rspan = parse_span(tok);
    } else if (tok == ">") {
      place(open_slot, open_rspan, open_cspan);
      open_td = false;
    }
  }
  return out;
}

} // namespace turbo_ocr::table
