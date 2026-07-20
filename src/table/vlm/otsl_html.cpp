// OTSL-v1.0 → HTML converter (paddlex convert_otsl_to_html port).
#include "turbo_ocr/common/string_utils.h"
#include "turbo_ocr/table/vlm/vlm_table.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace turbo_ocr::table {

namespace {

// HTML escape for cell text.
std::string html_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&':  out += "&amp;";  break;
      case '<':  out += "&lt;";   break;
      case '>':  out += "&gt;";   break;
      case '"':  out += "&quot;"; break;
      case '\'': out += "&#39;";  break;
      default:   out += c;
    }
  }
  return out;
}

// Trim whitespace from both ends.

} // namespace

// ---------------------------------------------------------------------------
// OTSL → HTML
// Port of paddlex's convert_otsl_to_html (uilts.py). Token set:
//   <fcel>  full cell  (followed by text until next token)
//   <ecel>  empty cell
//   <nl>    new row
//   <lcel>  left-merge  (extends previous cell to the right)
//   <ucel>  up-merge    (extends cell above downward)
//   <xcel>  corner merge (both left and up)
// ---------------------------------------------------------------------------

namespace {

constexpr std::string_view kFcel = "<fcel>";
constexpr std::string_view kEcel = "<ecel>";
constexpr std::string_view kNl   = "<nl>";
constexpr std::string_view kLcel = "<lcel>";
constexpr std::string_view kUcel = "<ucel>";
constexpr std::string_view kXcel = "<xcel>";

enum class OtslTok { Fcel, Ecel, Nl, Lcel, Ucel, Xcel };

struct OtslElement {
  OtslTok     tok = OtslTok::Ecel;
  std::string text;  // only meaningful for Fcel
};

// Parse OTSL into a flat element list (Fcel/Ecel/Lcel/Ucel/Xcel/Nl).
// Each Fcel absorbs any text up to the next tag.
std::vector<OtslElement> parse_otsl(const std::string &otsl) {
  std::vector<OtslElement> out;
  static const std::regex re(R"((<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>))");
  auto begin = std::sregex_iterator(otsl.begin(), otsl.end(), re);
  auto end   = std::sregex_iterator();
  std::vector<std::pair<size_t, std::string>> matches;
  for (auto it = begin; it != end; ++it) {
    matches.emplace_back(static_cast<size_t>(it->position(0)),
                         it->str(0));
  }
  for (size_t i = 0; i < matches.size(); ++i) {
    const std::string &tag = matches[i].second;
    OtslTok t;
    if (tag == kFcel) t = OtslTok::Fcel;
    else if (tag == kEcel) t = OtslTok::Ecel;
    else if (tag == kNl)   t = OtslTok::Nl;
    else if (tag == kLcel) t = OtslTok::Lcel;
    else if (tag == kUcel) t = OtslTok::Ucel;
    else                    t = OtslTok::Xcel;
    OtslElement e{t, {}};
    if (t == OtslTok::Fcel) {
      size_t text_start = matches[i].first + tag.size();
      size_t text_end   = (i + 1 < matches.size()) ? matches[i + 1].first
                                                   : otsl.size();
      if (text_end > text_start) {
        e.text = trim(otsl.substr(text_start, text_end - text_start));
      }
    }
    out.push_back(std::move(e));
  }
  return out;
}

// Pad each row to the dominant width by appending <ecel>. Mirrors
// otsl_pad_to_sqr_v2 in spirit but keeps the implementation simple: pick
// the modal row length (ties broken by the longest row).
struct Row {
  std::vector<OtslElement> cells;  // never contains Nl
};

std::vector<Row> split_rows(const std::vector<OtslElement> &elems) {
  std::vector<Row> rows;
  Row cur;
  for (const auto &e : elems) {
    if (e.tok == OtslTok::Nl) {
      if (!cur.cells.empty()) rows.push_back(std::move(cur));
      cur = {};
    } else {
      cur.cells.push_back(e);
    }
  }
  if (!cur.cells.empty()) rows.push_back(std::move(cur));
  return rows;
}

void pad_rows(std::vector<Row> &rows) {
  if (rows.empty()) return;
  size_t max_w = 0;
  for (const auto &r : rows) max_w = std::max(max_w, r.cells.size());
  for (auto &r : rows) {
    if (r.cells.empty()) {
      r.cells.resize(max_w, OtslElement{OtslTok::Ecel, ""});
      continue;
    }
    // A short row usually means the decoder truncated the trailing colspan
    // tokens of the row's LAST cell, so pad with <lcel> to left-merge the
    // missing columns into it (appending <ecel> would fabricate phantom columns
    // that shift every <ucel> rowspan below). EXCEPTION: a pure <ucel>
    // rowspan-continuation cannot root a horizontal merge, so an <lcel> chained
    // off it is malformed — in that one case pad with standalone <ecel> (same
    // column count, valid geometry). Decide once from the last *real* cell.
    const OtslTok pad_tok =
        (r.cells.back().tok == OtslTok::Ucel) ? OtslTok::Ecel : OtslTok::Lcel;
    while (r.cells.size() < max_w) {
      r.cells.push_back(OtslElement{pad_tok, ""});
    }
  }
}

bool is_l(OtslTok t) { return t == OtslTok::Lcel || t == OtslTok::Xcel; }
bool is_u(OtslTok t) { return t == OtslTok::Ucel || t == OtslTok::Xcel; }

} // namespace

std::string otsl_to_html(const std::string &otsl_in) {
  std::string otsl = trim(otsl_in);
  if (otsl.empty()) return "";
  // If no <nl> at all, treat the entire string as a single-row table.
  if (otsl.find(kNl) == std::string::npos) otsl += std::string(kNl);

  auto elems = parse_otsl(otsl);
  if (elems.empty()) return "";

  auto rows = split_rows(elems);
  if (rows.empty()) return "";

  // Cap the padded grid BEFORE pad_rows materializes it: model output fully
  // controls row count and max row width, and a sparse OTSL (one wide row +
  // many 1-cell rows) otherwise expands quadratically during padding.
  constexpr size_t kMaxTableCells = 1u << 16;
  size_t max_w = 0;
  for (const auto &r : rows) max_w = std::max(max_w, r.cells.size());
  if (max_w == 0 || rows.size() > kMaxTableCells / max_w) return "";

  pad_rows(rows);

  const size_t nrows = rows.size();
  const size_t ncols = rows.front().cells.size();
  if (ncols == 0) return "";

  // Build a 2D grid keyed by (row, col) -> origin (root cell idx + spans).
  // For each Fcel/Ecel root, compute col_span by counting Lcel/Xcel to the
  // right and row_span by counting Ucel/Xcel below in same column.
  struct CellInfo {
    bool        is_root  = false;
    bool        empty    = false;
    int         row_span = 1;
    int         col_span = 1;
    std::string text;
  };
  std::vector<std::vector<CellInfo>> grid(nrows,
      std::vector<CellInfo>(ncols, CellInfo{}));

  for (size_t r = 0; r < nrows; ++r) {
    for (size_t c = 0; c < ncols; ++c) {
      const auto &e = rows[r].cells[c];
      if (e.tok == OtslTok::Fcel || e.tok == OtslTok::Ecel) {
        CellInfo info;
        info.is_root = true;
        info.empty   = (e.tok == OtslTok::Ecel);
        info.text    = e.text;
        // Count Lcel/Xcel to the right on this row.
        size_t cc = c + 1;
        while (cc < ncols && is_l(rows[r].cells[cc].tok)) {
          info.col_span += 1;
          ++cc;
        }
        // Count Ucel/Xcel below in column c.
        size_t rr = r + 1;
        while (rr < nrows && is_u(rows[rr].cells[c].tok)) {
          info.row_span += 1;
          ++rr;
        }
        grid[r][c] = info;
      }
    }
  }

  // Emit HTML.
  std::string out = "<table>";
  for (size_t r = 0; r < nrows; ++r) {
    out += "<tr>";
    for (size_t c = 0; c < ncols; ++c) {
      const auto &g = grid[r][c];
      if (!g.is_root) continue;
      std::string tag = "<td";
      if (g.row_span > 1) tag += " rowspan=\"" + std::to_string(g.row_span) + "\"";
      if (g.col_span > 1) tag += " colspan=\"" + std::to_string(g.col_span) + "\"";
      tag += ">";
      out += tag;
      out += html_escape(g.text);
      out += "</td>";
    }
    out += "</tr>";
  }
  out += "</table>";
  return out;
}

} // namespace turbo_ocr::table
