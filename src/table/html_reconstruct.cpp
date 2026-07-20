#include "turbo_ocr/table/html_reconstruct.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace turbo_ocr::table {

namespace {

// ---- Table-HTML sanitizer -------------------------------------------------

std::string ascii_lower(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

// Tags that may appear in a reconstructed table, with structural meaning.
// Anything else is dropped (its inner text is kept).
bool is_allowed_table_tag(std::string_view name) {
  static constexpr std::array<std::string_view, 13> kAllowed = {
      "table", "thead", "tbody", "tr", "td", "th", "col",
      "colgroup", "caption", "b", "i", "strong", "em"};
  const std::string lower = ascii_lower(name);
  return std::find(kAllowed.begin(), kAllowed.end(), lower) != kAllowed.end();
}

// Re-emit an allowed tag keeping only colspan/rowspan/align/valign attributes
// (the only ones table rendering needs); on*= handlers, style, and any
// href/src carrying javascript: are dropped by omission.
std::string sanitize_tag(std::string_view inner) {
  // inner is the text between < and > (may start with '/').
  bool closing = !inner.empty() && inner.front() == '/';
  std::string_view body = closing ? inner.substr(1) : inner;
  size_t sp = body.find_first_of(" \t\r\n");
  std::string_view name = body.substr(0, sp);
  if (!is_allowed_table_tag(name)) return {};  // drop tag, keep text
  std::string out = "<";
  if (closing) out += '/';
  out += ascii_lower(name);
  if (!closing && sp != std::string_view::npos) {
    // Keep only the whitelisted span/align attributes, values re-quoted.
    static constexpr std::array<std::string_view, 4> kAttrs = {
        "colspan", "rowspan", "align", "valign"};
    std::string_view attrs = body.substr(sp);
    const std::string lowattrs = ascii_lower(attrs);
    for (std::string_view a : kAttrs) {
      const std::string needle = std::string(a) + "=";
      // Match the attribute only at a token boundary so `align=` doesn't
      // substring-match inside `valign=` and emit a spurious duplicate.
      size_t p = std::string::npos;
      for (size_t i = lowattrs.find(needle); i != std::string::npos;
           i = lowattrs.find(needle, i + 1)) {
        if (i == 0 || std::isspace((unsigned char)lowattrs[i - 1]) ||
            lowattrs[i - 1] == '"' || lowattrs[i - 1] == '\'') {
          p = i;
          break;
        }
      }
      if (p == std::string::npos) continue;
      p += needle.size();
      // value: quoted or bare up to whitespace
      std::string val;
      if (p < attrs.size() && (attrs[p] == '"' || attrs[p] == '\'')) {
        char q = attrs[p++];
        while (p < attrs.size() && attrs[p] != q) val += attrs[p++];
      } else {
        while (p < attrs.size() && !std::isspace((unsigned char)attrs[p]))
          val += attrs[p++];
      }
      // keep only digits for span, alnum for align — never quotes/brackets
      std::string clean;
      for (char c : val)
        if (std::isalnum((unsigned char)c)) clean += c;
      if (!clean.empty()) { out += ' '; out += std::string(a); out += "=\""; out += clean; out += '"'; }
    }
  }
  out += '>';
  return out;
}

} // namespace

std::string sanitize_table_html(const std::string& html) {
  std::string out;
  out.reserve(html.size());
  // Lower-cased ONCE for the close-tag scans: recomputing it per
  // <script>/<style> element made an input of many such elements O(n²) —
  // a CPU-DoS lever on this semi-trusted (VLM-produced) input.
  const std::string html_lower = ascii_lower(html);
  size_t i = 0, n = html.size();
  while (i < n) {
    if (html[i] != '<') { out += html[i++]; continue; }
    size_t end = html.find('>', i);
    if (end == std::string::npos) {  // stray '<' — emit escaped, stop tag parse
      out += "&lt;";
      ++i;
      continue;
    }
    std::string_view inner(html.data() + i + 1, end - i - 1);
    std::string lower = ascii_lower(inner);
    // Drop <script>/<style> ELEMENTS including their content.
    if (lower.rfind("script", 0) == 0 || lower.rfind("style", 0) == 0) {
      // skip to matching close tag (or end)
      std::string close = lower.rfind("script", 0) == 0 ? "</script" : "</style";
      size_t c = html_lower.find(close, end);
      i = (c == std::string::npos) ? n : html.find('>', c);
      i = (i == std::string::npos) ? n : i + 1;
      continue;
    }
    out += sanitize_tag(inner);
    i = end + 1;
  }
  return out;
}

namespace {

// Structure streams from SLANeXt are wrapped <html><body><table> ... on each
// side; below this we can't slice out the body and must fall back to a raw
// concatenation.
constexpr std::size_t kWrapTokens = 3;
constexpr std::size_t kMinWrapped = 2 * kWrapTokens;

std::string_view td_content(std::string_view text) {
    // A page-OCR fragment carries its own leading space and <b>…</b> emphasis
    // wrapper; when several fragments are joined into one cell we trim those so
    // the cell can carry a single outer <b> and single-space joins. Each trim is
    // a pure prefix/suffix slice, so a view avoids copying + O(n) erase shifts.
    if (!text.empty() && text.front() == ' ') text.remove_prefix(1);
    if (text.starts_with("<b>")) text.remove_prefix(3);
    if (text.ends_with("</b>")) text.remove_suffix(4);
    return text;
}

// Escape &, <, > in cell text while preserving the intentional <b>/</b>
// emphasis wrappers the cell matcher emits — OCR'd angle brackets and
// ampersands must not become live markup in the reconstructed table HTML.
std::string escape_cell_html(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        if (text.compare(i, 3, "<b>") == 0)  { out += "<b>";  i += 3; continue; }
        if (text.compare(i, 4, "</b>") == 0) { out += "</b>"; i += 4; continue; }
        const char c = text[i++];
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;";  break;
            case '>': out += "&gt;";  break;
            default:  out += c;
        }
    }
    return out;
}

} // namespace

std::string reconstruct_html(
    const std::vector<std::string>& structure,
    const std::vector<MatchedCell>& cells,
    const std::vector<std::string>& ocr_texts) {
    // Single allocation: tag bytes + substituted text + slack for the per-cell
    // <b></b> wrapper, join spaces (≤ one per matched fragment) and <td></td>
    // expansion. A guaranteed upper bound means no reallocation while appending.
    std::size_t budget = cells.size() * 8;
    for (const auto& tag : structure) budget += tag.size();
    // 5x: worst-case escape_cell_html expansion ('&' -> "&amp;").
    for (const auto& text : ocr_texts) budget += text.size() * 5;
    for (const auto& c : cells) budget += c.ocr_indices.size();

    std::string out;
    out.reserve(budget);

    const std::size_t n = structure.size();
    if (n < kMinWrapped) {
        for (const auto& t : structure) out += t;
        return out;
    }

    for (std::size_t i = 0; i < kWrapTokens; ++i) out += structure[i];

    std::size_t td_index = 0;
    for (std::size_t k = kWrapTokens; k + kWrapTokens < n; ++k) {
        const std::string& tag = structure[k];
        // The single-token empty cell "<td></td>" and the closing "</td>" of a
        // split (attributed) cell are the two text-insertion points; every other
        // token — <tr>, </tr>, <td, attribute chunks, > — passes through verbatim.
        const bool is_open_close = (tag == "<td></td>");
        const bool is_td_slot = is_open_close || tag.find("</td>") != std::string::npos;
        if (!is_td_slot) {
            out += tag;
            continue;
        }

        // Split cells have already emitted <td...>; only the fused token needs
        // its opening <td> synthesized here before the text is inserted.
        if (is_open_close) out += "<td>";

        if (td_index < cells.size()) {
            const std::vector<std::size_t>& idx = cells[td_index].ocr_indices;
            const std::size_t nm = idx.size();
            if (nm == 1) {
                // Sole fragment keeps its emphasis wrapper; everything else is
                // escaped (see escape_cell_html).
                if (idx[0] < ocr_texts.size())
                    out += escape_cell_html(ocr_texts[idx[0]]);
            } else if (nm > 1) {
                const std::size_t first = idx.front();
                const bool wrap_b =
                    first < ocr_texts.size() &&
                    ocr_texts[first].find("<b>") != std::string::npos;
                if (wrap_b) out += "<b>";
                for (std::size_t j = 0; j < nm; ++j) {
                    std::string_view frag =
                        idx[j] < ocr_texts.size()
                            ? std::string_view{ocr_texts[idx[j]]}
                            : std::string_view{};
                    frag = td_content(frag);
                    if (frag.empty()) continue;
                    out += escape_cell_html(frag);
                    if (j + 1 != nm && frag.back() != ' ') out.push_back(' ');
                }
                if (wrap_b) out += "</b>";
            }
        }

        // Fused token closes with a synthesized </td>; a split cell reuses its
        // own closing </td> token (which is exactly `tag` here).
        out += is_open_close ? std::string_view{"</td>"} : std::string_view{tag};
        ++td_index;
    }

    for (std::size_t i = n - kWrapTokens; i < n; ++i) out += structure[i];
    return out;
}

} // namespace turbo_ocr::table
