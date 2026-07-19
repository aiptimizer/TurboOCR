#include "turbo_ocr/table/html_reconstruct.h"

#include <string_view>

namespace turbo_ocr::table {

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
