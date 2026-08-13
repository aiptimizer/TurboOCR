#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/analysis/table/cell_matcher.h"

namespace turbo_ocr::table {

// Strip the decorations a page-OCR fragment carries — a leading space and the
// <b>…</b> emphasis wrapper — leaving the bare cell text. Shared by
// reconstruct_html (joining fragments inside a <td>) and build_table_cells
// (joining them into the plain-text cell), so the HTML and the structured cell
// list can never disagree on what a cell says. Each trim is a pure prefix/
// suffix slice, so a view avoids copying + O(n) erase shifts.
[[nodiscard]] inline std::string_view td_fragment_text(std::string_view text) noexcept {
    if (!text.empty() && text.front() == ' ') text.remove_prefix(1);
    if (text.starts_with("<b>")) text.remove_prefix(3);
    if (text.ends_with("</b>")) text.remove_suffix(4);
    return text;
}

// Build the final <html>...</html> string by walking `structure` (already
// wrapped with <html><body><table> ... by SLANeXt) and substituting OCR text
// into each <td> slot in order.
//
// `ocr_texts` is the original OCR strings. `cells[i]` corresponds to the i-th
// <td>-family token in the structure stream.
std::string reconstruct_html(
    const std::vector<std::string>& structure,
    const std::vector<MatchedCell>& cells,
    const std::vector<std::string>& ocr_texts);

// Sanitize model-produced table HTML before it flows into an output document.
// A VLM given an adversarial page image can be induced to emit
// `<table>…<script>…</script>…` (or `onerror=`/`javascript:` attributes),
// which becomes live markup if the surrounding Markdown/HTML is rendered
// downstream — a stored-XSS-class vector. Drops <script>/<style> elements
// (including their content), strips on*= event-handler attributes and
// javascript: URIs, and removes any other non-table-structural tags while
// keeping their text. Table-structural tags (table/thead/tbody/tr/td/th/
// col/colgroup/caption/b/i) and their span/align attributes pass through.
std::string sanitize_table_html(const std::string& html);

} // namespace turbo_ocr::table
