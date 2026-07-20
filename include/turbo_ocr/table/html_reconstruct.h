#pragma once

#include <string>
#include <vector>

#include "turbo_ocr/table/cell_matcher.h"

namespace turbo_ocr::table {

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
