#pragma once

// Internals shared by the markdown_* TUs in this directory.

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/base/string_utils.h"

namespace turbo_ocr::markdown::mddetail {

using turbo_ocr::trim;


// Minimum body cells before the column-aware pass even looks at a page.
// Shared by the detector (markdown_columns.cpp) and the driver's early-out.
constexpr int kColMinBodyBlocks = 4;

// Conservative structural KaTeX safety check (markdown_latex.cpp).
[[nodiscard]] bool latex_is_render_safe(const std::string &s);

// Runaway / mode-collapse garbage detector, Markdown view only
// (markdown_latex.cpp).
[[nodiscard]] bool latex_is_mode_collapsed(const std::string &latex);

// Wrap `s` in a backtick span, widening the delimiter past the longest
// backtick run inside (markdown_latex.cpp).
[[nodiscard]] std::string inline_code(const std::string &s);

// Wrap `s` in a fenced code block whose fence is wider than any backtick run
// in the content — a fixed ``` fence is escapable by its own payload
// (markdown_latex.cpp). `info` is the info string ("", "latex", ...).
[[nodiscard]] std::string fenced_block(const std::string &s,
                                       const std::string &info);

// Escape untrusted document text for Markdown output: HTML metacharacters to
// entities (& first) + a leading block marker neutralized. See the rationale
// at the definition (markdown_latex.cpp).
[[nodiscard]] std::string escape_md_text(const std::string &s);

// Column-major emission order for a clear multi-column body, or nullopt
// (markdown_columns.cpp).
[[nodiscard]] std::optional<std::vector<int>>
column_major_order(const std::vector<std::array<int, 4>> &rects);

} // namespace turbo_ocr::markdown::mddetail
