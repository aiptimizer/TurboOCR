#pragma once

// Internals shared by the markdown_* TUs in this directory.

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "turbo_ocr/common/string_utils.h"

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

// Wrap `s` in a backtick span, widening the fence when it contains one.
[[nodiscard]] std::string inline_code(const std::string &s);

// Column-major emission order for a clear multi-column body, or nullopt
// (markdown_columns.cpp).
[[nodiscard]] std::optional<std::vector<int>>
column_major_order(const std::vector<std::array<int, 4>> &rects);

} // namespace turbo_ocr::markdown::mddetail
