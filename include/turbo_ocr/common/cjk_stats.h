#pragma once

#include <string_view>

namespace turbo_ocr::formula {

// True once `s` (UTF-8) contains a CJK ideograph codepoint. Used by the
// auto-CJK composite to scan a single formula crop's -S output. No allocation.
[[nodiscard]] bool text_has_cjk(std::string_view s) noexcept;

// CJK-vs-total codepoint counts for `s` (UTF-8). The pipeline accumulates these
// across a page's recognized text to decide the per-page routing hint with a
// THRESHOLD — a single stray CJK glyph from an OCR misrecognition on a
// math-heavy EN page (measured: 1 CJK in 1700 chars) must NOT escalate the
// whole page, while a genuine Chinese page (measured: 23–88% CJK) must.
struct CjkStat { int cjk = 0; int total = 0; };
[[nodiscard]] CjkStat cjk_stats(std::string_view s) noexcept;

} // namespace turbo_ocr::formula
