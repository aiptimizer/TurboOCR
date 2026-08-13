#include "turbo_ocr/analysis/formula/cjk_stats.h"

#include <cstdint>

namespace turbo_ocr::formula {

// Is codepoint `cp` a CJK ideograph? CJK Unified (incl. Ext A), Compatibility
// Ideographs, and Ext B+ (SIP). Kana/hangul are deliberately excluded — the
// escalation signal is Chinese formula content, which -S mangles into
// ideograph garbage.
inline bool is_cjk_cp(uint32_t cp) noexcept {
  return (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0x20000 && cp <= 0x2FFFF);
}

CjkStat cjk_stats(std::string_view s) noexcept {
  CjkStat st;
  const auto *p = reinterpret_cast<const unsigned char *>(s.data());
  const auto *end = p + s.size();
  while (p < end) {
    uint32_t cp;
    int len;
    const unsigned char c = *p;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
    else { ++p; continue; }  // stray continuation / invalid lead
    if (p + len > end) break;
    bool ok = true;
    for (int i = 1; i < len; ++i) {
      if ((p[i] & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (p[i] & 0x3F);
    }
    if (!ok) { ++p; continue; }
    p += len;
    ++st.total;
    if (is_cjk_cp(cp)) ++st.cjk;
  }
  return st;
}

bool text_has_cjk(std::string_view s) noexcept { return cjk_stats(s).cjk > 0; }

} // namespace turbo_ocr::formula
