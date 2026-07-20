#pragma once

#include <cmath>
#include <cstdio>
#include <string>

#include "turbo_ocr/common/types.h"

namespace turbo_ocr {

namespace detail {

// Full JSON-string escape. Tables emit HTML (quotes, ampersands) and
// formulas emit LaTeX (backslashes, braces) — both need every escape
// the OCR text branch uses. Caller writes the surrounding quotes.
inline void append_escaped_string(std::string &j, const std::string &s) {
  const size_t n = s.size();
  for (size_t i = 0; i < n;) {
    const auto uc = static_cast<unsigned char>(s[i]);
    if (uc < 0x80) {  // ASCII: JSON-escape control/special chars, pass the rest through
      const char c = s[i++];
      switch (c) {
        case '"':  j += "\\\""; break;
        case '\\': j += "\\\\"; break;
        case '\b': j += "\\b";  break;
        case '\f': j += "\\f";  break;
        case '\n': j += "\\n";  break;
        case '\r': j += "\\r";  break;
        case '\t': j += "\\t";  break;
        default:
          if (uc < 0x20) {
            char buf[7];
            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(uc));
            j += buf;
          } else {
            j += c;
          }
      }
      continue;
    }
    // Multi-byte: copy a well-formed UTF-8 sequence verbatim, else emit U+FFFD. A backstop so
    // no producer can ever ship RFC-8259-invalid bytes that a strict JSON client would reject
    // (valid output is byte-identical to before).
    const int len = (uc >> 5) == 0x6 ? 2 : (uc >> 4) == 0xE ? 3 : (uc >> 3) == 0x1E ? 4 : 0;
    bool ok = len >= 2 && i + static_cast<size_t>(len) <= n;
    for (int k = 1; k < len && ok; ++k)
      ok = (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
    if (ok) { j.append(s, i, static_cast<size_t>(len)); i += static_cast<size_t>(len); }
    else { j += "\xEF\xBF\xBD"; ++i; }
  }
}

// Serialize a confidence/score as JSON. snprintf("%.5g", NaN|Inf) emits the
// bare tokens `nan`/`inf`, which are NOT valid JSON and break the entire
// response for a strict client — a single degraded region must not do that.
// Non-finite -> 0.
inline void append_score(std::string &j, double v) {
  char buf[16];
  if (std::isfinite(v))
    std::snprintf(buf, sizeof(buf), "%.5g", v);
  else
    std::snprintf(buf, sizeof(buf), "0");
  j += buf;
}

// Append `[[x,y],[x,y],[x,y],[x,y]]` to j — shared by text + layout writers.
inline void append_box(std::string &j, const Box &box) {
  j += '[';
  for (int k = 0; k < 4; ++k) {
    if (k > 0) j += ',';
    j += '[';
    j += std::to_string(box[k][0]);
    j += ',';
    j += std::to_string(box[k][1]);
    j += ']';
  }
  j += ']';
}

} // namespace detail

} // namespace turbo_ocr
