// pdf_searchable_encoding.cpp — codepoints, encodings and size statistics.
//
// Everything here is a pure function of its arguments: no PDFium handle, no
// page, no document. That is why it is its own TU and not merely its own
// section — this is the part of writing a searchable PDF that can be reasoned
// about, and tested, without a PDF at all.
//
// See pdf_searchable_detail.h for the whole split.

#include "pdf_searchable_detail.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include <fpdf_annot.h>

#include "turbo_ocr/base/log/logger.h"
#include "pdf_text_internal.h"

namespace turbo_ocr::pdf {
namespace searchable_detail {

// Decode UTF-8, dropping control characters (a content stream cannot carry
// them and OCR output occasionally contains them). Returns false on malformed
// input so the caller can drop the run rather than emit mojibake.
bool decode_utf8(const std::string &s, std::vector<unsigned> &out) {
  out.clear();
  for (size_t i = 0; i < s.size();) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    unsigned cp;
    int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1Fu; n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0Fu; n = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07u; n = 4; }
    else return false;
    if (i + n > s.size()) return false;
    for (int k = 1; k < n; ++k) {
      unsigned char cc = static_cast<unsigned char>(s[i + k]);
      if ((cc & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (cc & 0x3Fu);
    }
    i += n;
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
    if (cp == '\t' || cp == '\n' || cp == '\r') cp = ' ';
    else if (cp < 0x20 || cp == 0x7F) continue;
    out.push_back(cp);
  }
  return !out.empty();
}

void append_hex4(std::string &out, unsigned v) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  out += kHex[(v >> 12) & 0xF];
  out += kHex[(v >> 8) & 0xF];
  out += kHex[(v >> 4) & 0xF];
  out += kHex[v & 0xF];
}

// CID -> Unicode map for the document's characters. Written as a PDF CMap
// program; PDFium embeds it as the font's /ToUnicode, which is what every
// extractor (and Ctrl-F) reads.
std::string build_to_unicode(const std::vector<std::pair<unsigned, unsigned>> &entries) {
  std::string cmap =
      "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
      "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
      "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
      "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n";
  for (size_t i = 0; i < entries.size(); i += 100) {
    const size_t n = std::min<size_t>(100, entries.size() - i);
    cmap += std::to_string(n);
    cmap += " beginbfchar\n";
    for (size_t k = i; k < i + n; ++k) {
      cmap += '<';
      append_hex4(cmap, entries[k].second);
      cmap += "> <";
      const unsigned cp = entries[k].first;
      if (cp >= 0x10000) { // UTF-16 surrogate pair
        const unsigned v = cp - 0x10000;
        append_hex4(cmap, 0xD800 + (v >> 10));
        append_hex4(cmap, 0xDC00 + (v & 0x3FF));
      } else {
        append_hex4(cmap, cp);
      }
      cmap += ">\n";
    }
    cmap += "endbfchar\n";
  }
  cmap += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n";
  return cmap;
}

bool keep(const OCRResultItem &r, float min_confidence) {
  return r.source != "pdf" && r.confidence >= min_confidence && !r.text.empty();
}

// The standard-14 fonts are Latin-1 only. A codepoint outside that cannot be
// drawn in them, and drawing the wrong glyph is worse than leaving the scan to
// speak for itself, so such a line stays invisible over its original print.
//
// The C1 range 0x80-0x9F is excluded on purpose: PDF's WinAnsi encoding puts
// typographic punctuation there rather than control characters, so a codepoint
// landing in it would come out as an unrelated glyph.
bool spellable_in_standard14(const std::vector<uint32_t> &cps) {
  return std::ranges::all_of(cps, [](uint32_t cp) {
    return (cp >= 0x20 && cp <= 0x7E) || (cp >= 0xA0 && cp <= 0xFF);
  });
}

// UTF-16LE for FPDFText_SetText. Every codepoint here is Latin-1 (see
// spellable_in_standard14), so no surrogate pair can arise.
std::vector<unsigned short> to_utf16(const std::vector<uint32_t> &cps) {
  std::vector<unsigned short> out(cps.size() + 1, 0);
  for (size_t i = 0; i < cps.size(); ++i)
    out[i] = static_cast<unsigned short>(cps[i]);
  return out;
}

// Split a run on spaces. The spaces themselves are dropped: when this is used,
// the gaps are about to be recomputed, and a kept space would be counted twice.
std::vector<std::vector<uint32_t>>
split_words(const std::vector<uint32_t> &cps) {
  std::vector<std::vector<uint32_t>> words;
  std::vector<uint32_t> cur;
  for (uint32_t cp : cps) {
    if (cp == ' ') {
      if (!cur.empty()) words.push_back(std::exchange(cur, {}));
      continue;
    }
    cur.push_back(cp);
  }
  if (!cur.empty()) words.push_back(std::move(cur));
  return words;
}

// One piece per character, for letter spacing. Only ever reached when the run
// has no word gaps to widen, so there are no spaces left to drop.
std::vector<std::vector<uint32_t>>
split_glyphs(const std::vector<uint32_t> &cps) {
  std::vector<std::vector<uint32_t>> out;
  out.reserve(cps.size());
  for (uint32_t cp : cps)
    if (cp != ' ') out.push_back({cp});
  return out;
}

float median_of(std::vector<float> &v) {
  if (v.empty()) return 0.0F;
  const size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
  return v[mid];
}

// Collapses per-line sizes onto the small set of sizes the document actually
// uses, so lines set alike come out alike.
//
// Each line arrives with the size that would fit its own ink exactly. That is
// nearly right — and "nearly" is the problem. The detector's boxes wobble by a
// pixel or two from line to line, so a run of labels set in one size arrives as
// twelve slightly different sizes. Nobody sees that until they click into a
// line and type, and get text a size out from the line above.
//
// Distinct sizes are still kept apart: a heading really is bigger than its
// body, so lines are grouped and each group snapped to its own median, rather
// than the document being flattened onto one size.
std::vector<float> snap_line_sizes(const std::vector<float> &sizes) {
  std::vector<float> out(sizes.size(), 0.0F);
  // Within this fraction of its group's smallest member, a line is the same
  // size as the rest of it. Comfortably above the couple of per cent that box
  // wobble contributes, and comfortably below the step from a body size to a
  // heading. Bounded against the group's SMALLEST member rather than a running
  // mean, so a document of gently increasing sizes cannot chain itself into one
  // enormous group.
  // 0.25 was chosen when sizes came from the measured x-height, which capitals
  // inflate by about 15%. Sizes now come from the ink a face actually makes for
  // that string, which cancels the content dependence, so the tolerance no
  // longer has to absorb it — and at 0.25 it merged 10 pt with 12 pt, since
  // 12 <= 10 * 1.25.
  constexpr float kSameSize = 0.14F;

  std::vector<size_t> order;
  for (size_t i = 0; i < sizes.size(); ++i)
    if (sizes[i] > 0.0F) order.push_back(i);
  if (order.empty()) return out;
  std::ranges::sort(order,
                    [&](size_t a, size_t b) { return sizes[a] < sizes[b]; });

  size_t start = 0;
  while (start < order.size()) {
    size_t end = start;
    const float base = sizes[order[start]];
    while (end + 1 < order.size() && sizes[order[end + 1]] <= base * (1.0F + kSameSize))
      ++end;
    std::vector<float> group;
    group.reserve(end - start + 1);
    for (size_t k = start; k <= end; ++k) group.push_back(sizes[order[k]]);
    const float snapped = median_of(group);
    for (size_t k = start; k <= end; ++k) out[order[k]] = snapped;
    start = end + 1;
  }
  return out;
}

} // namespace searchable_detail
} // namespace turbo_ocr::pdf
