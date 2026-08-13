#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

// Order statistics used by more than one subsystem.
//
// median_of() lived here twice before this header existed — byte-identical
// copies in src/pdf/text/font_style.cpp and src/pdf/text/pdf_searchable.cpp.
// (They sat in the same directory then; the searchable-PDF writer has since moved
// to document/, which only widens the gap the shared definition closes.) Both are
// load-bearing (one picks the
// document's stroke weight and slant, the other snaps line sizes), so the two
// copies were a silent invitation to fix a rounding or empty-input decision in
// one place and leave the other reading differently.
namespace turbo_ocr::stats {

// Upper median (element at size/2) — for an even count this is the higher of
// the two middle values, NOT their average. That choice is deliberate and both
// original copies made it: these medians pick a REPRESENTATIVE measurement
// (a stroke width, a font size) out of a sample, and averaging two neighbours
// can produce a value no line actually had.
//
// Takes a mutable reference and PARTIALLY REORDERS it (std::nth_element), which
// is why it is not const: callers pass scratch vectors they do not read again.
// Empty input is 0.0F rather than undefined.
[[nodiscard]] inline float median_of(std::vector<float> &v) {
  if (v.empty()) return 0.0F;
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
  return v[mid];
}

} // namespace turbo_ocr::stats
