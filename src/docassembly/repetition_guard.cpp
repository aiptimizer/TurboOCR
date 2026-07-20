#include "turbo_ocr/common/string_utils.h"
#include "turbo_ocr/docassembly/repetition_guard.h"

#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace turbo_ocr::docassembly {

namespace {


// Shortest unit u such that u repeated fills the whole string (e.g. "abab" → "ab").
std::optional<std::string> find_shortest_repeating_substring(const std::string& s) {
  const std::size_t n = s.size();
  for (std::size_t i = 1; i <= n / 2; ++i) {
    if (n % i != 0) continue;
    std::string_view unit(s.data(), i);
    bool all = true;
    for (std::size_t off = i; off < n; off += i) {
      if (std::string_view(s.data() + off, i) != unit) { all = false; break; }
    }
    if (all) return s.substr(0, i);
  }
  return std::nullopt;
}

// If s ends with (unit * >=min_repeats), return (prefix, unit, count) where
// count is the maximal trailing repeat count. Mirrors find_repeating_suffix.
std::optional<std::tuple<std::string, std::string, int>>
find_repeating_suffix(const std::string& s, int min_len = 8, int min_repeats = 5) {
  const int n = static_cast<int>(s.size());
  for (int i = n / min_repeats; i >= min_len; --i) {
    if (i <= 0) break;
    std::string_view unit(s.data() + (n - i), i);
    // s.endswith(unit * min_repeats)
    bool ok = true;
    for (int r = 1; r <= min_repeats; ++r) {
      int off = n - r * i;
      if (off < 0 || std::string_view(s.data() + off, i) != unit) { ok = false; break; }
    }
    if (!ok) continue;
    int count = 0;
    int end = n;
    while (end - i >= 0 && std::string_view(s.data() + (end - i), i) == unit) {
      end -= i;
      ++count;
    }
    int start_index = n - count * i;
    return std::make_tuple(s.substr(0, start_index), std::string(unit), count);
  }
  return std::nullopt;
}

} // namespace

std::string truncate_repetitive_content(const std::string& content,
                                        int line_threshold,
                                        int char_threshold,
                                        int min_len,
                                        int min_count) {
  if (static_cast<int>(content.size()) < min_count) return content;

  const std::string stripped = turbo_ocr::trim(content);
  if (stripped.empty()) return content;

  const bool single_line = stripped.find('\n') == std::string::npos;

  // The two single-line passes below are O(n^2) in the line length. Untrusted
  // OCR can collapse a whole page into one multi-MB line, which would spin the
  // quadratic scan into a per-page hang — so skip them above a bound. A real
  // hallucinated-repetition tail is caught long before this; 64 KB is orders
  // of magnitude past any genuine document line.
  constexpr std::size_t kMaxSingleLineScan = 64 * 1024;
  const bool scannable_single_line =
      single_line && stripped.size() <= kMaxSingleLineScan;

  // Priority 1: phrase-level suffix repetition in long single lines.
  if (scannable_single_line && stripped.size() > 100) {
    if (auto m = find_repeating_suffix(stripped, 8, 5)) {
      const auto& [prefix, unit, count] = *m;
      if (static_cast<double>(unit.size()) * count >
          static_cast<double>(stripped.size()) * 0.5) {
        return prefix;
      }
    }
  }

  // Priority 2: full-string character-level repetition (e.g. 'ababab').
  if (scannable_single_line && static_cast<int>(stripped.size()) > min_len) {
    if (auto unit = find_shortest_repeating_substring(stripped)) {
      int count = static_cast<int>(stripped.size() / unit->size());
      if (count >= char_threshold) return *unit;
    }
  }

  // Priority 3: line-level repetition (same line repeated many times).
  std::vector<std::string> lines;
  {
    std::size_t pos = 0;
    while (pos <= content.size()) {
      std::size_t nl = content.find('\n', pos);
      std::string line =
          content.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
      std::string t = turbo_ocr::trim(line);
      if (!t.empty()) lines.push_back(std::move(t));
      if (nl == std::string::npos) break;
      pos = nl + 1;
    }
  }
  if (lines.empty()) return content;
  const int total_lines = static_cast<int>(lines.size());
  if (total_lines < line_threshold) return content;

  std::unordered_map<std::string, int> counts;
  const std::string* best = nullptr;
  int best_count = 0;
  for (const auto& l : lines) {
    int c = ++counts[l];
    if (c > best_count) { best_count = c; best = &l; }
  }
  if (best && best_count >= line_threshold &&
      static_cast<double>(best_count) / total_lines >= 0.8) {
    return *best;
  }

  return content;
}

} // namespace turbo_ocr::docassembly
