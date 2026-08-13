// Label reading and the label-then-gap detector (detector 3).
//
// Ported from the Java heuristic in stirling-pdf's TurboOcrFieldDetector: band
// the OCR runs into visual lines, then look for a run that ends in a colon
// followed by enough blank space on the same baseline. That heuristic is the
// whole of what a text-only client can see, which is exactly why it misses
// every printed rule — the detectors in field_masks.cpp cover what it cannot.

#include "turbo_ocr/analysis/forms/field_detector.h"

#include <algorithm>
#include <cmath>

#include "forms_internal.h"

namespace turbo_ocr::forms {

namespace {

// Confidence for a label+gap proposal. The weakest of the four detectors: a
// colon followed by whitespace is also what the end of a sentence looks like,
// so this needs corroboration from a rule or a box to become convincing.
constexpr float kLabelGapConfidence = 0.55f;
// A blank the OCR actually read as underscores. Stronger — the document drew
// something there — but rarer than the morphological rule detector suggests,
// because recognisers usually drop the run entirely.
constexpr float kRuleTextConfidence = 0.70f;

[[nodiscard]] bool is_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}

} // namespace

namespace detail {

std::string trim_copy(std::string_view s) {
  size_t b = 0, e = s.size();
  while (b < e && is_space(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && is_space(static_cast<unsigned char>(s[e - 1]))) --e;
  return std::string(s.substr(b, e - b));
}

cv::Rect item_rect(const OCRResultItem &item) {
  const auto r = aabb(item.box);
  return cv::Rect(r[0], r[1], std::max(0, r[2] - r[0]),
                  std::max(0, r[3] - r[1]));
}

bool rect_is_empty(const cv::Rect &r, const std::vector<OCRResultItem> &text,
                   float max_overlap) {
  const double area = static_cast<double>(r.width) * r.height;
  if (area <= 0.0) return false;
  for (const auto &item : text) {
    if (detail::trim_copy(item.text).empty()) continue;
    // Centroid inside: the run belongs to this rect, whatever its extent.
    const auto [cx, cy] = quad_centroid(item.box);
    if (cx >= static_cast<float>(r.x) &&
        cx <= static_cast<float>(r.x + r.width) &&
        cy >= static_cast<float>(r.y) &&
        cy <= static_cast<float>(r.y + r.height))
      return false;
    // Or the rect lies ON the run — which is how the counter of an 'O' in a
    // label gets proposed as a checkbox. Its centroid test passes (the run's
    // centre is elsewhere), but it is covered by recognised text and so is
    // part of a glyph, not a place to write.
    const cv::Rect ir = detail::item_rect(item) & r;
    if (static_cast<double>(ir.width) * ir.height > max_overlap * area)
      return false;
  }
  return true;
}

} // namespace detail

bool is_label_text(std::string_view text) {
  const std::string t = detail::trim_copy(text);
  if (t.size() < 2) return false;
  if (t.back() == ':') return true;
  // U+FF1A FULLWIDTH COLON, the CJK form-label convention.
  return t.size() >= 3 && t.compare(t.size() - 3, 3, "\xEF\xBC\x9A") == 0;
}

bool is_rule_text(std::string_view text) {
  const std::string t = detail::trim_copy(text);
  if (t.size() < 3) return false;
  return std::ranges::all_of(t, [](char c) {
    return c == '_' || c == '-' || c == '.';
  });
}

std::vector<std::vector<int>>
group_into_lines(const std::vector<OCRResultItem> &text,
                 const FieldOptions &opt) {
  std::vector<int> order;
  order.reserve(text.size());
  for (int i = 0; i < static_cast<int>(text.size()); ++i)
    if (!detail::trim_copy(text[i].text).empty()) order.push_back(i);

  std::ranges::stable_sort(order, [&](int a, int b) {
    return quad_centroid(text[a].box).second <
           quad_centroid(text[b].box).second;
  });

  std::vector<std::vector<int>> lines;
  std::vector<int> current;
  double line_y = 0.0, line_h = 1.0;
  for (int idx : order) {
    const double cy = quad_centroid(text[idx].box).second;
    const double h = std::max(1, detail::item_rect(text[idx]).height);
    if (current.empty()) {
      line_y = cy;
      line_h = h;
      current.push_back(idx);
    } else if (std::abs(cy - line_y) <= line_h * opt.same_line_tol) {
      current.push_back(idx);
    } else {
      lines.push_back(std::move(current));
      current = {idx};
      line_y = cy;
      line_h = h;
    }
  }
  if (!current.empty()) lines.push_back(std::move(current));

  for (auto &line : lines)
    std::ranges::stable_sort(line, [&](int a, int b) {
      return detail::item_rect(text[a]).x < detail::item_rect(text[b]).x;
    });
  return lines;
}

std::string find_label(const cv::Rect &field,
                       const std::vector<OCRResultItem> &text, float text_h,
                       const FieldOptions &opt, bool label_follows) {
  const double fcy = field.y + field.height * 0.5;
  const double line_tol =
      std::max<double>(field.height, text_h) * opt.same_line_tol;
  const double max_left = opt.label_max_left_gap * text_h;
  // A label may overhang the field's left edge by a hair when the field was
  // derived from a rule that starts under the colon.
  const double slack = 0.25 * text_h;

  // Nearest word on the same line, on the requested side.
  const auto nearest = [&](bool to_the_right) -> int {
    int best = -1;
    double best_gap = 1e18;
    for (int i = 0; i < static_cast<int>(text.size()); ++i) {
      const std::string t = detail::trim_copy(text[i].text);
      if (t.empty() || is_rule_text(t)) continue;
      const cv::Rect r = detail::item_rect(text[i]);
      const double cy = r.y + r.height * 0.5;
      if (std::abs(cy - fcy) > line_tol) continue;
      // Which side the word EXTENDS to, not where it begins. A tick box and
      // its label are usually read as one run — "[ ] Oats" comes back as a
      // single box starting on the tick — so a test for "begins after the
      // field ends" rejects the very word that names it and takes the next
      // option along, labelling every box in the row one to the right.
      const double wl = r.x;
      const double wr = r.x + r.width;
      double gap = 0;
      if (to_the_right) {
        if (wr <= field.x + field.width + slack) continue;
        gap = std::max(0.0, wl - (field.x + field.width));
      } else {
        if (wl >= field.x - slack) continue;
        gap = std::max(0.0, field.x - wr);
      }
      if (gap > max_left) continue;
      if (gap < best_gap) {
        best_gap = gap;
        best = i;
      }
    }
    return best;
  };

  int best = nearest(label_follows);
  if (best < 0) best = nearest(!label_follows);
  if (best >= 0) return detail::trim_copy(text[best].text);

  // Nothing to the left — try directly above, which is how column-headed forms
  // ("Datum" over a ruled line) label their blanks.
  const double max_above = opt.label_max_above_gap * text_h;
  double best_bottom = -1e18;
  best = -1;
  for (int i = 0; i < static_cast<int>(text.size()); ++i) {
    const std::string t = detail::trim_copy(text[i].text);
    if (t.empty() || is_rule_text(t)) continue;
    const cv::Rect r = detail::item_rect(text[i]);
    const double bottom = r.y + r.height;
    if (bottom > field.y + slack) continue;
    if (field.y - bottom > max_above) continue;
    // Horizontal spans must actually overlap; "near and above" is not enough.
    const int ox = std::min(r.x + r.width, field.x + field.width) -
                   std::max(r.x, field.x);
    if (ox <= 0) continue;
    if (bottom > best_bottom) {
      best_bottom = bottom;
      best = i;
    }
  }
  return best >= 0 ? detail::trim_copy(text[best].text) : std::string{};
}

namespace detail {

void collect_label_gap_fields(const std::vector<OCRResultItem> &text,
                              int page_width, float text_h,
                              const FieldOptions &opt,
                              std::vector<FormField> &out) {
  const double page_right = page_width * static_cast<double>(opt.right_margin);

  for (const auto &line : group_into_lines(text, opt)) {
    for (size_t i = 0; i < line.size(); ++i) {
      const OCRResultItem &item = text[line[i]];
      const std::string t = trim_copy(item.text);
      const cv::Rect r = item_rect(item);
      const double h = std::max(1, r.height);

      // A drawn blank that OCR did read as characters IS the field, so it
      // takes precedence over treating the run as a word.
      if (is_rule_text(t)) {
        if (r.width < opt.min_field_width * text_h) continue;
        FormField f;
        f.box = rect_to_box(r);
        f.confidence = kRuleTextConfidence;
        f.source = "rule_text";
        out.push_back(std::move(f));
        continue;
      }
      if (!is_label_text(t)) continue;

      // The next run on this line bounds the blank; past the last run the
      // blank ends at the page's right margin.
      const double gap_start = r.x + r.width;
      double gap_end = page_right;
      if (i + 1 < line.size()) {
        const std::string next_t = trim_copy(text[line[i + 1]].text);
        // Leave a run of underscores to the branch above — it will be visited
        // on the next iteration and describes the blank more precisely.
        if (is_rule_text(next_t)) continue;
        // A label followed by a gap and then CONTENT ("Status: bereits
        // ausgefuellt") is a filled entry, not a blank: the whitespace is
        // layout, and proposing a field in it would offer to overwrite a value
        // the document already carries. A gap followed by another LABEL is the
        // two-fields-per-line case and stays.
        if (opt.require_blank_after_gap && !is_label_text(next_t)) continue;
        gap_end = item_rect(text[line[i + 1]]).x;
      }
      if (gap_end - gap_start < h * opt.min_gap) continue;

      const double left = gap_start + h * 0.25;
      const double width =
          std::min(gap_end - left, h * static_cast<double>(opt.max_field_width));
      if (width < opt.min_field_width * text_h) continue;

      FormField f;
      f.box = rect_to_box(cv::Rect(static_cast<int>(std::lround(left)), r.y,
                                   static_cast<int>(std::lround(width)),
                                   r.height));
      f.label = t;
      f.confidence = kLabelGapConfidence;
      f.source = "label_gap";
      out.push_back(std::move(f));
    }
  }
}

} // namespace detail

} // namespace turbo_ocr::forms
