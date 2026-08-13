// Ruling-line and closed-box recovery from the page raster (detectors 1 + 2).
//
// A printed form draws its blanks. Those strokes are invisible to OCR — on the
// reference scan the recogniser returns "Unterschrift:" and stops, with no
// underscore run for the rule beside it — so morphology is the only thing that
// can see them. Opening the binarised page with a long 1-D kernel keeps runs
// that are straight and long in that direction and erases everything else.

#include "turbo_ocr/analysis/forms/field_detector.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "forms_internal.h"

namespace turbo_ocr::forms {

namespace {

// Fraction of `strip` that is ink. Used both for "is the band above this rule
// blank" and for the four-edge coverage test.
[[nodiscard]] double ink_fraction(const cv::Mat &mask, const cv::Rect &r) {
  const cv::Rect clipped = r & cv::Rect(0, 0, mask.cols, mask.rows);
  if (clipped.width <= 0 || clipped.height <= 0) return 0.0;
  const double area = static_cast<double>(clipped.width) * clipped.height;
  return cv::countNonZero(mask(clipped)) / area;
}

// Fraction of a box side that is actually covered by a line, measured by
// scanning the side's own axis and asking whether the mask has ink anywhere in
// a small band across it. Coverage — not ink area — is the right question: a
// 2px rule over a 5px band would score 0.4 on area but 1.0 on coverage.
[[nodiscard]] double edge_coverage(const cv::Mat &mask, const cv::Rect &side,
                                   bool horizontal) {
  const cv::Rect clipped = side & cv::Rect(0, 0, mask.cols, mask.rows);
  if (clipped.width <= 0 || clipped.height <= 0) return 0.0;
  const cv::Mat sub = mask(clipped);
  // Reduce across the thin axis: any ink in that column/row counts as covered.
  cv::Mat profile;
  cv::reduce(sub, profile, horizontal ? 0 : 1, cv::REDUCE_MAX, CV_8U);
  const int covered = cv::countNonZero(profile);
  const int total = horizontal ? clipped.width : clipped.height;
  return total > 0 ? static_cast<double>(covered) / total : 0.0;
}

} // namespace

cv::Mat binarize_page(const cv::Mat &page) {
  if (page.empty()) return {};
  cv::Mat gray;
  if (page.channels() == 3)
    cv::cvtColor(page, gray, cv::COLOR_BGR2GRAY);
  else if (page.channels() == 4)
    cv::cvtColor(page, gray, cv::COLOR_BGRA2GRAY);
  else
    gray = page;
  if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8U);

  cv::Mat bin;
  cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  // A near-blank page has no bimodal histogram to split, so Otsu picks a
  // threshold inside the paper noise and half the page comes back as "ink".
  // Anything over half is not a document; fall back to a fixed dark cut so the
  // morphology downstream sees strokes rather than a solid block.
  if (ink_fraction(bin, cv::Rect(0, 0, bin.cols, bin.rows)) > 0.5)
    cv::threshold(gray, bin, 160, 255, cv::THRESH_BINARY_INV);
  return bin;
}

LineMasks extract_line_masks(const cv::Mat &binary, int h_kernel,
                             int v_kernel) {
  LineMasks out;
  if (binary.empty()) return out;
  if (h_kernel > 0) {
    const cv::Mat hk = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(std::max(3, h_kernel), 1));
    cv::morphologyEx(binary, out.horizontal, cv::MORPH_OPEN, hk);
  }
  if (v_kernel > 0) {
    const cv::Mat vk = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(1, std::max(3, v_kernel)));
    cv::morphologyEx(binary, out.vertical, cv::MORPH_OPEN, vk);
  }
  return out;
}

std::vector<cv::Rect> find_rule_segments(const cv::Mat &horizontal,
                                         float text_h,
                                         const FieldOptions &opt) {
  std::vector<cv::Rect> out;
  if (horizontal.empty() || text_h <= 0.0f) return out;

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(horizontal, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  const double min_len = opt.min_rule_len * text_h;
  const double max_thick =
      std::max(2.0, static_cast<double>(opt.max_rule_thickness) * text_h);
  out.reserve(contours.size());
  for (const auto &c : contours) {
    const cv::Rect r = cv::boundingRect(c);
    if (r.width < min_len) continue;
    if (r.height > max_thick) continue;
    out.push_back(r);
  }
  std::ranges::sort(out, [](const cv::Rect &a, const cv::Rect &b) {
    return a.y != b.y ? a.y < b.y : a.x < b.x;
  });
  return out;
}

std::vector<cv::Rect> find_closed_boxes(const LineMasks &masks, float text_h,
                                        const FieldOptions &opt) {
  std::vector<cv::Rect> out;
  if (masks.horizontal.empty() || masks.vertical.empty() || text_h <= 0.0f)
    return out;

  cv::Mat grid;
  cv::bitwise_or(masks.horizontal, masks.vertical, grid);

  // The CELLS are the holes in the line grid, so ask for the hierarchy and
  // keep the contours that have a parent. This proposes; the four-edge test
  // below is what actually decides, because a letter counter is a hole too.
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(grid, contours, hierarchy, cv::RETR_CCOMP,
                   cv::CHAIN_APPROX_SIMPLE);

  const double min_side = opt.min_box_side * text_h;
  // The edge band has to be thick enough to catch a rule that bows by a pixel
  // or two on a scan, but thin enough that it is still testing the edge.
  const int band = std::max(2, static_cast<int>(std::lround(0.15 * text_h)));

  for (size_t i = 0; i < contours.size(); ++i) {
    if (hierarchy[i][3] < 0) continue; // outer contour of the ink, not a cell
    cv::Rect r = cv::boundingRect(contours[i]);
    if (r.width < min_side || r.height < min_side) continue;
    // A hole that does not fill its own bounding rect is not a rectangle.
    const double fill = cv::contourArea(contours[i]) /
                        (static_cast<double>(r.width) * r.height);
    if (fill < 0.65) continue;

    // The hole is the INTERIOR, so its own edges sit just outside it.
    const cv::Rect top(r.x, r.y - band, r.width, band * 2);
    const cv::Rect bottom(r.x, r.y + r.height - band, r.width, band * 2);
    const cv::Rect left(r.x - band, r.y, band * 2, r.height);
    const cv::Rect right(r.x + r.width - band, r.y, band * 2, r.height);
    if (edge_coverage(masks.horizontal, top, true) < opt.min_edge_coverage)
      continue;
    if (edge_coverage(masks.horizontal, bottom, true) < opt.min_edge_coverage)
      continue;
    if (edge_coverage(masks.vertical, left, false) < opt.min_edge_coverage)
      continue;
    if (edge_coverage(masks.vertical, right, false) < opt.min_edge_coverage)
      continue;
    out.push_back(r);
  }
  std::ranges::sort(out, [](const cv::Rect &a, const cv::Rect &b) {
    return a.y != b.y ? a.y < b.y : a.x < b.x;
  });
  return out;
}

// ── Used by field_detector.cpp ────────────────────────────────────────────
namespace detail {

double band_ink_fraction(const cv::Mat &binary, const cv::Rect &r) {
  return ink_fraction(binary, r);
}

} // namespace detail

} // namespace turbo_ocr::forms

namespace turbo_ocr::forms {

void trim_fields_off_text(std::vector<FormField> &fields,
                          const std::vector<OCRResultItem> &text,
                          const FieldOptions &opt) {
  if (fields.empty() || text.empty()) return;

  // A field may legitimately sit over a word — an empty table cell whose header
  // OCR read across it, a rule under a caption. Only an EDGE bite is trimmed:
  // the field keeps most of itself and gives up the sliver that lapped onto the
  // word. A field mostly covered by a word is a different mistake and is left
  // to the container and merge passes.
  constexpr double kMaxBite = 0.45;
  // Keep a hair of daylight so the widget's border does not touch the glyph.
  constexpr double kGap = 1.0;

  for (FormField &f : fields) {
    cv::Rect r = box_to_rect(f.box);
    if (r.width <= 2 || r.height <= 2) continue;

    for (const OCRResultItem &item : text) {
      if (detail::trim_copy(item.text).empty()) continue;
      const cv::Rect w = detail::item_rect(item);
      const cv::Rect hit = r & w;
      if (hit.width <= 0 || hit.height <= 0) continue;
      // Only worth trimming when the word overlaps most of the field's height;
      // a clipped corner is not the field sitting on the text.
      if (hit.height < 0.5 * r.height) continue;
      const double bite = static_cast<double>(hit.width) / r.width;
      if (bite <= 0.0 || bite > kMaxBite) continue;

      const int word_cx = w.x + w.width / 2;
      const int field_cx = r.x + r.width / 2;
      if (word_cx > field_cx) {
        // The word is to the right: pull the field's right edge back.
        const int right = static_cast<int>(w.x - kGap);
        if (right > r.x + 2) r.width = right - r.x;
      } else {
        const int left = static_cast<int>(w.x + w.width + kGap);
        if (left < r.x + r.width - 2) {
          r.width = (r.x + r.width) - left;
          r.x = left;
        }
      }
    }
    f.box = rect_to_box(r);
  }
  (void)opt;
}

} // namespace turbo_ocr::forms

namespace turbo_ocr::forms {

void name_fields_from_columns(std::vector<FormField> &fields,
                              const std::vector<OCRResultItem> &text,
                              const FieldOptions &opt) {
  (void)opt;
  if (fields.empty() || text.empty()) return;

  // Which unnamed fields there are, and where.
  std::vector<size_t> unnamed;
  for (size_t i = 0; i < fields.size(); ++i)
    if (detail::trim_copy(fields[i].label).empty()) unnamed.push_back(i);
  if (unnamed.empty()) return;

  for (size_t idx : unnamed) {
    const cv::Rect r = box_to_rect(fields[idx].box);
    if (r.width <= 2 || r.height <= 2) continue;

    // The nearest text ABOVE that shares most of this field's width. Sharing
    // the width is what makes it a column header rather than a neighbour that
    // merely happens to be higher up the page.
    int best = -1;
    int best_bottom = -1;
    for (size_t k = 0; k < text.size(); ++k) {
      const std::string t = detail::trim_copy(text[k].text);
      if (t.empty() || is_rule_text(t)) continue;
      const cv::Rect w = detail::item_rect(text[k]);
      const int bottom = w.y + w.height;
      if (bottom > r.y) continue;  // not above
      const int ox = std::min(w.x + w.width, r.x + r.width) - std::max(w.x, r.x);
      if (ox <= 0) continue;
      // Most of the narrower of the two has to be shared, or a long heading
      // spanning the page would claim every column under it.
      const int narrower = std::min(w.width, r.width);
      if (narrower <= 0 || ox * 2 < narrower) continue;
      if (bottom > best_bottom) {
        best_bottom = bottom;
        best = static_cast<int>(k);
      }
    }
    if (best < 0) continue;

    // Which row of that column this is: count the fields sharing the column
    // that sit above it — but only ones of the same KIND and roughly the same
    // size. A column of table cells usually has unrelated fields somewhere
    // above it (a row of tick boxes, a wide header blank), and counting those
    // numbered the second row of the table "11".
    int row = 1;
    for (size_t j = 0; j < fields.size(); ++j) {
      if (j == idx) continue;
      if (fields[j].type != fields[idx].type) continue;
      const cv::Rect o = box_to_rect(fields[j].box);
      if (o.y + o.height > r.y) continue;
      const int ox = std::min(o.x + o.width, r.x + r.width) - std::max(o.x, r.x);
      if (ox <= 0) continue;
      const int narrower = std::min(o.width, r.width);
      if (narrower <= 0 || ox * 2 < narrower) continue;
      // Same column means comparable width, not merely a shared span.
      const double ratio = static_cast<double>(std::min(o.width, r.width)) /
                           std::max(o.width, r.width);
      if (ratio < 0.6) continue;
      ++row;
    }

    fields[idx].label =
        detail::trim_copy(text[static_cast<size_t>(best)].text) + " " +
        std::to_string(row);
  }
}

} // namespace turbo_ocr::forms
