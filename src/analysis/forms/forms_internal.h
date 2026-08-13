#pragma once

// Shared across the src/analysis/forms/ TUs only — not part of the public surface in
// include/turbo_ocr/analysis/forms/.

#include <opencv2/core.hpp>

#include "turbo_ocr/analysis/forms/field_detector.h"

namespace turbo_ocr::forms::detail {

// Fraction of `r` that is ink in `binary`. Defined in field_masks.cpp.
[[nodiscard]] double band_ink_fraction(const cv::Mat &binary,
                                       const cv::Rect &r);

// Axis-aligned rect of an OCR item's quad. Defined in field_labels.cpp.
[[nodiscard]] cv::Rect item_rect(const OCRResultItem &item);

// Does any OCR item sit inside `r`? The emptiness test both the box detector
// and the table-cell detector use, so "empty" means the same thing to both.
[[nodiscard]] bool rect_is_empty(const cv::Rect &r,
                                 const std::vector<OCRResultItem> &text,
                                 float max_overlap);

// Detector 3. Defined in field_labels.cpp.
void collect_label_gap_fields(const std::vector<OCRResultItem> &text,
                              int page_width, float text_h,
                              const FieldOptions &opt,
                              std::vector<FormField> &out);

[[nodiscard]] std::string trim_copy(std::string_view s);

} // namespace turbo_ocr::forms::detail
