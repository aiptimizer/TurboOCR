#pragma once

#include <string>

#include "turbo_ocr/base/geometry/box.h"

namespace turbo_ocr::forms {

// What a caller would create in the PDF form dictionary for this rectangle.
//
// Text and Checkbox are the two kinds PAGE GEOMETRY alone can tell apart: a
// place to write, and a small square to tick. Signature is the third class the
// FFDetr model predicts; nothing in the morphology distinguishes a signature
// line from any other write-on rule, so a Signature only ever comes from the
// model. Radio groups and dropdowns still need the form's own semantics, which
// a raster does not carry, and are not guessed at.
enum class FieldType { Text, Checkbox, Signature };

[[nodiscard]] constexpr const char *field_type_name(FieldType t) noexcept {
  switch (t) {
  case FieldType::Checkbox: return "checkbox";
  case FieldType::Signature: return "signature";
  case FieldType::Text: break;
  }
  return "text";
}

// One proposed fillable region, in PAGE PIXELS of the raster it was found in
// (the same space as OCRResultItem::box), so a caller converts to PDF user
// space exactly once, using that page's own scale.
//
// Everything here is a PROPOSAL. A wrong field is worse than a missing one —
// it silently captures input the document never meant to have — which is why
// `source` and `confidence` travel with the geometry instead of being dropped
// once a rectangle is emitted.
struct FormField {
  FieldType type = FieldType::Text;
  Box box{};
  // Nearest plausible label: the OCR text to the left on the same line, else
  // the one directly above. Empty when nothing plausible was near — the field
  // is still emitted, because an unlabelled blank is still a blank.
  std::string label;
  float confidence = 0.0f;
  // WHICH detector(s) proposed this rectangle, '+'-joined after a merge
  // ("rule+label_gap"). Carried so a wrong field can be traced back to the
  // geometry that argued for it rather than being an anonymous rectangle.
  std::string source;

  // Index of the aligned run of choice buttons this one belongs to, or -1.
  //
  // A row or column of equally-sized, evenly-spaced checkboxes IS one control
  // in the document's mind — "Oats / Barley / Wheat / Corn". That much is a
  // geometric fact and is reported here.
  //
  // Whether the run is EXCLUSIVE (a radio group) is deliberately NOT decided:
  // nothing on a raster distinguishes pick-one from pick-many, and guessing
  // wrong is asymmetric — turning a multi-select into a radio group silently
  // stops the user ticking two boxes, while leaving them independent costs
  // nothing but the grouping. So this reports the run and lets a caller that
  // knows the form's semantics make it exclusive.
  int group = -1;
};

} // namespace turbo_ocr::forms
