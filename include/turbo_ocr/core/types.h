#pragma once

#include "turbo_ocr/base/geometry/box.h"
#include <string>

namespace turbo_ocr {

struct OCRResultItem {
  std::string text;
  float confidence = 0.0f;
  Box box{};
  // Optional provenance marker for /ocr/pdf modes that can mix native PDF
  // text extraction with OCR. Default empty == "ocr" (the serializer
  // omits this field when empty, so non-PDF endpoints see zero change).
  // Values used: "" (implicit ocr) | "pdf" (from PDFium text layer).
  std::string source;
  // Cross-reference IDs used ONLY when layout detection is enabled.
  // Default -1 → serializer omits the field, so responses without layout
  // stay byte-identical to pre-layout clients. When >= 0, emitted as
  // `"id"` / `"layout_id"` in the JSON response.
  int id = -1;
  int layout_id = -1;
};

/// Minimum confidence to keep a recognition result (matching PaddleOCR Python).
inline constexpr float kDropScore = 0.5f;

} // namespace turbo_ocr
