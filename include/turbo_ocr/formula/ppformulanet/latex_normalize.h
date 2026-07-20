#pragma once

#include <string>

namespace turbo_ocr::formula {

// LaTeX text normalization applied after tokenizer decode (PaddleX parity):
// strip \text{...} wrapping around Chinese runs, drop quotes, collapse
// whitespace between non-letter tokens while preserving "\ " escapes, and
// de-space wrapper macros (operatorname/mathrm/text/mathbf). Distinct from
// tokenization itself — see src/formula/latex_normalize.cpp.
[[nodiscard]] std::string latex_post_process(const std::string &s);

} // namespace turbo_ocr::formula
