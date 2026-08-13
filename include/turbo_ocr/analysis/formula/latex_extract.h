#pragma once

#include <string>

namespace turbo_ocr::formula {

// Pull LaTeX out of a VLM assistant message. Tries, in order: a ```latex
// fence, $$...$$ display math, \[...\] display math, $...$ inline math, and
// finally the bare message with a leading "LaTeX:"/"Answer:" prefix stripped
// and whitespace trimmed. The ONE implementation shared by the vLLM sidecar
// backend (vlm_formula) and the generic OpenAI endpoint — these were
// hand-mirrored copies before and had already begun to drift.
[[nodiscard]] std::string extract_latex(const std::string &msg);

} // namespace turbo_ocr::formula
