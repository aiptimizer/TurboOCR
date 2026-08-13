#pragma once

// Internal to src/backends/intel/stages/: the helpers shared by the two stage
// translation units (intel_stages.cpp — detector + recognizer;
// intel_stages_structure.cpp — classifier + layout).
//
// The pair was ONE file until it crossed the 900-line gate
// (tools/checks/architecture.sh). Splitting it moved these out of an anonymous
// namespace, where a copy per TU would have been the obvious and wrong fix:
// this arm already paid for duplicated helpers once (see the norm-factory note
// in intel_stages.cpp). `inline` keeps exactly one definition across both TUs.

#include <cstddef>
#include <string>
#include <vector>

namespace turbo_ocr::intel::stagesdetail {

// OpenVINO port names are positional; a model whose metadata omits one still
// has to bind. Falling back to the documented default name beats an empty
// string, which OpenVINO reports as "port not found" several frames later.
[[nodiscard]] inline std::string tensor_name(const std::vector<std::string> &names,
                                             std::size_t i, const char *fallback) {
  return i < names.size() ? names[i] : std::string(fallback);
}

} // namespace turbo_ocr::intel::stagesdetail
