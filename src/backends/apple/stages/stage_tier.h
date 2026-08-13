#pragma once

// Tier name embedded in a model/dict path, or "" when it carries no tier hint.
// PP-OCRv6 ships three tiers whose det box_thresh and rec DICTIONARY differ, and
// nothing in the file format identifies which one an artefact belongs to — the
// path is the only signal available at this seam. Shared by MpsDetector (per-tier
// DB thresholds) and MpsRecognizer (per-tier ANE package resolution).

#include <string>

namespace turbo_ocr::apple {

inline std::string tier_from_path(const std::string &p) {
  if (p.find("medium") != std::string::npos) return "medium";
  if (p.find("small") != std::string::npos) return "small";
  if (p.find("tiny") != std::string::npos) return "tiny";
  return {};
}

} // namespace turbo_ocr::apple
