#pragma once

// Shared textline-orientation classifier options (GPU + CPU pipelines).

#include <cctype>
#include <string>
#include <string_view>

#include "turbo_ocr/base/env_utils.h"

namespace turbo_ocr::classification {

// True for every value env_bool_strict accepts as true (1/true/yes/on,
// case-insensitive).
//
// It DELEGATES rather than restating the set. The comment that used to sit here
// said this predicate "must stay in sync with env_utils.h" — a copy that must
// stay in sync is a copy that will not, and the failure it invited is silent:
// CLS_ALL_BOXES=true validates at boot and then reads as false at runtime, so
// the server reports a feature on while running it off. One definition is what
// makes the requirement true instead of merely asked for.
[[nodiscard]] inline bool truthy_env_value(const char *v) {
  return v && *v && env::is_truthy(v);
}

// CLS_ALL_BOXES=1 — run the 0/180 orientation classifier on every crop
// instead of only vertical-looking ones (h >= 1.5*w). Off by default: the
// vertical-only gate exists because upright documents gain nothing from
// classifying horizontal lines, but it also means an upside-down horizontal
// line is never checked — scans with mixed per-line orientations need this.
// The value is validated strictly at boot (server_config); this helper only
// re-reads the already-validated env.
[[nodiscard]] inline bool cls_all_boxes_enabled() {
  static const bool e = env::env_truthy("CLS_ALL_BOXES");
  return e;
}

// CLS_ONNX (GPU) / CLS_MODEL (CPU) accept a filesystem path or one of these
// shorthand names for the shipped textline-orientation variants. Returns the
// resolved path (shorthand -> bundled file), or `value` unchanged when it is
// not a known shorthand.
[[nodiscard]] inline std::string resolve_cls_shorthand(std::string_view value) {
  if (value == "x0_25") return "models/cls.onnx";
  if (value == "x1_0")  return "models/cls_x1_0.onnx";
  return std::string(value);
}

} // namespace turbo_ocr::classification
