#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace turbo_ocr::env {

/// Read an environment variable with a fallback default. Set-but-empty is
/// treated as unset, matching every other parser here (an empty `VAR=` line
/// in a compose file means "not configured", not "configure the empty
/// string").
[[nodiscard]] inline std::string env_or(const char *name,
                                        std::string_view def) {
  if (const char *v = std::getenv(name); v && *v)
    return std::string(v);
  return std::string(def);
}

/// Check if an environment variable equals "1".
[[nodiscard]] inline bool env_enabled(const char *name) noexcept {
  const char *v = std::getenv(name);
  return v && v[0] == '1' && v[1] == '\0';
}

/// Lenient integer parse with bounds clamping. Garbage input → returns def.
/// Out-of-range input → clamps to [min_val, max_val]. Used by call sites that
/// genuinely want forgiving parsing; ServerConfig uses env_int_strict instead.
[[nodiscard]] inline int env_int(const char *name, int def,
                                  int min_val = 1, int max_val = 65535) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  char *end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') return def;
  if (val < min_val) return min_val;
  if (val > max_val) return max_val;
  return static_cast<int>(val);
}

/// Strict integer parse. Behavior:
///   - unset / empty       → returns def (no error)
///   - well-formed in-range → returns parsed value
///   - malformed / out-of-range → pushes a descriptive error into `errors`
///     and returns def so the rest of the loader can keep collecting.
/// The caller (typically ServerConfig::from_env) inspects the final error
/// vector and refuses to start if it is non-empty.
[[nodiscard]] inline int env_int_strict(const char *name, int def, int min_val,
                                         int max_val,
                                         std::vector<std::string> &errors) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  char *end = nullptr;
  // strtoll + ERANGE catches both numeric overflow and out-of-int values;
  // we then bounds-check against the caller-provided [min, max] window.
  errno = 0;
  long long val = std::strtoll(v, &end, 10);
  if (end == v || *end != '\0') {
    errors.push_back(std::string(name) + "=\"" + v +
                     "\" is not a valid integer");
    return def;
  }
  if (errno == ERANGE || val < static_cast<long long>(min_val) ||
      val > static_cast<long long>(max_val)) {
    errors.push_back(std::string(name) + "=\"" + v +
                     "\" is outside the allowed range [" +
                     std::to_string(min_val) + ", " +
                     std::to_string(max_val) + "]");
    return def;
  }
  return static_cast<int>(val);
}

/// Strict float parse. Same contract as env_int_strict: unset/empty → def,
/// malformed or out-of-range → error pushed, def returned.
[[nodiscard]] inline float env_float_strict(const char *name, float def,
                                             float min_val, float max_val,
                                             std::vector<std::string> &errors) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  char *end = nullptr;
  errno = 0;
  float val = std::strtof(v, &end);
  // Reject non-finite explicitly: NaN passes every range comparison (all NaN
  // compares are false), and inf would pass an unbounded range — no config
  // knob ever wants either, so treat both as malformed independent of range.
  if (end == v || *end != '\0' || !std::isfinite(val)) {
    errors.push_back(std::string(name) + "=\"" + v +
                     "\" is not a valid number");
    return def;
  }
  if (errno == ERANGE || val < min_val || val > max_val) {
    errors.push_back(std::string(name) + "=\"" + v +
                     "\" is outside the allowed range [" +
                     std::to_string(min_val) + ", " +
                     std::to_string(max_val) + "]");
    return def;
  }
  return val;
}

/// Strict boolean parse. Accepts (case-insensitive): 1/0, true/false,
/// yes/no, on/off. Anything else pushes an error. Unset → returns def.
[[nodiscard]] inline bool env_bool_strict(const char *name, bool def,
                                           std::vector<std::string> &errors) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (s == "1" || s == "true"  || s == "yes" || s == "on")  return true;
  if (s == "0" || s == "false" || s == "no"  || s == "off") return false;
  errors.push_back(std::string(name) + "=\"" + v +
                   "\" is not a boolean (use 1/0, true/false, yes/no, on/off)");
  return def;
}

/// Strict choice parse. `v` must be one of `choices` (case-sensitive).
/// Unset → returns def. Anything else pushes an error.
[[nodiscard]] inline std::string env_choice_strict(
    const char *name, std::string_view def,
    std::initializer_list<std::string_view> choices,
    std::vector<std::string> &errors) {
  const char *v = std::getenv(name);
  if (!v || !*v) return std::string(def);
  for (auto c : choices)
    if (c == v) return std::string(v);
  std::string msg = std::string(name) + "=\"" + v + "\" must be one of {";
  bool first = true;
  for (auto c : choices) {
    if (!first) msg += ", ";
    msg += std::string(c);
    first = false;
  }
  msg += "}";
  errors.push_back(std::move(msg));
  return std::string(def);
}

/// True iff the env var is set to a non-empty string.
[[nodiscard]] inline bool env_present(const char *name) noexcept {
  const char *v = std::getenv(name);
  return v && *v;
}

} // namespace turbo_ocr::env
