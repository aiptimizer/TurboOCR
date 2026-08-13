#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>
#include <string>
#include <string_view>
#include <vector>

namespace turbo_ocr::env {

// EVERY KNOB THIS PROCESS ACTUALLY READ, in the order first seen.
//
// The problem this solves: ServerConfig logs an "Effective server config" line
// that operators read as the truth about a running server, but ~80 reads across
// ~39 files go straight to std::getenv behind its back — TURBO_OCR_CUDA_GRAPHS,
// GPU_CCL, DET_DB_THRESH, TURBO_POOL_*, TURBO_DET_BATCH and more. None of them
// appear in that line, none are validated, and a typo is indistinguishable from
// the default. Someone debugging "why is my override not working" has nothing
// to look at.
//
// Recording here rather than at 80 call sites means a knob becomes visible the
// moment it is read through these helpers, with no per-site bookkeeping. It is
// deliberately NOT validation — that stays ServerConfig's job for the knobs it
// owns; this is the inventory that shows what else exists.
//
// Thread-safe because knobs are read from stage constructors on pool-warmup
// threads, not just from main().
namespace detail {
inline std::mutex &env_log_mu() {
  static std::mutex m;
  return m;
}
inline std::vector<std::pair<std::string, std::string>> &env_log() {
  static std::vector<std::pair<std::string, std::string>> v;
  return v;
}
inline void record(const char *name, std::string_view value) {
  std::lock_guard<std::mutex> lk(env_log_mu());
  auto &v = env_log();
  for (const auto &kv : v)
    if (kv.first == name) return;  // first read wins; later reads are the same
  v.emplace_back(name, std::string(value));
}
} // namespace detail

/// Every environment knob read through these helpers so far, name-sorted.
/// Call after startup to log what the process is actually running on.
[[nodiscard]] inline std::vector<std::pair<std::string, std::string>>
observed() {
  std::lock_guard<std::mutex> lk(detail::env_log_mu());
  auto v = detail::env_log();
  std::sort(v.begin(), v.end());
  return v;
}

/// Read an environment variable with a fallback default. Set-but-empty is
/// treated as unset, matching every other parser here (an empty `VAR=` line
/// in a compose file means "not configured", not "configure the empty
/// string").
[[nodiscard]] inline std::string env_or(const char *name,
                                        std::string_view def) {
  if (const char *v = std::getenv(name); v && *v) {
    detail::record(name, v);
    return std::string(v);
  }
  return std::string(def);
}

/// Check if an environment variable equals "1".
[[nodiscard]] inline bool env_enabled(const char *name) noexcept {
  const char *v = std::getenv(name);
  if (v && *v) detail::record(name, v);
  return v && v[0] == '1' && v[1] == '\0';
}

/// Lenient integer parse with bounds clamping. Garbage input → returns def.
/// Out-of-range input → clamps to [min_val, max_val]. Used by call sites that
/// genuinely want forgiving parsing; ServerConfig uses env_int_strict instead.
[[nodiscard]] inline int env_int(const char *name, int def,
                                  int min_val = 1, int max_val = 65535) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  detail::record(name, v);
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

/// Lenient float parse with bounds clamping — the float twin of env_int, and
/// the same bargain: garbage input → def, out-of-range → clamped. Used by the
/// tuning knobs (DET_DB_THRESH and friends) that were reading std::atof
/// directly, where atof's "garbage is 0.0" would silently set a threshold to
/// zero rather than leave the model default alone.
[[nodiscard]] inline float env_float(const char *name, float def, float min_val,
                                     float max_val) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  detail::record(name, v);
  char *end = nullptr;
  float val = std::strtof(v, &end);
  if (end == v || *end != '\0' || !std::isfinite(val)) return def;
  return std::clamp(val, min_val, max_val);
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

/// THE boolean vocabulary, in one place. Every reader of a boolean knob must
/// use these two, because a value that the boot validator accepts as true and
/// a runtime reader treats as false runs the server with a feature the operator
/// was told is on. (`CLS_ALL_BOXES=true` was exactly that shape: validated
/// against this set at boot, then re-read by a hand-written predicate.)
[[nodiscard]] inline bool is_truthy(std::string_view v) {
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s == "1" || s == "true" || s == "yes" || s == "on";
}
[[nodiscard]] inline bool is_falsy(std::string_view v) {
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s == "0" || s == "false" || s == "no" || s == "off";
}

/// Strict boolean parse. Accepts (case-insensitive): 1/0, true/false,
/// yes/no, on/off. Anything else pushes an error. Unset → returns def.
[[nodiscard]] inline bool env_bool_strict(const char *name, bool def,
                                           std::vector<std::string> &errors) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  if (is_truthy(v)) return true;
  if (is_falsy(v)) return false;
  errors.push_back(std::string(name) + "=\"" + v +
                   "\" is not a boolean (use 1/0, true/false, yes/no, on/off)");
  return def;
}

/// Lenient boolean read over the same vocabulary: unset, empty or anything not
/// truthy is false. The runtime counterpart to env_bool_strict for knobs the
/// boot validator has already checked.
[[nodiscard]] inline bool env_truthy(const char *name) {
  return is_truthy(env_or(name, ""));
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

/// True iff the env var is set to a non-empty string. Records, like every other
/// helper here: a knob whose mere presence changes behaviour is exactly the kind
/// the inventory exists to surface, and "it was only checked, not parsed" is not
/// a distinction an operator debugging an override can make.
[[nodiscard]] inline bool env_present(const char *name) {
  const char *v = std::getenv(name);
  if (v && *v) detail::record(name, v);
  return v && *v;
}

} // namespace turbo_ocr::env
