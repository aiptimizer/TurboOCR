#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>
#include <type_traits>

namespace turbo_ocr::log {

enum class Level : int { Debug = 0, Info = 1, Warn = 2, Error = 3 };

inline Level parse_level(const char *s) {
  if (!s || !*s) return Level::Info;
  if (s[0] == 'd' || s[0] == 'D') return Level::Debug;
  if (s[0] == 'w' || s[0] == 'W') return Level::Warn;
  if (s[0] == 'e' || s[0] == 'E') return Level::Error;
  return Level::Info;
}

enum class Format { Json, Text };

inline Format parse_format(const char *s) {
  if (!s || !*s) return Format::Json;
  if (s[0] == 't' || s[0] == 'T') return Format::Text;
  return Format::Json;
}

// ── Global config (initialized once on first use) ──────────────────────

struct Config {
  Level  min_level;
  Format format;
};

inline const Config &config() {
  static const Config cfg = {
      parse_level(std::getenv("LOG_LEVEL")),  // pre-commit-allow-getenv (logger bootstrap)
      parse_format(std::getenv("LOG_FORMAT")),  // pre-commit-allow-getenv (logger bootstrap)
  };
  return cfg;
}

// ── Thread-safe stderr writer ──────────────────────────────────────────

inline std::mutex &log_mutex() {
  static std::mutex mtx;
  return mtx;
}

// ── Timestamp formatting ───────────────────────────────────────────────

inline int format_timestamp_iso(char *buf, size_t cap) {
  auto now = std::chrono::system_clock::now();
  auto ms_since_epoch =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::time_t secs = static_cast<std::time_t>(ms_since_epoch / 1000);
  int millis = static_cast<int>(ms_since_epoch % 1000);
  std::tm tm{};
  gmtime_r(&secs, &tm);
  return std::snprintf(buf, cap, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
      tm.tm_hour, tm.tm_min, tm.tm_sec, millis);
}

inline int format_timestamp_text(char *buf, size_t cap) {
  auto now = std::chrono::system_clock::now();
  auto ms_since_epoch =
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::time_t secs = static_cast<std::time_t>(ms_since_epoch / 1000);
  int millis = static_cast<int>(ms_since_epoch % 1000);
  std::tm tm{};
  gmtime_r(&secs, &tm);
  return std::snprintf(buf, cap, "%04d-%02d-%02d %02d:%02d:%02d.%03d",
      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
      tm.tm_hour, tm.tm_min, tm.tm_sec, millis);
}

// ── Value serialization helpers ────────────────────────────────────────

namespace detail {

// Advance the write cursor by a formatter's return value, clamped to what
// can actually have been written. snprintf reports the WOULD-BE length on
// truncation (>= cap); advancing by that raw value walks p past the buffer
// and wraps rem (size_t) — an out-of-bounds write on the next append. Every
// cursor advance in this file goes through here so truncation degrades to a
// clipped line, never memory corruption. rem stays >= 1 (snprintf writes at
// most cap-1 chars plus NUL), preserving the caller's \n\0 reserve.
inline void advance(char *&p, size_t &rem, int n) {
  if (n < 0) return;                       // encoding error: nothing written
  size_t adv = static_cast<size_t>(n);
  if (rem == 0) return;
  if (adv >= rem) adv = rem - 1;           // truncated: cap-1 chars + NUL
  p += adv;
  rem -= adv;
}

// Append a JSON-escaped string value (with quotes) to buffer.
// Returns number of chars written.
inline int json_string(char *buf, size_t cap, std::string_view s) {
  if (cap < 3) return 0;
  size_t pos = 0;
  buf[pos++] = '"';
  for (char c : s) {
    if (pos + 3 >= cap) break;  // leave room for closing quote + null
    if (c == '"')       { buf[pos++] = '\\'; buf[pos++] = '"'; }
    else if (c == '\\') { buf[pos++] = '\\'; buf[pos++] = '\\'; }
    else if (c == '\n') { buf[pos++] = '\\'; buf[pos++] = 'n'; }
    else if (c == '\r') { buf[pos++] = '\\'; buf[pos++] = 'r'; }
    else if (c == '\t') { buf[pos++] = '\\'; buf[pos++] = 't'; }
    else                 buf[pos++] = c;
  }
  buf[pos++] = '"';
  buf[pos] = '\0';
  return static_cast<int>(pos);
}

// Append a value to buffer in JSON format. Returns chars written.
inline int append_json_value(char *buf, size_t cap, std::string_view v) {
  return json_string(buf, cap, v);
}
inline int append_json_value(char *buf, size_t cap, const char *v) {
  return json_string(buf, cap, std::string_view(v));
}
inline int append_json_value(char *buf, size_t cap, int v) {
  return std::snprintf(buf, cap, "%d", v);
}
inline int append_json_value(char *buf, size_t cap, long v) {
  return std::snprintf(buf, cap, "%ld", v);
}
inline int append_json_value(char *buf, size_t cap, long long v) {
  return std::snprintf(buf, cap, "%lld", v);
}
inline int append_json_value(char *buf, size_t cap, unsigned int v) {
  return std::snprintf(buf, cap, "%u", v);
}
inline int append_json_value(char *buf, size_t cap, unsigned long v) {
  return std::snprintf(buf, cap, "%lu", v);
}
inline int append_json_value(char *buf, size_t cap, unsigned long long v) {
  return std::snprintf(buf, cap, "%llu", v);
}
inline int append_json_value(char *buf, size_t cap, float v) {
  return std::snprintf(buf, cap, "%.3f", static_cast<double>(v));
}
inline int append_json_value(char *buf, size_t cap, double v) {
  return std::snprintf(buf, cap, "%.3f", v);
}

// Append a value to buffer in text format (key=value). Returns chars written.
inline int append_text_value(char *buf, size_t cap, std::string_view v) {
  return std::snprintf(buf, cap, "%.*s", static_cast<int>(v.size()), v.data());
}
inline int append_text_value(char *buf, size_t cap, const char *v) {
  return std::snprintf(buf, cap, "%s", v);
}
inline int append_text_value(char *buf, size_t cap, int v) {
  return std::snprintf(buf, cap, "%d", v);
}
inline int append_text_value(char *buf, size_t cap, long v) {
  return std::snprintf(buf, cap, "%ld", v);
}
inline int append_text_value(char *buf, size_t cap, long long v) {
  return std::snprintf(buf, cap, "%lld", v);
}
inline int append_text_value(char *buf, size_t cap, unsigned int v) {
  return std::snprintf(buf, cap, "%u", v);
}
inline int append_text_value(char *buf, size_t cap, unsigned long v) {
  return std::snprintf(buf, cap, "%lu", v);
}
inline int append_text_value(char *buf, size_t cap, unsigned long long v) {
  return std::snprintf(buf, cap, "%llu", v);
}
inline int append_text_value(char *buf, size_t cap, float v) {
  return std::snprintf(buf, cap, "%.3f", static_cast<double>(v));
}
inline int append_text_value(char *buf, size_t cap, double v) {
  return std::snprintf(buf, cap, "%.3f", v);
}

// ── Key-value pair writers (recursive variadic) ────────────────────────

inline void write_json_kvs(char *&, size_t &) {}

template <typename V, typename... Rest>
void write_json_kvs(char *&p, size_t &rem, std::string_view key, V &&val, Rest &&...rest) {
  // comma + key
  advance(p, rem, std::snprintf(p, rem, ","));
  advance(p, rem, json_string(p, rem, key));
  advance(p, rem, std::snprintf(p, rem, ":"));
  advance(p, rem, append_json_value(p, rem, std::forward<V>(val)));
  write_json_kvs(p, rem, std::forward<Rest>(rest)...);
}

inline void write_text_kvs(char *&, size_t &) {}

template <typename V, typename... Rest>
void write_text_kvs(char *&p, size_t &rem, std::string_view key, V &&val, Rest &&...rest) {
  advance(p, rem, std::snprintf(p, rem, " %.*s=",
                                static_cast<int>(key.size()), key.data()));
  advance(p, rem, append_text_value(p, rem, std::forward<V>(val)));
  write_text_kvs(p, rem, std::forward<Rest>(rest)...);
}

} // namespace detail

// ── Level name helpers ─────────────────────────────────────────────────

inline const char *level_name_json(Level lvl) {
  switch (lvl) {
    case Level::Debug: return "debug";
    case Level::Info:  return "info";
    case Level::Warn:  return "warn";
    case Level::Error: return "error";
  }
  return "info";
}

inline const char *level_name_text(Level lvl) {
  switch (lvl) {
    case Level::Debug: return "DEBUG";
    case Level::Info:  return "INFO";
    case Level::Warn:  return "WARN";
    case Level::Error: return "ERROR";
  }
  return "INFO";
}

// ── Main log function ──────────────────────────────────────────────────

template <typename... KVs>
void log_msg(Level lvl, std::string_view msg, KVs &&...kvs) {
  if (static_cast<int>(lvl) < static_cast<int>(config().min_level)) return;

  // Fixed 4KB thread-local buffer — zero allocation
  // Fixed 4KB thread-local buffer — zero allocation. Every cursor move goes
  // through detail::advance (truncation-safe); rem never reaches 0, and the
  // -3 reserve leaves room for the closing '}' + '\n' + '\0' even on a
  // fully clipped line.
  thread_local char buf[4096];
  char *p = buf;
  size_t rem = sizeof(buf) - 3;

  if (config().format == Format::Json) {
    // {"ts":"...","level":"...","msg":"...",...}\n
    detail::advance(p, rem, std::snprintf(p, rem, "{\"ts\":\""));
    detail::advance(p, rem, format_timestamp_iso(p, rem));
    detail::advance(p, rem, std::snprintf(p, rem, "\",\"level\":\"%s\",\"msg\":",
                                          level_name_json(lvl)));
    detail::advance(p, rem, detail::json_string(p, rem, msg));
    detail::write_json_kvs(p, rem, std::forward<KVs>(kvs)...);
    *p++ = '}';
  } else {
    // [2026-04-13 10:00:00.123] [INFO] message key=val ...\n
    *p++ = '['; rem--;
    detail::advance(p, rem, format_timestamp_text(p, rem));
    detail::advance(p, rem, std::snprintf(p, rem, "] [%s] %.*s",
                                          level_name_text(lvl),
                                          static_cast<int>(msg.size()),
                                          msg.data()));
    detail::write_text_kvs(p, rem, std::forward<KVs>(kvs)...);
  }
  *p++ = '\n';
  *p = '\0';

  size_t len = static_cast<size_t>(p - buf);
  std::lock_guard<std::mutex> lock(log_mutex());
  std::fwrite(buf, 1, len, stderr);
}

// ── Per-call-site rate limiting ─────────────────────────────────────────
// Goal: prevent a hot error path (e.g. 1000 qps × 3 lines) from flooding
// stderr / log aggregators. Each call site keeps its own atomic state;
// when a window rolls over, a single "[suppressed K logs]" line is emitted.

struct RateLimitConfig {
  int      max_logs_per_window;  // N (0 disables rate limiting)
  long     window_ms;            // W in milliseconds
};

inline const RateLimitConfig &ratelimit_config() {
  static const RateLimitConfig cfg = []() {
    RateLimitConfig c{10, 1000};
    if (const char *s = std::getenv("TOCR_LOG_RATELIMIT")) {  // pre-commit-allow-getenv (logger bootstrap)
      // "0" disables; "N" sets max logs per default 1s window;
      // "N:W_MS" sets both. Anything unparseable falls back to defaults.
      char *end = nullptr;
      long n = std::strtol(s, &end, 10);
      if (end != s) {
        c.max_logs_per_window = static_cast<int>(n);
        if (end && *end == ':') {
          char *end2 = nullptr;
          long w = std::strtol(end + 1, &end2, 10);
          if (end2 != end + 1 && w > 0) c.window_ms = w;
        }
      }
    }
    return c;
  }();
  return cfg;
}

struct RateLimitSlot {
  std::atomic<long long> window_start_ms{0};
  std::atomic<int>       count{0};
  std::atomic<int>       suppressed{0};
};

// Returns: 1 if the current call should log normally, 0 if it should be
// suppressed. Out-param suppressed_to_emit, if non-zero, is the count of
// logs suppressed during the previous window — caller should emit a
// rollup line for them.
inline int ratelimit_check(RateLimitSlot &slot, int &suppressed_to_emit) {
  suppressed_to_emit = 0;
  const auto &cfg = ratelimit_config();
  if (cfg.max_logs_per_window <= 0) return 1;  // disabled

  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  long long ws = slot.window_start_ms.load(std::memory_order_relaxed);
  if (ws == 0 || now_ms - ws >= cfg.window_ms) {
    // Try to roll the window. The thread that wins the CAS owns the
    // rollup emission for the prior window's suppressed count.
    if (slot.window_start_ms.compare_exchange_strong(
            ws, now_ms, std::memory_order_acq_rel)) {
      suppressed_to_emit = slot.suppressed.exchange(0, std::memory_order_acq_rel);
      slot.count.store(1, std::memory_order_release);
      return 1;
    }
    // Lost the race — fall through and treat as same window.
  }
  int prev = slot.count.fetch_add(1, std::memory_order_acq_rel);
  if (prev < cfg.max_logs_per_window) return 1;
  slot.suppressed.fetch_add(1, std::memory_order_relaxed);
  return 0;
}

} // namespace turbo_ocr::log

// ── Convenience macros ─────────────────────────────────────────────────
// Prefixed with TOCR_ to avoid collision with Drogon/trantor's LOG_* macros.

#define TOCR_LOG_DEBUG(msg, ...) ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Debug, msg, ##__VA_ARGS__)
#define TOCR_LOG_INFO(msg, ...)  ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Info,  msg, ##__VA_ARGS__)
#define TOCR_LOG_WARN(msg, ...)  ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Warn,  msg, ##__VA_ARGS__)
#define TOCR_LOG_ERROR(msg, ...) ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Error, msg, ##__VA_ARGS__)

// Rate-limited variant. Each expansion site has its own static atomic
// counter — when more than N logs would fire within W ms, extras are
// dropped and a single "[suppressed logs]" rollup is emitted at window
// roll. Set TOCR_LOG_RATELIMIT=0 to disable; "N" or "N:W_MS" to
// override (defaults N=10, W=1000).
#define TOCR_LOG_STRINGIZE_INNER(x) #x
#define TOCR_LOG_STRINGIZE(x) TOCR_LOG_STRINGIZE_INNER(x)

#define TOCR_LOG_RATELIMITED(level, msg, ...)                                      \
  do {                                                                             \
    static ::turbo_ocr::log::RateLimitSlot _tocr_rl_slot;                          \
    int _tocr_rl_drained = 0;                                                      \
    int _tocr_rl_pass = ::turbo_ocr::log::ratelimit_check(                         \
        _tocr_rl_slot, _tocr_rl_drained);                                          \
    if (_tocr_rl_drained > 0) {                                                    \
      ::turbo_ocr::log::log_msg(level, "[suppressed logs]",                        \
          "site", ::std::string_view(__FILE__ ":" TOCR_LOG_STRINGIZE(__LINE__)),   \
          "count", _tocr_rl_drained);                                              \
    }                                                                              \
    if (_tocr_rl_pass) {                                                           \
      ::turbo_ocr::log::log_msg(level, msg, ##__VA_ARGS__);                        \
    }                                                                              \
  } while (0)

#define TOCR_LOG_DEBUG_RL(msg, ...) TOCR_LOG_RATELIMITED(::turbo_ocr::log::Level::Debug, msg, ##__VA_ARGS__)
#define TOCR_LOG_INFO_RL(msg, ...)  TOCR_LOG_RATELIMITED(::turbo_ocr::log::Level::Info,  msg, ##__VA_ARGS__)
#define TOCR_LOG_WARN_RL(msg, ...)  TOCR_LOG_RATELIMITED(::turbo_ocr::log::Level::Warn,  msg, ##__VA_ARGS__)
#define TOCR_LOG_ERROR_RL(msg, ...) TOCR_LOG_RATELIMITED(::turbo_ocr::log::Level::Error, msg, ##__VA_ARGS__)
