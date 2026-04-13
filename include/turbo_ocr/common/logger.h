#pragma once

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
      parse_level(std::getenv("LOG_LEVEL")),
      parse_format(std::getenv("LOG_FORMAT")),
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
  int n;
  // comma + key
  n = std::snprintf(p, rem, ","); p += n; rem -= static_cast<size_t>(n);
  n = json_string(p, rem, key); p += n; rem -= static_cast<size_t>(n);
  n = std::snprintf(p, rem, ":"); p += n; rem -= static_cast<size_t>(n);
  n = append_json_value(p, rem, std::forward<V>(val)); p += n; rem -= static_cast<size_t>(n);
  write_json_kvs(p, rem, std::forward<Rest>(rest)...);
}

inline void write_text_kvs(char *&, size_t &) {}

template <typename V, typename... Rest>
void write_text_kvs(char *&p, size_t &rem, std::string_view key, V &&val, Rest &&...rest) {
  int n;
  n = std::snprintf(p, rem, " %.*s=", static_cast<int>(key.size()), key.data());
  p += n; rem -= static_cast<size_t>(n);
  n = append_text_value(p, rem, std::forward<V>(val)); p += n; rem -= static_cast<size_t>(n);
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
  thread_local char buf[4096];
  char *p = buf;
  size_t rem = sizeof(buf) - 2;  // reserve for \n\0
  int n;

  if (config().format == Format::Json) {
    // {"ts":"...","level":"...","msg":"...",...}\n
    n = std::snprintf(p, rem, "{\"ts\":\""); p += n; rem -= static_cast<size_t>(n);
    n = format_timestamp_iso(p, rem); p += n; rem -= static_cast<size_t>(n);
    n = std::snprintf(p, rem, "\",\"level\":\"%s\",\"msg\":", level_name_json(lvl));
    p += n; rem -= static_cast<size_t>(n);
    n = detail::json_string(p, rem, msg); p += n; rem -= static_cast<size_t>(n);
    detail::write_json_kvs(p, rem, std::forward<KVs>(kvs)...);
    *p++ = '}';
  } else {
    // [2026-04-13 10:00:00.123] [INFO] message key=val ...\n
    *p++ = '['; rem--;
    n = format_timestamp_text(p, rem); p += n; rem -= static_cast<size_t>(n);
    n = std::snprintf(p, rem, "] [%s] %.*s", level_name_text(lvl),
        static_cast<int>(msg.size()), msg.data());
    p += n; rem -= static_cast<size_t>(n);
    detail::write_text_kvs(p, rem, std::forward<KVs>(kvs)...);
  }
  *p++ = '\n';
  *p = '\0';

  size_t len = static_cast<size_t>(p - buf);
  std::lock_guard<std::mutex> lock(log_mutex());
  std::fwrite(buf, 1, len, stderr);
}

} // namespace turbo_ocr::log

// ── Convenience macros ─────────────────────────────────────────────────

#define LOG_DEBUG(msg, ...) ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Debug, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...)  ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Info,  msg, ##__VA_ARGS__)
#define LOG_WARN(msg, ...)  ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Warn,  msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) ::turbo_ocr::log::log_msg(::turbo_ocr::log::Level::Error, msg, ##__VA_ARGS__)
