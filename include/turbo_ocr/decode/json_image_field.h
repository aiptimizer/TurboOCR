#pragma once

// Locate the `image` member of a `{"image": "<base64>", ...}` request body
// without building a JSON document, so the base64 text is decoded straight
// from the request buffer instead of being copied into a Json::Value first
// (for a multi-MB image that copy was the largest transient allocation on
// the route after the pixels themselves).
//
// Deliberately narrow: it understands exactly the top-level object shape the
// /ocr route accepts and refuses anything it is not sure about (escapes inside
// the image string, malformed syntax), in which case the caller falls back to
// the full JSON parser. Refusing is always safe; misreading never happens.

#include <cstddef>
#include <optional>
#include <string_view>

namespace turbo_ocr::decode {

struct JsonImageField {
  std::string_view base64;   // the image string's content (no quotes, no escapes)
  bool has_routing = false;  // a top-level "routing" member exists
};

namespace json_scan_detail {

constexpr void skip_ws(std::string_view s, size_t &i) noexcept {
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

// Skip a JSON string starting at the opening quote. On success `i` points
// past the closing quote and the content span is returned; `escaped` tells
// whether the content contains any backslash escape.
constexpr std::optional<std::string_view> skip_string(std::string_view s, size_t &i,
                                                      bool &escaped) noexcept {
  if (i >= s.size() || s[i] != '"') return std::nullopt;
  const size_t start = ++i;
  escaped = false;
  while (i < s.size()) {
    const char c = s[i];
    if (c == '"') {
      auto out = s.substr(start, i - start);
      ++i;
      return out;
    }
    if (c == '\\') {
      escaped = true;
      i += 2;  // skip the escaped character (a \uXXXX still advances safely below)
      continue;
    }
    ++i;
  }
  return std::nullopt;  // unterminated
}

// Skip any JSON value (object/array nesting included). Returns false on
// malformed input.
constexpr bool skip_value(std::string_view s, size_t &i) noexcept {
  skip_ws(s, i);
  if (i >= s.size()) return false;
  const char c = s[i];
  if (c == '"') {
    bool esc = false;
    return skip_string(s, i, esc).has_value();
  }
  if (c == '{' || c == '[') {
    int depth = 0;
    bool in_str = false;
    for (; i < s.size(); ++i) {
      const char d = s[i];
      if (in_str) {
        if (d == '\\') { ++i; continue; }
        if (d == '"') in_str = false;
        continue;
      }
      if (d == '"') in_str = true;
      else if (d == '{' || d == '[') ++depth;
      else if (d == '}' || d == ']') { if (--depth == 0) { ++i; return true; } }
    }
    return false;
  }
  // number / true / false / null: run to the next delimiter
  while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
         s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r')
    ++i;
  return true;
}

} // namespace json_scan_detail

// nullopt when the body is not a plain top-level object with a string
// `image` member free of escapes; the caller then uses the full parser.
[[nodiscard]] constexpr std::optional<JsonImageField>
find_json_image_field(std::string_view body) noexcept {
  using namespace json_scan_detail;
  size_t i = 0;
  skip_ws(body, i);
  if (i >= body.size() || body[i] != '{') return std::nullopt;
  ++i;
  JsonImageField out;
  bool found = false;
  for (;;) {
    skip_ws(body, i);
    if (i >= body.size()) return std::nullopt;
    if (body[i] == '}') break;
    bool key_escaped = false;
    auto key = skip_string(body, i, key_escaped);
    if (!key || key_escaped) return std::nullopt;
    skip_ws(body, i);
    if (i >= body.size() || body[i] != ':') return std::nullopt;
    ++i;
    skip_ws(body, i);
    if (*key == "image") {
      if (found) return std::nullopt;  // duplicate key: let the parser decide
      bool esc = false;
      auto v = skip_string(body, i, esc);
      if (!v || esc || v->empty()) return std::nullopt;
      out.base64 = *v;
      found = true;
    } else {
      if (*key == "routing") out.has_routing = true;
      if (!skip_value(body, i)) return std::nullopt;
    }
    skip_ws(body, i);
    if (i >= body.size()) return std::nullopt;
    if (body[i] == ',') { ++i; continue; }
    if (body[i] == '}') break;
    return std::nullopt;
  }
  ++i;
  skip_ws(body, i);
  if (i != body.size()) return std::nullopt;  // trailing garbage
  if (!found) return std::nullopt;
  return out;
}

} // namespace turbo_ocr::decode
