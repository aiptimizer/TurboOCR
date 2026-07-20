#pragma once

#include <cctype>
#include <string>
#include <string_view>

namespace turbo_ocr {

[[nodiscard]] inline std::string_view trim_view(std::string_view s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

[[nodiscard]] inline std::string trim(std::string_view s) {
  return std::string(trim_view(s));
}

} // namespace turbo_ocr
