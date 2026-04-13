#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "simdutf.h"

namespace turbo_ocr {

// SIMD-accelerated base64 decoder using simdutf (AVX2/AVX-512).
// ~5-10x faster than scalar 4-at-a-time loop on large payloads (images).
[[nodiscard]] inline std::string base64_decode(const char *in, size_t len) {
  // Strip trailing padding/whitespace
  while (len > 0 && (in[len - 1] == '=' || in[len - 1] == '\n' || in[len - 1] == '\r'))
    --len;

  size_t max_out = simdutf::maximal_binary_length_from_base64(in, len);
  std::string out(max_out, '\0');
  auto result = simdutf::base64_to_binary(in, len, out.data());
  if (result.error) return {};
  out.resize(result.count);
  return out;
}

// Convenience overload for std::string_view
[[nodiscard]] inline std::string base64_decode(std::string_view in) {
  return base64_decode(in.data(), in.size());
}

// Convenience overload for std::string (avoids ambiguity with string_view)
[[nodiscard]] inline std::string base64_decode(const std::string &in) {
  return base64_decode(in.data(), in.size());
}

} // namespace turbo_ocr
