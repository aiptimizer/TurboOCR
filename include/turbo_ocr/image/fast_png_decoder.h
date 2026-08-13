#pragma once

#include <cstddef>

#include <opencv2/core.hpp>

namespace turbo_ocr::decode {

// Fast PNG decoder using Wuffs (Google, ~2.75x faster than libpng/OpenCV)
class FastPngDecoder {
public:
  // Decode PNG data to cv::Mat (BGR, 8-bit).
  // Returns empty Mat on failure.
  [[nodiscard]] static cv::Mat decode(const unsigned char *data, std::size_t len);

  // Check if data looks like PNG (magic bytes)
  [[nodiscard]] static bool is_png(const unsigned char *data, std::size_t len) noexcept {
    return len >= 8 && data[0] == 0x89 && data[1] == 'P' &&
           data[2] == 'N' && data[3] == 'G';
  }
};

} // namespace turbo_ocr::decode
