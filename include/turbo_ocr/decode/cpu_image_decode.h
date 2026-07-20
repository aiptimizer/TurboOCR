#pragma once

#include <climits>
#include <cstddef>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/decode/fast_png_decoder.h"

namespace turbo_ocr::decode {

/// CPU decode for any supported image format: PNG via Wuffs (fast path);
/// every other format (JPEG, WebP, BMP, TIFF, GIF, …) via cv::imdecode.
/// OpenCV's imgcodecs is linked to libwebp / libtiff so it covers the rest.
/// Shared tail of the CPU decoder and the GPU decoder's nvJPEG fallback.
[[nodiscard]] inline cv::Mat decode_cpu_fallback(const unsigned char *data,
                                                 size_t len) {
  if (FastPngDecoder::is_png(data, len))
    return FastPngDecoder::decode(data, len);
  if (len > static_cast<size_t>(INT_MAX)) return {};
  return cv::imdecode(
      cv::Mat(1, static_cast<int>(len), CV_8UC1,
              const_cast<unsigned char *>(data)),
      cv::IMREAD_COLOR);
}

} // namespace turbo_ocr::decode
