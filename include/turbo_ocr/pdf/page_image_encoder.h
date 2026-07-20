#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace turbo_ocr::pdf {

/// Image formats supported for page image encoding.
enum class PageImageFormat {
  Jpeg,
  Png,
  WebP,
};

/// Parse format string (case-insensitive). Returns Png on unknown.
PageImageFormat parse_page_image_format(const char *s) noexcept;

/// Strict request-parsing companion: true iff `s` names a supported format.
/// Route handlers 400 on explicit unknown values instead of coercing to Png.
bool is_valid_page_image_format(const char *s) noexcept;

/// Human-readable format name used in Content-Type and URL paths.
const char *page_image_format_name(PageImageFormat fmt) noexcept;
const char *page_image_content_type(PageImageFormat fmt) noexcept;

struct EncodeOptions {
  PageImageFormat format   = PageImageFormat::Png;
  // PNG is always lossless. `lossless` only affects WebP (default true there too).
  bool            lossless = true;
  int             quality  = 85;   // Used for JPEG and lossy WebP. Ignored for PNG.
  // PNG compression level: 0 = no compression (largest, fastest),
  // 9 = max compression (smallest, slowest). 3 = good speed-size sweet spot.
  int             png_compression = 3;
  int             max_side = 0;    // 0 = no resize; >0 = fit within max_side px
};

/// Encode a BGR cv::Mat to compressed bytes.
/// Uses libjpeg-turbo for JPEG (fastest path), OpenCV imencode for PNG/WebP.
/// Returns empty vector on failure.
[[nodiscard]] std::vector<uint8_t>
encode_page_image(const cv::Mat &bgr, const EncodeOptions &opts);

} // namespace turbo_ocr::pdf
