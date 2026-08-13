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

/// An optional device JPEG encoder. Returns the encoded bytes, or an EMPTY
/// vector to mean "not this time" — encode_page_image then falls back to
/// libjpeg-turbo, so a device that is busy, absent or unhappy costs a branch
/// rather than a failed request.
using JpegEncodeHook = std::vector<uint8_t> (*)(const cv::Mat &bgr, int quality);

/// Install the device JPEG encoder. Call once, before serving.
///
/// WHY A HOOK AND NOT AN #include: this TU used to `#include
/// "nvidia/support/nvjpeg_encoder.h"` under `#ifndef USE_CPU_ONLY`, which was
/// the one place in the tree where a file outside src/backends/ named a vendor
/// arm — the exact rule src/README.md states and tools/checks/architecture.sh
/// exists to hold. It is also the same shape as the bug that check was written
/// for (a device-neutral header hard-wired to one vendor behind a legacy
/// build flag), and it would have had to be undone anyway the moment a second
/// vendor grew a hardware JPEG encoder.
///
/// The NVIDIA arm installs nvJPEG at backend load; every other build leaves it
/// null and encodes on the host, which is what those builds already did.
void set_jpeg_encode_hook(JpegEncodeHook hook) noexcept;

} // namespace turbo_ocr::pdf
