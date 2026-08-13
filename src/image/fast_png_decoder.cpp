#include "turbo_ocr/image/fast_png_decoder.h"
#include "turbo_ocr/image/image_config.h"

#include <climits>

#include <opencv2/imgcodecs.hpp>

// Wuffs — Google's fastest PNG decoder (Apache 2.0 / MIT)
// Single-file C library, used in Chrome
#define WUFFS_IMPLEMENTATION
#define WUFFS_CONFIG__STATIC_FUNCTIONS
#include "../../third_party/wuffs/wuffs-v0.4.c"

using namespace turbo_ocr::decode;

namespace {
// The IHDR bit-depth byte lives at a fixed offset: 8-byte PNG signature +
// 4-byte chunk length + 4-byte "IHDR" type + 4-byte width + 4-byte height,
// i.e. offset 24 (color type follows at 25). A 16-bit-depth PNG (PIL mode
// 'I;16' grayscale, or 16-bit RGB/RGBA) reports bit_depth == 16 here. The
// signature check makes offset 24 meaningful for a non-PNG blob whose byte
// 24 happens to be 16 (self-contained; callers also gate on is_png).
[[nodiscard]] bool png_is_16bit(const unsigned char *data, std::size_t len) noexcept {
  return len >= 26 && FastPngDecoder::is_png(data, len) && data[24] == 16;
}

// IHDR width/height (big-endian u32 at offsets 16/20), for cap enforcement on
// paths that bypass the Wuffs config probe.
[[nodiscard]] uint32_t png_be32(const unsigned char *p) noexcept {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) |
         uint32_t(p[3]);
}

// OpenCV fallback for inputs the BGR-8 Wuffs fast path can't represent
// losslessly (16-bit depth). cv::imdecode(IMREAD_COLOR) routes through libpng,
// which strips 16->8 bit and emits an 8-bit BGR Mat just like the 8-bit path.
[[nodiscard]] cv::Mat opencv_png_decode(const unsigned char *data, std::size_t len) {
  if (len > static_cast<std::size_t>(INT_MAX))
    return {};
  return cv::imdecode(
      cv::Mat(1, static_cast<int>(len), CV_8UC1, const_cast<unsigned char *>(data)),
      cv::IMREAD_COLOR);
}
} // namespace

cv::Mat FastPngDecoder::decode(const unsigned char *data, std::size_t len) {
  // 16-bit-depth PNGs can't be decoded by the BGR-8 Wuffs fast path below
  // (it produces a blank Mat that would silently pass the empty() guard).
  // Route them through OpenCV/libpng, which downconverts 16->8 correctly —
  // but only after the same per-side and pixel-area caps the Wuffs path
  // enforces, read straight from the IHDR; otherwise a crafted 16-bit PNG
  // allocates its full declared raster inside cv::imdecode.
  if (png_is_16bit(data, len)) {
    const uint32_t w = png_be32(data + 16);
    const uint32_t h = png_be32(data + 20);
    const uint32_t cap = static_cast<uint32_t>(decode::max_image_dim());
    if (w == 0 || h == 0 || w > cap || h > cap)
      return {};
    if (decode::exceeds_pixel_cap(w, h))
      return {};
    return opencv_png_decode(data, len);
  }

  // 1. Initialize Wuffs PNG decoder
  wuffs_png__decoder dec;
  wuffs_base__status status = wuffs_png__decoder__initialize(
      &dec, sizeof(dec), WUFFS_VERSION, WUFFS_INITIALIZE__DEFAULT_OPTIONS);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  // 2. Set source
  wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader(
      const_cast<uint8_t *>(data), len, true);

  // 3. Read image config (dimensions, pixel format)
  wuffs_base__image_config ic;
  status = wuffs_png__decoder__decode_image_config(&dec, &ic, &src);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  uint32_t w = wuffs_base__pixel_config__width(&ic.pixcfg);
  uint32_t h = wuffs_base__pixel_config__height(&ic.pixcfg);

  // Track the configurable cap (env MAX_IMAGE_DIM) so valid in-spec PNGs
  // aren't falsely rejected when the operator raised the limit above 16384.
  const uint32_t cap = static_cast<uint32_t>(decode::max_image_dim());
  if (w == 0 || h == 0 || w > cap || h > cap)
    return {};
  // Pixel-AREA cap (MAX_IMAGE_PIXELS_MP): a solid-colour PNG can pass the
  // per-side cap yet still decode to hundreds of MB. Refuse before allocating
  // the cv::Mat below, so a decompression bomb never reaches the host buffer.
  if (decode::exceeds_pixel_cap(w, h))
    return {};

  // 4. Set up pixel buffer — decode directly to BGR (OpenCV CV_8UC3)
  wuffs_base__pixel_config pc;
  wuffs_base__pixel_config__set(&pc, WUFFS_BASE__PIXEL_FORMAT__BGR,
                                 WUFFS_BASE__PIXEL_SUBSAMPLING__NONE, w, h);

  // Allocate cv::Mat directly — Wuffs writes into it, zero extra copies
  cv::Mat bgr(h, w, CV_8UC3);
  wuffs_base__slice_u8 pixslice = {bgr.data, static_cast<size_t>(bgr.total() * bgr.elemSize())};

  wuffs_base__pixel_buffer pb;
  status = wuffs_base__pixel_buffer__set_from_slice(&pb, &pc, pixslice);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  // 5. Thread-local work buffer (reused across decodes, avoids per-call alloc)
  uint64_t workbuf_len = wuffs_png__decoder__workbuf_len(&dec).max_incl;
  thread_local std::vector<uint8_t> workbuf;
  if (workbuf.size() < workbuf_len)
    workbuf.resize(workbuf_len);
  wuffs_base__slice_u8 work = {workbuf.data(), workbuf.size()};

  // 6. Decode frame
  wuffs_base__frame_config fc;
  status = wuffs_png__decoder__decode_frame_config(&dec, &fc, &src);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  status = wuffs_png__decoder__decode_frame(&dec, &pb, &src,
      WUFFS_BASE__PIXEL_BLEND__SRC, work, nullptr);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  // 7. Already decoded to BGR cv::Mat — zero copies
  return bgr;
}
