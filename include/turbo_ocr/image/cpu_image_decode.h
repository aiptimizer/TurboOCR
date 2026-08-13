#pragma once

#include <climits>
#include <cstddef>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/image/fast_png_decoder.h"

namespace turbo_ocr::decode {

/// Does the byte prefix match a raster format we actually serve?
///
/// This is the surface-shrinking guard in front of cv::imdecode. OpenCV's
/// imgcodecs dispatches by content, and — depending on how it was built — links
/// exotic codecs the OCR server never wants: a fuzzer handed it a 142-byte blob
/// that OpenCV routed into GDCM (DICOM), which hit `assert(0 && "Should not
/// happen")` and abort()ed the process. No try/catch stops a C assert, so the
/// only defence is to not reach it. Decoding only PNG (handled upstream by
/// Wuffs) + JPEG/WebP/BMP/GIF/TIFF, and rejecting everything else, both closes
/// that path and makes decode deterministic across OpenCV builds instead of
/// depending on which codecs happen to be compiled in.
[[nodiscard]] inline bool is_supported_image_magic(const unsigned char *d,
                                                   size_t n) noexcept {
  if (d == nullptr) return false;
  auto has = [&](size_t k) { return n >= k; };
  // JPEG: FF D8 FF
  if (has(3) && d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF) return true;
  // BMP: "BM"
  if (has(2) && d[0] == 'B' && d[1] == 'M') return true;
  // GIF: "GIF87a" / "GIF89a"
  if (has(6) && d[0]=='G'&&d[1]=='I'&&d[2]=='F'&&d[3]=='8'&&
      (d[4]=='7'||d[4]=='9')&&d[5]=='a') return true;
  // TIFF: "II*\0" little-endian or "MM\0*" big-endian
  if (has(4) && ((d[0]=='I'&&d[1]=='I'&&d[2]==0x2A&&d[3]==0x00) ||
                 (d[0]=='M'&&d[1]=='M'&&d[2]==0x00&&d[3]==0x2A))) return true;
  // WebP: "RIFF" .... "WEBP"
  if (has(12) && d[0]=='R'&&d[1]=='I'&&d[2]=='F'&&d[3]=='F'&&
      d[8]=='W'&&d[9]=='E'&&d[10]=='B'&&d[11]=='P') return true;
  return false;
}

/// CPU decode for any supported image format: PNG via Wuffs (fast path);
/// every other format (JPEG, WebP, BMP, TIFF, GIF, …) via cv::imdecode.
/// OpenCV's imgcodecs is linked to libwebp / libtiff so it covers the rest.
/// Shared tail of the CPU decoder and the GPU decoder's nvJPEG fallback.
///
/// CONTRACT: returns an empty Mat on any undecodable input and NEVER throws.
/// Every caller relies on this — they test `.empty()` and answer
/// IMAGE_DECODE_FAILED — so it must hold for the whole untrusted byte range,
/// zero-length included. It did not: cv::imdecode(empty buffer) does not return
/// an empty Mat, it ASSERTS (`!buf.empty()`) and throws a cv::Exception, which
/// on the gRPC path — decode is called before the `.empty()` guard, not after —
/// propagated out of the handler as an uncaught exception. A fuzzer found it on
/// a zero-byte input in 24 iterations. Fixing it here rather than per-caller is
/// deliberate: one shared decoder, one guarantee, every transport covered.
[[nodiscard]] inline cv::Mat decode_cpu_fallback(const unsigned char *data,
                                                 size_t len) {
  if (data == nullptr || len == 0) return {};   // imdecode asserts on empty
  if (FastPngDecoder::is_png(data, len))
    return FastPngDecoder::decode(data, len);
  if (len > static_cast<size_t>(INT_MAX)) return {};
  // Only hand imdecode a format we serve. Anything else (DICOM, EXR, Sun
  // raster, …) returns empty here instead of reaching a codec that may abort.
  if (!is_supported_image_magic(data, len)) return {};
  // Even within the allowlist OpenCV can throw (a truncated-but-valid-magic
  // JPEG); catch so the contract holds for every byte sequence.
  try {
    return cv::imdecode(
        cv::Mat(1, static_cast<int>(len), CV_8UC1,
                const_cast<unsigned char *>(data)),
        cv::IMREAD_COLOR);
  } catch (const cv::Exception &) {
    return {};
  }
}

} // namespace turbo_ocr::decode
