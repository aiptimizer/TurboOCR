#pragma once

// CUDA-free vocabulary shared by the JPEG decode paths (GPU decoder, routes,
// gRPC) so policy can be unit-tested without a device.

#include <cstddef>

namespace turbo_ocr::decode {

// Outcome of a JPEG decode on the GPU. The split is the whole point:
//  - Unsupported is a property of the BITSTREAM. nvJPEG's hardware path
//    decodes baseline and extended-sequential JPEG; progressive, arithmetic,
//    lossless, 12-bit and CMYK are outside it. Such an image may be decoded by
//    the host codec, and that is by specification, not a fallback.
//  - Failed is a device or runtime fault (allocator, execution, context). It
//    must surface as an error the client can see and retry. Decoding such an
//    image on the CPU instead would hide a broken GPU behind slower requests
//    and slightly different pixels, which is exactly what happened in v3.5.1.
enum class JpegDecodeStatus { Ok, Unsupported, Failed };

// nvjpegStatus_t values (nvjpeg.h), spelled out so this header stays
// CUDA-free. Anything not listed is treated as a fault.
namespace nvjpeg_status {
constexpr int kSuccess = 0;
constexpr int kBadJpeg = 3;
constexpr int kJpegNotSupported = 4;
constexpr int kIncompleteBitstream = 10;
} // namespace nvjpeg_status

[[nodiscard]] constexpr JpegDecodeStatus classify_nvjpeg_status(int st) noexcept {
  switch (st) {
    case nvjpeg_status::kSuccess: return JpegDecodeStatus::Ok;
    case nvjpeg_status::kBadJpeg:
    case nvjpeg_status::kJpegNotSupported:
    case nvjpeg_status::kIncompleteBitstream: return JpegDecodeStatus::Unsupported;
    default: return JpegDecodeStatus::Failed;
  }
}

[[nodiscard]] constexpr const char *to_string(JpegDecodeStatus s) noexcept {
  switch (s) {
    case JpegDecodeStatus::Ok: return "ok";
    case JpegDecodeStatus::Unsupported: return "unsupported";
    case JpegDecodeStatus::Failed: return "failed";
  }
  return "?";
}

// JPEG magic (SOI marker). The only sniff the routes need to pick a path.
[[nodiscard]] constexpr bool looks_like_jpeg(const unsigned char *data, size_t len) noexcept {
  return data != nullptr && len >= 2 && data[0] == 0xFF && data[1] == 0xD8;
}

} // namespace turbo_ocr::decode
