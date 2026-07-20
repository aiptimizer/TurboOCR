#pragma once

#include <format>
#include <string>

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/decode/image_config.h"

// The decompression-bomb size verdict: ONE predicate + ONE message source for
// every transport (HTTP routes, gRPC RPCs, batch slots). The per-transport
// emission (error_response callback, grpc::Status, throw, errors[i]) stays a
// thin adapter at the call site — but the classification order (dimension cap
// before pixel cap) and the wire text live only here, so they cannot drift
// between endpoints again.
namespace turbo_ocr::decode {

enum class ImageSizeVerdict { kOk, kDimTooLarge, kPixelsTooLarge };

[[nodiscard]] inline ImageSizeVerdict classify_image_size(int w, int h) {
  if (w > max_image_dim() || h > max_image_dim())
    return ImageSizeVerdict::kDimTooLarge;
  if (exceeds_pixel_cap(w, h)) return ImageSizeVerdict::kPixelsTooLarge;
  return ImageSizeVerdict::kOk;
}

[[nodiscard]] inline const char *
image_size_error_code(ImageSizeVerdict v) {
  return v == ImageSizeVerdict::kDimTooLarge ? "DIMENSIONS_TOO_LARGE"
                                             : "PIXELS_TOO_LARGE";
}

[[nodiscard]] inline std::string
image_size_error_message(ImageSizeVerdict v, int w, int h) {
  if (v == ImageSizeVerdict::kDimTooLarge)
    return std::format("Image dimensions {}x{} exceed maximum of {}x{}", w, h,
                       max_image_dim(), max_image_dim());
  return std::format("Image area {}x{} exceeds maximum of {} pixels", w, h,
                     max_image_pixels());
}

// Worker-thread adapter: dispatcher lambdas can't touch the HTTP/gRPC reply,
// so they surface the verdict as ImageTooLargeError (mapped to 400
// DIMENSIONS_TOO_LARGE by both transports' catch chains).
inline void throw_if_image_too_large(int w, int h) {
  auto v = classify_image_size(w, h);
  if (v != ImageSizeVerdict::kOk)
    throw turbo_ocr::ImageTooLargeError(image_size_error_message(v, w, h));
}

} // namespace turbo_ocr::decode
