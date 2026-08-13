#pragma once

#include <cassert>
#include <cstdio>
#include <format>
#include <string>

#include "turbo_ocr/base/errors.h"
#include "turbo_ocr/image/image_config.h"

// The decompression-bomb size verdict: ONE predicate + ONE message source for
// every transport (HTTP routes, gRPC RPCs, batch slots). The per-transport
// emission (error_response callback, grpc::Status, throw, errors[i]) stays a
// thin adapter at the call site — but the classification order (dimension cap
// before pixel cap) and the wire text live only here, so they cannot drift
// between endpoints again.
namespace turbo_ocr::decode {

enum class ImageSizeVerdict { kOk, kDimTooLarge, kPixelsTooLarge };

// Order is load-bearing and not merely stylistic: an image over BOTH caps must
// report DIMENSIONS_TOO_LARGE, so the per-side test comes first. Swapping the
// two ifs silently changes the wire code every transport emits for that input.
[[nodiscard]] inline ImageSizeVerdict classify_image_size(int w, int h) {
  if (w > max_image_dim() || h > max_image_dim())
    return ImageSizeVerdict::kDimTooLarge;
  if (exceeds_pixel_cap(w, h)) return ImageSizeVerdict::kPixelsTooLarge;
  return ImageSizeVerdict::kOk;
}

namespace detail {
// kOk reaching an error FORMATTER is a caller bug: the verdict must be branched
// on (or `!= kOk`-guarded) before asking for the wire code/text. Both helpers
// used to be bare ternaries whose else-branch answered "PIXELS_TOO_LARGE" for
// kOk — a plausible-looking wrong answer that would ship a bomb rejection for
// an image that passed. Loud like backend/kernels.h's report_unhonoured:
// asserts in debug so a test can never miss it, logs in release so the
// divergence shows up in the log rather than in a client's error handler.
inline void report_ok_verdict_misuse(const char *fn) {
  std::fprintf(stderr,
               "[size_classify] %s called with ImageSizeVerdict::kOk — the "
               "caller must check the verdict before formatting an error.\n",
               fn);
  assert(false && "image size error formatter called with kOk — see stderr");
}
} // namespace detail

// Wire error code for a rejecting verdict. Never called with kOk (see above);
// if it is, the release-build fallback is the empty string — an obviously
// absent code a client/log can diagnose, and safe to assign to the
// std::string error_code fields the callers hold (a nullptr there would be UB).
[[nodiscard]] inline const char *
image_size_error_code(ImageSizeVerdict v) {
  switch (v) {
  case ImageSizeVerdict::kDimTooLarge:
    return "DIMENSIONS_TOO_LARGE";
  case ImageSizeVerdict::kPixelsTooLarge:
    return "PIXELS_TOO_LARGE";
  case ImageSizeVerdict::kOk:
    break;
  }
  detail::report_ok_verdict_misuse("image_size_error_code");
  return "";
}

[[nodiscard]] inline std::string
image_size_error_message(ImageSizeVerdict v, int w, int h) {
  switch (v) {
  case ImageSizeVerdict::kDimTooLarge:
    return std::format("Image dimensions {}x{} exceed maximum of {}x{}", w, h,
                       max_image_dim(), max_image_dim());
  case ImageSizeVerdict::kPixelsTooLarge:
    return std::format("Image area {}x{} exceeds maximum of {} pixels", w, h,
                       max_image_pixels());
  case ImageSizeVerdict::kOk:
    break;
  }
  detail::report_ok_verdict_misuse("image_size_error_message");
  return {};
}

// Per-SLOT wire text for a batch response, which is a different contract from
// the two above: /ocr/batch keeps the response array aligned with images[] and
// tags a failed slot with a snake_case string carrying the measured dims and the
// cap it broke (docs/reference/http.md documents the exact shape). Every batch stage
// — the pre-decode header sniff, the post-decode net, the nvJPEG header gate —
// must emit the SAME string for the same verdict; they used to spell it
// themselves and the nvJPEG gate drifted to a bare tag with no detail at all.
[[nodiscard]] inline std::string
image_size_slot_error(ImageSizeVerdict v, int w, int h) {
  switch (v) {
  case ImageSizeVerdict::kDimTooLarge:
    return std::format("dimensions_too_large ({}x{} > {}x{})", w, h,
                       max_image_dim(), max_image_dim());
  case ImageSizeVerdict::kPixelsTooLarge:
    return std::format("pixels_too_large ({}x{} > {} px)", w, h,
                       max_image_pixels());
  case ImageSizeVerdict::kOk:
    break;
  }
  detail::report_ok_verdict_misuse("image_size_slot_error");
  return {};
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
